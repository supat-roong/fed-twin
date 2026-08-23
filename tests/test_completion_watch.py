"""
P3 Task 4: MinIO-based completion detection for topology='multi' workers.

The host cannot reliably watch a pod Karmada has propagated to a member
cluster (see src/orchestration/dispatch.py), but train_worker.py already
uploads weights, then a delta, then a metrics JSON per worker per round
(_push_weights) -- and the aggregator (src/aggregator/collect.py) already
treats the metrics object's presence as "this worker is done". These tests
cover wait_for_worker_artifact, the function that turns that same signal into
launch_and_watch_pod's multi-cluster completion check.

The subtlety that matters: weights land *before* metrics, so only the
metrics key may ever be read as completion -- see
test_does_not_treat_the_weights_object_alone_as_completion below.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from minio.error import S3Error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.activities import WorkerJobFailed, wait_for_worker_artifact

FL_ROUND = 4
WORKER_ID = 1
BUCKET = "bucket"
METRICS_KEY = f"round_{FL_ROUND}/workers/worker_{WORKER_ID}_metrics.json"
WEIGHTS_KEY = f"round_{FL_ROUND}/workers/worker_{WORKER_ID}_weights.pt"


def _s3_error(code: str) -> S3Error:
    return S3Error(
        response=None, code=code, message="test", resource=f"/{BUCKET}/x",
        request_id="req-1", host_id="host-1",
    )


class FakeMinioClient:
    """Scripted stand-in for minio.Minio's stat_object, per test.

    - `keys`: object keys considered "present", but only once the call count
      reaches `present_from_call` (default: present from the very first
      call) -- lets a test simulate an object landing after N polls.
    - `error_calls`: {call_number: S3Error code | Exception} to raise instead
      of the normal present/absent check on that specific call. A str value
      is treated as an S3Error code (transient, non-"not found" errors); an
      Exception instance is raised as-is, for scripting connection-level
      errors (Finding 4) that never reach the S3Error layer at all.
    - `last_modified`: {key: datetime} overriding the default (now, UTC) a
      present object reports as its last_modified -- lets a test plant an
      object that predates a given `not_before` cutoff (Finding 3).
    """

    def __init__(
        self, keys=frozenset(), present_from_call=1, error_calls=None, last_modified=None
    ):
        self.keys = set(keys)
        self.present_from_call = present_from_call
        self.error_calls = dict(error_calls or {})
        self.last_modified = dict(last_modified or {})
        self.calls = 0
        self.queried_keys: list[str] = []

    def stat_object(self, bucket, key):
        self.calls += 1
        self.queried_keys.append(key)
        if self.calls in self.error_calls:
            scripted = self.error_calls[self.calls]
            raise _s3_error(scripted) if isinstance(scripted, str) else scripted
        if key in self.keys and self.calls >= self.present_from_call:
            lm = self.last_modified.get(key, datetime.now(timezone.utc))
            return SimpleNamespace(object_name=key, bucket_name=bucket, last_modified=lm)
        raise _s3_error("NoSuchKey")


def _patch_heartbeat(monkeypatch):
    """wait_for_worker_artifact calls activity.heartbeat() every failed poll.
    These tests call it directly (not through a live Temporal activity), so
    without this, any test that polls more than once would hit
    `RuntimeError: Not in activity context`. Returns the list of recorded
    calls so tests can assert heartbeat cadence."""
    import src.orchestration.activities as activities_module

    calls: list[tuple] = []
    monkeypatch.setattr(
        activities_module.activity, "heartbeat", lambda *a, **k: calls.append((a, k))
    )
    return calls


async def test_returns_true_as_soon_as_the_metrics_object_appears(monkeypatch):
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={METRICS_KEY})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01
    )

    assert result is True
    assert client.calls == 1  # succeeded on the very first poll -- no waiting


async def test_polls_the_exact_metrics_key_train_worker_uploads(monkeypatch):
    # Regression guard for the key format itself, matching train_worker.py's
    # _push_weights and collect.py's collect_worker_updates.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={METRICS_KEY})

    await wait_for_worker_artifact(client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01)

    assert client.queried_keys == [METRICS_KEY]


async def test_raises_timeout_error_instead_of_returning_false(monkeypatch):
    # THE constraint: Temporal only retries an activity on a *raised*
    # exception (see WorkerJobFailed's docstring in activities.py). A MinIO-
    # completion timeout that returned False instead of raising would
    # silently turn RetryPolicy(maximum_attempts=3) into a single attempt for
    # every multi-cluster worker whose artifact never lands -- the exact bug
    # that shipped once already for the single-topology Job-watch path.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys=set())  # metrics object never shows up

    with pytest.raises(TimeoutError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=0.03, poll_s=0.01
        )


async def test_heartbeats_on_every_failed_poll(monkeypatch):
    calls = _patch_heartbeat(monkeypatch)
    # Present only from the 4th call onward: 3 failed polls (each must
    # heartbeat) before the 4th call succeeds without an additional one.
    client = FakeMinioClient(keys={METRICS_KEY}, present_from_call=4)

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.001
    )

    assert result is True
    assert client.calls == 4
    assert len(calls) == 3


async def test_tolerates_a_transient_s3_error_and_keeps_polling(monkeypatch):
    _patch_heartbeat(monkeypatch)
    # Calls 1-2 raise a transient (non-"not found") S3Error; call 3 finds the
    # object. A loop that aborted on the first non-NoSuchKey S3Error would
    # never reach that successful 3rd call.
    client = FakeMinioClient(keys={METRICS_KEY}, error_calls={1: "InternalError", 2: "SlowDown"})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.001
    )

    assert result is True
    assert client.calls == 3


# ---------------------------------------------------------------------------
# Finding 4 (p3-task-4-review.md): the transient-error tolerance above only
# catches S3Error, but a MinIO pod restart or network blip raises a
# connection-level urllib3 exception that never reaches the S3 protocol
# layer at all. Verified against the installed minio==7.2.20: pointing a
# real Minio client at a refused connection raises
# urllib3.exceptions.MaxRetryError, not S3Error. ProtocolError (e.g. a
# connection dropped mid-response) is the other realistic "network blip"
# shape. Deliberately not testing a bare Exception/AttributeError here being
# tolerated -- the whole point is that those must still propagate.
# ---------------------------------------------------------------------------


async def test_tolerates_a_connection_level_error_and_keeps_polling(monkeypatch):
    from urllib3.exceptions import MaxRetryError

    _patch_heartbeat(monkeypatch)
    conn_error = MaxRetryError(pool=None, url="http://minio:9000/bucket/key", reason=None)
    # Call 1 raises the connection-level error (a MinIO pod restart or
    # network blip); call 2 finds the object. A loop that only caught
    # S3Error would let this escape on the very first poll.
    client = FakeMinioClient(keys={METRICS_KEY}, error_calls={1: conn_error})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.001
    )

    assert result is True
    assert client.calls == 2


async def test_tolerates_a_urllib3_protocol_error_and_keeps_polling(monkeypatch):
    from urllib3.exceptions import ProtocolError

    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(
        keys={METRICS_KEY}, error_calls={1: ProtocolError("Connection aborted.")}
    )

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.001
    )

    assert result is True
    assert client.calls == 2


async def test_does_not_swallow_a_genuine_bug_in_the_poll_loop(monkeypatch):
    # THE guardrail: catching connection-level errors must not widen into
    # catching bare Exception. An AttributeError (or any other real bug) in
    # this loop must still propagate -- swallowing it here already cost this
    # project a debugging session once (see p3-task-4-review.md, Finding 4).
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={METRICS_KEY}, error_calls={1: AttributeError("boom")})

    with pytest.raises(AttributeError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.001
        )


async def test_connection_level_errors_still_respect_the_overall_timeout(monkeypatch):
    from urllib3.exceptions import MaxRetryError

    # If every single poll hits a connection error and the artifact never
    # appears, the loop must still bail out at timeout_s -- tolerance must
    # not turn into an unbounded retry.
    _patch_heartbeat(monkeypatch)

    class AlwaysConnectionError:
        def __init__(self):
            self.calls = 0

        def stat_object(self, bucket, key):
            self.calls += 1
            raise MaxRetryError(pool=None, url="http://minio:9000/x", reason=None)

    client = AlwaysConnectionError()

    with pytest.raises(TimeoutError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=0.03, poll_s=0.01
        )

    assert client.calls > 0


async def test_does_not_treat_the_weights_object_alone_as_completion(monkeypatch):
    # THE subtlety: train_worker.py's _push_weights uploads weights, then the
    # delta, then the metrics JSON, in that order -- so only the metrics
    # object's appearance means "done". Treating worker_i_weights.pt as
    # completion would read a partial (weights-uploaded-but-not-finished)
    # worker as a success.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={WEIGHTS_KEY})  # weights present, metrics never uploaded

    with pytest.raises(TimeoutError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=0.03, poll_s=0.01
        )

    # It must be the metrics key specifically being polled for -- never
    # satisfied merely because the weights key happens to already exist.
    assert client.queried_keys
    assert all(k == METRICS_KEY for k in client.queried_keys)


# ---------------------------------------------------------------------------
# Finding 2 (p3-task-4-review.md): a crashed multi-cluster worker was only
# ever detected via the full timeout_s, because wait_for_worker_artifact had
# no way to distinguish "crashed" from "still training". failure_check is an
# optional zero-arg async callable, polled once per iteration alongside the
# MinIO check: returning a diagnostic string means "terminally failed, stop
# waiting now"; returning None means "no news" and must never raise -- that
# covers the absent/lagging Karmada-status case explicitly (production wires
# this to _karmada_terminal_failure; these tests exercise the generic
# contract with a plain async callable, keeping wait_for_worker_artifact
# itself ignorant of Karmada).
# ---------------------------------------------------------------------------


async def test_raises_immediately_when_failure_check_reports_terminal_failure(monkeypatch):
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys=set())  # metrics object never appears

    async def failed_check():
        return "Job aflw-xxx-r4-w1 is terminally Failed"

    with pytest.raises(WorkerJobFailed) as exc_info:
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01,
            failure_check=failed_check,
        )

    assert "terminally Failed" in str(exc_info.value)
    # The whole point: raised on (about) the first poll, nowhere near
    # timeout_s=5 / poll_s=0.01's ~500 iterations.
    assert client.calls <= 2


async def test_absent_or_lagging_failure_check_never_short_circuits_the_wait(monkeypatch):
    # THE safety property: right after dispatch, before Karmada has
    # propagated the Job, there is no aggregated status yet -- absent means
    # "not yet", never "failed". A failure_check that always reports "no
    # news" (absent, lagging, or genuinely still running) must never raise;
    # the wait must still time out normally once the artifact really never
    # appears, exactly as it did before failure_check existed.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys=set())

    async def never_fails():
        return None

    with pytest.raises(TimeoutError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=0.03, poll_s=0.01,
            failure_check=never_fails,
        )


async def test_failure_check_is_optional_and_defaults_to_never_firing(monkeypatch):
    # Backward compatibility: every test above (and every pre-Finding-2
    # caller) invokes wait_for_worker_artifact without failure_check at all.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={METRICS_KEY})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01
    )

    assert result is True


# ---------------------------------------------------------------------------
# Finding 3 (p3-task-4-review.md): the completion key
# (round_{fl_round}/workers/worker_{worker_id}_metrics.json) has no run
# identifier, so a round resumed via run_pipeline.py's --bucket can find a
# *previous*, abandoned attempt's metrics object and report a brand-new Job
# as instantly succeeded. not_before is the fix: an object whose
# last_modified predates it (beyond a small clock-skew tolerance) is treated
# as if it weren't there. This is the reviewer's own repro
# (pre-seed the object before the Job is created; the wait returned True
# immediately) turned into a real test.
# ---------------------------------------------------------------------------


async def test_ignores_a_stale_artifact_pre_seeded_before_this_attempt(monkeypatch):
    # THE repro: the metrics object already exists -- e.g. left over from a
    # previous, abandoned attempt at this same round/worker -- *before* this
    # attempt's Job is even created. not_before is captured at that point (in
    # production, right before dispatch). A stale object must never satisfy
    # the wait; it must behave exactly as if the key were absent.
    _patch_heartbeat(monkeypatch)
    stale_time = datetime.now(timezone.utc) - timedelta(hours=1)
    client = FakeMinioClient(keys={METRICS_KEY}, last_modified={METRICS_KEY: stale_time})
    not_before = datetime.now(timezone.utc)

    with pytest.raises(TimeoutError):
        await wait_for_worker_artifact(
            client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=0.03, poll_s=0.01,
            not_before=not_before,
        )

    # It must actually have been polling for the key the whole time, not
    # failing for some unrelated reason.
    assert client.calls > 0


async def test_accepts_a_fresh_artifact_at_or_after_not_before(monkeypatch):
    _patch_heartbeat(monkeypatch)
    not_before = datetime.now(timezone.utc)
    fresh_time = not_before + timedelta(seconds=1)
    client = FakeMinioClient(keys={METRICS_KEY}, last_modified={METRICS_KEY: fresh_time})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01, not_before=not_before,
    )

    assert result is True


async def test_accepts_an_artifact_within_clock_skew_tolerance(monkeypatch):
    # A tiny bit *before* not_before must still count as fresh -- this
    # process's clock and the MinIO server's clock are two different
    # machines, and an exact >= comparison would be one NTP hiccup away from
    # rejecting a perfectly legitimate, brand-new object.
    _patch_heartbeat(monkeypatch)
    not_before = datetime.now(timezone.utc)
    barely_before = not_before - timedelta(seconds=1)
    client = FakeMinioClient(keys={METRICS_KEY}, last_modified={METRICS_KEY: barely_before})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01, not_before=not_before,
    )

    assert result is True


async def test_not_before_is_optional_and_defaults_to_no_freshness_check(monkeypatch):
    # Backward compatibility: every pre-Finding-3 caller (and every test
    # above that doesn't pass not_before) never provided a last_modified at
    # all in some fakes -- the freshness check must be entirely skippable.
    _patch_heartbeat(monkeypatch)
    client = FakeMinioClient(keys={METRICS_KEY})

    result = await wait_for_worker_artifact(
        client, BUCKET, FL_ROUND, WORKER_ID, timeout_s=5, poll_s=0.01
    )

    assert result is True
