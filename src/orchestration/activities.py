"""
Temporal activities: the only orchestration code that touches Kubernetes.

Workflow code must be deterministic because Temporal replays it, so every
side effect lives here. `build_job_manifest` and `job_name_for` are pure and
separately testable; the activities themselves are thin wrappers around the
Kubernetes API plus heartbeating.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from temporalio import activity

from src.orchestration.types import WorkerResult, WorkerSpec

log = logging.getLogger(__name__)

# How long a worker pod may run before we give up on it.
POD_WATCH_TIMEOUT_S = 3600
# How often to poll pod state and emit a heartbeat.
POLL_INTERVAL_S = 5
# Tolerance for clock skew between this
# process and the MinIO server's own clock -- last_modified is stamped by
# MinIO when it receives the upload, not by the worker pod, so what matters
# is skew between this process and the MinIO server, not the worker.
# NTP-synced hosts in the same (or a nearby) cluster typically drift well
# under a second; 5s is generous headroom for that jitter while still being
# tiny next to how old a genuinely stale object from a previous, abandoned
# attempt would be (at least one full training round -- minutes, not
# seconds).
_CLOCK_SKEW_TOLERANCE_S = 5.0

# A Kubernetes object name segment: RFC-1123 lowercase alphanumerics and '-'.
# kfp_run_id is truncated to 8 chars before use, so this only needs to accept
# that fragment, not a full name.
_VALID_RUN_ID_FRAGMENT = re.compile(r"[a-z0-9]+")


def job_name_for(spec: WorkerSpec) -> str:
    """Deterministic Job name.

    Deterministic on purpose: a retried activity must re-attach to the existing
    Job rather than launch a second fleet. The previous implementation embedded
    uuid4() and raced 2N workers onto the same MinIO keys on retry.

    Validates the run-id fragment before building the name.
    A KFP deployment that never substitutes dsl.PIPELINE_JOB_ID_PLACEHOLDER
    previously let the literal string "{{$.pipeline_job_uuid}}" flow straight
    through into this f-string, producing "ftwn-{{$.pipe-r0-w0" -- a name
    Kubernetes' own API rejects outright (422, not a lowercase RFC 1123
    subdomain), but only after a Job create call, and identically on every one
    of Temporal's retries. Rejecting `{`, `}`, `$`, `.` and uppercase here,
    synchronously and before any I/O, turns that into an immediate, clear
    ValueError at the one place this value is turned into a Kubernetes name.
    """
    fragment = spec.kfp_run_id[:8]
    if not _VALID_RUN_ID_FRAGMENT.fullmatch(fragment):
        raise ValueError(
            f"kfp_run_id must be lowercase alphanumeric to form a valid Kubernetes "
            f"name; got {spec.kfp_run_id!r} (an unsubstituted KFP placeholder like "
            f"'{{{{$.pipeline_job_uuid}}}}' looks like this -- pass an explicit "
            f"run_uid instead of dsl.PIPELINE_JOB_ID_PLACEHOLDER)"
        )
    return f"ftwn-{fragment}-r{spec.fl_round}-w{spec.worker_id}"


def build_job_manifest(spec: WorkerSpec) -> dict:
    """Render the batch/v1 Job for one worker. Pure — no I/O."""
    name = job_name_for(spec)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": spec.namespace,
            "labels": {
                "app": "fed-twin-worker",
                "fl-round": str(spec.fl_round),
                "worker-id": str(spec.worker_id),
            },
        },
        "spec": {
            # Temporal owns retry entirely. A non-zero backoffLimit would let
            # the Job controller silently replace a failed/deleted pod under
            # its own umbrella, invisible to launch_and_watch_pod's poll.
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "fed-twin-worker",
                        "fl-round": str(spec.fl_round),
                        "worker-id": str(spec.worker_id),
                    }
                },
                "spec": {
                    # Never, not OnFailure: with Temporal owning retry,
                    # restarting the container in place would mask a crash
                    # from launch_and_watch_pod's Job-level poll the same way
                    # a non-zero backoffLimit would.
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "worker",
                            "image": spec.worker_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-m", "src.core.worker_entrypoint"],
                            "env": [
                                {"name": "RANK", "value": str(spec.worker_id)},
                                {"name": "FL_ROUND", "value": str(spec.fl_round)},
                                {"name": "LOCAL_EPISODES", "value": str(spec.local_episodes)},
                                {"name": "EVAL_EPISODES", "value": str(spec.eval_episodes)},
                                {"name": "MINIO_ENDPOINT", "value": spec.minio_endpoint},
                                {"name": "MINIO_ACCESS_KEY", "value": spec.minio_access_key},
                                {"name": "MINIO_SECRET_KEY", "value": spec.minio_secret_key},
                                {"name": "MINIO_BUCKET", "value": spec.minio_bucket},
                                {"name": "MLFLOW_TRACKING_URI", "value": spec.mlflow_tracking_uri},
                                {
                                    "name": "MLFLOW_EXPERIMENT_NAME",
                                    "value": spec.mlflow_experiment_name,
                                },
                            ],
                        }
                    ],
                },
            },
        },
    }


def classify_job_status(status: Any, backoff_limit: int) -> str:
    """Classify a Job's status as 'succeeded', 'failed', or 'running'. Pure.

    Conditions ('Complete'/'Failed') are the primary signal: they are the Job
    controller's own terminal determination and are restartPolicy-independent.
    `status.succeeded` is kept only as a secondary success signal.

    Comparing `status.failed > backoff_limit` was previously the *only* failure
    signal, and it is unreliable: measured against a live cluster running a
    Job with `restartPolicy: OnFailure` and `backoffLimit: 2` whose container
    always exits 1, `status.failed` settled at 1 and never exceeded
    `backoff_limit`, so that branch never fired. A genuinely exhausted Job kept
    polling until POD_WATCH_TIMEOUT_S and was misreported as a timeout. The
    failed-count comparison is kept only as a defense-in-depth fallback for the
    case where the Failed condition hasn't propagated yet.
    """
    conditions = {c.type: c.status for c in (status.conditions or [])}
    if conditions.get("Complete") == "True" or status.succeeded:
        return "succeeded"
    if conditions.get("Failed") == "True":
        return "failed"
    if status.failed and status.failed > backoff_limit:
        return "failed"
    return "running"


def _k8s_batch_and_core():
    """Import and configure the Kubernetes client lazily.

    Kept out of module scope so the pure functions above remain importable
    (and unit-testable) without a kubeconfig present.
    """
    from kubernetes import client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()


# How long to wait for a stale, terminally-failed Job to finish deleting
# before recreating it, and how often to poll while waiting.
JOB_DELETE_TIMEOUT_S = 60
JOB_DELETE_POLL_S = 2


async def _ensure_job(batch, spec: WorkerSpec, manifest: dict) -> None:
    """Create the worker Job, handling the case where it already exists.

    Job names are deterministic (job_name_for), so a Temporal *activity*
    retry re-runs this against a name that may already be sitting in the
    cluster from the previous attempt. Two distinct cases:

    - The existing Job is still active (or already succeeded): re-attach, as
      before -- do nothing further here and let the watch loop below observe
      its current/eventual state.
    - The existing Job is terminally Failed: with Temporal now owning retry
      (backoffLimit=0), a Failed Job here is exactly what a previous, real
      failure left behind. `create_namespaced_job` cannot resurrect it (still
      409s), and simply re-attaching would make the watch loop observe an
      already-Failed object forever, so the retry would fail instantly and
      permanently instead of actually retrying. It must be deleted and
      recreated -- and deleted *and confirmed gone* before recreating, or the
      create races a half-deleted object (a spurious 409, or orphaned pods
      from the old generation attaching to the new Job via a stale label
      selector timing window).
    """
    from kubernetes.client.exceptions import ApiException

    name = manifest["metadata"]["name"]
    namespace = spec.namespace

    try:
        batch.create_namespaced_job(namespace=namespace, body=manifest)
        activity.logger.info(f"created Job {name}")
        return
    except ApiException as e:
        if e.status != 409:  # 409 = already exists
            raise

    existing = batch.read_namespaced_job_status(name=name, namespace=namespace)
    outcome = classify_job_status(existing.status, manifest["spec"]["backoffLimit"])
    if outcome != "failed":
        activity.logger.info(f"Job {name} already exists, re-attaching")
        return

    activity.logger.info(f"Job {name} exists but is terminally failed; deleting and recreating")
    batch.delete_namespaced_job(name=name, namespace=namespace, propagation_policy="Background")
    await _await_job_deleted(
        batch, namespace, name, timeout_s=JOB_DELETE_TIMEOUT_S, poll_s=JOB_DELETE_POLL_S
    )
    batch.create_namespaced_job(namespace=namespace, body=manifest)
    activity.logger.info(f"recreated Job {name} after clearing the failed attempt")


async def _await_job_deleted(
    batch, namespace: str, name: str, timeout_s: float, poll_s: float
) -> None:
    """Block until a deleted Job (and, via cascade, its pods) is fully gone.

    Background propagation returns as soon as the delete is *accepted*, not
    once the object and its pods are actually removed -- recreating
    immediately would race that in-flight deletion. Polls
    read_namespaced_job_status until it 404s; raises TimeoutError rather than
    looping forever if deletion is somehow wedged, so a stuck cluster fails
    the activity (and gets retried/surfaced) instead of hanging it.
    """
    from kubernetes.client.exceptions import ApiException

    waited = 0.0
    while True:
        try:
            batch.read_namespaced_job_status(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return
            raise
        if waited >= timeout_s:
            raise TimeoutError(
                f"Job {name} in {namespace} did not finish deleting within {timeout_s}s"
            )
        await asyncio.sleep(poll_s)
        waited += poll_s


# How many trailing lines of a failed pod's log to fold into the raised
# failure. Bounded on purpose: the diagnostic must survive into Temporal's
# history, not reproduce the whole log there.
FAILURE_LOG_TAIL_LINES = 20


class WorkerJobFailed(Exception):
    """A worker's Kubernetes Job reached a terminal Failed state.

    Raised instead of returned as
    WorkerResult(succeeded=False, ...), because a normal activity return is an
    activity *success* to Temporal no matter what the payload says --
    RetryPolicy(maximum_attempts=3) on this activity (workflows.py) only ever
    fires for raised exceptions. Returning here would silently drop a worker
    from 3 attempts to 1, since Kubernetes' own retry
    (backoffLimit 2->0, restartPolicy OnFailure->Never) leaves nothing to
    own retry at all.

    Deliberately left retryable (not added to non_retryable_error_types in
    workflows.py): a crashed container is not assumed permanent -- it may be a
    transient node/resource issue -- and _ensure_job's delete-and-recreate
    branch exists precisely so a retried attempt gets a fresh Job/pod rather
    than re-observing the same dead one. Contrast job_name_for's ValueError,
    which is a deterministic input-validation failure -- retrying it three
    times cannot change the outcome, so that one *is* marked non-retryable.
    """


def _is_fresh(last_modified: datetime | None, not_before: datetime) -> bool:
    """True if an object's last_modified is at/after not_before, within
    _CLOCK_SKEW_TOLERANCE_S. Pure -- no I/O, directly unit-testable.

    Both datetimes must be timezone-aware
    UTC -- minio.time.from_http_header (what populates a real stat_object
    result's last_modified) always attaches tzinfo, and every caller of
    wait_for_worker_artifact must pass an aware not_before (e.g.
    datetime.now(timezone.utc)) for the same reason: comparing a naive and
    an aware datetime raises TypeError, and this function does nothing to
    paper over that mismatch -- it is the caller's job to never create it.
    last_modified being None (no real stat_object result should ever lack
    it, but a test double might) is treated as "can't prove freshness",
    i.e. not fresh -- the same as the object not existing at all.
    """
    if last_modified is None:
        return False
    return last_modified >= not_before - timedelta(seconds=_CLOCK_SKEW_TOLERANCE_S)


async def wait_for_worker_artifact(
    minio_client,
    bucket: str,
    fl_round: int,
    worker_id: int,
    timeout_s: float,
    poll_s: float,
    failure_check: Callable[[], Awaitable[str | None]] | None = None,
    not_before: datetime | None = None,
) -> bool:
    """Poll MinIO for the one object that means a multi-cluster worker is done.

    The host cannot reliably watch a pod Karmada has propagated to a member
    cluster (see dispatch.py), but the worker entrypoint
    (src/core/worker_entrypoint.py) already uploads, per worker per round, in
    this order:
      1. round_{fl_round}/workers/worker_{worker_id}_weights.pt
      2. round_{fl_round}/workers/worker_{worker_id}_metrics.json
    and the aggregator (the aggregate_and_evaluate KFP component) already
    treats the metrics object's presence as "this worker succeeded". Only the
    metrics key is checked here -- weights land *first*, so a check that
    stopped at worker_{worker_id}_weights.pt would read a partial,
    still-in-flight upload as a finished worker.

    Heartbeats every failed poll (matching launch_and_watch_pod's
    single-topology loop below) so Temporal can distinguish a slow
    member-cluster worker from a wedged one.

    Raises TimeoutError instead of returning False when the artifact never
    appears within timeout_s. This matters exactly as much as
    WorkerJobFailed does for the single-topology path above: Temporal's
    RetryPolicy(maximum_attempts=3) only ever fires for a *raised*
    exception, so returning False here would silently turn 3 retry attempts
    into 1 for every multi-cluster worker whose artifact never lands.

    failure_check: an optional zero-arg
    async callable, invoked once per iteration alongside the MinIO check.
    MinIO can only ever prove "the artifact isn't here yet" -- never "the
    worker is dead" -- so without an independent signal, a crashed worker is
    indistinguishable from a slow one and costs a full timeout_s to detect,
    every single retry attempt. Returning a non-None string from
    failure_check means "terminally failed, stop waiting", and raises
    WorkerJobFailed with that string immediately. Returning None -- whether
    because the check has no verdict yet (e.g. the Karmada aggregated Job
    status hasn't propagated), is still lagging, or the worker is genuinely
    still running -- must never be treated as failure; it simply means "keep
    polling", identical to not passing failure_check at all. Keeping the
    contract this narrow (a plain callable returning str | None) is what
    lets this function stay ignorant of Karmada/dispatch.py and directly
    unit-testable with a bare lambda/async def.

    not_before: the completion key has no
    run identifier, and run_pipeline.py's --bucket is documented as a resume
    mechanism, so a resumed round can find a *previous*, abandoned attempt's
    metrics object already sitting at this same key. When not_before is
    given, an object whose last_modified predates it (beyond
    _CLOCK_SKEW_TOLERANCE_S) is treated as though it weren't there at all --
    it never satisfies the wait. Pass the activity's own wall-clock
    timestamp captured before dispatch; ordinary datetime.now() is fine here
    because this is activity code, not workflow code (only *workflow* code
    must be deterministic for Temporal's replay). Left as None (the
    default), no freshness check is performed, matching every caller that
    doesn't need one.

    Besides S3Error, this also tolerates
    urllib3.exceptions.MaxRetryError and ProtocolError -- the connection-
    level exceptions a real MinIO pod restart or network blip actually
    raises (verified against the installed minio==7.2.20; these never reach
    the S3 protocol layer, so they are not S3Error instances). Deliberately
    not a bare `except Exception`, which would also mask a genuine bug in
    this loop.
    """
    from minio.error import S3Error
    from urllib3.exceptions import MaxRetryError, ProtocolError

    key = f"round_{fl_round}/workers/worker_{worker_id}_metrics.json"
    waited = 0.0
    while waited < timeout_s:
        try:
            stat = minio_client.stat_object(bucket, key)
            if not_before is None or _is_fresh(stat.last_modified, not_before):
                return True
            log.warning(
                f"{key} exists but predates this attempt (likely a stale artifact "
                f"from an earlier, abandoned attempt at this round/worker) -- "
                f"ignoring it and continuing to poll"
            )
        except S3Error as e:
            if e.code not in ("NoSuchKey", "NoSuchObject"):
                log.warning(f"transient MinIO error polling for {key}: {e}")
            # else: simply not uploaded yet -- keep polling.
        except (MaxRetryError, ProtocolError) as e:
            # A MinIO pod restart or network
            # blip never reaches the S3 protocol layer, so it doesn't raise
            # S3Error at all -- verified against the installed minio==7.2.20,
            # a refused connection raises urllib3.exceptions.MaxRetryError
            # (connect-phase failures, after urllib3's own internal retries
            # are exhausted) or ProtocolError (a connection dropped mid-
            # response). Deliberately not `except Exception`: that would also
            # swallow a genuine bug in this loop (an AttributeError here
            # already cost this project a full debugging session once), so
            # only these two specific, connection-level classes are caught.
            log.warning(f"transient connection error polling for {key}: {e}")

        if failure_check is not None:
            reason = await failure_check()
            if reason is not None:
                raise WorkerJobFailed(
                    f"worker {worker_id} round {fl_round}: job reached a terminal "
                    f"failed state before producing {key}: {reason}"
                )

        activity.heartbeat(
            {"worker_id": worker_id, "fl_round": fl_round, "waited_s": waited, "artifact": key}
        )
        await asyncio.sleep(poll_s)
        waited += poll_s

    raise TimeoutError(
        f"worker {worker_id} round {fl_round}: {key} did not appear in MinIO "
        f"within {timeout_s}s"
    )


def _minio_client_for(spec: WorkerSpec):
    """Lazily build a MinIO client from a WorkerSpec's credentials.

    Kept out of module scope (like _k8s_batch_and_core below) so the pure
    functions in this module stay importable without the minio package's
    transitive dependencies configured.
    """
    from minio import Minio

    return Minio(
        endpoint=spec.minio_endpoint,
        access_key=spec.minio_access_key,
        secret_key=spec.minio_secret_key,
        secure=False,
    )


@activity.defn
async def launch_and_watch_pod(spec: WorkerSpec) -> WorkerResult:
    """Create the worker Job if absent, then watch it to completion.

    topology='single' (below): watches the Job directly
    via the local Kubernetes API, heartbeating every poll so Temporal can
    distinguish a slow worker from a wedged one.

    topology='multi': the pod runs on a Karmada member cluster the host
    cannot reliably watch (see dispatch.py), so this dispatches the Job via
    dispatcher_for(spec) and waits on its MinIO completion artifact instead
    -- see wait_for_worker_artifact and _launch_and_watch_pod_multi.
    """
    if spec.topology == "multi":
        return await _launch_and_watch_pod_multi(spec)

    from kubernetes.client.exceptions import ApiException

    batch, core = _k8s_batch_and_core()
    name = job_name_for(spec)
    manifest = build_job_manifest(spec)

    await _ensure_job(batch, spec, manifest)

    waited = 0
    while waited < POD_WATCH_TIMEOUT_S:
        try:
            job = batch.read_namespaced_job_status(name=name, namespace=spec.namespace)
        except ApiException as e:
            if e.status == 404:  # Job vanished mid-watch (e.g. deleted out-of-band)
                return WorkerResult(
                    worker_id=spec.worker_id, succeeded=False, attempts=0,
                    failure_reason=f"Job {name} disappeared mid-watch (404)", job_name=name,
                )
            raise
        status = job.status
        outcome = classify_job_status(status, manifest["spec"]["backoffLimit"])
        if outcome == "succeeded":
            await _log_tail(core, spec, name)
            return WorkerResult(
                worker_id=spec.worker_id, succeeded=True,
                attempts=await _attempt_count(core, spec, name), failure_reason="", job_name=name,
            )
        if outcome == "failed":
            # Capture the pod's log tail *before* returning -- WorkerWorkflow's
            # `finally` deletes the Job (and cascades to its pods) within seconds
            # of this activity completing, so this is the last chance to read it.
            reason = await _failure_reason(core, spec, name)
            tail = await _log_tail(core, spec, name, lines=FAILURE_LOG_TAIL_LINES)
            message = f"worker {spec.worker_id} job {name} failed: {reason}"
            if tail:
                message += f"\n--- last {FAILURE_LOG_TAIL_LINES} log lines ---\n{tail}"
            raise WorkerJobFailed(message)

        activity.heartbeat(
            {"worker_id": spec.worker_id, "active": int(status.active or 0), "waited_s": waited}
        )
        await asyncio.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S

    return WorkerResult(
        worker_id=spec.worker_id, succeeded=False, attempts=0,
        failure_reason=f"timed out after {POD_WATCH_TIMEOUT_S}s", job_name=name,
    )


def _resolve_host_node_ip(core) -> str:
    """Return the InternalIP address of any Ready node on the cluster `core`
    points at.

    A Karmada member cluster is a separate Kubernetes
    cluster with its own DNS, so WorkerSpec.minio_endpoint/
    mlflow_tracking_uri -- in-cluster DNS names belonging to the *host*
    cluster -- never resolve there (this is exactly the live-gate failure:
    NameResolutionError on mlflow-service.fed-twin.svc.cluster.local from
    inside a member pod). Every kind cluster in this project's multi-cluster
    environment shares one Docker bridge network, so a member pod *can*
    reach the host's MinIO/MLflow NodePorts via the host's own node IP
    (verified live: curl http://172.18.0.2:30500/ -> 200). `core` must be the
    CoreV1Api this module's own `_k8s_batch_and_core()` builds -- pointed at
    the host cluster the Temporal worker pod itself runs on, never the
    Karmada apiserver `dispatch.py._karmada_clients()` builds, since a node
    IP is only meaningful read from the API of the cluster it belongs to.

    Resolved fresh on every call rather than read from config: the IP is not
    stable across a kind/VM restart (observed live: 172.18.0.3 before a
    restart, 172.18.0.2 after), and this project has repeatedly been bitten
    by exactly this class of stale-value-in-config bug.

    Raises RuntimeError naming exactly what could not be resolved (the node
    listing call itself failing, zero nodes, no Ready node, or a Ready node
    with no InternalIP) rather than falling back to any default -- a silent
    fallback here would reproduce the original in-cluster-DNS failure as an
    unreachable IP, discovered only after a full POD_WATCH_TIMEOUT_S, instead
    of an immediate and actionable error.
    """
    try:
        nodes = core.list_node().items
    except Exception as e:
        raise RuntimeError(
            "topology='multi' could not resolve a host node IP for the "
            f"MinIO/MLflow NodePort rewrite: listing nodes on the host cluster "
            f"failed: {e}"
        ) from e

    if not nodes:
        raise RuntimeError(
            "topology='multi' could not resolve a host node IP for the "
            "MinIO/MLflow NodePort rewrite: the host cluster's API returned "
            "zero nodes"
        )

    for node in nodes:
        conditions = {c.type: c.status for c in (node.status.conditions or [])}
        if conditions.get("Ready") != "True":
            continue
        for addr in node.status.addresses or []:
            if addr.type == "InternalIP":
                return addr.address

    names = [n.metadata.name for n in nodes]
    raise RuntimeError(
        "topology='multi' could not resolve a host node IP for the MinIO/"
        f"MLflow NodePort rewrite: no Ready node exposes an InternalIP address "
        f"(checked {names})"
    )


def _rewrite_endpoints_for_multi(spec: WorkerSpec) -> WorkerSpec:
    """Rewrite spec's MinIO endpoint and MLflow tracking URI onto the host
    node's NodePorts, for topology='multi' only.

    WorkerSpec is a frozen dataclass (Temporal replays workflow arguments, so
    immutability keeps replay deterministic) -- dataclasses.replace returns a
    new instance with only these two fields changed rather than mutating
    spec in place. Called once, at dispatch time, from
    _launch_and_watch_pod_multi -- before dispatcher_for(spec).ensure_job
    ever calls build_job_manifest(spec), so the rewritten values are what
    actually lands in the worker Job's env, not the unreachable in-cluster
    DNS names WorkerSpec carries by default.

    spec.minio_nodeport/mlflow_nodeport (types.py) must both be configured
    (nonzero) -- this is checked before the node lookup runs, so a
    misconfigured spec fails immediately with a clear message naming which
    value is missing, rather than resolving a real node IP only to build a
    silently-broken "<ip>:0" endpoint.
    """
    if not spec.minio_nodeport or not spec.mlflow_nodeport:
        # ValueError, not RuntimeError: a deterministic input-validation
        # failure exactly like job_name_for's kfp_run_id check above -- every
        # retry would rebuild the identical spec and fail identically, so
        # this is listed non-retryable in workflows.py
        # (non_retryable_error_types=["ValueError"]) to fail fast instead of
        # burning retry backoff on an outcome that cannot change.
        raise ValueError(
            "topology='multi' requires minio_nodeport and mlflow_nodeport to be "
            "configured (nonzero) to rewrite the worker's MinIO/MLflow "
            f"endpoints onto the host node's NodePorts; got "
            f"minio_nodeport={spec.minio_nodeport!r}, "
            f"mlflow_nodeport={spec.mlflow_nodeport!r}"
        )

    _, core = _k8s_batch_and_core()
    host_ip = _resolve_host_node_ip(core)
    return dataclasses.replace(
        spec,
        minio_endpoint=f"{host_ip}:{spec.minio_nodeport}",
        mlflow_tracking_uri=f"http://{host_ip}:{spec.mlflow_nodeport}",
    )


async def _launch_and_watch_pod_multi(spec: WorkerSpec) -> WorkerResult:
    """topology='multi': rewrite spec's MinIO/MLflow endpoints onto the host
    node's NodePorts, dispatch to spec.member_cluster via Karmada, then wait
    on the MinIO completion artifact rather than watching the pod -- the
    host cannot reliably watch a pod on a member cluster (dispatch.py).

    The endpoint rewrite (_rewrite_endpoints_for_multi) happens first, and
    exactly once per activity invocation: spec.minio_endpoint/
    mlflow_tracking_uri default to in-cluster DNS names that belong to the
    host cluster and never resolve from a Karmada member cluster's own DNS
    -- see _resolve_host_node_ip for the full story. Every subsequent use of
    `spec` in this function (dispatch, MinIO polling) uses the rewritten
    version, so the Job manifest build_job_manifest produces inside
    dispatcher.ensure_job carries the reachable NodePort endpoints, not the
    unreachable originals.

    The Karmada aggregated API is consulted only as best-effort failure
    enrichment, inside a try/except that can never turn a real failure into
    a reported success -- matching _failure_reason/_log_tail's contract
    above for the single-topology path. Whatever wait_for_worker_artifact
    raises (timeout or otherwise) is what drives the outcome; the Karmada
    lookup only adds context to the message.

    not_before is captured here, before
    dispatch, using ordinary wall-clock time -- this is activity code, which
    Temporal does not replay, so datetime.now() is safe (only *workflow*
    code must be deterministic). Capturing it before ensure_job rather than
    after means any object already sitting in MinIO at this round/worker's
    key -- left over from an earlier, abandoned attempt reached via
    run_pipeline.py's --bucket resume path -- is unambiguously older than
    this attempt's cutoff, so wait_for_worker_artifact can never mistake it
    for this attempt's own completion.
    """
    from datetime import timezone

    from src.orchestration.dispatch import dispatcher_for

    spec = _rewrite_endpoints_for_multi(spec)
    not_before = datetime.now(timezone.utc)

    dispatcher = dispatcher_for(spec)
    name = await dispatcher.ensure_job(spec)
    minio_client = _minio_client_for(spec)

    try:
        await wait_for_worker_artifact(
            minio_client,
            spec.minio_bucket,
            spec.fl_round,
            spec.worker_id,
            timeout_s=POD_WATCH_TIMEOUT_S,
            poll_s=POLL_INTERVAL_S,
            failure_check=lambda: _karmada_terminal_failure(spec, name),
            not_before=not_before,
        )
    except Exception as e:
        reason = await _karmada_failure_reason(spec, name)
        message = (
            f"worker {spec.worker_id} job {name} on member cluster "
            f"{spec.member_cluster!r} produced no completion artifact: {e}"
        )
        if reason:
            message += f"\n--- Karmada aggregated Job status ---\n{reason}"
        raise WorkerJobFailed(message) from e

    return WorkerResult(
        worker_id=spec.worker_id, succeeded=True, attempts=1, failure_reason="", job_name=name,
    )


async def _karmada_job_status(spec: WorkerSpec, job_name: str) -> tuple[str | None, str]:
    """Fetch and classify the Karmada aggregated Job status for one worker.

    Shared by _karmada_failure_reason (best-effort enrichment consulted only
    after wait_for_worker_artifact has already given up) and
    _karmada_terminal_failure (the fast-fail check, consulted
    *during* the wait via wait_for_worker_artifact's failure_check
    parameter). Returns (outcome, message):

    - outcome is classify_job_status's result ('succeeded'/'failed'/
      'running'), or None if no verdict could be reached at all -- the Job
      hasn't propagated to the member cluster yet (the read 404s, which is
      completely normal for the first few seconds after dispatch), the
      Karmada apiserver is unreachable, or FED_KARMADA_CONFIG isn't set.
      None must never be read as "failed": conflating an absent/lagging
      status with failure would fast-fail a perfectly healthy round the
      moment it's dispatched, before Karmada has had any chance to catch up.
    - message is always a human-readable string, safe to fold into a raised
      exception either way.
    """
    try:
        from src.orchestration.dispatch import _karmada_clients

        batch, _ = _karmada_clients()
        job = batch.read_namespaced_job_status(name=job_name, namespace=spec.namespace)
        status = job.status
        conditions = ", ".join(f"{c.type}={c.status}" for c in (status.conditions or []))
        message = (
            f"active={status.active or 0} succeeded={status.succeeded or 0} "
            f"failed={status.failed or 0} conditions=[{conditions}]"
        )
        backoff_limit = build_job_manifest(spec)["spec"]["backoffLimit"]
        return classify_job_status(status, backoff_limit), message
    except Exception as e:
        return None, f"unavailable: {e}"


async def _karmada_failure_reason(spec: WorkerSpec, job_name: str) -> str:
    """Best-effort diagnostics from the Karmada aggregated API for a failed
    multi-cluster worker.

    Consulted only after wait_for_worker_artifact has already given up, and
    only to enrich the raised message -- never able to turn a real failure
    into a reported success. Any failure here (aggregated API unreachable,
    FED_KARMADA_CONFIG unset, member cluster unreachable, ...) is swallowed
    and surfaces as a plain diagnostic string, exactly like
    _failure_reason/_log_tail's contract for the single-topology path.
    """
    _, message = await _karmada_job_status(spec, job_name)
    return message


async def _karmada_terminal_failure(spec: WorkerSpec, job_name: str) -> str | None:
    """Fast-fail check wired into
    wait_for_worker_artifact's poll loop via its failure_check parameter.

    Returns a diagnostic message only when the Karmada aggregated Job status
    is definitely, terminally Failed -- mirroring what classify_job_status
    already does for the single-topology Job-watch loop above. Every other
    case -- absent (Karmada hasn't propagated the Job to the member cluster
    yet), still running, succeeded, or the aggregated API being unreachable
    -- returns None, which wait_for_worker_artifact treats as "keep
    polling". This is what lets a crashed worker be caught within roughly
    one poll interval instead of the full POD_WATCH_TIMEOUT_S, without any
    risk of a merely-absent-or-lagging status killing a healthy round.
    """
    outcome, message = await _karmada_job_status(spec, job_name)
    return message if outcome == "failed" else None


async def _attempt_count(core, spec: WorkerSpec, job_name: str) -> int:
    """Container attempts (restarts + 1) for the job's pod.

    Chosen over `status.failed`/`status.succeeded`: under the old
    `restartPolicy: OnFailure`, Kubernetes restarted the *container* in place
    rather than replacing the Pod, so those Job-level counters counted Pods,
    not attempts, and could not report the retry count they were previously
    assumed to give. `containerStatuses[0].restartCount` was the accurate
    source for that in-place-restart world.

    Since restartPolicy is now `Never` with `backoffLimit: 0` (see
    build_job_manifest), a Job's pod is never restarted in place, so this
    will normally read 0 (-> return 1) for whichever Job generation is
    currently live -- Kubernetes no longer retries at all. The authoritative
    per-worker retry count now lives one layer up, in Temporal's own activity
    attempt number (`activity.info().attempt`), across the delete+recreate
    cycle in `_ensure_job`. This function still answers a real, distinct
    question (how many times did *this* container attempt run, which is
    always 0-or-1 now) but is no longer the retry-count signal; left as-is,
    out of scope for this fix.
    """
    try:
        pods = core.list_namespaced_pod(
            namespace=spec.namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
            statuses = pod.status.container_statuses or []
            if statuses:
                return statuses[0].restart_count + 1
        return 1
    except Exception:  # best-effort; never fail the activity over a diagnostic
        return 1


async def _failure_reason(core, spec: WorkerSpec, job_name: str) -> str:
    """Best-effort human-readable cause from the most recent pod."""
    try:
        pods = core.list_namespaced_pod(
            namespace=spec.namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
            for cs in pod.status.container_statuses or []:
                term = cs.state.terminated
                if term is not None and term.reason:
                    return f"{term.reason} (exit {term.exit_code})"
        return "unknown"
    except Exception as e:  # diagnostics must never mask the real failure
        return f"unavailable: {e}"


async def _log_tail(
    core, spec: WorkerSpec, job_name: str, lines: int = FAILURE_LOG_TAIL_LINES
) -> str:
    """Best-effort tail of the job's pod logs, bounded to `lines`.

    Returns the captured text so callers can fold it into a failure message
    -- never raises, matching _failure_reason's contract that a
    diagnostic failure must never mask the real outcome.
    """
    try:
        pods = core.list_namespaced_pod(
            namespace=spec.namespace, label_selector=f"job-name={job_name}"
        )
        chunks = []
        for pod in pods.items:
            text = core.read_namespaced_pod_log(
                name=pod.metadata.name, namespace=spec.namespace, tail_lines=lines
            )
            activity.logger.info(f"[{pod.metadata.name}] {text}")
            chunks.append(f"[{pod.metadata.name}] {text}")
        return "\n".join(chunks)
    except Exception as e:
        activity.logger.warning(f"could not read logs for {job_name}: {e}")
        return ""


@activity.defn
async def cleanup_worker_job(spec: WorkerSpec) -> None:
    """Delete the Job and its pods. Safe to call when already gone.

    topology='multi' delegates to dispatcher_for(spec) (KarmadaJobDispatcher),
    which also deletes the PropagationPolicy that pinned the Job to
    spec.member_cluster -- leaving that behind would leak one Karmada object
    per finished worker. topology='single' still keeps
    calling _k8s_batch_and_core() directly rather than routing through
    LocalJobDispatcher, which duplicates the same lazy client builder under a
    distinct module-level name in dispatch.py -- existing tests monkeypatch
    *this* module's _k8s_batch_and_core specifically, and routing through
    dispatch's own copy would silently bypass that seam (and, in production,
    would be equally correct but pointlessly indirect for a case that never
    needs Karmada at all).
    """
    if spec.topology == "multi":
        from src.orchestration.dispatch import dispatcher_for

        dispatcher_for(spec).delete_job(spec)
        return

    from kubernetes.client.exceptions import ApiException

    batch, _ = _k8s_batch_and_core()
    try:
        batch.delete_namespaced_job(
            name=job_name_for(spec), namespace=spec.namespace, propagation_policy="Background"
        )
    except ApiException as e:
        if e.status != 404:
            raise
