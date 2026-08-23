import dataclasses
import os
import sys

import pytest

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.types import RoundReport, RoundSpec, WorkerResult, WorkerSpec


def _round_spec(**overrides):
    base = dict(
        fl_round=0,
        num_workers=3,
        min_workers=2,
        local_episodes=10,
        eval_episodes=5,
        namespace="ns",
        worker_image="img:v1",
        minio_endpoint="minio:9000",
        minio_access_key="a",
        minio_secret_key="b",
        minio_bucket="bucket",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp",
        kfp_run_id="run-1",
    )
    base.update(overrides)
    return RoundSpec(**base)


def _result(worker_id: int, succeeded: bool) -> WorkerResult:
    return WorkerResult(
        worker_id=worker_id,
        succeeded=succeeded,
        attempts=1,
        failure_reason="" if succeeded else "OOMKilled",
        job_name=f"job-{worker_id}",
    )


def test_worker_spec_inherits_round_fields_and_sets_id():
    spec = _round_spec().worker_spec(2)
    assert spec.worker_id == 2
    assert spec.fl_round == 0
    assert spec.num_workers == 3
    assert spec.minio_bucket == "bucket"


def test_round_spec_is_immutable():
    spec = _round_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.fl_round = 5  # type: ignore[misc]


def test_worker_spec_defaults_topology_to_single_with_no_member_cluster():
    spec = WorkerSpec(
        fl_round=0, worker_id=0, num_workers=1, local_episodes=1, eval_episodes=1,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="run-1",
    )
    assert spec.topology == "single"
    assert spec.member_cluster == ""


def test_worker_spec_topology_and_member_cluster_are_frozen():
    spec = WorkerSpec(
        fl_round=0, worker_id=0, num_workers=1, local_episodes=1, eval_episodes=1,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="run-1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.member_cluster = "member1"  # type: ignore[misc]


def test_round_spec_defaults_topology_to_single_with_no_member_cluster():
    spec = _round_spec()
    assert spec.topology == "single"
    assert spec.member_cluster == ""


def test_worker_spec_inherits_round_topology():
    spec = _round_spec(
        topology="multi", member_count=2, member_prefix="active-fed-member"
    ).worker_spec(1)
    assert spec.topology == "multi"


# ---------------------------------------------------------------------------
# minio_nodeport/mlflow_nodeport. Only meaningful
# under topology="multi" (see src/orchestration/activities.py's endpoint
# rewrite, which reads them off the WorkerSpec it dispatches), but carried on
# both RoundSpec and WorkerSpec the same way member_count/member_prefix
# already are, so worker_spec() threads them through unchanged just like every
# other RoundSpec-driving field.
# ---------------------------------------------------------------------------


def test_worker_spec_defaults_nodeports_to_zero():
    spec = WorkerSpec(
        fl_round=0, worker_id=0, num_workers=1, local_episodes=1, eval_episodes=1,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="run-1",
    )
    assert spec.minio_nodeport == 0
    assert spec.mlflow_nodeport == 0


def test_round_spec_defaults_nodeports_to_zero():
    spec = _round_spec()
    assert spec.minio_nodeport == 0
    assert spec.mlflow_nodeport == 0


def test_round_spec_threads_nodeports_into_worker_spec():
    spec = _round_spec(
        topology="multi", member_count=2, member_prefix="active-fed-member",
        minio_nodeport=30900, mlflow_nodeport=30500,
    ).worker_spec(0)
    assert spec.minio_nodeport == 30900
    assert spec.mlflow_nodeport == 30500


# ---------------------------------------------------------------------------
# Round-robin worker -> member assignment (Task 5). Task 3 deliberately left
# RoundSpec.worker_spec() passing a single static member_cluster straight
# through to every worker; member_count/member_prefix (added here) let it
# compute a distinct member per worker instead. physics_seed derives from
# worker_id, so one worker per member cluster is what gives each member a
# distinct physical variation -- the whole point of the multi topology.
# ---------------------------------------------------------------------------


def test_round_robin_assigns_distinct_members_across_workers():
    spec = _round_spec(topology="multi", member_count=2, member_prefix="active-fed-member")
    assert spec.worker_spec(0).member_cluster == "active-fed-member1"
    assert spec.worker_spec(1).member_cluster == "active-fed-member2"


def test_round_robin_wraps_when_workers_outnumber_members():
    # 3 workers over 2 members -> member1, member2, member1.
    spec = _round_spec(topology="multi", member_count=2, member_prefix="active-fed-member")
    assignments = [spec.worker_spec(i).member_cluster for i in range(3)]
    assert assignments == ["active-fed-member1", "active-fed-member2", "active-fed-member1"]


def test_single_topology_worker_spec_ignores_member_count_and_prefix():
    # topology="single" must keep yielding an empty member_cluster
    # regardless of member_count/member_prefix -- those fields are only
    # meaningful under topology="multi".
    spec = _round_spec(member_count=2, member_prefix="active-fed-member")
    assert spec.topology == "single"
    assert spec.worker_spec(0).member_cluster == ""


def test_multi_topology_with_zero_member_count_raises():
    # The critical safety property, enforced one layer earlier than
    # dispatch.py: member_count=0 must never silently produce an empty
    # member_cluster (which would fan a worker's Job out to every joined
    # member -- see dispatch.py's build_propagation_policy/dispatcher_for).
    spec = _round_spec(topology="multi", member_count=0, member_prefix="active-fed-member")
    with pytest.raises(ValueError):
        spec.worker_spec(0)


def test_multi_topology_with_blank_member_prefix_raises():
    spec = _round_spec(topology="multi", member_count=2, member_prefix="")
    with pytest.raises(ValueError):
        spec.worker_spec(0)


def test_multi_topology_with_negative_member_count_raises():
    spec = _round_spec(topology="multi", member_count=-1, member_prefix="active-fed-member")
    with pytest.raises(ValueError):
        spec.worker_spec(0)


# --- Whitespace/format defeats truthiness -----------------------------------
# `if not self.member_prefix:` is Python truthiness -- "   "/"\t\n" are
# non-empty strings and sail straight through, producing a member_cluster
# like "   1" that selects zero real Karmada clusters (see dispatch.py's
# build_propagation_policy) instead of raising here, one layer earlier. The
# fix validates the full character-class format, not just blankness, so it
# also catches non-blank-but-still-invalid prefixes a bare `.strip()`
# truthiness check would still miss.

@pytest.mark.parametrize("garbage", ["   ", "\t\n", " "])
def test_multi_topology_with_whitespace_only_member_prefix_raises(garbage):
    spec = _round_spec(topology="multi", member_count=2, member_prefix=garbage)
    with pytest.raises(ValueError):
        spec.worker_spec(0)


@pytest.mark.parametrize("garbage", ["ACTIVE-FED-MEMBER", "active fed member", "-active-member"])
def test_multi_topology_with_non_blank_but_invalid_member_prefix_raises(garbage):
    # Non-blank but not RFC-1123-style: uppercase, an embedded space, or a
    # leading '-'. Each would still produce a member_cluster name selecting
    # zero real clusters, exactly like whitespace does.
    spec = _round_spec(topology="multi", member_count=2, member_prefix=garbage)
    with pytest.raises(ValueError):
        spec.worker_spec(0)


def test_multi_topology_with_trailing_hyphen_member_prefix_is_accepted():
    # A trailing '-' in member_prefix is fine (unlike a full RFC 1123 label,
    # which must end alphanumeric): worker_spec() always appends a digit, so
    # the concatenated member_cluster name always ends alphanumeric
    # regardless of what member_prefix itself ends with.
    spec = _round_spec(topology="multi", member_count=2, member_prefix="active-fed-member-")
    assert spec.worker_spec(0).member_cluster == "active-fed-member-1"


def test_report_partitions_succeeded_and_failed():
    report = RoundReport(
        fl_round=0,
        results=[_result(0, True), _result(1, False), _result(2, True)],
    )
    assert report.succeeded_ids == [0, 2]
    assert report.failed_ids == [1]


def test_quorum_met_when_enough_workers_succeed():
    report = RoundReport(
        fl_round=0,
        results=[_result(0, True), _result(1, True), _result(2, False)],
    )
    assert report.meets_quorum(2) is True


def test_quorum_not_met_below_threshold():
    report = RoundReport(
        fl_round=0,
        results=[_result(0, True), _result(1, False), _result(2, False)],
    )
    assert report.meets_quorum(2) is False


def test_quorum_with_zero_successes_is_never_met():
    report = RoundReport(fl_round=0, results=[_result(0, False)])
    assert report.meets_quorum(1) is False


def test_worker_spec_carries_eval_episodes_distinct_from_local_episodes():
    spec = _round_spec(local_episodes=10, eval_episodes=3).worker_spec(0)
    assert spec.local_episodes == 10
    assert spec.eval_episodes == 3
