import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.orchestration.activities as activities_module
from src.orchestration.activities import cleanup_worker_job, launch_and_watch_pod
from src.orchestration.types import RoundSpec, WorkerResult, WorkerSpec
from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow, WorkerWorkflow


def _round_spec(num_workers=3, min_workers=2, kfp_backend_run_id="") -> RoundSpec:
    return RoundSpec(
        fl_round=0, num_workers=num_workers, min_workers=min_workers, local_episodes=5,
        eval_episodes=5,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234",
        kfp_backend_run_id=kfp_backend_run_id,
    )


def _worker_spec(worker_id=0) -> WorkerSpec:
    return WorkerSpec(
        fl_round=0, worker_id=worker_id, num_workers=1, local_episodes=5,
        eval_episodes=5,
        namespace="ns", worker_image="img:v1", minio_endpoint="m:9000",
        minio_access_key="a", minio_secret_key="b", minio_bucket="bkt",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        kfp_run_id="abcdef1234",
    )


def _ok(spec: WorkerSpec) -> WorkerResult:
    return WorkerResult(worker_id=spec.worker_id, succeeded=True, attempts=1,
                        failure_reason="", job_name=f"j{spec.worker_id}")


async def _run(env: WorkflowEnvironment, acts, spec: RoundSpec):
    async with Worker(
        env.client, task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=acts,
    ):
        return await env.client.execute_workflow(
            TrainRoundWorkflow.run, spec,
            id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
        )


async def _run_worker(env: WorkflowEnvironment, acts, spec: WorkerSpec):
    async with Worker(
        env.client, task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=acts,
    ):
        return await env.client.execute_workflow(
            WorkerWorkflow.run, spec,
            id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
        )


@pytest.mark.asyncio
async def test_all_workers_succeed():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec())
    assert report.succeeded_ids == [0, 1, 2]
    assert report.failed_ids == []


@pytest.mark.asyncio
async def test_partial_failure_above_quorum_still_returns_survivors():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 1:
            raise RuntimeError("pod OOMKilled")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=2))
    assert report.succeeded_ids == [0, 2]
    assert report.failed_ids == [1]
    assert report.meets_quorum(2) is True


@pytest.mark.asyncio
async def test_below_quorum_fails_the_round():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 0:
            return _ok(spec)
        raise RuntimeError("boom")

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    from temporalio.client import WorkflowFailureError

    async with await WorkflowEnvironment.start_time_skipping() as env:
        with pytest.raises(WorkflowFailureError):
            await _run(env, [launch, cleanup], _round_spec(min_workers=2))


@pytest.mark.asyncio
async def test_worker_retries_then_succeeds():
    calls: dict[int, int] = {}

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        calls[spec.worker_id] = calls.get(spec.worker_id, 0) + 1
        if spec.worker_id == 1 and calls[spec.worker_id] == 1:
            raise RuntimeError("transient")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=3))
    assert report.succeeded_ids == [0, 1, 2]
    assert calls[1] == 2  # proves it actually retried rather than passing first time


@pytest.mark.asyncio
async def test_cleanup_runs_for_every_worker():
    cleaned: list[int] = []

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        cleaned.append(spec.worker_id)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        await _run(env, [launch, cleanup], _round_spec())
    assert sorted(cleaned) == [0, 1, 2]


@pytest.mark.asyncio
async def test_cleanup_still_runs_when_worker_activity_raises():
    from temporalio.client import WorkflowFailureError

    cleaned: list[int] = []

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        raise RuntimeError("boom")

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        cleaned.append(spec.worker_id)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        with pytest.raises(WorkflowFailureError):
            await _run_worker(env, [launch, cleanup], _worker_spec(worker_id=7))
    assert cleaned == [7]


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_worker_failure_reason():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return WorkerResult(worker_id=spec.worker_id, succeeded=False, attempts=1,
                            failure_reason="OOMKilled (exit 137)", job_name=f"j{spec.worker_id}")

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        raise RuntimeError("cleanup exploded")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run_worker(env, [launch, cleanup], _worker_spec(worker_id=3))
    assert result.succeeded is False
    assert result.failure_reason == "OOMKilled (exit 137)"


@pytest.mark.asyncio
async def test_round_status_query_reports_populated_map():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client, task_queue=TASK_QUEUE,
            workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=[launch, cleanup],
        ):
            handle = await env.client.start_workflow(
                TrainRoundWorkflow.run, _round_spec(),
                id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            await handle.result()
            statuses = await handle.query(TrainRoundWorkflow.status)
    assert set(statuses.keys()) == {0, 1, 2}
    assert all(s.phase == "Succeeded" for s in statuses.values())


@pytest.mark.asyncio
async def test_live_status_query_reports_root_cause_for_failed_worker():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 1:
            raise RuntimeError("pod OOMKilled")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client, task_queue=TASK_QUEUE,
            workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=[launch, cleanup],
        ):
            handle = await env.client.start_workflow(
                TrainRoundWorkflow.run, _round_spec(min_workers=2),
                id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            await handle.result()
            statuses = await handle.query(TrainRoundWorkflow.status)
    assert "OOMKilled" in statuses[1].message
    assert statuses[1].message != "Child Workflow execution failed"


@pytest.mark.asyncio
async def test_gather_exception_reports_root_cause_not_generic_wrapper():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        if spec.worker_id == 1:
            raise RuntimeError("pod OOMKilled")
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(env, [launch, cleanup], _round_spec(min_workers=2))

    failed = next(r for r in report.results if r.worker_id == 1)
    assert "OOMKilled" in failed.failure_reason
    assert failed.failure_reason != "Child Workflow execution failed"


# ---------------------------------------------------------------------------
# I1 consequence check: with launch_and_watch_pod now raising on a genuine Job
# failure (rather than returning WorkerResult(succeeded=False, ...)), confirm
# end-to-end -- through the *real* activity, not a hand-rolled fake -- that
# asyncio.gather(..., return_exceptions=True) still turns the exhausted-retry
# exception into a useful synthetic WorkerResult, that the C1 log tail
# survives all the way into RoundReport.failure_reason, and that quorum
# tolerance for the surviving workers is unaffected.
# ---------------------------------------------------------------------------
class _MixedBatchApi:
    """Fails only the Job whose deterministic name ends in the given worker
    suffix; every other worker's Job succeeds immediately. Job names are
    unique per worker (job_name_for), so the suffix check is enough to steer
    the *real* launch_and_watch_pod per-worker without a hand-rolled fake."""

    def __init__(self, failing_suffix: str):
        self._failing_suffix = failing_suffix

    def create_namespaced_job(self, namespace, body):
        pass

    def read_namespaced_job_status(self, name, namespace):
        if name.endswith(self._failing_suffix):
            return SimpleNamespace(
                status=SimpleNamespace(
                    conditions=[SimpleNamespace(type="Failed", status="True")],
                    succeeded=None, failed=1,
                )
            )
        return SimpleNamespace(
            status=SimpleNamespace(
                conditions=[SimpleNamespace(type="Complete", status="True")],
                succeeded=1, failed=None,
            )
        )

    def delete_namespaced_job(self, name, namespace, propagation_policy):
        pass


class _MixedCoreApi:
    def __init__(self, failing_suffix: str):
        self._failing_suffix = failing_suffix

    def list_namespaced_pod(self, namespace, label_selector):
        job_name = label_selector.split("=", 1)[1]
        failing = job_name.endswith(self._failing_suffix)
        state = SimpleNamespace(
            terminated=SimpleNamespace(
                reason="OOMKilled" if failing else "Completed",
                exit_code=137 if failing else 0,
            )
        )
        pod = SimpleNamespace(
            metadata=SimpleNamespace(name=f"{job_name}-pod"),
            status=SimpleNamespace(container_statuses=[SimpleNamespace(
                restart_count=0, state=state,
            )]),
        )
        return SimpleNamespace(items=[pod])

    def read_namespaced_pod_log(self, name, namespace, tail_lines):
        return "worker crashed with MemoryError\n" if self._failing_suffix in name else "ok\n"


@pytest.mark.asyncio
async def test_real_activity_failure_reaches_round_report_with_useful_message(monkeypatch):
    monkeypatch.setattr(
        activities_module, "_k8s_batch_and_core",
        lambda: (_MixedBatchApi("-w1"), _MixedCoreApi("-w1")),
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        report = await _run(
            env, [launch_and_watch_pod, cleanup_worker_job],
            _round_spec(num_workers=3, min_workers=2),
        )

    # Quorum tolerance for the survivors is unaffected by the fix.
    assert report.succeeded_ids == [0, 2]
    assert report.failed_ids == [1]
    assert report.meets_quorum(2) is True

    failed = next(r for r in report.results if r.worker_id == 1)
    assert failed.succeeded is False
    # The C1 log tail and the reason both survive all the way into the round
    # report's failure_reason, through real retries and the real activity.
    assert "OOMKilled" in failed.failure_reason
    assert "MemoryError" in failed.failure_reason


# ---------------------------------------------------------------------------
# P4 Task 2: kfp_run_id carried in the workflow memo, so the Temporal UI can
# reverse-link back to the KFP round without opening the workflow's history.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_train_round_workflow_memo_carries_kfp_run_id():
    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client, task_queue=TASK_QUEUE,
            workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=[launch, cleanup],
        ):
            handle = await env.client.start_workflow(
                TrainRoundWorkflow.run,
                # kfp_run_id ("abcdef1234") is run_uid, which names Jobs but
                # does not resolve in KFP's UI. The memo must carry the backend
                # id instead -- a memo built from run_uid links nowhere while
                # looking correct, which is exactly what the live P4 gate found.
                _round_spec(kfp_backend_run_id="3b66067f-040b-461f-8b1a-a153cc6a13b4"),
                id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            await handle.result()
            desc = await handle.describe()
    assert await desc.memo_value("kfp_run_id") == "3b66067f-040b-461f-8b1a-a153cc6a13b4"


@pytest.mark.asyncio
async def test_train_round_workflow_skips_the_memo_when_the_backend_id_is_unknown():
    """No memo beats a memo pointing at a run KFP cannot find."""

    @activity.defn(name="launch_and_watch_pod")
    async def launch(spec: WorkerSpec) -> WorkerResult:
        return _ok(spec)

    @activity.defn(name="cleanup_worker_job")
    async def cleanup(spec: WorkerSpec) -> None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client, task_queue=TASK_QUEUE,
            workflows=[TrainRoundWorkflow, WorkerWorkflow], activities=[launch, cleanup],
        ):
            handle = await env.client.start_workflow(
                TrainRoundWorkflow.run, _round_spec(),  # kfp_backend_run_id defaults to ""
                id=f"t-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            await handle.result()
            desc = await handle.describe()
    # memo_value raises rather than returning None when the key is absent --
    # which is the assertion: nothing was written at all.
    with pytest.raises(KeyError):
        await desc.memo_value("kfp_run_id")


@pytest.mark.asyncio
async def test_real_activity_failure_below_quorum_still_fails_the_round(monkeypatch):
    from temporalio.client import WorkflowFailureError

    monkeypatch.setattr(
        activities_module, "_k8s_batch_and_core",
        lambda: (_MixedBatchApi("-w1"), _MixedCoreApi("-w1")),
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        with pytest.raises(WorkflowFailureError):
            await _run(
                env, [launch_and_watch_pod, cleanup_worker_job],
                _round_spec(num_workers=2, min_workers=2),
            )
