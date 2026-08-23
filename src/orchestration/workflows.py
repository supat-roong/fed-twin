"""
Temporal workflows owning the worker fleet within one federated round.

Scope boundary: KFP sequences rounds and owns the DAG and artifact lineage;
these workflows own the fleet inside a round. The two never overlap, so there
is no two-schedulers conflict.

Workflow code is replayed by Temporal and must stay deterministic — no I/O, no
clocks, no randomness. Everything with a side effect is an activity.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from src.orchestration.types import (
        RoundReport,
        RoundSpec,
        WorkerResult,
        WorkerSpec,
        WorkerStatus,
    )

TASK_QUEUE = "fed-twin"


def _root_cause_message(exc: BaseException) -> str:
    """Walk an exception's cause chain to the innermost non-empty message.

    A child-workflow failure's own message is always the generic wrapper text
    ("Child Workflow execution failed") regardless of what actually killed the
    worker; the real reason is further down the `.cause` chain (e.g.
    ChildWorkflowError -> ActivityError -> ApplicationError("boom")). Without
    this, RoundReport tells you a worker died but never why. Pure and
    deterministic — only walks `__cause__`, no I/O.
    """
    message = str(exc)
    current: BaseException | None = exc
    while current is not None:
        text = str(current)
        if text:
            message = text
        current = current.__cause__
    return message


@workflow.defn
class WorkerWorkflow:
    """One durable entity per (round, worker).

    Gives per-pod retry, a per-pod failure reason in Temporal history, and a
    queryable live status the dashboard and the KFP component both read.
    """

    def __init__(self) -> None:
        self._status = WorkerStatus(worker_id=-1)

    @workflow.run
    async def run(self, spec: WorkerSpec) -> WorkerResult:
        self._status = WorkerStatus(worker_id=spec.worker_id, phase="Pending")
        try:
            result: WorkerResult = await workflow.execute_activity(
                "launch_and_watch_pod",
                spec,
                result_type=WorkerResult,
                start_to_close_timeout=timedelta(seconds=3900),
                heartbeat_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=10),
                    # job_name_for's ValueError is a deterministic input-
                    # validation failure (an unsubstituted KFP placeholder or
                    # similar): every retry would build the exact same
                    # invalid name and fail identically, so retrying it three
                    # times only adds latency. A genuine worker/Job failure
                    # (WorkerJobFailed, activities.py) is deliberately *not*
                    # listed here -- it may be transient, and is exactly the
                    # case this retry policy exists to cover.
                    non_retryable_error_types=["ValueError"],
                ),
            )
            self._status.phase = "Succeeded" if result.succeeded else "Failed"
            self._status.message = result.failure_reason
            return result
        finally:
            # Runs on success, failure and cancellation. Without it a failed
            # round leaves orphaned Jobs that collide with the next attempt's
            # deterministic names. Cleanup is best-effort housekeeping: if it
            # fails, that failure must never overwrite or mask the worker's
            # real outcome, which is exactly the diagnostic this phase exists
            # to surface.
            try:
                await workflow.execute_activity(
                    "cleanup_worker_job",
                    spec,
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
            except Exception as e:  # noqa: BLE001 - cleanup must never mask the real result
                workflow.logger.warning(f"cleanup failed for worker {spec.worker_id}: {e}")

    @workflow.query
    def status(self) -> WorkerStatus:
        return self._status


@workflow.defn
class TrainRoundWorkflow:
    """Fans out one WorkerWorkflow child per worker and gathers the outcomes."""

    def __init__(self) -> None:
        self._statuses: dict[int, WorkerStatus] = {}

    @workflow.run
    async def run(self, spec: RoundSpec) -> RoundReport:
        parent_id = workflow.info().workflow_id

        # Carry kfp_run_id in the workflow memo so the Temporal UI
        # can reverse-link back to the KFP round (KFP -> Temporal is already
        # covered by the URL train_workers prints; this is the other
        # direction) without opening the workflow's full event history.
        # Deterministic and replay-safe: spec is the workflow's own input,
        # and upsert_memo is a workflow command, not I/O. Skipped when empty
        # for the same reason log_run_context skips empty ids -- an empty
        # memo value is worse than no memo entry.
        # kfp_backend_run_id, not kfp_run_id: the latter holds run_uid, which
        # names Jobs but does not resolve in KFP's UI, so a memo built from it
        # linked nowhere while looking correct (seen live: memo value
        # "23859aa5" against a real run id of 3b66067f-...).
        if spec.kfp_backend_run_id:
            workflow.upsert_memo({"kfp_run_id": spec.kfp_backend_run_id})

        async def _one(worker_id: int) -> WorkerResult:
            self._statuses[worker_id] = WorkerStatus(worker_id=worker_id, phase="Running")
            try:
                res = await workflow.execute_child_workflow(
                    WorkerWorkflow.run,
                    spec.worker_spec(worker_id),
                    id=f"{parent_id}-w{worker_id}",
                    task_queue=TASK_QUEUE,
                )
            except Exception as e:
                self._statuses[worker_id] = WorkerStatus(
                    worker_id=worker_id, phase="Failed", message=_root_cause_message(e)
                )
                raise
            self._statuses[worker_id] = WorkerStatus(
                worker_id=worker_id,
                phase="Succeeded" if res.succeeded else "Failed",
                attempt=res.attempts,
                message=res.failure_reason,
            )
            return res

        # return_exceptions=True so one dead worker does not abort the fleet;
        # the quorum check below decides whether the round can still proceed.
        raw = await asyncio.gather(
            *[_one(i) for i in range(spec.num_workers)], return_exceptions=True
        )

        results: list[WorkerResult] = []
        for worker_id, item in enumerate(raw):
            if isinstance(item, BaseException):
                results.append(
                    WorkerResult(
                        worker_id=worker_id, succeeded=False, attempts=0,
                        failure_reason=_root_cause_message(item), job_name="",
                    )
                )
            else:
                results.append(item)

        report = RoundReport(fl_round=spec.fl_round, results=results)
        if not report.meets_quorum(spec.min_workers):
            raise ApplicationError(
                f"round {spec.fl_round}: only {len(report.succeeded_ids)} of "
                f"{spec.num_workers} workers succeeded, need {spec.min_workers}",
                non_retryable=True,
            )
        return report

    @workflow.query
    def status(self) -> dict[int, WorkerStatus]:
        return self._statuses
