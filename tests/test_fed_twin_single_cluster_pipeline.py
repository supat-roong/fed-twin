import json
import os
import sys
from types import SimpleNamespace

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError

# Add src to Python path (matches this repo's other tests/test_*.py files --
# there is no root conftest.py, pyproject pythonpath setting, or editable
# install that puts `src` on sys.path otherwise).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.types import RoundReport, WorkerResult, WorkerStatus
from src.pipelines.fed_twin_single_cluster_pipeline import train_workers


class _FakeHandle:
    def __init__(self, statuses, cause, report=None):
        self.id = "ftwn-train-abcdef12-r0"
        self._statuses = statuses
        self._cause = cause
        self._report = report

    async def result(self):
        if self._cause is not None:
            raise WorkflowFailureError(cause=self._cause)
        return self._report

    async def query(self, query_fn, *args, **kwargs):
        return self._statuses


class _FakeClient:
    def __init__(self, handle):
        self._handle = handle

    async def start_workflow(self, *args, **kwargs):
        return self._handle


def _base_kwargs(worker_report) -> dict:
    return dict(
        fl_round=0, num_workers=3, local_episodes=2, eval_episodes=2,
        namespace="kubeflow", temporal_address="temporal:7233",
        kfp_run_id="abcdef1234", mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp", mlflow_run_id="run-1",
        minio_endpoint="minio:9000", minio_access_key="minio",
        minio_secret_key="minio123", minio_bucket="mlflow-artifacts",
        worker_image="fed-twin-app:v1", learning_rate=0.003, gamma=0.99,
        entropy_coeff=0.01, max_grad_norm=0.5, worker_report=worker_report,
    )


def test_worker_report_artifact_written_when_round_fails_quorum(tmp_path, monkeypatch):
    statuses = {
        0: WorkerStatus(worker_id=0, phase="Succeeded", attempt=1, message=""),
        1: WorkerStatus(worker_id=1, phase="Failed", attempt=1, message="worker 1 failed: OOMKilled"),
        2: WorkerStatus(worker_id=2, phase="Succeeded", attempt=1, message=""),
    }
    cause = ApplicationError("round 0: only 2 of 3 workers succeeded, need 3", non_retryable=True)
    handle = _FakeHandle(statuses, cause)

    async def fake_connect(target_host, **kwargs):
        return _FakeClient(handle)

    import temporalio.client
    monkeypatch.setattr(temporalio.client.Client, "connect", fake_connect)

    report_path = tmp_path / "worker_report.json"
    worker_report = SimpleNamespace(path=str(report_path))

    with pytest.raises(Exception):  # noqa: B017 - WorkflowFailureError re-raised as-is
        train_workers.python_func(**_base_kwargs(worker_report))

    assert report_path.exists(), (
        "worker_report artifact must be written even when the round fails "
        "quorum, not just on success"
    )
    payload = json.loads(report_path.read_text())
    assert payload["succeeded"] == [0, 2]
    assert payload["failed"] == [1]
    assert "OOMKilled" in json.dumps(payload)


def test_worker_report_artifact_written_on_success(tmp_path, monkeypatch):
    report = RoundReport(
        fl_round=0,
        results=[
            WorkerResult(worker_id=0, succeeded=True, attempts=1, failure_reason="", job_name="j0"),
            WorkerResult(worker_id=1, succeeded=True, attempts=1, failure_reason="", job_name="j1"),
            WorkerResult(worker_id=2, succeeded=True, attempts=1, failure_reason="", job_name="j2"),
        ],
    )
    handle = _FakeHandle(statuses={}, cause=None, report=report)

    async def fake_connect(target_host, **kwargs):
        return _FakeClient(handle)

    import temporalio.client
    monkeypatch.setattr(temporalio.client.Client, "connect", fake_connect)

    report_path = tmp_path / "worker_report.json"
    worker_report = SimpleNamespace(path=str(report_path))

    train_workers.python_func(**_base_kwargs(worker_report))

    payload = json.loads(report_path.read_text())
    assert payload["succeeded"] == [0, 1, 2]
    assert payload["failed"] == []
    assert payload["temporal_workflow_id"] == "ftwn-train-abcdef12-r0"
