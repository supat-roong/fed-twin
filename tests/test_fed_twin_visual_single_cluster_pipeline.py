import csv
import json
import os
import sys
from types import SimpleNamespace

import pytest

# Add repo root to Python path (matches this repo's other tests/test_*.py
# files -- there is no root conftest.py, pyproject pythonpath setting, or
# editable install that puts `src` on sys.path otherwise).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.types import WorkerResult
from src.pipelines.fed_twin_visual_single_cluster_pipeline import (
    make_worker_ids,
    run_worker_via_temporal,
)


class _FakeHandle:
    def __init__(self, result):
        self.id = "ftwn-vis-abcdef12-r0-w1"
        self._result = result

    async def result(self):
        return self._result


class _FakeClient:
    def __init__(self, handle):
        self._handle = handle
        self.started_specs = []

    async def start_workflow(self, *args, **kwargs):
        self.started_specs.append(args[1])
        return self._handle


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeMinioClient:
    """In-memory stand-in for minio.Minio; constructor-compatible so it can
    monkeypatch the Minio class itself."""

    _store: dict = {}

    def __init__(self, endpoint=None, access_key=None, secret_key=None, secure=False):
        pass

    def get_object(self, bucket, key):
        if (bucket, key) not in self._store:
            raise KeyError(f"no such object: {bucket}/{key}")
        return _FakeResponse(self._store[(bucket, key)])


def _base_kwargs(metrics) -> dict:
    return dict(
        worker_id=1, fl_round=0, visual_round=1, num_workers=2,
        local_episodes=2, eval_episodes=2, namespace="kubeflow",
        temporal_address="temporal:7233", kfp_run_id="abcdef1234",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        mlflow_run_id="run-1", minio_endpoint="minio:9000",
        minio_access_key="minio", minio_secret_key="minio123",
        minio_bucket="mlflow-artifacts", worker_image="fed-twin-app:v1",
        learning_rate=0.003, gamma=0.99, entropy_coeff=0.01,
        max_grad_norm=0.5, metrics=metrics,
    )


def test_make_worker_ids_lists_training_twins():
    assert make_worker_ids.python_func(num_workers=3) == [1, 2, 3]


def test_run_worker_writes_the_nodes_csv_and_builds_the_fleet_spec(tmp_path, monkeypatch):
    result = WorkerResult(worker_id=1, succeeded=True, attempts=1, failure_reason="", job_name="j1")
    handle = _FakeHandle(result)
    client = _FakeClient(handle)

    async def fake_connect(target_host, **kwargs):
        return client

    import temporalio.client
    monkeypatch.setattr(temporalio.client.Client, "connect", fake_connect)

    payload = {
        "round": 0, "rank": 1, "twin_id": "train-twin-1",
        "rows": [
            {"mode": "TRAIN", "reward": 2.0, "loss": 0.5, "num_examples": 10},
            {"mode": "EVAL", "reward": 2.5, "loss": 0.0},
        ],
    }
    FakeMinioClient._store = {
        ("mlflow-artifacts", "round_0/workers/worker_1_metrics.json"):
            json.dumps(payload).encode("utf-8"),
    }
    import minio
    monkeypatch.setattr(minio, "Minio", FakeMinioClient)

    metrics_path = tmp_path / "metrics.csv"
    status = run_worker_via_temporal.python_func(
        **_base_kwargs(SimpleNamespace(path=str(metrics_path)))
    )

    assert len(client.started_specs) == 1
    spec = client.started_specs[0]
    assert spec.num_workers == 3, "fleet must be num_workers(2) + 1 eval twin"
    assert spec.worker_id == 1
    assert spec.fl_round == 0

    with open(metrics_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["round", "twin_id", "mode", "reward", "loss"]
    assert rows[1] == ["1", "train-twin-1", "TRAIN", "2.0", "0.5"]
    assert rows[2] == ["1", "train-twin-1", "EVAL", "2.5", "0.0"]
    assert "worker 1" in status


def test_run_worker_raises_when_the_worker_failed(tmp_path, monkeypatch):
    result = WorkerResult(
        worker_id=1, succeeded=False, attempts=3,
        failure_reason="OOMKilled (exit 137)", job_name="j1",
    )
    handle = _FakeHandle(result)
    client = _FakeClient(handle)

    async def fake_connect(target_host, **kwargs):
        return client

    import temporalio.client
    monkeypatch.setattr(temporalio.client.Client, "connect", fake_connect)

    FakeMinioClient._store = {}
    import minio
    monkeypatch.setattr(minio, "Minio", FakeMinioClient)

    with pytest.raises(RuntimeError, match="OOMKilled"):
        run_worker_via_temporal.python_func(
            **_base_kwargs(SimpleNamespace(path=str(tmp_path / "metrics.csv")))
        )
