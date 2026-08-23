import io
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/core")))

import pytest
import torch

from engine import PolicyNet
from worker_entrypoint import _download_checkpoint, _upload_state_dict, run_worker

BUCKET = "fed-twin-test-bucket"


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
    """In-memory stand-in for minio.Minio, just enough surface for worker_entrypoint.py."""

    def __init__(self):
        self._store: dict[tuple[str, str], bytes] = {}

    def get_object(self, bucket, key):
        if (bucket, key) not in self._store:
            raise KeyError(f"no such object: {bucket}/{key}")
        return _FakeResponse(self._store[(bucket, key)])

    def put_object(self, bucket, key, data, length, content_type=None):
        self._store[(bucket, key)] = data.read()

    def has(self, bucket, key) -> bool:
        return (bucket, key) in self._store

    def get_bytes(self, bucket, key) -> bytes:
        return self._store[(bucket, key)]


def test_download_checkpoint_round_zero_returns_fresh_model_without_any_minio_call():
    fake = FakeMinioClient()  # empty — get_object would raise if ever called
    model = _download_checkpoint(fake, BUCKET, fl_round=0)
    assert isinstance(model, PolicyNet)


def test_download_checkpoint_loads_the_previous_rounds_global_checkpoint():
    fake = FakeMinioClient()
    seeded = PolicyNet()
    torch.manual_seed(123)
    for p in seeded.parameters():
        p.data.fill_(0.5)
    _upload_state_dict(fake, BUCKET, "round_1/global.pt", seeded.state_dict())

    model = _download_checkpoint(fake, BUCKET, fl_round=2)

    for key, value in model.state_dict().items():
        assert torch.equal(value, seeded.state_dict()[key])


def test_eval_worker_rank_zero_uploads_metrics_but_no_weights():
    fake = FakeMinioClient()
    _upload_state_dict(fake, BUCKET, "round_0/global.pt", PolicyNet().state_dict())

    run_worker(
        fake, BUCKET, rank=0, fl_round=1, local_episodes=1, eval_episodes=1
    )

    assert fake.has(BUCKET, "round_1/workers/worker_0_metrics.json")
    assert not fake.has(BUCKET, "round_1/workers/worker_0_weights.pt")

    payload = json.loads(fake.get_bytes(BUCKET, "round_1/workers/worker_0_metrics.json"))
    assert payload["twin_id"] == "eval-twin-global"
    assert payload["round"] == 1
    modes = {row["mode"] for row in payload["rows"]}
    assert modes == {"EVAL"}


def test_training_worker_uploads_weights_and_both_train_and_eval_rows():
    fake = FakeMinioClient()
    _upload_state_dict(fake, BUCKET, "round_0/global.pt", PolicyNet().state_dict())

    run_worker(
        fake, BUCKET, rank=1, fl_round=1, local_episodes=1, eval_episodes=1
    )

    assert fake.has(BUCKET, "round_1/workers/worker_1_weights.pt")
    assert fake.has(BUCKET, "round_1/workers/worker_1_metrics.json")

    payload = json.loads(fake.get_bytes(BUCKET, "round_1/workers/worker_1_metrics.json"))
    assert payload["twin_id"] == "train-twin-1"
    modes = {row["mode"] for row in payload["rows"]}
    assert modes == {"TRAIN", "EVAL"}

    # uploaded weights must be a real, loadable PolicyNet state dict
    buf = io.BytesIO(fake.get_bytes(BUCKET, "round_1/workers/worker_1_weights.pt"))
    reloaded = PolicyNet()
    reloaded.load_state_dict(torch.load(buf))


def test_round_zero_training_worker_needs_no_prior_checkpoint():
    fake = FakeMinioClient()  # no round_-1 checkpoint exists, and none should be sought

    run_worker(
        fake, BUCKET, rank=1, fl_round=0, local_episodes=1, eval_episodes=1
    )

    assert fake.has(BUCKET, "round_0/workers/worker_1_weights.pt")
    assert fake.has(BUCKET, "round_0/workers/worker_1_metrics.json")
