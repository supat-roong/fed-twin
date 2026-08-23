import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/core")))

import pytest
import torch
from aggregate import aggregate_state_dicts, run_aggregate_round

BUCKET = "fed-twin-test-bucket"


def test_aggregate_state_dicts_means_each_key():
    a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([10.0])}
    b = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([20.0])}
    result = aggregate_state_dicts([a, b])
    assert torch.equal(result["w"], torch.tensor([2.0, 3.0]))
    assert torch.equal(result["b"], torch.tensor([15.0]))


def test_aggregate_state_dicts_matches_the_original_inline_math():
    """Golden-reference test: reproduces fed_twin_visual_single_cluster_pipeline.py's
    aggregate_models logic independently (torch.stack + mean per key) and checks
    aggregate_state_dicts produces bit-identical output — the math must not drift
    from what today's visual pipeline already does.
    """
    torch.manual_seed(0)
    state_dicts = [
        {"layer.weight": torch.randn(4, 3), "layer.bias": torch.randn(3)}
        for _ in range(3)
    ]

    expected = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        expected[key] = torch.mean(stacked, dim=0)

    result = aggregate_state_dicts(state_dicts)
    for key in expected:
        assert torch.equal(result[key], expected[key])


def test_aggregate_state_dicts_requires_at_least_one_input():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_state_dicts([])


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
    """In-memory stand-in for minio.Minio, just enough surface for aggregate.py."""

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


def _seed_weights(fake: FakeMinioClient, fl_round: int, rank: int, state_dict: dict) -> None:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    data = buf.getvalue()
    fake.put_object(
        BUCKET, f"round_{fl_round}/workers/worker_{rank}_weights.pt", io.BytesIO(data), length=len(data)
    )


def test_run_aggregate_round_writes_the_mean_of_training_workers_only():
    fake = FakeMinioClient()
    _seed_weights(fake, fl_round=0, rank=1, state_dict={"w": torch.tensor([1.0, 2.0])})
    _seed_weights(fake, fl_round=0, rank=2, state_dict={"w": torch.tensor([3.0, 4.0])})

    run_aggregate_round(fake, BUCKET, fl_round=0, num_workers=3)

    assert fake.has(BUCKET, "round_0/global.pt")
    result = torch.load(io.BytesIO(fake.get_bytes(BUCKET, "round_0/global.pt")))
    assert torch.equal(result["w"], torch.tensor([2.0, 3.0]))


def test_run_aggregate_round_never_reads_rank_zero_the_eval_twin():
    fake = FakeMinioClient()
    # No weights ever uploaded for rank 0 (the eval twin never writes weights,
    # D5) -- if run_aggregate_round tried to read it, this would raise
    # KeyError before the assertion below ever runs.
    _seed_weights(fake, fl_round=0, rank=1, state_dict={"w": torch.tensor([5.0])})

    run_aggregate_round(fake, BUCKET, fl_round=0, num_workers=2)

    assert fake.has(BUCKET, "round_0/global.pt")
    result = torch.load(io.BytesIO(fake.get_bytes(BUCKET, "round_0/global.pt")))
    assert torch.equal(result["w"], torch.tensor([5.0]))


def test_run_aggregate_round_raises_when_no_training_workers_exist():
    fake = FakeMinioClient()
    with pytest.raises(ValueError, match="at least one"):
        # num_workers=1 means only rank 0 (eval-only) exists -- range(1, 1) is
        # empty, so aggregate_state_dicts gets nothing to average.
        run_aggregate_round(fake, BUCKET, fl_round=0, num_workers=1)
