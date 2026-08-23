import io
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/core")))

from metrics_csv import collect_metrics_rows

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
    """In-memory stand-in for minio.Minio, just enough surface for metrics_csv.py."""

    def __init__(self):
        self._store: dict[tuple[str, str], bytes] = {}

    def get_object(self, bucket, key):
        if (bucket, key) not in self._store:
            raise KeyError(f"no such object: {bucket}/{key}")
        return _FakeResponse(self._store[(bucket, key)])

    def put_object(self, bucket, key, data, length, content_type=None):
        self._store[(bucket, key)] = data.read()


def _seed_metrics(fake: FakeMinioClient, fl_round: int, rank: int, twin_id: str, rows: list) -> None:
    payload = {"round": fl_round, "rank": rank, "twin_id": twin_id, "rows": rows}
    data = json.dumps(payload).encode("utf-8")
    fake.put_object(
        BUCKET, f"round_{fl_round}/workers/worker_{rank}_metrics.json", io.BytesIO(data), length=len(data)
    )


def test_collect_metrics_rows_flattens_every_worker_in_one_round():
    fake = FakeMinioClient()
    _seed_metrics(
        fake, fl_round=0, rank=0, twin_id="eval-twin-global",
        rows=[{"mode": "EVAL", "reward": 1.0, "loss": 0.0}],
    )
    _seed_metrics(
        fake, fl_round=0, rank=1, twin_id="train-twin-1",
        rows=[
            {"mode": "TRAIN", "reward": 2.0, "loss": 0.5, "num_examples": 10},
            {"mode": "EVAL", "reward": 2.5, "loss": 0.0},
        ],
    )

    rows = collect_metrics_rows(fake, BUCKET, fl_rounds=1, num_workers=2)

    assert rows == [
        [1, "eval-twin-global", "EVAL", 1.0, 0.0],
        [1, "train-twin-1", "TRAIN", 2.0, 0.5],
        [1, "train-twin-1", "EVAL", 2.5, 0.0],
    ]


def test_collect_metrics_rows_spans_multiple_rounds_in_order():
    fake = FakeMinioClient()
    for fl_round in (0, 1):
        _seed_metrics(
            fake, fl_round=fl_round, rank=0, twin_id="eval-twin-global",
            rows=[{"mode": "EVAL", "reward": float(fl_round), "loss": 0.0}],
        )

    rows = collect_metrics_rows(fake, BUCKET, fl_rounds=2, num_workers=1)

    assert [r[0] for r in rows] == [1, 2]


def test_collect_metrics_rows_returns_empty_list_for_zero_rounds():
    fake = FakeMinioClient()
    assert collect_metrics_rows(fake, BUCKET, fl_rounds=0, num_workers=3) == []
