"""
Reads every round's worker metrics.json from MinIO and flattens them into
the same (round, twin_id, mode, reward, loss) rows the flower-launcher
pipelines already write directly from their log-scrape.

Kept separate from aggregate.py: this never runs per-round (only once, after
every round in the pipeline's round loop has finished) and never touches
PyTorch or the aggregation math.
"""

from __future__ import annotations

import json


def collect_metrics_rows(minio_client, minio_bucket: str, fl_rounds: int, num_workers: int) -> list:
    """Flatten every round's worker metrics.json into CSV-ready rows.

    Schema read matches worker_entrypoint.py's _upload_metrics exactly:
    {"round": int, "rank": int, "twin_id": str, "rows": [{"mode": str,
    "reward": float, "loss": float, ...}]}. Every worker (including the
    eval-only rank 0) always uploads exactly one metrics.json per round --
    that upload is itself the completion signal Phase 1's
    wait_for_worker_artifact polls for, so it is always present by the time
    this function is called (after the whole round loop has finished).

    A missing key therefore raises out of get_object and fails the component
    loudly -- deliberately unhandled, since silence here was exactly the old
    log-scrape's failure mode.
    """
    rows = []
    for fl_round in range(fl_rounds):
        for rank in range(num_workers):
            key = f"round_{fl_round}/workers/worker_{rank}_metrics.json"
            response = minio_client.get_object(minio_bucket, key)
            try:
                payload = json.loads(response.read())
            finally:
                response.close()
                response.release_conn()
            twin_id = payload["twin_id"]
            for row in payload["rows"]:
                # fl_round is the 0-based MinIO key index; the CSV's round
                # column is 1-based everywhere else in this repo (Flower's
                # server_round, both visual pipelines), so shift here -- at
                # the presentation boundary, never in the storage keys.
                rows.append([fl_round + 1, twin_id, row["mode"], row["reward"], row["loss"]])
    return rows
