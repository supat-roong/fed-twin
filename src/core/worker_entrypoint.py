"""
One-shot Job body for one FL round's worker: downloads the current global
checkpoint, trains (or, for the eval twin, only evaluates), and uploads its
weights/metrics to MinIO. Replaces client.py's Flower start_numpy_client()
entrypoint — twin.py's TwinClient class itself is reused completely
unchanged, only how it's invoked changes.

Twin identity is computed from RANK, not read from an env var: RANK==0 is
always the eval twin ("eval-twin-global"); every other RANK is a training
twin ("train-twin-{RANK}"). This mirrors today's PyTorchJob hostname-suffix
convention exactly, without needing to parse a hostname at all.

LEARNING_RATE/GAMMA/ENTROPY_COEFF/MAX_GRAD_NORM are deliberately not read
anywhere in this file: twin.py's own module-level os.getenv(...) constants
already pick them up the moment "from twin import TwinClient" executes,
as long as they're already in the process environment (build_job_manifest
guarantees this in production).
"""

from __future__ import annotations

import io
import json
import os

import torch
from engine import PolicyNet, get_parameters
from twin import TwinClient


def _download_checkpoint(minio_client, bucket: str, fl_round: int) -> PolicyNet:
    """Fresh PolicyNet for round 0; otherwise loads round_{fl_round-1}/global.pt."""
    model = PolicyNet()
    if fl_round == 0:
        return model

    key = f"round_{fl_round - 1}/global.pt"
    response = minio_client.get_object(bucket, key)
    try:
        buf = io.BytesIO(response.read())
    finally:
        response.close()
        response.release_conn()
    model.load_state_dict(torch.load(buf))
    return model


def _upload_state_dict(minio_client, bucket: str, key: str, state_dict: dict) -> None:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    data = buf.getvalue()
    minio_client.put_object(bucket, key, io.BytesIO(data), length=len(data))


def _upload_metrics(minio_client, bucket: str, key: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    minio_client.put_object(
        bucket, key, io.BytesIO(data), length=len(data), content_type="application/json"
    )


def run_worker(
    minio_client,
    minio_bucket: str,
    rank: int,
    fl_round: int,
    local_episodes: int,
    eval_episodes: int,
) -> None:
    twin_id = "eval-twin-global" if rank == 0 else f"train-twin-{rank}"
    eval_only = rank == 0

    model = _download_checkpoint(minio_client, minio_bucket, fl_round)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=eval_only)
    params = get_parameters(model)

    rows = []
    if eval_only:
        _, _, eval_results = client.evaluate(
            params, {"server_round": fl_round, "eval_episodes": eval_episodes}
        )
        rows.append({"mode": "EVAL", "reward": eval_results["reward"], "loss": 0.0})
    else:
        new_params, num_examples, fit_results = client.fit(
            params, {"server_round": fl_round, "local_episodes": local_episodes}
        )
        rows.append(
            {
                "mode": "TRAIN",
                "reward": fit_results["reward"],
                "loss": fit_results["loss"],
                "num_examples": num_examples,
            }
        )
        _, _, eval_results = client.evaluate(
            new_params, {"server_round": fl_round, "eval_episodes": eval_episodes}
        )
        rows.append({"mode": "EVAL", "reward": eval_results["reward"], "loss": 0.0})
        # fit() trains self.model (== model) in place via optimizer.step(), so
        # model.state_dict() already holds the post-training weights here.
        _upload_state_dict(
            minio_client,
            minio_bucket,
            f"round_{fl_round}/workers/worker_{rank}_weights.pt",
            model.state_dict(),
        )

    _upload_metrics(
        minio_client,
        minio_bucket,
        f"round_{fl_round}/workers/worker_{rank}_metrics.json",
        {"round": fl_round, "rank": rank, "twin_id": twin_id, "rows": rows},
    )


def main() -> None:
    from minio import Minio

    rank = int(os.environ["RANK"])
    fl_round = int(os.environ["FL_ROUND"])
    local_episodes = int(os.environ["LOCAL_EPISODES"])
    eval_episodes = int(os.environ["EVAL_EPISODES"])
    minio_endpoint = os.environ["MINIO_ENDPOINT"]
    minio_bucket = os.environ["MINIO_BUCKET"]

    # tracking.py's setup_mlflow() (called by TwinClient.__init__) only configures
    # S3 artifact credentials when this is set — same MinIO instance, just needs
    # the URL scheme MLflow's S3 client expects.
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"

    minio_client = Minio(
        minio_endpoint,
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )
    run_worker(minio_client, minio_bucket, rank, fl_round, local_episodes, eval_episodes)


if __name__ == "__main__":
    main()
