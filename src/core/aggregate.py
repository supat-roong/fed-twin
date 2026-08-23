"""
Plain-mean FedAvg aggregation, with clean separation of math from I/O.

Promoted from fed_twin_visual_single_cluster_pipeline.py's inline aggregate_models
KFP component — same math (torch.stack + mean, per state_dict key), extracted into
a shared module for reuse between the visual and functional pipelines (Phase 3).

The module maintains clear separation of concerns: `aggregate_state_dicts` is the
pure, dependency-free math function (works on in-memory state dicts); `run_aggregate_round`
is the deliberate I/O wrapper that reads training-worker weights from MinIO, delegates
the averaging to `aggregate_state_dicts`, and writes the aggregated result back to MinIO.
"""

from __future__ import annotations

import io
import torch


def aggregate_state_dicts(state_dicts: list[dict]) -> dict:
    """Mean of N PyTorch state dicts, key by key.

    Every state dict must have the same keys (they're all snapshots of the same
    model architecture — PolicyNet — trained from the same global checkpoint).
    """
    if not state_dicts:
        raise ValueError("aggregate_state_dicts requires at least one state dict")

    avg_state_dict = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state_dict[key] = torch.mean(stacked, dim=0)
    return avg_state_dict


def run_aggregate_round(minio_client, minio_bucket: str, fl_round: int, num_workers: int) -> None:
    """Mean this round's training-worker weights and write the next round's
    global checkpoint.

    Reads ranks 1..num_workers-1 (rank 0 is always the eval-only twin, D5 --
    it never uploads weights, so it's never read here). Delegates the actual
    math to aggregate_state_dicts, which already raises ValueError if there
    is nothing to average (e.g. num_workers == 1: no training workers at
    all).
    """
    state_dicts = []
    for rank in range(1, num_workers):
        response = minio_client.get_object(
            minio_bucket, f"round_{fl_round}/workers/worker_{rank}_weights.pt"
        )
        try:
            buf = io.BytesIO(response.read())
        finally:
            response.close()
            response.release_conn()
        state_dicts.append(torch.load(buf))

    avg_state_dict = aggregate_state_dicts(state_dicts)

    out_buf = io.BytesIO()
    torch.save(avg_state_dict, out_buf)
    data = out_buf.getvalue()
    minio_client.put_object(
        minio_bucket, f"round_{fl_round}/global.pt", io.BytesIO(data), length=len(data)
    )
