"""
Plain-mean FedAvg aggregation.

Promoted from fed_twin_visual_single_cluster_pipeline.py's inline aggregate_models
KFP component — same math (torch.stack + mean, per state_dict key), extracted into
a pure, dependency-free function so it can be shared between the visual and
functional pipelines (Phase 3) instead of living only inside one generated file.

No I/O here deliberately: reading worker weights from MinIO and writing the
aggregated result back is Phase 3's job (the aggregate_and_evaluate KFP component),
once there's a real pipeline to drive it. This module only does the math.
"""

from __future__ import annotations

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
