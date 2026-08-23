import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/core")))

import pytest
import torch

from aggregate import aggregate_state_dicts


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
