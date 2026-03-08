import numpy as np
import sys
import os
from unittest.mock import patch

# Add src and src/core to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/core"))
)

from core.engine import PolicyNet
from core.client import compute_returns, TwinClient


def test_compute_returns():
    rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    episode_ends = [2, 4]
    # Episode 1: ends at index 2 (length 3). Rewards: 1, 1, 1
    # Returns (gamma=0.9): 1 + 0.9 + 0.81 = 2.71, 1 + 0.9 = 1.9, 1
    # Episode 2: ends at index 4 (length 2). Rewards: 1, 1
    # Returns (gamma=0.9): 1.9, 1.0
    returns = compute_returns(rewards, episode_ends, gamma=0.9)
    assert len(returns) == 5
    assert np.allclose(returns, [2.71, 1.9, 1.0, 1.9, 1.0])


def test_twin_client_init():
    model = PolicyNet()
    client = TwinClient(model, twin_id="test_init", eval_only=False)
    assert client.twin_id == "test_init"
    assert not client.eval_only


def test_twin_client_get_set_parameters():
    model = PolicyNet()
    client = TwinClient(model, twin_id="test_params")

    # Get parameters
    params = client.get_parameters(config={})
    assert isinstance(params, list)
    assert len(params) > 0
    assert isinstance(params[0], np.ndarray)

    # Change first parameter slightly
    new_params = [p.copy() for p in params]
    new_params[0] += 0.1

    # Set parameters
    client.set_parameters(new_params)
    updated_params = client.get_parameters(config={})

    assert np.allclose(updated_params[0], new_params[0])


def test_twin_client_fit_eval_only():
    model = PolicyNet()
    client = TwinClient(model, twin_id="test_eval_fit", eval_only=True)

    parameters = client.get_parameters(config={})
    new_parameters, num_examples, metrics = client.fit(parameters, {"server_round": 1})

    assert num_examples == 0
    assert metrics == {}


@patch("core.client.log_metrics")
@patch("core.client.DigitalTwinEnv.collect_experience")
def test_twin_client_evaluate(mock_collect, mock_log):
    # Mock collect_experience to return dummy reward for speed
    mock_collect.return_value = (None, None, [100.0], None)

    model = PolicyNet()
    client = TwinClient(model, twin_id="test_eval")

    parameters = client.get_parameters(config={})
    # This will use the mocked collect_experience and be instant
    loss, num_examples, metrics = client.evaluate(
        parameters, {"server_round": 1, "eval_episodes": 1}
    )

    assert num_examples == 1
    assert "reward" in metrics
    assert metrics["reward"] == 100.0


@patch("core.client.log_metrics")
@patch("core.client.DigitalTwinEnv.collect_experience")
def test_twin_client_fit_train(mock_collect, mock_log):
    # Mock collect_experience to return dummy data for speed
    mock_collect.return_value = (
        np.zeros((10, 4)),  # states
        np.zeros(10, dtype=int),  # actions
        np.ones(10),  # rewards
        [4, 9],  # episode ends
    )

    model = PolicyNet()
    client = TwinClient(model, twin_id="test_fit", eval_only=False)

    parameters = client.get_parameters(config={})
    # This will use the mocked collect_experience and run the optimizer forward/backward over the dummy data
    new_parameters, num_examples, metrics = client.fit(
        parameters, {"server_round": 1, "local_episodes": 1}
    )

    assert num_examples == 10  # length of states returned by our mock
    assert "reward" in metrics
    assert "loss" in metrics
