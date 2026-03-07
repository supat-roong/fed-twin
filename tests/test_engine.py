import torch
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.engine import DigitalTwinEnv, PolicyNet


def test_digital_twin_env_eval_mode():
    env = DigitalTwinEnv(twin_id="test_eval", eval_only=True)

    # Check if physics are neutral
    assert env.env.unwrapped.gravity == 9.8
    assert env.env.unwrapped.masscart == 1.0
    assert env.env.unwrapped.masspole == 0.1
    assert env.env.unwrapped.length == 0.5


def test_digital_twin_env_train_mode():
    env = DigitalTwinEnv(twin_id="test_train", eval_only=False)

    # Check if physics are randomized but within expected bounds (±15%)
    assert 0.85 * 9.8 <= env.env.unwrapped.gravity <= 1.15 * 9.8
    assert 0.85 * 1.0 <= env.env.unwrapped.masscart <= 1.15 * 1.0
    assert 0.85 * 0.1 <= env.env.unwrapped.masspole <= 1.15 * 0.1
    assert 0.85 * 0.5 <= env.env.unwrapped.length <= 1.15 * 0.5


def test_policy_net_forward():
    policy = PolicyNet(state_dim=4, action_dim=2)
    # CartPole state is a 4-dimensional array
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    action_probs = policy(state)

    # Output should have shape (1, 2)
    assert action_probs.shape == (1, 2)
    # Probabilities should sum to 1
    assert torch.allclose(action_probs.sum(), torch.tensor(1.0))
    # Probabilities should be between 0 and 1
    assert torch.all((action_probs >= 0) & (action_probs <= 1))


def test_collect_experience():
    env = DigitalTwinEnv(twin_id="test_exp", eval_only=True)
    policy = PolicyNet(state_dim=4, action_dim=2)

    states, actions, rewards, episode_ends = env.collect_experience(
        policy, n_episodes=2
    )

    # Check that we have transitions
    assert len(states) > 0
    assert len(actions) == len(states)
    assert len(rewards) == len(states)

    # Check that we captured 2 episodes
    assert len(episode_ends) == 2
    # The last episode end should be at the final index
    assert episode_ends[-1] == len(states) - 1
