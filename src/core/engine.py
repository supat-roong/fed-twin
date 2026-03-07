import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np


# --- Digital Twin Simulation Engine ---
class DigitalTwinEnv:
    """
    Wrapper around Gymnasium to simulate a robot/agent as a digital twin.
    Each twin pod can have slightly different dynamics (e.g., friction, mass).
    """

    def __init__(self, twin_id, eval_only=False):
        self.twin_id = twin_id
        self.eval_only = eval_only

        # Deterministic seed based on twin_id string
        # 'robot-01' -> some integer
        import hashlib

        hash_object = hashlib.md5(twin_id.encode())
        self.seed = int(hash_object.hexdigest(), 16) % 10000

        # Initialize env with this seed
        self.env = gym.make("CartPole-v1", render_mode=None)
        self.env.reset(seed=self.seed)

        # Customize physics based on twin ID and mode
        rng = np.random.default_rng(self.seed)

        if eval_only:
            # EVALUATION TWIN: Use neutral/default physics for unbiased testing
            self.env.unwrapped.gravity = 9.8
            self.env.unwrapped.masscart = 1.0
            self.env.unwrapped.masspole = 0.1
            self.env.unwrapped.length = 0.5
            print(
                f"[{twin_id}] EVAL MODE - Neutral physics: Gravity=9.8, CartMass=1.0, PoleMass=0.1, Length=0.5"
            )
        else:
            # TRAINING TWIN: Use MODERATE physics variations for effective FL
            # Moderate ranges allow FL workers to learn compatible strategies

            # Gravity: 9.8 ± 15% (balanced variation for FL effectiveness)
            gravity_noise = rng.uniform(0.85, 1.15)
            self.env.unwrapped.gravity = 9.8 * gravity_noise

            # Cart Mass: 1.0 ± 15% (moderate variation)
            mass_noise = rng.uniform(0.85, 1.15)
            self.env.unwrapped.masscart = 1.0 * mass_noise

            # Pole Mass: 0.1 ± 15% (moderate variation)
            pole_mass_noise = rng.uniform(0.85, 1.15)
            self.env.unwrapped.masspole = 0.1 * pole_mass_noise

            # Pole Length: 0.5 ± 15% (moderate variation)
            pole_length_noise = rng.uniform(0.85, 1.15)
            self.env.unwrapped.length = 0.5 * pole_length_noise

            print(
                f"[{twin_id}] TRAIN MODE - Moderate physics (±15%): Gravity={self.env.unwrapped.gravity:.2f}, "
                f"CartMass={self.env.unwrapped.masscart:.2f}, PoleMass={self.env.unwrapped.masspole:.3f}, "
                f"PoleLen={self.env.unwrapped.length:.3f}"
            )

    def collect_experience(self, policy, n_episodes=5):
        """
        Collects experience for local training.
        Returns states, actions, rewards, and episode_ends (indices where episodes terminate).
        """
        states, actions, rewards = [], [], []
        episode_ends = []  # Track where episodes end

        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                state_t = torch.tensor(state, dtype=torch.float32)
                action_probs = policy(state_t)
                action = torch.multinomial(action_probs, 1).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            # Mark the end of this episode
            episode_ends.append(len(rewards) - 1)

        return np.array(states), np.array(actions), np.array(rewards), episode_ends


# --- Policy Network ---
class PolicyNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
