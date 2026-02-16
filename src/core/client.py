import os
import flwr as fl
from engine import PolicyNet, DigitalTwinEnv, get_parameters, set_parameters
import torch
import torch.optim as optim
import numpy as np

TWIN_ID = os.getenv("TWIN_ID", "robot-01")
SERVER_ADDR = os.getenv("SERVER_ADDR", "fl-server:8080")
LOCAL_EPISODES = int(os.getenv("LOCAL_EPISODES", "50"))
EVAL_EPISODES = int(os.getenv("EVAL_EPISODES", "10"))

EVAL_ONLY = os.getenv("EVAL_ONLY", "false").lower() == "true"

# RL Hyperparameters (configurable via environment variables)
GAMMA = float(os.getenv("GAMMA", "0.99"))
ENTROPY_COEFF = float(os.getenv("ENTROPY_COEFF", "0.01"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "0.5"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))

def compute_returns(rewards, episode_ends, gamma=GAMMA):
    """
    Compute discounted returns (reward-to-go) for each timestep.
    Args:
        rewards: array of rewards for all timesteps
        episode_ends: list of indices where episodes end
        gamma: discount factor
    Returns:
        array of discounted returns
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    episode_start = 0
    
    for episode_end in episode_ends:
        # Extract rewards for this episode
        episode_rewards = rewards[episode_start:episode_end + 1]
        episode_length = len(episode_rewards)
        
        # Compute discounted returns backward from end
        episode_returns = np.zeros(episode_length, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(episode_length)):
            running_return = episode_rewards[t] + gamma * running_return
            episode_returns[t] = running_return
        
        # Store in main returns array
        returns[episode_start:episode_end + 1] = episode_returns
        episode_start = episode_end + 1
    
    return returns

class TwinClient(fl.client.NumPyClient):
    def __init__(self, model, twin_id=TWIN_ID, eval_only=EVAL_ONLY):
        self.model = model
        self.twin_id = twin_id
        self.eval_only = eval_only
        self.env = DigitalTwinEnv(twin_id, eval_only=eval_only)
        print(f"Twin {self.twin_id} initialized. Mode: {'EVALUATION' if self.eval_only else 'TRAINING'}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        sv_round = config.get("server_round", "?")
        episodes = int(config.get("local_episodes", LOCAL_EPISODES))
        
        if self.eval_only:
            print(f"Twin {self.twin_id} [Round {sv_round}] [METRIC] EVAL-ONLY-SKIP Reward: 0.0 Loss: 0.0")
            return self.get_parameters(config={}), 0, {}

        # Training Worker Logic
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # 1. Simulate - collect experience with episode boundaries
        states, actions, rewards, episode_ends = self.env.collect_experience(self.model, n_episodes=episodes)
        avg_reward = sum(rewards) / episodes if episodes > 0 else 0.0

        # 2. Compute discounted returns (reward-to-go)
        returns = compute_returns(rewards, episode_ends, gamma=GAMMA)
        
        # 3. Normalize returns for stability (baseline subtraction via standardization)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Convert to tensors
        state_t = torch.tensor(states, dtype=torch.float32)
        action_t = torch.tensor(actions, dtype=torch.long)
        return_t = torch.tensor(returns, dtype=torch.float32)

        # 4. Train (single epoch for on-policy REINFORCE)
        loss_history = []
        optimizer.zero_grad()
        
        # Forward pass
        action_probs = self.model(state_t)
        log_probs = torch.log(action_probs.gather(1, action_t.unsqueeze(1)).squeeze() + 1e-8)
        
        # Policy gradient loss (maximize log_prob * return)
        pg_loss = -(log_probs * return_t).mean()
        
        # Entropy regularization (encourage exploration)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -ENTROPY_COEFF * entropy
        
        # Total loss
        loss = pg_loss + entropy_loss
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        loss_history.append(loss.item())
            
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0
        
        # Consolidated Metric Line for the monitor
        print(f"Twin {self.twin_id} [Round {sv_round}] [METRIC] TRAIN Reward: {avg_reward:.2f} Loss: {avg_loss:.4f}")
        return self.get_parameters(config={}), len(states), {"reward": avg_reward, "loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        sv_round = config.get("server_round", "?")
        # Use eval_episodes for evaluation (separate from training episodes)
        episodes = int(config.get("eval_episodes", EVAL_EPISODES))
        
        mode = "EVAL-ONLY" if self.eval_only else "TRAIN-WORKER"
        _, _, rewards, _ = self.env.collect_experience(self.model, n_episodes=episodes)
        avg_reward = sum(rewards) / episodes
        
        # Consolidated Metric Line
        print(f"Twin {self.twin_id} [Round {sv_round}] [METRIC] EVAL Reward: {avg_reward:.2f} Loss: 0.0")
            
        return float(-avg_reward), episodes, {"reward": avg_reward}

if __name__ == "__main__":
    model = PolicyNet()
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=TwinClient(model))
