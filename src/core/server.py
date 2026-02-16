import flwr as fl
import os

ROUNDS = int(os.getenv("FL_ROUNDS", "5"))
EVAL_EPISODES = int(os.getenv("EVAL_EPISODES", "20"))

def main():
    MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2")) 

    print(f"Starting Federated Digital Twin Aggregator...")
    print(f"Rounds: {ROUNDS}, Minimum Clients Required: {MIN_CLIENTS}")
    print(f"Evaluation Episodes: {EVAL_EPISODES}")

    def fit_config(server_round: int):
        print(f"--- ROUND {server_round} START ---")
        return {"server_round": server_round}

    def eval_config(server_round: int):
        return {"server_round": server_round, "eval_episodes": EVAL_EPISODES}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Sample all available clients for training
        fraction_evaluate=1.0,      # Sample all for evaluation
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
