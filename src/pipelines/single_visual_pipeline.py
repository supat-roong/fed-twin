from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Model, Artifact
import json

# Load Config Defaults
try:
    with open("config/config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"fl_rounds": 5, "local_episodes": 10}

BASE_IMAGE = "fed-twin-app:v1"


@dsl.component(base_image=BASE_IMAGE)
def initialize_model_visual(model: Output[Model]):
    import torch
    from engine import PolicyNet

    net = PolicyNet()
    torch.save(net.state_dict(), model.path)
    print(f"Initialized global model at {model.path}")


@dsl.component(base_image=BASE_IMAGE)
def eval_step(
    twin_id: str,
    input_model: Input[Model],
    output_model: Output[Model],
    metrics: Output[Artifact],
    round_num: int,
    eval_episodes: int,
    learning_rate: float = 0.003,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
):
    import torch
    import csv
    import os
    from engine import PolicyNet, get_parameters
    from client import TwinClient

    # Set hyperparameters as environment variables
    os.environ["LEARNING_RATE"] = str(learning_rate)
    os.environ["GAMMA"] = str(gamma)
    os.environ["ENTROPY_COEFF"] = str(entropy_coeff)
    os.environ["MAX_GRAD_NORM"] = str(max_grad_norm)

    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))
    params = get_parameters(model)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=True)

    loss_neg, num_samples, results = client.evaluate(
        params, {"server_round": round_num, "eval_episodes": eval_episodes}
    )
    reward = results["reward"]
    loss = 0.0
    torch.save(model.state_dict(), output_model.path)

    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])
        writer.writerow([round_num, twin_id, "EVAL", reward, loss])


@dsl.component(base_image=BASE_IMAGE)
def train_step(
    twin_id: str,
    input_model: Input[Model],
    output_model: Output[Model],
    metrics: Output[Artifact],
    round_num: int,
    local_episodes: int,
    learning_rate: float = 0.003,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
):
    import torch
    import csv
    import os
    from engine import PolicyNet, get_parameters
    from client import TwinClient

    # Set hyperparameters as environment variables
    os.environ["LEARNING_RATE"] = str(learning_rate)
    os.environ["GAMMA"] = str(gamma)
    os.environ["ENTROPY_COEFF"] = str(entropy_coeff)
    os.environ["MAX_GRAD_NORM"] = str(max_grad_norm)

    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))
    params = get_parameters(model)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=False)

    new_params, num_samples, results = client.fit(
        params, {"server_round": round_num, "local_episodes": local_episodes}
    )
    train_reward = results["reward"]
    train_loss = results["loss"]

    # Post-Training Evaluation on Training Environment (Local Eval)
    # We switch the client/env to eval mode or just run evaluate.
    # client.evaluate uses 'local_episodes' from config.
    _, _, eval_results = client.evaluate(
        new_params, {"server_round": round_num, "local_episodes": local_episodes}
    )
    eval_reward = eval_results["reward"]

    torch.save(model.state_dict(), output_model.path)

    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])
        writer.writerow([round_num, twin_id, "TRAIN", train_reward, train_loss])
        writer.writerow([round_num, twin_id, "EVAL", eval_reward, 0.0])


@dsl.pipeline(
    name="Single Visual Pipeline",
    description="Visual DAG representation of single twin training rounds reusing core logic",
)
def single_twin_visual_pipeline(
    fl_rounds: int = config.get("fl_rounds", 10),
    local_episodes: int = config.get("local_episodes", 10),
    eval_episodes: int = config.get("eval_episodes", 20),
):
    init_task = initialize_model_visual()

    # Track models per round for clean dependencies
    round_models = [init_task.outputs["model"]]

    # No Round 0 Eval. Start with Training in Loop.

    fl_rounds_static = (
        int(fl_rounds)
        if isinstance(fl_rounds, (int, float))
        else config.get("fl_rounds", 10)
    )

    for r in range(1, fl_rounds_static + 1):
        # 1. Training Step (On INCOMING model M_{r-1} -> produces M_r)
        train_task = train_step(
            twin_id="train-twin-1",
            input_model=round_models[-1],
            round_num=r,
            local_episodes=local_episodes,
        )

        # 2. Global Evaluation Step (on OUTGOING model M_r)
        eval_task = eval_step(
            twin_id="eval-twin-global",
            input_model=train_task.outputs["output_model"],
            round_num=r,
            eval_episodes=eval_episodes,
        )

        # Ensure eval happens after training
        eval_task.after(train_task)

        # Carry the trained model to the next round
        round_models.append(train_task.outputs["output_model"])


if __name__ == "__main__":
    compiler.Compiler().compile(
        single_twin_visual_pipeline, "pipeline_specs/single_visual_pipeline.yaml"
    )
