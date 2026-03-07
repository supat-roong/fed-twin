from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Model, Artifact
import torch

BASE_IMAGE = "fed-twin-app:v1"


@dsl.component(base_image=BASE_IMAGE)
def initialize_model(model: Output[Model]):
    from engine import PolicyNet

    net = PolicyNet()
    torch.save(net.state_dict(), model.path)
    print(f"Initialized global model at {model.path}")


@dsl.component(base_image=BASE_IMAGE)
def train_twin(
    twin_id: str,
    input_model: Input[Model],
    output_model: Output[Model],
    metrics: Output[Artifact],
    round_num: int,
    local_episodes: int,
    eval_episodes: int,
    learning_rate: float = 0.003,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
):
    import csv
    import os
    from engine import PolicyNet, get_parameters
    from client import TwinClient

    # Set hyperparameters as environment variables
    os.environ["LEARNING_RATE"] = str(learning_rate)
    os.environ["GAMMA"] = str(gamma)
    os.environ["ENTROPY_COEFF"] = str(entropy_coeff)
    os.environ["MAX_GRAD_NORM"] = str(max_grad_norm)

    print(f"[{twin_id}] Loading global model from {input_model.path}")
    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))

    # Initialize Core Client (Training Mode)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=False)
    params = get_parameters(model)

    # 1. Train
    new_params, num_samples, results = client.fit(
        params, {"server_round": round_num, "local_episodes": local_episodes}
    )
    train_reward = results["reward"]
    train_loss = results["loss"]

    # 2. Post-Training Local Evaluation
    print(f"[{twin_id}] Running Post-Training Local Evaluation...")
    _, _, eval_results = client.evaluate(
        new_params, {"server_round": round_num, "eval_episodes": eval_episodes}
    )
    local_eval_reward = eval_results["reward"]

    torch.save(model.state_dict(), output_model.path)
    print(f"[{twin_id}] Local weights saved to {output_model.path}")

    # Write metrics
    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])
        writer.writerow([round_num, twin_id, "TRAIN", train_reward, train_loss])
        writer.writerow([round_num, twin_id, "EVAL", local_eval_reward, 0.0])


@dsl.component(base_image=BASE_IMAGE)
def eval_twin(
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
    import csv
    import os
    from engine import PolicyNet, get_parameters
    from client import TwinClient

    # Set hyperparameters as environment variables
    os.environ["LEARNING_RATE"] = str(learning_rate)
    os.environ["GAMMA"] = str(gamma)
    os.environ["ENTROPY_COEFF"] = str(entropy_coeff)
    os.environ["MAX_GRAD_NORM"] = str(max_grad_norm)

    print(f"[{twin_id}] Loading global model from {input_model.path}")
    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))

    # Initialize Core Client (Eval Mode)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=True)
    params = get_parameters(model)

    print(f"[{twin_id}] running global evaluation.")
    loss_neg, num_samples, results = client.evaluate(
        params, {"server_round": round_num, "eval_episodes": eval_episodes}
    )
    eval_reward = results["reward"]

    # Pass through model state (identity)
    torch.save(model.state_dict(), output_model.path)

    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])
        writer.writerow([round_num, twin_id, "EVAL", eval_reward, 0.0])


@dsl.component(base_image=BASE_IMAGE)
def aggregate_models(
    model_0: Input[Model],
    model_1: Input[Model],
    output_model: Output[Model],
    round_num: int,
):

    paths = [model_0.path, model_1.path]
    print(f"Aggregating {len(paths)} models for Round {round_num}")

    state_dicts = [torch.load(p) for p in paths]

    avg_state_dict = {}
    for key in state_dicts[0].keys():
        metas = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state_dict[key] = torch.mean(metas, dim=0)

    torch.save(avg_state_dict, output_model.path)
    print(f"New Global Model saved to {output_model.path}")


@dsl.pipeline(
    name="Federated Visual Pipeline",
    description="Dynamically generated visual pipeline for 2 workers.",
)
def visual_fl_pipeline():
    init_task = initialize_model()
    current_model = init_task.outputs["model"]

    for r in range(1, 3 + 1):
        # Parallel Training

        t0 = train_twin(
            twin_id="train-twin-1",
            input_model=current_model,
            local_episodes=2,
            eval_episodes=10,
            round_num=r,
        )

        t1 = train_twin(
            twin_id="train-twin-2",
            input_model=current_model,
            local_episodes=2,
            eval_episodes=10,
            round_num=r,
        )

        # Aggregation (waits for all training tasks to complete)
        agg = aggregate_models(
            model_0=t0.outputs["output_model"],
            model_1=t1.outputs["output_model"],
            round_num=r,
        )

        # Global Evaluation (Eval Twin)
        eval_twin(
            twin_id="eval-twin-global",
            input_model=agg.outputs["output_model"],
            eval_episodes=10,
            round_num=r,
        )

        current_model = agg.outputs["output_model"]


if __name__ == "__main__":
    compiler.Compiler().compile(
        visual_fl_pipeline, "pipeline_specs/fl_visual_pipeline.yaml"
    )
