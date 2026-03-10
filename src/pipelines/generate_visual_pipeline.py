import json


def generate_pipeline_code():
    # Load config
    try:
        with open("config/config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"fl_rounds": 3, "num_workers": 3, "local_episodes": 5}

    num_workers = config.get("num_workers", 3)
    fl_rounds = config.get("fl_rounds", 3)
    local_episodes = config.get("local_episodes", 5)

    # Generate the argument list for aggregate_models
    agg_args = ", ".join([f"model_{i}: Input[Model]" for i in range(num_workers)])

    # Generate the list of paths for aggregation logic
    agg_paths = "[" + ", ".join([f"model_{i}.path" for i in range(num_workers)]) + "]"

    # Generate the task calls in the pipeline
    pipeline_tasks = ""
    for i in range(num_workers):
        pipeline_tasks += f"""
        t{i} = train_twin(
            twin_id="train-twin-{i + 1}", 
            input_model=current_model, 
            local_episodes={local_episodes},
            round_num=r,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_exp_name=mlflow_exp_name
        ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
"""

    # Generate the aggregation call
    agg_call_args = ", ".join(
        [f"model_{i}=t{i}.outputs['output_model']" for i in range(num_workers)]
    )

    code = f"""from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Model, Artifact
import torch

BASE_IMAGE = 'fed-twin-app:v1'
MLFLOW_URI = 'http://mlflow-service.kubeflow:5000'

@dsl.component(
    base_image=BASE_IMAGE
)
def initialize_model(run_name: str, mlflow_run_id: str, mlflow_exp_name: str, model: Output[Model]):
    import torch
    import os
    from engine import PolicyNet
    from tracking import setup_mlflow

    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
    net = PolicyNet()
    torch.save(net.state_dict(), model.path)
    print(f"Initialized global model at {{model.path}}")

@dsl.component(
    base_image=BASE_IMAGE
)
def train_twin(
    twin_id: str, 
    input_model: Input[Model], 
    output_model: Output[Model],
    metrics: Output[Artifact],
    round_num: int,
    local_episodes: int,
    run_name: str,
    mlflow_run_id: str,
    mlflow_exp_name: str
):
    import torch
    import csv
    import os
    from engine import PolicyNet, get_parameters
    from client import TwinClient
    from tracking import setup_mlflow

    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
    print(f"[{{twin_id}}] Loading global model from {{input_model.path}}")
    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))
    
    # Initialize Core Client (Training Mode)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=False)
    params = get_parameters(model)
    
    # 1. Train
    new_params, num_samples, results = client.fit(params, {{
        "server_round": round_num, 
        "local_episodes": local_episodes
    }})
    train_reward = results["reward"]
    train_loss = results["loss"]
    
    # 2. Post-Training Local Evaluation
    print(f"[{{twin_id}}] Running Post-Training Local Evaluation...")
    _, _, eval_results = client.evaluate(new_params, {{"server_round": round_num, "local_episodes": local_episodes}})
    local_eval_reward = eval_results["reward"]
    
    torch.save(model.state_dict(), output_model.path)
    print(f"[{{twin_id}}] Local weights saved to {{output_model.path}}")

    # Write metrics
    with open(metrics.path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'twin_id', 'mode', 'reward', 'loss'])
        writer.writerow([round_num, twin_id, "TRAIN", train_reward, train_loss])
        writer.writerow([round_num, twin_id, "EVAL", local_eval_reward, 0.0])

@dsl.component(
    base_image=BASE_IMAGE
)
def eval_twin(
    twin_id: str, 
    input_model: Input[Model], 
    output_model: Output[Model],
    metrics: Output[Artifact],
    round_num: int,
    local_episodes: int,
    run_name: str,
    mlflow_run_id: str,
    mlflow_exp_name: str
):
    import torch
    import csv
    import os
    from engine import PolicyNet, get_parameters
    from client import TwinClient
    from tracking import setup_mlflow

    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
    print(f"[{{twin_id}}] Loading global model from {{input_model.path}}")
    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))
    
    # Initialize Core Client (Eval Mode)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=True)
    params = get_parameters(model)
    
    print(f"[{{twin_id}}] running global evaluation.")
    loss_neg, num_samples, results = client.evaluate(params, {{"server_round": round_num, "local_episodes": local_episodes}})
    eval_reward = results["reward"]
    
    # Pass through model state (identity)
    torch.save(model.state_dict(), output_model.path)
    
    with open(metrics.path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'twin_id', 'mode', 'reward', 'loss'])
        writer.writerow([round_num, twin_id, "EVAL", eval_reward, 0.0])

@dsl.component(
    base_image=BASE_IMAGE
)
def aggregate_models(
    {agg_args}, 
    output_model: Output[Model],
    round_num: int,
    run_name: str,
    mlflow_run_id: str,
    mlflow_exp_name: str
):
    import torch
    import os
    from tracking import setup_mlflow
    
    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
    paths = {agg_paths}
    print(f"Aggregating {{len(paths)}} models for Round {{round_num}}")
    
    state_dicts = [torch.load(p) for p in paths]
    
    avg_state_dict = {{}}
    for key in state_dicts[0].keys():
        metas = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state_dict[key] = torch.mean(metas, dim=0)
    
    torch.save(avg_state_dict, output_model.path)
    print(f"New Global Model saved to {{output_model.path}}")


@dsl.pipeline(
    name="Federated Visual Pipeline",
    description="Dynamically generated visual pipeline for {num_workers} workers."
)
def visual_fl_pipeline(
    run_name: str = "visual_run_default",
    mlflow_run_id: str = "",
    mlflow_exp_name: str = "Fed-Twin-FL-Visual"
):
    init_task = initialize_model(
        run_name=run_name,
        mlflow_run_id=mlflow_run_id,
        mlflow_exp_name=mlflow_exp_name
    ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
    current_model = init_task.outputs['model']

    for r in range(1, {fl_rounds} + 1):
        # Parallel Training
        {pipeline_tasks}
        
        # Aggregation (waits for all training tasks to complete)
        agg = aggregate_models(
            {agg_call_args},
            round_num=r,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_exp_name=mlflow_exp_name
        ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
        
        # Global Evaluation (Eval Twin)
        t_eval = eval_twin(
            twin_id="eval-twin-global",
            input_model=agg.outputs['output_model'],
            local_episodes={local_episodes},
            round_num=r,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_exp_name=mlflow_exp_name
        ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
        
        current_model = agg.outputs['output_model']

if __name__ == "__main__":
    compiler.Compiler().compile(visual_fl_pipeline, "pipeline_specs/fl_visual_k8s_pipeline.yaml")
"""

    with open("src/pipelines/fl_visual_k8s_pipeline.py", "w") as f:
        f.write(code)

    print("Generated src/pipelines/fl_visual_k8s_pipeline.py")


if __name__ == "__main__":
    generate_pipeline_code()
