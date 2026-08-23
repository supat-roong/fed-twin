"""Visual DAG for federated-twin training rounds -- hand-written (the
generator, generate_fed_twin_visual_pipeline.py, is gone, D3).

The minio branch's worker nodes are unrolled at trace time (a plain Python
for loop over config's num_workers), not dsl.ParallelFor: the KFP backend
pinned by infra.env (2.4.0) rejects dsl.Collected fan-in from ParallelFor at
submission (Argo DAG-sort "invalid dependency", needs backend >= 2.5.0 --
kubeflow/pipelines#10050), found live at Phase 3c's gate (spec D3, as
amended).

The flower branch below is frozen at the last-generated constants
(num_workers=2, fl_rounds=2, local_episodes=2) until Phase 3d deletes it.
"""

import json

from kfp import compiler, dsl
from kfp.dsl import Artifact, Input, Model, Output

# Load Config Defaults
try:
    with open("config/config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"fl_rounds": 3, "num_workers": 3, "local_episodes": 5}

# D6: gates the Flower->MinIO cutover -- see fed_twin_single_cluster_pipeline.py's
# identical comment for the full trace-time-vs-runtime-parameter rationale.
WORKER_LAUNCHER = config.get("worker_launcher", "flower")

BASE_IMAGE = "fed-twin-app:v1"
MLFLOW_URI = "http://mlflow-service.kubeflow:5000"


@dsl.component(base_image=BASE_IMAGE)
def initialize_model(run_name: str, mlflow_run_id: str, mlflow_exp_name: str, model: Output[Model]):
    import os

    import torch
    from engine import PolicyNet
    from tracking import setup_mlflow

    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
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
    run_name: str,
    mlflow_run_id: str,
    mlflow_exp_name: str
):
    import csv
    import os

    import torch
    from client import TwinClient
    from engine import PolicyNet, get_parameters
    from tracking import setup_mlflow

    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
    print(f"[{twin_id}] Loading global model from {input_model.path}")
    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))

    # Initialize Core Client (Training Mode)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=False)
    params = get_parameters(model)

    # 1. Train
    new_params, num_samples, results = client.fit(params, {
        "server_round": round_num,
        "local_episodes": local_episodes
    })
    train_reward = results["reward"]
    train_loss = results["loss"]

    # 2. Post-Training Local Evaluation
    print(f"[{twin_id}] Running Post-Training Local Evaluation...")
    _, _, eval_results = client.evaluate(new_params, {"server_round": round_num, "local_episodes": local_episodes})
    local_eval_reward = eval_results["reward"]

    torch.save(model.state_dict(), output_model.path)
    print(f"[{twin_id}] Local weights saved to {output_model.path}")

    # Write metrics
    with open(metrics.path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'twin_id', 'mode', 'reward', 'loss'])
        writer.writerow([round_num, twin_id, "TRAIN", train_reward, train_loss])
        writer.writerow([round_num, twin_id, "EVAL", local_eval_reward, 0.0])


@dsl.component(base_image=BASE_IMAGE)
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
    import csv
    import os

    import torch
    from client import TwinClient
    from engine import PolicyNet, get_parameters
    from tracking import setup_mlflow

    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
    print(f"[{twin_id}] Loading global model from {input_model.path}")
    model = PolicyNet()
    model.load_state_dict(torch.load(input_model.path))

    # Initialize Core Client (Eval Mode)
    client = TwinClient(model=model, twin_id=twin_id, eval_only=True)
    params = get_parameters(model)

    print(f"[{twin_id}] running global evaluation.")
    loss_neg, num_samples, results = client.evaluate(
        params, {"server_round": round_num, "local_episodes": local_episodes}
    )
    eval_reward = results["reward"]

    # Pass through model state (identity)
    torch.save(model.state_dict(), output_model.path)

    with open(metrics.path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'twin_id', 'mode', 'reward', 'loss'])
        writer.writerow([round_num, twin_id, "EVAL", eval_reward, 0.0])


@dsl.component(base_image=BASE_IMAGE)
def aggregate_models(
    model_0: Input[Model], model_1: Input[Model],
    output_model: Output[Model],
    round_num: int,
    run_name: str,
    mlflow_run_id: str,
    mlflow_exp_name: str
):
    import os

    import torch
    from tracking import setup_mlflow

    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_exp_name
    os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
    setup_mlflow()
    paths = [model_0.path, model_1.path]
    print(f"Aggregating {len(paths)} models for Round {round_num}")

    state_dicts = [torch.load(p) for p in paths]

    avg_state_dict = {}
    for key in state_dicts[0].keys():
        metas = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state_dict[key] = torch.mean(metas, dim=0)

    torch.save(avg_state_dict, output_model.path)
    print(f"New Global Model saved to {output_model.path}")


@dsl.component(base_image="fed-twin-app:v1", packages_to_install=[])
def run_worker_via_temporal(
    worker_id: int,
    fl_round: int,
    visual_round: int,
    num_workers: int,
    local_episodes: int,
    eval_episodes: int,
    namespace: str,
    temporal_address: str,
    kfp_run_id: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_run_id: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    worker_image: str,
    learning_rate: float,
    gamma: float,
    entropy_coeff: float,
    max_grad_norm: float,
    metrics: Output[Artifact],
) -> str:
    """Run one worker (a train twin for worker_id >= 1, the eval twin for
    worker_id == 0) as a single WorkerWorkflow, then surface its MinIO
    metrics as this node's own CSV artifact -- today's exact artifact shape.

    The eval node is dispatched with fl_round = visual_round (one past the
    train nodes' visual_round - 1), so worker_entrypoint.py downloads
    round_{visual_round - 1}/global.pt -- the aggregate this round just
    wrote -- preserving the visual pipelines' evaluate-the-round's-own-output
    semantics with zero entrypoint changes (spec 3.3).
    """
    import asyncio
    import csv
    import json
    import sys

    sys.path.insert(0, "/app")

    from minio import Minio
    from temporalio.client import Client
    from temporalio.common import WorkflowIDConflictPolicy

    from src.orchestration.types import WorkerSpec
    from src.orchestration.workflows import TASK_QUEUE, WorkerWorkflow

    # num_workers counts training twins (config.json's meaning under the
    # flower launcher); the fleet adds rank 0, the eval-only twin, on top --
    # same convention as the functional pipelines.
    spec = WorkerSpec(
        fl_round=fl_round,
        worker_id=worker_id,
        num_workers=num_workers + 1,
        local_episodes=local_episodes,
        eval_episodes=eval_episodes,
        namespace=namespace,
        worker_image=worker_image,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_id=mlflow_run_id,
        kfp_run_id=kfp_run_id,
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_coeff=entropy_coeff,
        max_grad_norm=max_grad_norm,
    )

    async def _run():
        client = await Client.connect(temporal_address)
        handle = await client.start_workflow(
            WorkerWorkflow.run,
            spec,
            id=f"ftwn-vis-{kfp_run_id[:8]}-r{fl_round}-w{worker_id}",
            task_queue=TASK_QUEUE,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
        print(f"started Temporal workflow {handle.id}")
        print(
            "Temporal workflow: "
            f"http://localhost:8233/namespaces/default/workflows/{handle.id}"
        )
        return await handle.result()

    result = asyncio.run(_run())
    if not result.succeeded:
        raise RuntimeError(
            f"worker {worker_id} (round {fl_round}) failed after "
            f"{result.attempts} attempts: {result.failure_reason}"
        )

    minio_client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    response = minio_client.get_object(
        minio_bucket, f"round_{fl_round}/workers/worker_{worker_id}_metrics.json"
    )
    try:
        payload = json.loads(response.read())
    finally:
        response.close()
        response.release_conn()

    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])
        for row in payload["rows"]:
            writer.writerow(
                [visual_round, payload["twin_id"], row["mode"], row["reward"], row["loss"]]
            )

    return f"worker {worker_id} round {fl_round} ok ({len(payload['rows'])} metric rows)"


@dsl.component(base_image="fed-twin-app:v1", packages_to_install=[])
def aggregate_round(
    fl_round: int,
    num_workers: int,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
):
    """Mean the round's training-worker weights and write the next round's
    global checkpoint. Runs .after() every train node in the round -- a plain
    task dependency; the pinned KFP backend (2.4.0) rejects dsl.Collected
    fan-in at submission (needs >= 2.5.0), so worker nodes are statically
    unrolled at trace time and fanned in here (spec D3, as amended).
    """
    import sys

    sys.path.insert(0, "/app")

    from aggregate import run_aggregate_round
    from minio import Minio

    minio_client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    # num_workers counts training twins; the fleet is num_workers + 1 (rank 0
    # is the eval twin). run_aggregate_round reads ranks 1..fleet-1, i.e.
    # exactly the num_workers training twins.
    run_aggregate_round(minio_client, minio_bucket, fl_round, num_workers + 1)


@dsl.pipeline(
    name="Federated Twin Visual Pipeline",
    description="Visual DAG of federated training rounds; one KFP node per worker.",
)
def visual_fed_twin_pipeline(
    run_name: str = "visual_run_default",
    mlflow_run_id: str = "",
    mlflow_exp_name: str = "Fed-Twin-Visual-Single-Cluster",
    local_episodes: int = config.get("local_episodes", 10),
    eval_episodes: int = config.get("eval_episodes", 20),
    namespace: str = "kubeflow",
    temporal_address: str = "temporal-frontend.kubeflow.svc.cluster.local:7233",
    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    minio_bucket: str = "mlflow-artifacts",
    worker_image: str = "fed-twin-app:v1",
):
    if WORKER_LAUNCHER == "minio":
        import uuid

        # The 8 chars consumed by workflow ids and Job names need real
        # entropy: a truncated epoch timestamp changes only every 100s, and
        # USE_EXISTING would silently attach a second same-window submission
        # to the first's running workflows. Same fix as the functional
        # pipelines (Phase 3b final review, commit 37cc58b).
        job_id = uuid.uuid4().hex[:8]
        # num_workers is trace-time (config), matching fl_rounds: the KFP
        # backend pinned by infra.env (2.4.0) rejects dsl.Collected fan-in
        # from dsl.ParallelFor at submission (Argo DAG-sort "invalid
        # dependency", needs backend >= 2.5.0 -- kubeflow/pipelines#10050),
        # so worker nodes are unrolled at trace time and the aggregate node
        # depends on them via plain .after(*train_ops). Found live at 3c's
        # gate; see the spec's D3 amendment.
        num_workers_static = config.get("num_workers", 3)
        prev_op = None
        for r in range(1, config.get("fl_rounds", 3) + 1):
            train_ops = []
            for wid in range(1, num_workers_static + 1):
                train_op = run_worker_via_temporal(
                    worker_id=wid,
                    fl_round=r - 1,
                    visual_round=r,
                    num_workers=num_workers_static,
                    local_episodes=local_episodes,
                    eval_episodes=eval_episodes,
                    namespace=namespace,
                    temporal_address=temporal_address,
                    kfp_run_id=job_id,
                    mlflow_tracking_uri="http://mlflow-service.kubeflow:5000",
                    mlflow_experiment_name=mlflow_exp_name,
                    mlflow_run_id=mlflow_run_id,
                    minio_endpoint=minio_endpoint,
                    minio_access_key=minio_access_key,
                    minio_secret_key=minio_secret_key,
                    minio_bucket=minio_bucket,
                    worker_image=worker_image,
                    learning_rate=config.get("learning_rate", 0.003),
                    gamma=config.get("gamma", 0.99),
                    entropy_coeff=config.get("entropy_coeff", 0.01),
                    max_grad_norm=config.get("max_grad_norm", 0.5),
                ).set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
                train_op.set_display_name(f"train-twin-{wid}-round-{r}")
                if prev_op is not None:
                    train_op.after(prev_op)
                train_ops.append(train_op)

            agg_op = aggregate_round(
                fl_round=r - 1,
                num_workers=num_workers_static,
                minio_endpoint=minio_endpoint,
                minio_access_key=minio_access_key,
                minio_secret_key=minio_secret_key,
                minio_bucket=minio_bucket,
            ).set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
            agg_op.after(*train_ops)
            agg_op.set_display_name(f"aggregate-round-{r}")

            # fl_round = r (one past this round's r-1): downloads
            # round_{r-1}/global.pt, the aggregate agg_op just wrote.
            eval_op = (
                run_worker_via_temporal(
                    worker_id=0,
                    fl_round=r,
                    visual_round=r,
                    num_workers=num_workers_static,
                    local_episodes=local_episodes,
                    eval_episodes=eval_episodes,
                    namespace=namespace,
                    temporal_address=temporal_address,
                    kfp_run_id=job_id,
                    mlflow_tracking_uri="http://mlflow-service.kubeflow:5000",
                    mlflow_experiment_name=mlflow_exp_name,
                    mlflow_run_id=mlflow_run_id,
                    minio_endpoint=minio_endpoint,
                    minio_access_key=minio_access_key,
                    minio_secret_key=minio_secret_key,
                    minio_bucket=minio_bucket,
                    worker_image=worker_image,
                    learning_rate=config.get("learning_rate", 0.003),
                    gamma=config.get("gamma", 0.99),
                    entropy_coeff=config.get("entropy_coeff", 0.01),
                    max_grad_norm=config.get("max_grad_norm", 0.5),
                )
                .after(agg_op)
                .set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
            )
            eval_op.set_display_name(f"eval-twin-global-round-{r}")
            prev_op = eval_op
    else:
        # Flower branch: the last-generated body, verbatim (frozen at
        # num_workers=2 / fl_rounds=2 / local_episodes=2 -- the generator is
        # deleted, D3; this whole branch is deleted in 3d).
        init_task = initialize_model(
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_exp_name=mlflow_exp_name
        ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
        current_model = init_task.outputs['model']

        for r in range(1, 2 + 1):
            # Parallel Training

            t0 = train_twin(
                twin_id="train-twin-1",
                input_model=current_model,
                local_episodes=2,
                round_num=r,
                run_name=run_name,
                mlflow_run_id=mlflow_run_id,
                mlflow_exp_name=mlflow_exp_name
            ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)

            t1 = train_twin(
                twin_id="train-twin-2",
                input_model=current_model,
                local_episodes=2,
                round_num=r,
                run_name=run_name,
                mlflow_run_id=mlflow_run_id,
                mlflow_exp_name=mlflow_exp_name
            ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)


            # Aggregation (waits for all training tasks to complete)
            agg = aggregate_models(
                model_0=t0.outputs['output_model'], model_1=t1.outputs['output_model'],
                round_num=r,
                run_name=run_name,
                mlflow_run_id=mlflow_run_id,
                mlflow_exp_name=mlflow_exp_name
            ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)

            # Global Evaluation (Eval Twin)
            eval_twin(
                twin_id="eval-twin-global",
                input_model=agg.outputs['output_model'],
                local_episodes=2,
                round_num=r,
                run_name=run_name,
                mlflow_run_id=mlflow_run_id,
                mlflow_exp_name=mlflow_exp_name
            ).set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)

            current_model = agg.outputs['output_model']


if __name__ == "__main__":
    compiler.Compiler().compile(visual_fed_twin_pipeline, "pipeline_specs/fed_twin_visual_single_cluster_pipeline.yaml")
