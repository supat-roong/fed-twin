import json

from kfp import compiler, dsl
from kfp.dsl import Artifact, Output

# Load Config Defaults
try:
    with open("config/config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"fl_rounds": 5, "local_episodes": 10}


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
    global checkpoint (identity mean over this pipeline's single training
    twin). Required even here: the next round's worker downloads
    round_{fl_round}/global.pt, which only this step writes (spec 3.3).
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
    name="Single Twin Visual Single Cluster Pipeline",
    description="Visual DAG representation of single twin training rounds in a single cluster",
)
def single_twin_visual_single_cluster_pipeline(
    local_episodes: int = config.get("local_episodes", 10),
    eval_episodes: int = config.get("eval_episodes", 20),
    mlflow_run_id: str = "",
    mlflow_exp_name: str = "Single-Twin-Visual-Single-Cluster",
    namespace: str = "kubeflow",
    temporal_address: str = "temporal-frontend.kubeflow.svc.cluster.local:7233",
    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    minio_bucket: str = "mlflow-artifacts",
    worker_image: str = "fed-twin-app:v1",
):
    import uuid

    # The 8 chars consumed by workflow ids and Job names need real
    # entropy: a truncated epoch timestamp changes only every 100s, and
    # USE_EXISTING would silently attach a second same-window submission
    # to the first's running workflows. Same fix as the functional
    # pipelines (Phase 3b final review, commit 37cc58b).
    job_id = uuid.uuid4().hex[:8]
    # 1 training twin; the components derive the fleet as num_workers + 1
    # (rank 0 is the eval twin), so each round runs 2 Jobs -- mirroring
    # the functional single_twin pipeline exactly.
    num_workers = 1
    prev_op = None
    # The round count is trace-time by design: KFP freezes the DAG shape
    # (how many train_workers tasks exist) the moment this function is
    # defined, so it must come from config, never from a runtime value.
    for r in range(1, config.get("fl_rounds", 10) + 1):
        train_op = run_worker_via_temporal(
            worker_id=1,
            fl_round=r - 1,
            visual_round=r,
            num_workers=num_workers,
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
        train_op.set_display_name(f"train-twin-1-round-{r}")
        if prev_op is not None:
            train_op.after(prev_op)

        agg_op = (
            aggregate_round(
                fl_round=r - 1,
                num_workers=num_workers,
                minio_endpoint=minio_endpoint,
                minio_access_key=minio_access_key,
                minio_secret_key=minio_secret_key,
                minio_bucket=minio_bucket,
            )
            .after(train_op)
            .set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
        )
        agg_op.set_display_name(f"aggregate-round-{r}")

        # fl_round = r (one past this round's r-1): downloads
        # round_{r-1}/global.pt, the aggregate agg_op just wrote.
        eval_op = (
            run_worker_via_temporal(
                worker_id=0,
                fl_round=r,
                visual_round=r,
                num_workers=num_workers,
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


if __name__ == "__main__":
    compiler.Compiler().compile(
        single_twin_visual_single_cluster_pipeline,
        "pipeline_specs/single_twin_visual_single_cluster_pipeline.yaml",
    )
