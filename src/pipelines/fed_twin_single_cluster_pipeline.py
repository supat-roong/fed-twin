import json

from kfp import compiler, dsl
from kfp.dsl import Artifact, Output


@dsl.component(base_image="fed-twin-app:v1", packages_to_install=[])
def train_workers(
    fl_round: int,
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
    worker_report: Output[Artifact],
):
    """Run one round's worker fleet via Temporal, blocking on the result.

    Starts a TrainRoundWorkflow with a deterministic workflow id (kfp_run_id
    + fl_round), so a retried KFP step reattaches to the already-running
    round instead of launching a second one.
    """
    import asyncio
    import json
    import sys

    sys.path.insert(0, "/app")

    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import WorkflowIDConflictPolicy

    from src.orchestration.types import RoundSpec
    from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow

    # num_workers counts training twins (config.json's meaning under the
    # flower launcher); the fleet adds rank 0, the eval-only twin, on top --
    # preserving flower's `replicas: num_workers + 1` semantics (D5) and its
    # exact per-round CSV row count. The +1 lives here, inside the component,
    # because KFP cannot do arithmetic on a dsl parameter at trace time.
    fleet_size = num_workers + 1
    spec = RoundSpec(
        fl_round=fl_round,
        num_workers=fleet_size,
        min_workers=fleet_size,
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

    async def _run() -> dict:
        client = await Client.connect(temporal_address)
        handle = await client.start_workflow(
            TrainRoundWorkflow.run,
            spec,
            id=f"ftwn-train-{kfp_run_id[:8]}-r{fl_round}",
            task_queue=TASK_QUEUE,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
        print(f"started Temporal workflow {handle.id}")
        print(
            "Temporal workflow: "
            f"http://localhost:8233/namespaces/default/workflows/{handle.id}"
        )
        try:
            report = await handle.result()
        except WorkflowFailureError as e:
            # Mirrors active-fed's own fix for the identical gap: without
            # this, a quorum failure raises before the per-worker
            # attribution the workflow already tracked ever reaches an
            # artifact. The round must still fail -- re-raise after writing
            # what's known.
            statuses = await handle.query(TrainRoundWorkflow.status)
            payload = {
                "fl_round": fl_round,
                "succeeded": sorted(
                    wid for wid, s in statuses.items() if s.phase == "Succeeded"
                ),
                "failed": sorted(
                    wid for wid, s in statuses.items() if s.phase != "Succeeded"
                ),
                "results": [vars(statuses[wid]) for wid in sorted(statuses)],
                "temporal_workflow_id": handle.id,
                "error": str(e),
            }
            print(json.dumps(payload, indent=2))
            with open(worker_report.path, "w") as f:
                json.dump(payload, f, indent=2)
            raise
        return {
            "fl_round": report.fl_round,
            "succeeded": report.succeeded_ids,
            "failed": report.failed_ids,
            "results": [vars(r) for r in report.results],
            "temporal_workflow_id": handle.id,
        }

    payload = asyncio.run(_run())
    print(json.dumps(payload, indent=2))
    with open(worker_report.path, "w") as f:
        json.dump(payload, f, indent=2)


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
    global checkpoint. No evaluation logic lives here -- the eval-twin
    (worker 0, RANK==0) already ran inside the worker fleet itself.
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


@dsl.component(base_image="fed-twin-app:v1", packages_to_install=[])
def collect_metrics_csv(
    fl_rounds: int,
    num_workers: int,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    metrics: Output[Artifact],
):
    """Read every round's worker metrics.json from MinIO and write the same
    (round, twin_id, mode, reward, loss) CSV the flower-launcher path already
    produces from its log-scrape -- runs once, after every round has
    finished, not per-round.
    """
    import csv
    import sys

    sys.path.insert(0, "/app")

    from metrics_csv import collect_metrics_rows
    from minio import Minio

    minio_client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    # num_workers counts training twins; the fleet is num_workers + 1 (rank 0
    # is the eval twin). collect_metrics_rows reads ranks 0..fleet-1.
    rows = collect_metrics_rows(minio_client, minio_bucket, fl_rounds, num_workers + 1)

    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])
        writer.writerows(rows)

    print(f"Collected {len(rows)} metric rows across {fl_rounds} rounds.")


# Load Config Defaults
try:
    with open("config/config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"fl_rounds": 5, "num_workers": 3, "local_episodes": 5}


@dsl.pipeline(
    name="Federated Twin Single Cluster Pipeline",
    description="Orchestrates distributed training for robot fleet in a single cluster",
)
def fed_twin_single_cluster_pipeline(
    namespace: str = "kubeflow",
    num_workers: int = config.get("num_workers", 3),
    local_episodes: int = config.get("local_episodes", 10),
    eval_episodes: int = config.get("eval_episodes", 20),
    mlflow_run_id: str = "",
    mlflow_exp_name: str = "Fed-Twin-Single-Cluster",
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
    # to the first's running round. uuid hex is lowercase alphanumeric,
    # satisfying the Job-name fragment rule. Computed at trace time, so
    # resubmitting one compiled YAML (KFP UI clone / recurring run)
    # reuses the id -- recompile per run, as run_pipeline.sh already does.
    job_id = uuid.uuid4().hex[:8]

    # The round count is trace-time by design: KFP freezes the DAG shape
    # (how many train_workers tasks exist) the moment this function is
    # defined, so it must come from config, never from a runtime value.
    prev_op = None
    for round_idx in range(config.get("fl_rounds", 5)):
        train_op = train_workers(
            fl_round=round_idx,
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
        if prev_op is not None:
            train_op.after(prev_op)

        agg_op = (
            aggregate_round(
                fl_round=round_idx,
                num_workers=num_workers,
                minio_endpoint=minio_endpoint,
                minio_access_key=minio_access_key,
                minio_secret_key=minio_secret_key,
                minio_bucket=minio_bucket,
            )
            .after(train_op)
            .set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
        )
        prev_op = agg_op

    collect_metrics_csv(
        fl_rounds=config.get("fl_rounds", 5),
        num_workers=num_workers,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
    ).after(prev_op)


if __name__ == "__main__":
    compiler.Compiler().compile(
        fed_twin_single_cluster_pipeline,
        "pipeline_specs/fed_twin_single_cluster_pipeline.yaml",
    )
