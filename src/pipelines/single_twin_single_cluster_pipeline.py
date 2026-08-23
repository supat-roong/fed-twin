import json

from kfp import compiler, dsl
from kfp.dsl import Artifact, Output

# Reuse config if available, or defaults
try:
    with open("config/config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"fl_rounds": 10, "num_workers": 1, "local_episodes": 10}

print(f"Compiling Single-Twin-FL Pipeline with Config: {config}")

# D6: gates the Flower->MinIO cutover -- see fed_twin_single_cluster_pipeline.py's
# identical comment for the full trace-time-vs-runtime-parameter rationale.
WORKER_LAUNCHER = config.get("worker_launcher", "flower")


@dsl.component(
    base_image="python:3.9-slim", packages_to_install=["jinja2", "requests", "pyyaml"]
)
def train_single_twin(
    namespace: str,
    fl_rounds: int,
    local_episodes: int,
    eval_episodes: int,
    job_id: str,
    run_name: str,
    mlflow_run_id: str,
    mlflow_exp_name: str,
    metrics: Output[Artifact],
):
    import csv
    import os
    import re
    import subprocess
    import time

    import requests
    from jinja2 import Template

    kubectl_path = "/tmp/kubectl"
    if not os.path.exists(kubectl_path):
        url = "https://dl.k8s.io/release/v1.28.0/bin/linux/amd64/kubectl"
        response = requests.get(url)
        with open(kubectl_path, "wb") as f:
            f.write(response.content)
        os.chmod(kubectl_path, 0o755)

    pytorch_job_template = """
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {{ job_name }}
  namespace: {{ namespace }}
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          serviceAccountName: default
          containers:
          - name: pytorch
            image: fed-twin-app:v1
            imagePullPolicy: IfNotPresent
            command: ["python", "server.py"]
            env:
            - name: FL_ROUNDS
              value: "{{ rounds }}"
            - name: MIN_CLIENTS
              value: "2"
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-service.kubeflow:5000"
            - name: MLFLOW_EXPERIMENT_NAME
              value: "{{ mlflow_exp_name }}"
            - name: MLFLOW_RUN_ID
              value: "{{ mlflow_run_id }}"
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: fed-twin-app:v1
            imagePullPolicy: IfNotPresent
            command: ["/bin/bash", "-c"]
            args:
              - |
                if [[ $HOSTNAME =~ -worker-0$ ]]; then
                  echo "IDENTIFIED: GLOBAL EVALUATION TWIN"
                  export EVAL_ONLY=true
                  export TWIN_ID=eval-twin-global
                else
                  echo "IDENTIFIED: SINGLE TRAINING TWIN"
                  export TWIN_ID="train-twin-1"
                fi
                python client.py
            env:
            - name: SERVER_ADDR
              value: "{{ job_name }}-master-0:8080"
            - name: LOCAL_EPISODES
              value: "{{ local_episodes }}"
            - name: EVAL_EPISODES
              value: "{{ eval_episodes }}"
            - name: LEARNING_RATE
              value: "{{ learning_rate }}"
            - name: GAMMA
              value: "{{ gamma }}"
            - name: ENTROPY_COEFF
              value: "{{ entropy_coeff }}"
            - name: MAX_GRAD_NORM
              value: "{{ max_grad_norm }}"
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-service.kubeflow:5000"
            - name: MLFLOW_EXPERIMENT_NAME
              value: "{{ mlflow_exp_name }}"
            - name: MLFLOW_RUN_ID
              value: "{{ mlflow_run_id }}"
            - name: MLFLOW_S3_ENDPOINT_URL
              value: "http://minio-service.kubeflow:9000"
            - name: AWS_ACCESS_KEY_ID
              value: "minio"
            - name: AWS_SECRET_ACCESS_KEY
              value: "minio123"
            - name: MLFLOW_S3_IGNORE_TLS
              value: "true"
    """

    job_name = f"single-job-{job_id}"
    template = Template(pytorch_job_template)
    manifest = template.render(
        job_name=job_name,
        rounds=fl_rounds,
        namespace=namespace,
        local_episodes=local_episodes,
        eval_episodes=eval_episodes,
        learning_rate=0.003,
        gamma=0.99,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
        run_name=run_name,
        mlflow_run_id=mlflow_run_id,
        mlflow_exp_name=mlflow_exp_name,
    )

    with open("/tmp/job.yaml", "w") as f:
        f.write(manifest)

    print(f"Deploying Single Twin (via FL logic) for {fl_rounds} rounds...")
    subprocess.run(
        [kubectl_path, "apply", "-f", "/tmp/job.yaml", "--force"], check=True
    )

    # Start streaming logs immediately - don't wait for pods to be ready
    print(f"Starting log stream for job {job_name}...")
    time.sleep(5)  # Brief wait for pods to start being created

    # Prepare CSV
    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])

    metric_pattern = re.compile(
        r"Twin ([\w-]+)\s+\[Round (\d+)\]\s+\[METRIC\]\s+(\S+)\s+Reward:\s+([-\d.]+)\s+Loss:\s+([-\d.]+)"
    )

    cmd = [
        kubectl_path,
        "logs",
        "-l",
        f"training.kubeflow.org/job-name={job_name}",
        "-n",
        namespace,
        "--all-containers",
        "--prefix=true",
        "--tail=-1",
        "-f",
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    start_time = time.time()
    last_check_time = 0
    metric_count = 0

    # Calculate expected duration: rounds * (train + eval episodes) * ~5 sec per episode
    # Add 50% buffer for safety
    expected_duration = int(fl_rounds * (local_episodes + eval_episodes) * 5 * 1.5)
    timeout = max(3600, expected_duration)  # At least 1 hour, or calculated duration
    print(
        f"Timeout set to {timeout} seconds (~{timeout // 60} minutes) based on {fl_rounds} rounds"
    )

    expected_metrics = (fl_rounds * 1 * 2) + (fl_rounds * 1)
    job_completed = False

    try:
        for line in process.stdout:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"[WARNING] Timeout reached after {elapsed:.0f} seconds")
                break

            match = metric_pattern.search(line)
            if match:
                twin_id, rd, mode, reward, loss = match.groups()
                csv_mode = "EVAL" if "EVAL" in mode else "TRAIN"
                if mode == "EVAL-ONLY-SKIP":
                    continue

                metric_count += 1
                if metric_count % 10 == 0 or metric_count <= 5:
                    print(
                        f"[OK] Metric #{metric_count}: R={rd}, Twin={twin_id}, Mode={mode}, Rew={reward}"
                    )
                with open(metrics.path, "a", newline="") as f:
                    csv.writer(f).writerow([rd, twin_id, csv_mode, reward, loss])

            # Check if job finished every 10 seconds
            current_time = time.time()
            if current_time - last_check_time >= 10:
                last_check_time = current_time

                # Check both job status AND pod phases
                job_res = subprocess.run(
                    [
                        kubectl_path,
                        "get",
                        "pytorchjob",
                        job_name,
                        "-n",
                        namespace,
                        "-o",
                        "jsonpath={.status.conditions[?(@.type=='Succeeded')].status}",
                    ],
                    capture_output=True,
                    text=True,
                )

                pods_res = subprocess.run(
                    [
                        kubectl_path,
                        "get",
                        "pods",
                        "-l",
                        f"training.kubeflow.org/job-name={job_name}",
                        "-n",
                        namespace,
                        "-o",
                        "jsonpath={.items[*].status.phase}",
                    ],
                    capture_output=True,
                    text=True,
                )

                # Job is complete when status is Succeeded AND all pods are in terminal state
                if "True" in job_res.stdout:
                    pod_phases = pods_res.stdout.split()
                    all_terminal = all(
                        phase in ["Succeeded", "Failed"] for phase in pod_phases
                    )

                    if all_terminal and metric_count >= expected_metrics:
                        print(
                            f"[SUCCESS] Job completed successfully and expected metrics "
                            f"({metric_count}/{expected_metrics}) captured."
                        )
                        job_completed = True
                        # Extended grace period to ensure all logs are flushed
                        time.sleep(10)
                        break
                    elif all_terminal:
                        print(
                            f"Job pods terminal but waiting for metric count ({metric_count}/{expected_metrics})..."
                        )
                    else:
                        print(
                            f"Job marked Succeeded but pods still running: {pod_phases}. Continuing to stream..."
                        )
    finally:
        process.terminate()
        print(f"Log streaming finished. Total metrics captured: {metric_count}")

        if not job_completed:
            print(
                "[WARNING] Warning: Log streaming ended before job completion was confirmed"
            )

        # Final verification: check if we got expected number of metrics
        if metric_count < expected_metrics * 0.8:  # Allow 20% tolerance
            print(
                f"[WARNING] Warning: Only captured {metric_count}/{expected_metrics} expected metrics"
            )


    # A run that captured no metrics at all is a failed run, not a warning --
    # see the identical guard in fed_twin_single_cluster_pipeline.py for why.
    if metric_count == 0:
        raise RuntimeError(
            f"captured 0 of {expected_metrics} expected metrics: the log scrape "
            f"produced no data, so this run has no results. Check the training "
            f"pods' logs -- the training itself may well have succeeded."
        )

    print("Training job finished.")


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


@dsl.pipeline(
    name="Single Twin Single Cluster Pipeline",
    description="Runs single twin training in a single cluster",
)
def single_twin_single_cluster_pipeline(
    namespace: str = "kubeflow",
    # NOTE: consumed only by the flower branch. Under worker_launcher=minio
    # the round count is frozen at trace time from config (see the loop
    # below) and this runtime parameter is accepted but ignored.
    fl_rounds: int = config.get("fl_rounds", 10),
    local_episodes: int = config.get("local_episodes", 10),
    eval_episodes: int = config.get("eval_episodes", 20),
    run_name: str = "single_run_default",
    mlflow_run_id: str = "",
    mlflow_exp_name: str = "Single-Twin-Single-Cluster",
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
        # to the first's running round. uuid hex is lowercase alphanumeric,
        # satisfying the Job-name fragment rule. Computed at trace time, so
        # resubmitting one compiled YAML (KFP UI clone / recurring run)
        # reuses the id -- recompile per run, as run_pipeline.sh already does.
        job_id = uuid.uuid4().hex[:8]

        # 1 training twin; the components derive the fleet as num_workers + 1
        # (rank 0 is the eval twin), so this yields 2 Jobs per round --
        # matching today's flower-path template's hardcoded "replicas: 2".
        # single_twin has never exposed num_workers as a parameter, and this
        # phase changes wiring, not scope.
        num_workers = 1
        prev_op = None
        for round_idx in range(config.get("fl_rounds", 10)):
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
            fl_rounds=config.get("fl_rounds", 10),
            num_workers=num_workers,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
        ).after(prev_op)
    else:
        import time

        job_id = str(int(time.time()))

        train_single_twin(
            namespace=namespace,
            fl_rounds=fl_rounds,
            local_episodes=local_episodes,
            eval_episodes=eval_episodes,
            job_id=job_id,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_exp_name=mlflow_exp_name,
        ).set_env_variable("MLFLOW_TRACKING_URI", "http://mlflow-service.kubeflow:5000")


if __name__ == "__main__":
    compiler.Compiler().compile(
        single_twin_single_cluster_pipeline,
        "pipeline_specs/single_twin_single_cluster_pipeline.yaml",
    )
