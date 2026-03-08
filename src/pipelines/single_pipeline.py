from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Artifact
import json

# Reuse config if available, or defaults
try:
    with open("config/config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"fl_rounds": 10, "num_workers": 1, "local_episodes": 10}

print(f"Compiling Single-Twin-FL Pipeline with Config: {config}")


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
    import os
    import subprocess
    import requests
    import time
    import re
    import csv
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

    job_completed = False

    try:
        for line in process.stdout:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"⚠️ Timeout reached after {elapsed:.0f} seconds")
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
                        f"✓ Metric #{metric_count}: R={rd}, Twin={twin_id}, Mode={mode}, Rew={reward}"
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

                    if all_terminal:
                        print(
                            f"✅ Job completed successfully. Waiting for final logs... ({metric_count} metrics captured)"
                        )
                        job_completed = True
                        # Extended grace period to ensure all logs are flushed
                        time.sleep(30)
                        break
                    else:
                        print(
                            f"Job marked Succeeded but pods still running: {pod_phases}. Continuing to stream..."
                        )
    finally:
        process.terminate()
        print(f"Log streaming finished. Total metrics captured: {metric_count}")

        if not job_completed:
            print("⚠️ Warning: Log streaming ended before job completion was confirmed")

        # Final verification: check if we got expected number of metrics
        expected_metrics = fl_rounds * 2  # Each round has TRAIN + EVAL
        if metric_count < expected_metrics * 0.8:  # Allow 20% tolerance
            print(
                f"⚠️ Warning: Only captured {metric_count}/{expected_metrics} expected metrics"
            )

    print("Training job finished.")


@dsl.pipeline(
    name="Single Twin Pipeline",
    description="Runs single twin training using the Federated architecture (1 client).",
)
def single_twin_pipeline(
    namespace: str = "kubeflow",
    fl_rounds: int = config.get("fl_rounds", 10),
    local_episodes: int = config.get("local_episodes", 10),
    eval_episodes: int = config.get("eval_episodes", 20),
    run_name: str = "single_run_default",
    mlflow_run_id: str = "",
    mlflow_exp_name: str = "Fed-Twin-Single",
):
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
        single_twin_pipeline, "pipeline_specs/single_pipeline.yaml"
    )
