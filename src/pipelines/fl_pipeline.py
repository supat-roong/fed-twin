import json
from kfp import dsl
from kfp import compiler
from kubernetes import config
from kfp.dsl import Output, Artifact


@dsl.component(
    base_image="python:3.9-slim", packages_to_install=["jinja2", "requests", "pyyaml"]
)
def train_federated(
    namespace: str,
    fl_rounds: int,
    num_workers: int,
    local_episodes: int,
    eval_episodes: int,
    job_id: str,
    metrics: Output[Artifact],
):
    import os
    import subprocess
    import requests
    import time
    import re
    import csv
    from jinja2 import Template

    # Install kubectl manually to /tmp since we might not have root
    kubectl_path = "/tmp/kubectl"
    if not os.path.exists(kubectl_path):
        print("Installing kubectl...")
        url = "https://dl.k8s.io/release/v1.28.0/bin/linux/amd64/kubectl"
        response = requests.get(url)
        with open(kubectl_path, "wb") as f:
            f.write(response.content)
        os.chmod(kubectl_path, 0o755)

    # Inline the PyTorchJob manifest as a template
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
              value: "{{ num_workers + 1 }}"
    Worker:
      replicas: {{ num_workers + 1 }}
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
                  INDEX=${HOSTNAME##*-}
                  echo "IDENTIFIED: TRAINING TWIN #$INDEX"
                  export TWIN_ID="train-twin-$INDEX"
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
    """

    job_name = f"fl-job-{job_id}"

    # Render and save
    template = Template(pytorch_job_template)
    manifest = template.render(
        job_name=job_name,
        rounds=fl_rounds,
        namespace=namespace,
        num_workers=num_workers,
        local_episodes=local_episodes,
        eval_episodes=eval_episodes,
        learning_rate=0.003,
        gamma=0.99,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
    )

    with open("/tmp/job.yaml", "w") as f:
        f.write(manifest)

    print(f"Deploying Federated Training for {fl_rounds} rounds in {namespace}...")
    print(f"Configuration: Workers={num_workers}, Episodes={local_episodes}")

    # Apply using kubectl (available in /tmp)
    subprocess.run(
        [kubectl_path, "apply", "-f", "/tmp/job.yaml", "--force"], check=True
    )

    # Start streaming logs immediately - don't wait for pods to be ready
    # The job might complete quickly, so we need to start streaming ASAP
    print(f"Starting log stream for job {job_name}...")

    # Wait just a moment for pods to start being created
    time.sleep(5)

    # Prepare CSV
    with open(metrics.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "twin_id", "mode", "reward", "loss"])

    # Get ALL logs from the beginning (not just new logs)
    # Using --since-time or --tail=-1 ensures we get everything
    cmd = [
        kubectl_path,
        "logs",
        "-l",
        f"training.kubeflow.org/job-name={job_name}",
        "-n",
        namespace,
        "--all-containers",
        "--prefix=true",
        "--max-log-requests=20",
        "--tail=-1",
        "-f",  # Follow for any new logs
    ]

    print(f"Streaming logs with command: {' '.join(cmd[:6])}...")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Atomic Metric Regex
    metric_pattern = re.compile(
        r"Twin ([\w-]+)\s+\[Round (\d+)\]\s+\[METRIC\]\s+(\S+)\s+Reward:\s+([-\d.]+)\s+Loss:\s+([-\d.]+)"
    )

    # Calculate expected duration: rounds * (train + eval episodes) * ~5 sec per episode
    # Add 50% buffer for safety. FL has more workers so slightly longer per round.
    expected_duration = int(fl_rounds * (local_episodes + eval_episodes) * 6 * 1.5)
    timeout_seconds = max(
        6000, expected_duration
    )  # At least 100 minutes, or calculated duration
    print(
        f"Timeout set to {timeout_seconds} seconds (~{timeout_seconds // 60} minutes) based on {fl_rounds} rounds"
    )

    start_time = time.time()
    last_check_time = 0
    line_count = 0
    metric_count = 0
    job_completed = False

    print(f"Monitor started for {job_name}. Parsing atomic metrics...")
    try:
        for line in process.stdout:
            line_count += 1
            if line_count % 100 == 0:
                print(
                    f"Processed {line_count} log lines, found {metric_count} metrics..."
                )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(
                    f"⚠️ Timeout reached after {elapsed:.0f} seconds. Processed {line_count} lines, found {metric_count} metrics."
                )
                break

            # Atomic Match
            match = metric_pattern.search(line)
            if match:
                twin_id = match.group(1)
                rd = match.group(2)
                mode = match.group(3)
                reward = match.group(4)
                loss = match.group(5)

                # Normalize mode for CSV
                csv_mode = "EVAL" if "EVAL" in mode else "TRAIN"
                if mode == "EVAL-ONLY-SKIP":
                    continue

                metric_count += 1
                if metric_count % 10 == 0 or metric_count <= 5:
                    print(
                        f"✓ Metric #{metric_count}: R={rd}, Twin={twin_id}, Mode={csv_mode}, Rew={reward}"
                    )
                with open(metrics.path, "a", newline="") as f:
                    csv.writer(f).writerow([rd, twin_id, csv_mode, reward, loss])

            # Check if job is finished every 10 seconds
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

    except Exception as e:
        print(f"Error reading logs: {e}")
    finally:
        process.terminate()
        print(
            f"Log streaming finished. Total: {line_count} lines processed, {metric_count} metrics captured"
        )

        if not job_completed:
            print("⚠️ Warning: Log streaming ended before job completion was confirmed")

        # Final verification: check if we got expected number of metrics
        # FL has num_workers training twins + 1 eval twin, each doing TRAIN and EVAL per round
        expected_metrics = fl_rounds * (num_workers + 1) * 2
        if metric_count < expected_metrics * 0.8:  # Allow 20% tolerance
            print(
                f"⚠️ Warning: Only captured {metric_count}/{expected_metrics} expected metrics"
            )

    print("Training job finished monitoring.")


# Load Config Defaults
try:
    with open("config/config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"fl_rounds": 5, "num_workers": 3, "local_episodes": 5}


@dsl.pipeline(
    name="Federated Digital Twin Pipeline",
    description="Orchestrates distributed training for robot fleet",
)
def fed_twin_pipeline(
    namespace: str = "kubeflow",
    fl_rounds: int = config.get("fl_rounds", 5),
    num_workers: int = config.get("num_workers", 3),
    local_episodes: int = config.get("local_episodes", 5),
    eval_episodes: int = config.get("eval_episodes", 20),
):
    import time

    job_id = str(int(time.time()))

    # Single component that does everything
    train_federated(
        namespace=namespace,
        fl_rounds=fl_rounds,
        num_workers=num_workers,
        local_episodes=local_episodes,
        eval_episodes=eval_episodes,
        job_id=job_id,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(fed_twin_pipeline, "pipeline_specs/fl_pipeline.yaml")
