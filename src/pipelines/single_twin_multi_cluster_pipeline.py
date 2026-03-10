import json
from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Artifact


@dsl.component(base_image="fed-twin-app:v1")
def train_single_karmada(
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
    import subprocess
    import time
    import re
    import csv

    kubectl_path = "/usr/local/bin/kubectl"

    # 1. Server Deployment (Master) -> Pinned to Host Cluster
    # Even in "single", we use the server to sync weights between trainer on member1 and evaluator on host.
    server_manifest_template = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ job_name }}-server
  namespace: {{ namespace }}
  labels:
    app: {{ job_name }}-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {{ job_name }}-server
  template:
    metadata:
      labels:
        app: {{ job_name }}-server
    spec:
      serviceAccountName: default
      containers:
      - name: pytorch
        image: fed-twin-app:v1
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-c"]
        args:
          - |
            python server.py
            echo "Server finished. Idling to prevent restart loop."
            tail -f /dev/null
        env:
        - name: FL_ROUNDS
          value: "{{ rounds }}"
        - name: MIN_CLIENTS
          value: "2"
        - name: MLFLOW_TRACKING_URI
          value: "http://multi-cluster-host-control-plane:30500"
        - name: MLFLOW_EXPERIMENT_NAME
          value: "{{ mlflow_exp_name }}"
        - name: MLFLOW_RUN_ID
          value: "{{ mlflow_run_id }}"
---
apiVersion: v1
kind: Service
metadata:
  name: fl-server-service
  namespace: {{ namespace }}
spec:
  type: NodePort
  selector:
    app: {{ job_name }}-server
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
      nodePort: 32444
---
apiVersion: policy.karmada.io/v1alpha1
kind: PropagationPolicy
metadata:
  name: {{ job_name }}-server-propagation
  namespace: {{ namespace }}
spec:
  resourceSelectors:
    - apiVersion: apps/v1
      kind: Deployment
      name: {{ job_name }}-server
    - apiVersion: v1
      kind: Service
      name: fl-server-service
  placement:
    clusterAffinity:
      clusterNames:
        - multi-cluster-host
"""

    worker_manifest_template = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ job_name }}-worker-{{ index }}
  namespace: {{ namespace }}
  labels:
    app: {{ job_name }}-worker
    twin-id: {{ twin_id }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {{ job_name }}-worker
      twin-id: {{ twin_id }}
  template:
    metadata:
      labels:
        app: {{ job_name }}-worker
        twin-id: {{ twin_id }}
    spec:
      containers:
      - name: pytorch
        image: fed-twin-app:v1
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-c"]
        args:
          - |
            export EVAL_ONLY="{{ eval_only }}"
            export TWIN_ID="{{ twin_id }}"
            
            echo "IDENTIFIED: {{ mode }} TWIN $TWIN_ID"
            
            python client.py
            
            echo "Worker finished. Idling to prevent restart loop and duplicate MLflow logs."
            tail -f /dev/null
        env:
        - name: SERVER_ADDR
          value: "{{ server_addr }}:32444"
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
          value: "http://multi-cluster-host-control-plane:30500"
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
---
apiVersion: policy.karmada.io/v1alpha1
kind: PropagationPolicy
metadata:
  name: {{ job_name }}-worker-{{ index }}-propagation
  namespace: {{ namespace }}
spec:
  resourceSelectors:
    - apiVersion: apps/v1
      kind: Deployment
      name: {{ job_name }}-worker-{{ index }}
  placement:
    clusterAffinity:
      clusterNames:
        - {{ target_cluster }}
"""

    job_name = f"single-job-{job_id}"

    # Fetch Configs via Kubernetes Secret created by automate_run.py
    import base64 as _b64

    secret_name = f"karmconfigs-{mlflow_run_id}"
    print(f"Fetching Karmada configs from secret: {secret_name}")
    try:
        karmada_b64 = subprocess.check_output(
            [
                kubectl_path,
                "get",
                "secret",
                secret_name,
                "-n",
                "kubeflow",
                "-o",
                "jsonpath={.data.karmada}",
            ]
        ).decode("utf-8")
        members_b64 = subprocess.check_output(
            [
                kubectl_path,
                "get",
                "secret",
                secret_name,
                "-n",
                "kubeflow",
                "-o",
                "jsonpath={.data.members}",
            ]
        ).decode("utf-8")
    except Exception as e:
        print(f"Failed to fetch secrets: {e}")
        raise e

    karmada_config = _b64.b64decode(karmada_b64).decode("utf-8")
    member_kubeconfigs = _b64.b64decode(members_b64).decode("utf-8")

    # Extract host cluster IPv4
    import json as _json_pre
    import re as _re

    try:
        _cfgs = _json_pre.loads(member_kubeconfigs)
        _host_kc = _cfgs.get("host", "")
        _ip_match = _re.search(r"https?://([\d.]+):", _host_kc)
        host_internal_ip = "multi-cluster-host-control-plane"
    except Exception:
        host_internal_ip = "multi-cluster-host-control-plane"

    print(f"Karmada Single Training Mode: Server on {host_internal_ip}:32444")

    # Render Server
    server_manifest = (
        server_manifest_template.replace("{{ job_name }}", job_name)
        .replace("{{ rounds }}", str(fl_rounds))
        .replace("{{ namespace }}", namespace)
        .replace("{{ mlflow_run_id }}", mlflow_run_id)
        .replace("{{ mlflow_exp_name }}", mlflow_exp_name)
    )
    with open("/tmp/server.yaml", "w") as f:
        f.write(server_manifest)

    # Render Workers (Exactly 2: 1 Evaluator on Host, 1 Trainer on Member1)
    with open("/tmp/worker.yaml", "w") as f:
        f.write("")

    # Index 0: global-eval -> host
    # Index 1: train-twin-1 -> member1
    for i in range(2):
        if i == 0:
            twin_id = "eval-twin-global"
            eval_only = "true"
            mode = "EVALUATION"
            target_cluster = "multi-cluster-host"
        else:
            twin_id = "train-twin-1"
            eval_only = "false"
            mode = "TRAINING"
            target_cluster = "multi-cluster-member1"

        worker_manifest = (
            worker_manifest_template.replace("{{ job_name }}", job_name)
            .replace("{{ index }}", str(i))
            .replace("{{ twin_id }}", twin_id)
            .replace("{{ eval_only }}", eval_only)
            .replace("{{ mode }}", mode)
            .replace("{{ target_cluster }}", target_cluster)
            .replace("{{ namespace }}", namespace)
            .replace("{{ local_episodes }}", str(local_episodes))
            .replace("{{ eval_episodes }}", str(eval_episodes))
            .replace("{{ learning_rate }}", "0.003")
            .replace("{{ gamma }}", "0.99")
            .replace("{{ entropy_coeff }}", "0.01")
            .replace("{{ max_grad_norm }}", "0.5")
            .replace("{{ mlflow_run_id }}", mlflow_run_id)
            .replace("{{ mlflow_exp_name }}", mlflow_exp_name)
            .replace("{{ server_addr }}", host_internal_ip)
        )
        with open("/tmp/worker.yaml", "a") as f:
            f.write(worker_manifest + "\n---\n")

    kubeconfig_data = karmada_config.replace(
        "https://127.0.0.1:32443",
        "https://karmada-apiserver.karmada-system.svc.cluster.local:5443",
    )
    with open("/tmp/karmada.config", "w") as f:
        f.write(kubeconfig_data)

    # Apply manifests
    print("Applying Single Twin manifests to Karmada...", flush=True)
    subprocess.run(
        [
            kubectl_path,
            "--kubeconfig",
            "/tmp/karmada.config",
            "--insecure-skip-tls-verify",
            "apply",
            "-f",
            "/tmp/server.yaml",
            "--force",
            "--validate=false",
        ],
        check=True,
    )
    subprocess.run(
        [
            kubectl_path,
            "--kubeconfig",
            "/tmp/karmada.config",
            "--insecure-skip-tls-verify",
            "apply",
            "-f",
            "/tmp/worker.yaml",
            "--force",
            "--validate=false",
        ],
        check=True,
    )

    # Metrics Scraping (Reuse robust FL logic)
    import json as _json

    metric_pattern = re.compile(
        r"Twin ([\w-]+)\s+\[Round (\d+)\]\s+\[METRIC\]\s+(\S+)\s+Reward:\s+([-\d.]+)\s+Loss:\s+([-\d.]+)"
    )
    with open(metrics.path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "twin_id", "mode", "reward", "loss"])

    try:
        cluster_configs = _json.loads(member_kubeconfigs)
    except Exception:
        cluster_configs = {}

    kube_paths = {}
    for cluster_name, kc_content in cluster_configs.items():
        kube_path = f"/tmp/kube_{cluster_name}.config"
        with open(kube_path, "w") as f:
            f.write(kc_content)
        kube_paths[cluster_name] = kube_path

    if not kube_paths:
        kube_paths["host"] = "/tmp/karmada.config"

    time.sleep(15)

    log_streams = []
    for cluster_name, kube_path in kube_paths.items():
        for label_selector in [f"app={job_name}-worker", f"app={job_name}-server"]:
            cmd = [
                kubectl_path,
                f"--kubeconfig={kube_path}",
                "--insecure-skip-tls-verify",
                "logs",
                "-f",
                "-l",
                label_selector,
                "-n",
                namespace,
                "--all-containers",
                "--prefix=true",
                "--pod-running-timeout=60s",
            ]
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            log_streams.append((cluster_name, label_selector, p))
            print(f"  Started log stream: {cluster_name}/{label_selector}", flush=True)

    start_time = time.time()
    metric_count = 0
    last_metric_time = time.time()
    # 1 server + 1 eval worker + 1 training worker = 3 pods total
    total_pods = 3
    idle_signals = 0
    # Training twin logs TRAIN + EVAL. Eval twin logs only EVAL.
    expected = (fl_rounds * 1 * 2) + (fl_rounds * 1)

    print(
        f"Monitoring Single training... (expecting {expected} metrics from {total_pods} pods)",
        flush=True,
    )
    import queue
    import threading

    log_queue = queue.Queue()

    def stream_reader(q, stream):
        for line in stream:
            q.put(line)

    for _, _, p in log_streams:
        t = threading.Thread(target=stream_reader, args=(log_queue, p.stdout))
        t.daemon = True
        t.start()

    try:
        while time.time() - start_time < 3600:
            try:
                line = log_queue.get(timeout=2.0)
            except queue.Empty:
                alive = [s for s in log_streams if s[2].poll() is None]
                if not alive and log_queue.empty():
                    print("All log streams ended.", flush=True)
                    break
                continue

            print(line, end="", flush=True)

            # Detect pods finishing — each pod prints this exactly once
            if "finished. Idling" in line:
                idle_signals += 1
                print(
                    f"[FINISH] Idle signal {idle_signals}/{total_pods} received.",
                    flush=True,
                )

            # Atomic Match
            match = metric_pattern.search(line)
            if match:
                twin_id, rd, mode, reward, loss = match.groups()
                csv_mode = "EVAL" if "EVAL" in mode else "TRAIN"
                if mode == "EVAL-ONLY-SKIP":
                    continue
                metric_count += 1
                last_metric_time = time.time()

                # Write to CSV
                with open(metrics.path, "a", newline="") as f:
                    csv.writer(f).writerow([rd, twin_id, csv_mode, reward, loss])

                if metric_count % 10 == 0 or metric_count <= 5:
                    print(
                        f"[OK] Metric #{metric_count}: R={rd}, Twin={twin_id}, Mode={mode}, Rew={reward}",
                        flush=True,
                    )

            # Exit 1: all pods signalled they are done AND expected metrics are captured
            if idle_signals >= total_pods and metric_count >= expected:
                print(
                    f"[SUCCESS] All {total_pods} pods finished and expected metrics ({metric_count}/{expected}) captured. Finishing.",
                    flush=True,
                )
                break

            # Exit 3: short stall — no new metrics for 30s after we already have some
            if metric_count > 0 and (time.time() - last_metric_time) > 45:
                print(
                    f"[WARNING] No new metrics for 45s (got {metric_count}/{expected}). Training likely done.",
                    flush=True,
                )
                break

        else:
            print(
                f"[WARNING] Timeout reached after 1 hour (got {metric_count}/{expected})",
                flush=True,
            )
    except Exception as e:
        print(f"Monitor error: {e}", flush=True)
    finally:
        for _, _, p in log_streams:
            try:
                p.terminate()
            except Exception:
                pass


# Load Config Defaults
try:
    with open("config/config.json", "r") as f:
        config_data = json.load(f)
except FileNotFoundError:
    config_data = {"fl_rounds": 5, "local_episodes": 5}


@dsl.pipeline(
    name="Single Twin Multi-Cluster Pipeline",
    description="Non-federated baseline on multi-cluster setup (Member1=Trainer, Host=Evaluator)",
)
def single_twin_multi_cluster_pipeline(
    namespace: str = "default",
    fl_rounds: int = config_data.get("fl_rounds", 5),
    local_episodes: int = config_data.get("local_episodes", 10),
    eval_episodes: int = config_data.get("eval_episodes", 20),
    run_name: str = "single_twin_multi_cluster_run_default",
    mlflow_run_id: str = "",
    mlflow_exp_name: str = "Single-Twin-Multi-Cluster",
):
    import time

    job_id = str(int(time.time()))
    train_single_karmada(
        namespace=namespace,
        fl_rounds=fl_rounds,
        local_episodes=local_episodes,
        eval_episodes=eval_episodes,
        job_id=job_id,
        run_name=run_name,
        mlflow_run_id=mlflow_run_id,
        mlflow_exp_name=mlflow_exp_name,
    ).set_env_variable(
        "MLFLOW_TRACKING_URI", "http://multi-cluster-host-control-plane:30500"
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        single_twin_multi_cluster_pipeline,
        "pipeline_specs/single_twin_multi_cluster_pipeline.yaml",
    )
