import mlflow
import kfp
import time
import os
import sys

CLIENT_HOST = "http://localhost:8080"


def run_experiment():
    # Detect Pipeline Choice
    pipeline_type = "fl_k8s"
    if len(sys.argv) > 1:
        pipeline_type = sys.argv[1].lower()

    pipeline_map = {
        "fl_k8s": {
            "file": "src/pipelines/fl_k8s_pipeline.py",
            "yaml": "pipeline_specs/fl_k8s_pipeline.yaml",
            "exp": "Fed-Twin-FL",
        },
        "single_k8s": {
            "file": "src/pipelines/single_k8s_pipeline.py",
            "yaml": "pipeline_specs/single_k8s_pipeline.yaml",
            "exp": "Fed-Twin-Single",
        },
        "single_visual_k8s": {
            "file": "src/pipelines/single_visual_k8s_pipeline.py",
            "yaml": "pipeline_specs/single_visual_k8s_pipeline.yaml",
            "exp": "Fed-Twin-Single-Visual",
        },
        "fl_visual_k8s": {
            "file": "src/pipelines/fl_visual_k8s_pipeline.py",
            "yaml": "pipeline_specs/fl_visual_k8s_pipeline.yaml",
            "exp": "Fed-Twin-FL-Visual",
        },
        "fl_karmada": {
            "file": "src/pipelines/fl_karmada_pipeline.py",
            "yaml": "pipeline_specs/fl_karmada_pipeline.yaml",
            "exp": "Fed-Twin-FL-Karmada",
        },
        "single_karmada": {
            "file": "src/pipelines/single_karmada_pipeline.py",
            "yaml": "pipeline_specs/single_karmada_pipeline.yaml",
            "exp": "Fed-Twin-Single-Karmada",
        },
    }

    if pipeline_type not in pipeline_map:
        print(
            f"Unknown pipeline type: {pipeline_type}. Choose 'fl_k8s', 'single_k8s', 'single_visual_k8s', 'fl_visual_k8s', 'fl_karmada', or 'single_karmada'."
        )
        sys.exit(1)

    cfg = pipeline_map[pipeline_type]
    print(f"--- Running {pipeline_type.upper()} Pipeline ---")

    print(f"Connecting to KFP at {CLIENT_HOST}...")
    try:
        client = kfp.Client(host=CLIENT_HOST)
    except Exception as e:
        print(f"Failed to connect to KFP: {e}")
        return

    print(f"Compiling pipeline from {cfg['file']}...")
    retval = os.system(f"{sys.executable} {cfg['file']}")
    if retval != 0:
        print("Compilation failed.")
        sys.exit(1)

    if not os.path.exists(cfg["yaml"]):
        print(f"Failed to compile pipeline. {cfg['yaml']} not found.")
        sys.exit(1)

    # Check/Create Experiment
    exp_name = cfg["exp"]
    experiment = None
    try:
        experiment = client.create_experiment(name=exp_name)
    except Exception as e:
        print(f"Experiment check: {e}")
        try:
            exps = client.list_experiments(filter=f"display_name='{exp_name}'")
            if exps.experiments:
                experiment = exps.experiments[0]
        except Exception as e2:
            print(f"Failed to list experiments: {e2}")

    if experiment is None:
        print("Failed to get experiment.")
        sys.exit(1)

    exp_id = getattr(experiment, "experiment_id", None) or getattr(
        experiment, "id", None
    )
    print(f"Using Experiment: {exp_name} ({exp_id})")

    # Submit Run
    run_name = f"{pipeline_type}_run_{int(time.time())}"

    # Set MLflow environment for local tracking and pod configuration
    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment(exp_name)

    # Create the run locally first to get the run_id
    with mlflow.start_run(run_name=run_name) as ml_run:
        mlflow_run_id = ml_run.info.run_id
        print(f"Created MLflow Run: {run_name} (ID: {mlflow_run_id})")

    print(f"Submitting run {run_name} to KFP...")

    try:
        # Build per-cluster kubeconfigs for Karmada log streaming
        import json
        try:
            with open(
                os.path.join(
                    os.path.dirname(__file__), "..", "config", "config.json"
                ),
                "r",
            ) as f:
                _cfg = json.load(f)
        except Exception:
            _cfg = {}
        _w = int(_cfg.get("num_workers", 3))

        # Build per-cluster kubeconfigs for Karmada log streaming
        member_kubeconfigs_json = "{}"
        if pipeline_type in ["fl_karmada", "single_karmada"]:
            import subprocess as _sp

            member_configs = {}
            # Define host cluster
            clusters_info = [
                ("host", "fed-twin-host-control-plane", "kind-fed-twin-host")
            ]
            # Read num_workers from config to deterministically generate member list

            for i in range(1, _w + 1):
                clusters_info.append(
                    (
                        f"member{i}",
                        f"fed-twin-member{i}-control-plane",
                        f"kind-fed-twin-member{i}",
                    )
                )
            for cluster_key, container_name, context_name in clusters_info:
                try:
                    # Get internal Docker bridge IP
                    ip_result = _sp.run(
                        [
                            "docker",
                            "inspect",
                            "-f",
                            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                            container_name,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    internal_ip = ip_result.stdout.strip()
                    if not internal_ip:
                        print(f"  Warning: No IP found for {container_name}, skipping.")
                        continue

                    # Export the kubeconfig for this context
                    kc_result = _sp.run(
                        [
                            "kubectl",
                            "config",
                            "view",
                            "--minify",
                            "--flatten",
                            f"--context={context_name}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    kc_content = kc_result.stdout
                    # Replace host-mapped port with internal port 6443
                    import re

                    kc_content = re.sub(
                        r"https://(?:127\.0\.0\.1|localhost):\d+",
                        f"https://{internal_ip}:6443",
                        kc_content,
                    )
                    member_configs[cluster_key] = kc_content
                    print(
                        f"  Member kubeconfig: {cluster_key} -> {container_name} ({internal_ip})"
                    )
                except Exception as e:
                    print(
                        f"  Warning: Could not build kubeconfig for {cluster_key}: {e}"
                    )

            member_kubeconfigs_json = json.dumps(member_configs)

        # Define strings dynamically instead of variables to prevent NameError
        k_config = open(os.path.expanduser("~/.karmada/karmada-apiserver.config")).read() if pipeline_type in ["fl_karmada", "single_karmada"] else ""
        m_configs = member_kubeconfigs_json if pipeline_type in ["fl_karmada", "single_karmada"] else "{}"

        if pipeline_type in ["fl_karmada", "single_karmada"]:
            # Create Kubernetes secret directly to bypass KFP parameter limitations
            import tempfile
            import subprocess as _sp
            secret_name = f"karmconfigs-{mlflow_run_id}"
            
            with tempfile.NamedTemporaryFile("w") as f_k, tempfile.NamedTemporaryFile("w") as f_m:
                f_k.write(k_config)
                f_k.flush()
                f_m.write(m_configs)
                f_m.flush()
                
                print(f"Creating Kubernetes secret {secret_name} for configs...")
                _sp.run(["kubectl", "delete", "secret", secret_name, "-n", "kubeflow", "--ignore-not-found"], check=False)
                _sp.run([
                    "kubectl", "create", "secret", "generic", secret_name,
                    f"--from-file=karmada={f_k.name}",
                    f"--from-file=members={f_m.name}",
                    "-n", "kubeflow"
                ], check=True)

        import yaml
        expected_params = set()
        try:
            with open(cfg["yaml"], "r") as f:
                yaml_content = yaml.safe_load(f)
                input_defs = yaml_content.get("root", {}).get("inputDefinitions", {}).get("parameters", {})
                expected_params = set(input_defs.keys())
        except Exception as e:
            print(f"Warning: Could not parse {cfg['yaml']}: {e}")

        all_args = {
            "run_name": run_name,
            "mlflow_run_id": mlflow_run_id,
            "mlflow_exp_name": exp_name,
            "fl_rounds": _cfg.get("fl_rounds", 5),
            "num_workers": _w,
            "local_episodes": _cfg.get("local_episodes", 10),
            "eval_episodes": _cfg.get("eval_episodes", 20),
        }

        # Filter arguments to only pass those expected by the pipeline
        if expected_params:
            final_args = {k: v for k, v in all_args.items() if k in expected_params}
        else:
            final_args = all_args
            
        print(f"Passing arguments: {list(final_args.keys())}")

        run = client.create_run_from_pipeline_package(
            pipeline_file=cfg["yaml"],
            arguments=final_args,
            run_name=run_name,
            experiment_name=exp_name,
            enable_caching=False,
        )

        print(f"Run submitted! Run ID: {run.run_id}")
        with open(f"metrics/last_run_id_{pipeline_type}.txt", "w") as f:
            f.write(run.run_id)

        # Wait for completion...
        print("Waiting for completion...")
        start_time = time.time()
        conn_errors = 0
        while True:
            try:
                r = client.get_run(run.run_id)
                status = getattr(r, "state", None) or (
                    r.run.status if hasattr(r, "run") else None
                )

                print(f"Status: {status}")
                conn_errors = 0  # Reset on success

                if status in ["Succeeded", "SUCCEEDED"]:
                    print("✅ Run SUCCEEDED!")
                    break
                elif status in ["Failed", "FAILED", "Error", "ERROR", "Skipped"]:
                    print(f"❌ Run ended with status: {status}")
                    sys.exit(1)
            except Exception as e:
                err_str = str(e).lower()
                if (
                    "failed to establish" in err_str
                    or "connection refused" in err_str
                    or "connection reset" in err_str
                ):
                    conn_errors += 1
                    if conn_errors > 20:  # Increased from 15
                        print(f"Too many connection errors: {e}")
                        sys.exit(1)
                    print(f"Connection lost, retrying ({conn_errors}/20)...")
                    time.sleep(15)  # Increased sleep to allow port-forward to recover
                    # Re-initialize client if connection was reset
                    try:
                        client = kfp.Client(host=CLIENT_HOST)
                    except Exception:
                        pass
                    continue
                else:
                    print(f"Error polling run: {e}")
                    sys.exit(1)

            if time.time() - start_time > 7200:  # 2 hours for 100-round FL training
                print("Timeout.")
                sys.exit(1)

            time.sleep(20)

    except Exception as e:
        print(f"Failed to submit or monitor run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_experiment()
