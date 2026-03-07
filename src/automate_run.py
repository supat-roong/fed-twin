import kfp
import time
import os
import sys

CLIENT_HOST = "http://localhost:8080"


def run_experiment():
    # Detect Pipeline Choice
    pipeline_type = "fl"
    if len(sys.argv) > 1:
        pipeline_type = sys.argv[1].lower()

    pipeline_map = {
        "fl": {
            "file": "src/pipelines/fl_pipeline.py",
            "yaml": "pipeline_specs/fl_pipeline.yaml",
            "exp": "Fed-Twin-FL",
        },
        "single": {
            "file": "src/pipelines/single_pipeline.py",
            "yaml": "pipeline_specs/single_pipeline.yaml",
            "exp": "Fed-Twin-Single",
        },
        "single_visual": {
            "file": "src/pipelines/single_visual_pipeline.py",
            "yaml": "pipeline_specs/single_visual_pipeline.yaml",
            "exp": "Fed-Twin-Single-Visual",
        },
        "fl_visual": {
            "file": "src/pipelines/fl_visual_pipeline.py",
            "yaml": "pipeline_specs/fl_visual_pipeline.yaml",
            "exp": "Fed-Twin-FL-Visual",
        },
    }

    if pipeline_type not in pipeline_map:
        print(
            f"Unknown pipeline type: {pipeline_type}. Choose 'fl', 'single', 'single_visual' or 'fl_visual'."
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

    # Compile the pipeline
    print(f"Compiling pipeline from {cfg['file']}...")
    retval = os.system(f"python {cfg['file']}")
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
    print(f"Submitting run {run_name}...")

    try:
        run = client.create_run_from_pipeline_package(
            pipeline_file=cfg["yaml"],
            arguments={},
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
