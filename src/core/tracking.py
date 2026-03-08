import os
import logging
import mlflow

log = logging.getLogger(__name__)


def setup_mlflow():
    """
    Configure MLflow tracking URI and experiment based on environment variables.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fed-twin-experiments")
    run_id = os.getenv("MLFLOW_RUN_ID")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Configure S3 for artifacts if running in K8s
    if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "minio")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY", "minio123")
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    if run_id:
        # Check if there is already an active run to avoid "Run already active" error
        if not mlflow.active_run():
            mlflow.start_run(run_id=run_id)
        return tracking_uri, experiment_name, run_id

    return tracking_uri, experiment_name, None


def log_metrics(metrics, step=None, run_name=None):
    """
    Log metrics to MLflow. If run_name is provided, it starts/stops a nested run.
    """
    try:
        if run_name:
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        log.warning(f"Failed to log to MLflow: {e}")
