import pytest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import automate_run


@patch("automate_run.mlflow")
@patch("kfp.Client")
@patch("os.system")
@patch("os.path.exists")
@patch("sys.argv", ["automate_run.py", "fl"])
@patch("time.time", return_value=1234567890)
@patch("time.sleep")
@patch("builtins.open", new_callable=mock_open)
def test_run_experiment_fl_success(
    mock_file,
    mock_sleep,
    mock_time,
    mock_exists,
    mock_system,
    mock_kfp_client,
    mock_mlflow,
):
    # Setup MLflow mock
    mock_mlflow_run = MagicMock()
    mock_mlflow_run.info.run_id = "mock-run-id-123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_mlflow_run

    # Setup mocks
    mock_exists.return_value = True  # YAML exists
    mock_system.return_value = 0  # Compilation success

    mock_client_instance = mock_kfp_client.return_value
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp-123"
    mock_client_instance.create_experiment.return_value = mock_experiment

    mock_run = MagicMock()
    mock_run.run_id = "run-456"
    mock_client_instance.create_run_from_pipeline_package.return_value = mock_run

    # Mock get_run status progression: Running -> Succeeded
    mock_run_status_running = MagicMock()
    mock_run_status_running.state = "Running"
    mock_run_status_succeeded = MagicMock()
    mock_run_status_succeeded.state = "Succeeded"
    mock_client_instance.get_run.side_effect = [
        mock_run_status_running,
        mock_run_status_succeeded,
    ]

    # Execute
    automate_run.run_experiment()

    # Verify
    mock_kfp_client.assert_called_once_with(host="http://localhost:8080")
    mock_system.assert_called_with("python src/pipelines/fl_pipeline.py")
    mock_client_instance.create_run_from_pipeline_package.assert_called_once()
    # Check that it passed the mlflow run id
    args = mock_client_instance.create_run_from_pipeline_package.call_args.kwargs
    assert args["arguments"]["mlflow_run_id"] == "mock-run-id-123"
    mock_file().write.assert_called_with("run-456")


@patch("automate_run.mlflow")
@patch("kfp.Client", side_effect=Exception("Connection failed"))
@patch("sys.argv", ["automate_run.py", "fl"])
def test_run_experiment_connection_failed(mock_kfp_client, mock_mlflow):
    # Execute (should not raise)
    automate_run.run_experiment()
    # It prints and returns early


@patch("automate_run.mlflow")
@patch("kfp.Client")
@patch("os.system", return_value=1)  # Compilation fails
@patch("sys.argv", ["automate_run.py", "fl"])
def test_run_experiment_compilation_failed(mock_system, mock_kfp_client, mock_mlflow):
    with pytest.raises(SystemExit) as e:
        automate_run.run_experiment()
    assert e.value.code == 1


@patch("automate_run.mlflow")
@patch("kfp.Client")
@patch("os.system", return_value=0)
@patch("os.path.exists", return_value=False)  # YAML missing
@patch("sys.argv", ["automate_run.py", "fl"])
def test_run_experiment_yaml_missing(
    mock_exists, mock_system, mock_kfp_client, mock_mlflow
):
    with pytest.raises(SystemExit) as e:
        automate_run.run_experiment()
    assert e.value.code == 1


@patch("automate_run.mlflow")
@patch("kfp.Client")
@patch("os.system", return_value=0)
@patch("os.path.exists", return_value=True)
@patch("sys.argv", ["automate_run.py", "invalid"])
def test_run_experiment_invalid_type(
    mock_exists, mock_system, mock_kfp_client, mock_mlflow
):
    with pytest.raises(SystemExit) as e:
        automate_run.run_experiment()
    assert e.value.code == 1


@patch("automate_run.mlflow")
@patch("kfp.Client")
@patch("os.system", return_value=0)
@patch("os.path.exists", return_value=True)
@patch("sys.argv", ["automate_run.py", "fl"])
@patch("time.sleep")
@patch("time.time", side_effect=[0, 100, 200, 300])  # For polling
def test_run_experiment_poll_failure(
    mock_time, mock_sleep, mock_exists, mock_system, mock_kfp_client, mock_mlflow
):
    # Setup MLflow mock
    mock_mlflow_run = MagicMock()
    mock_mlflow_run.info.run_id = "mock-run-id-poll"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_mlflow_run

    mock_client_instance = mock_kfp_client.return_value
    mock_client_instance.create_experiment.return_value = MagicMock(id="exp-1")
    mock_run = MagicMock(run_id="run-1")
    mock_client_instance.create_run_from_pipeline_package.return_value = mock_run

    mock_run_status_failed = MagicMock()
    mock_run_status_failed.state = "Failed"
    mock_client_instance.get_run.return_value = mock_run_status_failed

    with pytest.raises(SystemExit) as e:
        automate_run.run_experiment()
    assert e.value.code == 1
