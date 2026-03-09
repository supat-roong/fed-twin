import pytest


@pytest.fixture(autouse=True)
def mock_mlflow_env(monkeypatch):
    """
    Mock MLflow environment variables for all tests.
    This prevents `setup_mlflow()` from hanging when trying to connect
    to a non-existent local tracking server at http://localhost:5050.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///:memory:")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
