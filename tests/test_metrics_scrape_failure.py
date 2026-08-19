"""A run that captured no metrics must fail, not report success.

The pipelines scrape `[METRIC]` lines out of training-pod logs; that scrape is
how results leave the cluster. When it produces nothing the CSV is a bare
header and every downstream plot is empty -- but the components used to print
a warning and exit 0, so KFP reported Succeeded. A raced scrape was then
indistinguishable from a good run without opening the file. Seen live during
the P3/P4 gates: "0 metrics captured", exit 0, 32-byte CSV, KFP green.
"""

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

from pipelines.fed_twin_single_cluster_pipeline import train_federated  # noqa: E402


class _EmptyStream:
    """A log stream that yields nothing, i.e. the scrape captures no metrics."""

    def __init__(self):
        self.stdout = iter(())

    def terminate(self):
        pass

    def poll(self):
        return 0


def _silence_cluster_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    # kubectl is "already installed" so the component skips downloading it
    import os

    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path, "exists", lambda p: True if p == "/tmp/kubectl" else real_exists(p)
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _EmptyStream())


def test_zero_metrics_captured_fails_the_component(monkeypatch, tmp_path):
    _silence_cluster_calls(monkeypatch, tmp_path)
    metrics = SimpleNamespace(path=str(tmp_path / "metrics.csv"))

    with pytest.raises(RuntimeError, match="captured 0 of"):
        train_federated.python_func(
            namespace="kubeflow", fl_rounds=2, num_workers=2, local_episodes=2,
            eval_episodes=2, job_id="1", run_name="r", mlflow_run_id="",
            mlflow_exp_name="e", metrics=metrics,
        )


def test_the_failure_names_what_to_check(monkeypatch, tmp_path):
    """The message must point at the training pods, not just say 'failed'.

    The training itself usually succeeded -- it is the scrape that broke -- so
    an operator reading this needs to know where the real results are.
    """
    _silence_cluster_calls(monkeypatch, tmp_path)
    metrics = SimpleNamespace(path=str(tmp_path / "metrics.csv"))

    with pytest.raises(RuntimeError) as excinfo:
        train_federated.python_func(
            namespace="kubeflow", fl_rounds=2, num_workers=2, local_episodes=2,
            eval_episodes=2, job_id="1", run_name="r", mlflow_run_id="",
            mlflow_exp_name="e", metrics=metrics,
        )
    msg = str(excinfo.value)
    assert "no data" in msg
    assert "pods" in msg
