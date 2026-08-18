"""The multi-cluster contract is spread across three files that must agree.

infra.env.multi tells fed-infra which member clusters to *create*;
src/automate_run.py independently decides which member clusters to *talk to*,
deriving the count from config/config.json's num_workers and hardcoding the
name prefix and the Karmada kubeconfig path. Nothing but a comment kept the
three in sync, and a mismatch fails in the worst way available: the clusters
come up fine, the pipeline submits fine, and workers are addressed on
clusters that do not exist -- so the run stalls with no error naming the
cause. These tests make the coupling load-bearing instead of advisory.
"""

import json
import os
import re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key] = value
    return env


def _automate_run_source():
    with open(os.path.join(_ROOT, "src", "automate_run.py")) as f:
        return f.read()


def test_member_count_matches_config_num_workers():
    env = _read_env(os.path.join(_ROOT, "infra.env.multi"))
    with open(os.path.join(_ROOT, "config", "config.json")) as f:
        cfg = json.load(f)
    assert int(env["FED_MEMBER_COUNT"]) == int(cfg["num_workers"]), (
        "infra.env.multi's FED_MEMBER_COUNT (how many member clusters fed-infra "
        "creates) disagrees with config.json's num_workers (how many members "
        "automate_run.py builds kubeconfigs for)"
    )


def test_member_prefix_matches_the_name_automate_run_hardcodes():
    env = _read_env(os.path.join(_ROOT, "infra.env.multi"))
    prefix = env["FED_MEMBER_PREFIX"]
    src = _automate_run_source()
    assert f"{prefix}{{i}}-control-plane" in src, (
        f"automate_run.py does not build member names from FED_MEMBER_PREFIX "
        f"({prefix!r}); the clusters fed-infra creates would not be the ones it "
        f"addresses"
    )
    assert f"kind-{prefix}{{i}}" in src


def test_karmada_config_path_matches_the_one_automate_run_reads():
    env = _read_env(os.path.join(_ROOT, "infra.env.multi"))
    # infra.env.multi writes it as ${HOME}/...; automate_run.py expands ~/...
    declared = env["FED_KARMADA_CONFIG"].replace("${HOME}/", "").replace("$HOME/", "")
    src = _automate_run_source()
    matches = re.findall(r'os\.path\.expanduser\("~/([^"]+)"\)', src)
    assert declared in matches, (
        f"infra.env.multi declares FED_KARMADA_CONFIG={declared!r} but "
        f"automate_run.py reads {matches!r}"
    )


def test_host_cluster_name_matches_the_one_automate_run_expects():
    env = _read_env(os.path.join(_ROOT, "infra.env.multi"))
    host = env["FED_CLUSTER_NAME"]
    src = _automate_run_source()
    assert host in src, (
        f"infra.env.multi's FED_CLUSTER_NAME={host!r} appears nowhere in "
        f"automate_run.py, which streams logs from the host cluster by name"
    )
