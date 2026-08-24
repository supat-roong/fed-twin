"""Contract tests: infra.env.multi tells fed-infra which clusters to create;
the multi-cluster pipelines' parameter DEFAULTS decide which clusters the
Temporal path targets and which host NodePorts member-cluster workers get.
Nothing but these tests keeps the two in sync (spec 3.4).

The old version of this file grepped automate_run.py's source for hardcoded
cluster-name literals; Phase 4 deleted that machinery, so the contract now
binds the compiled pipelines' defaults instead (active-fed's pattern).
"""

import json
import os
import sys

from google.protobuf import json_format

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.fed_twin_multi_cluster_pipeline import fed_twin_multi_cluster_pipeline
from src.pipelines.single_twin_multi_cluster_pipeline import (
    single_twin_multi_cluster_pipeline,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _infra_env_multi() -> dict:
    values = {}
    with open(os.path.join(_ROOT, "infra.env.multi")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key] = value.strip().strip('"')
    return values


def _pipeline_defaults(pipeline_func) -> dict:
    spec = json_format.MessageToDict(pipeline_func.pipeline_spec)
    return {
        name: param.get("defaultValue")
        for name, param in spec["root"]["inputDefinitions"]["parameters"].items()
    }


ENV = _infra_env_multi()
PIPELINES = {
    "fed_twin_multi_cluster": _pipeline_defaults(fed_twin_multi_cluster_pipeline),
    "single_twin_multi_cluster": _pipeline_defaults(single_twin_multi_cluster_pipeline),
}


def test_members_default_matches_the_multi_infra_contract():
    for name, defaults in PIPELINES.items():
        assert int(defaults["members"]) == int(ENV["FED_MEMBER_COUNT"]), (
            f"{name}'s members default disagrees with infra.env.multi's "
            f"FED_MEMBER_COUNT -- workers would round-robin across clusters "
            f"fed-infra never created"
        )


def test_member_prefix_default_matches_the_multi_infra_contract():
    for name, defaults in PIPELINES.items():
        assert defaults["member_prefix"] == ENV["FED_MEMBER_PREFIX"], (
            f"{name}'s member_prefix default disagrees with infra.env.multi's "
            f"FED_MEMBER_PREFIX -- every PropagationPolicy would name clusters "
            f"that do not exist"
        )


def test_nodeport_defaults_match_the_multi_infra_contract():
    for name, defaults in PIPELINES.items():
        assert int(defaults["minio_nodeport"]) == int(ENV["FED_NODEPORT_MINIO_API"]), name
        assert int(defaults["mlflow_nodeport"]) == int(ENV["FED_NODEPORT_MLFLOW"]), name


def test_topology_defaults_to_multi():
    for name, defaults in PIPELINES.items():
        assert defaults["topology"] == "multi", name


def test_member_count_matches_config_num_workers():
    """Dev-profile convention, not a requirement: one member cluster per
    training twin. worker_spec() round-robins workers across members
    (worker_id % member_count + 1), so unequal values work fine at runtime --
    this test just keeps the local profile intentional. Loosen it
    deliberately if the two values ever need to diverge."""
    with open(os.path.join(_ROOT, "config", "config.json")) as f:
        config = json.load(f)
    assert int(ENV["FED_MEMBER_COUNT"]) == int(config["num_workers"])


def test_multi_profile_installs_temporal():
    """Guards Task 1's infra change: without `temporal` in FED_COMPONENTS the
    multi profile boots a cluster where every dispatch dies connecting to a
    Temporal frontend that was never installed (the exact gap Phase 4's
    pre-design survey found)."""
    assert "temporal" in ENV["FED_COMPONENTS"].split(",")
