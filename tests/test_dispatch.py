import os
import re
import sys
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.activities import job_name_for
from src.orchestration.dispatch import (
    KarmadaJobDispatcher,
    LocalJobDispatcher,
    build_propagation_policy,
    dispatcher_for,
)
from src.orchestration.types import WorkerSpec

# RFC-1123 subdomain: lowercase alphanumerics and '-', not starting/ending with '-'.
_VALID_K8S_NAME = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def _spec(**overrides) -> WorkerSpec:
    base = dict(
        fl_round=3, worker_id=2, num_workers=4, local_episodes=25, eval_episodes=5,
        namespace="active-fed", worker_image="active-fed-worker:v1",
        minio_endpoint="minio-service:9000", minio_access_key="ak",
        minio_secret_key="sk", minio_bucket="bucket",
        mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment_name="exp",
        mlflow_run_id="run-1",
        kfp_run_id="abcdef1234567890",
        topology="multi", member_cluster="active-fed-member1",
    )
    base.update(overrides)
    return WorkerSpec(**base)


# ---------------------------------------------------------------------------
# build_propagation_policy -- pure, no I/O.
# ---------------------------------------------------------------------------

def test_build_propagation_policy_targets_the_job_by_its_deterministic_name():
    spec = _spec()
    policy = build_propagation_policy(spec)
    selectors = policy["spec"]["resourceSelectors"]
    assert len(selectors) == 1
    assert selectors[0]["kind"] == "Job"
    assert selectors[0]["name"] == job_name_for(spec)
    assert selectors[0]["namespace"] == spec.namespace


def test_build_propagation_policy_cluster_names_contains_exactly_the_one_member():
    spec = _spec(member_cluster="active-fed-member2")
    policy = build_propagation_policy(spec)
    assert policy["spec"]["placement"]["clusterAffinity"]["clusterNames"] == [
        "active-fed-member2"
    ]


def test_build_propagation_policy_name_is_deterministic():
    assert build_propagation_policy(_spec())["metadata"]["name"] == build_propagation_policy(
        _spec()
    )["metadata"]["name"]


def test_build_propagation_policy_name_differs_by_worker_and_round():
    a = build_propagation_policy(_spec(worker_id=1))["metadata"]["name"]
    b = build_propagation_policy(_spec(worker_id=2))["metadata"]["name"]
    assert a != b


def test_build_propagation_policy_name_is_a_valid_kubernetes_name():
    name = build_propagation_policy(_spec())["metadata"]["name"]
    assert _VALID_K8S_NAME.fullmatch(name), name
    assert len(name) <= 63


def test_build_propagation_policy_name_reuses_job_name_fors_validation():
    # job_name_for already raises ValueError on a kfp_run_id that would
    # produce an invalid Kubernetes name. The policy name
    # must reuse that same discipline rather than inventing a second
    # validation rule -- so an input that job_name_for rejects must also be
    # rejected here, for the same reason.
    bad = _spec(kfp_run_id="{{$.pipeline_job_uuid}}")
    with pytest.raises(ValueError):
        build_propagation_policy(bad)


# --- The critical safety property ------------------------------------------

def test_build_propagation_policy_raises_on_empty_member_cluster():
    # An empty clusterNames list in a Karmada PropagationPolicy means *all*
    # clusters. Building a policy for a spec with no member_cluster must
    # raise rather than silently produce a policy that fans this worker's
    # Job out to every joined member.
    spec = _spec(member_cluster="")
    with pytest.raises(ValueError):
        build_propagation_policy(spec)


# --- Whitespace/format defeats truthiness -----------------------------------
# `if not spec.member_cluster:` is Python truthiness -- "   "/"\t\n" are
# non-empty strings and sail straight through. This doesn't reach the
# catastrophic empty-clusterNames case (a garbage, non-matching name selects
# *zero* real clusters, not all), but it silently produces a Job Karmada
# schedules nowhere, stalling the round with the root cause nowhere near the
# error. The fix validates the full RFC 1123 label format, not just
# blankness, so it also catches the adjacent non-blank-but-still-invalid
# cases (uppercase, embedded spaces) a bare `.strip()` truthiness check
# would still miss.

@pytest.mark.parametrize("garbage", ["   ", "\t\n", " "])
def test_build_propagation_policy_raises_on_whitespace_only_member_cluster(garbage):
    spec = _spec(member_cluster=garbage)
    with pytest.raises(ValueError):
        build_propagation_policy(spec)


@pytest.mark.parametrize(
    "garbage",
    ["ACTIVE-FED-MEMBER1", "active fed member1", "-active-fed-member1", "active-fed-member1-"],
)
def test_build_propagation_policy_raises_on_non_blank_but_invalid_member_cluster(garbage):
    # Non-blank but not a valid RFC 1123 label: uppercase, an embedded space,
    # or a leading/trailing '-'. Each of these selects zero real clusters
    # exactly like whitespace does, so each gets the same treatment.
    spec = _spec(member_cluster=garbage)
    with pytest.raises(ValueError):
        build_propagation_policy(spec)


def test_dispatcher_for_multi_topology_with_whitespace_only_member_cluster_raises():
    spec = _spec(topology="multi", member_cluster="\t\n")
    with pytest.raises(ValueError):
        dispatcher_for(spec)


# ---------------------------------------------------------------------------
# dispatcher_for
# ---------------------------------------------------------------------------

def test_dispatcher_for_single_topology_returns_local_dispatcher():
    spec = _spec(topology="single", member_cluster="")
    assert isinstance(dispatcher_for(spec), LocalJobDispatcher)


def test_dispatcher_for_multi_topology_returns_karmada_dispatcher():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    assert isinstance(dispatcher_for(spec), KarmadaJobDispatcher)


def test_dispatcher_for_multi_topology_with_empty_member_cluster_raises():
    # The same safety property as above, enforced at the dispatch-selection
    # boundary too: a caller must never be handed a working dispatcher for
    # an unsafe (multi, no member) spec.
    spec = _spec(topology="multi", member_cluster="")
    with pytest.raises(ValueError):
        dispatcher_for(spec)


def test_dispatcher_for_unknown_topology_raises():
    spec = _spec(topology="quantum", member_cluster="")
    with pytest.raises(ValueError):
        dispatcher_for(spec)


# ---------------------------------------------------------------------------
# LocalJobDispatcher -- wraps the create/delete Job path. Fake, in-memory
# stand-in for kubernetes.client.BatchV1Api, mirroring the style of
# test_orchestration_activities.py's FakeBatchApi.
# ---------------------------------------------------------------------------

class FakeBatchApi:
    def __init__(self, existing=False):
        self.existing = existing
        self.calls: list[str] = []
        # Records the body create_namespaced_job
        # was actually called with, so tests can assert against what the
        # dispatcher *applied* rather than a second, independent
        # build_job_manifest(spec) call that proves nothing about it.
        self.applied_body: dict | None = None

    def create_namespaced_job(self, namespace, body):
        self.calls.append("create")
        self.applied_body = body
        if self.existing:
            raise ApiException(status=409)
        self.existing = True

    def delete_namespaced_job(self, name, namespace, propagation_policy):
        self.calls.append(f"delete:{propagation_policy}")
        if not self.existing:
            raise ApiException(status=404)
        self.existing = False

    def read_namespaced_job_status(self, name, namespace):
        # Only exercised via _await_job_deleted's delete-then-poll loop: it
        # needs to see the Job gone once delete_namespaced_job has run, on
        # the *same* client used to
        # create/delete it -- the Karmada aggregated *status classification*
        # goes through a separate client (_karmada_clients, see
        # FakeKarmadaStatusApi below), so the content returned here is never
        # inspected, only whether this raises 404.
        self.calls.append("read")
        if not self.existing:
            raise ApiException(status=404)
        return SimpleNamespace(status=None)


def test_local_dispatcher_ensure_job_creates_and_returns_job_name():
    spec = _spec(topology="single", member_cluster="")
    batch = FakeBatchApi()
    name = LocalJobDispatcher()._ensure_job_with(batch, spec)
    assert name == job_name_for(spec)
    assert batch.calls == ["create"]


def test_local_dispatcher_ensure_job_is_idempotent_when_job_already_exists():
    spec = _spec(topology="single", member_cluster="")
    batch = FakeBatchApi(existing=True)
    name = LocalJobDispatcher()._ensure_job_with(batch, spec)
    assert name == job_name_for(spec)
    assert batch.calls == ["create"]  # 409 swallowed, no crash


def test_local_dispatcher_delete_job_tolerates_already_gone():
    spec = _spec(topology="single", member_cluster="")
    batch = FakeBatchApi(existing=False)
    LocalJobDispatcher()._delete_job_with(batch, spec)  # must not raise
    assert batch.calls == ["delete:Background"]


# ---------------------------------------------------------------------------
# KarmadaJobDispatcher -- applies both the Job and the PropagationPolicy.
# ---------------------------------------------------------------------------

class FakeCustomObjectsApi:
    def __init__(self, existing=False):
        self.existing = existing
        self.calls: list[tuple] = []

    def create_namespaced_custom_object(self, group, version, namespace, plural, body):
        self.calls.append(("create", plural, body["metadata"]["name"]))
        if self.existing:
            raise ApiException(status=409)
        self.existing = True

    def delete_namespaced_custom_object(self, group, version, namespace, plural, name):
        self.calls.append(("delete", plural, name))
        if not self.existing:
            raise ApiException(status=404)
        self.existing = False


async def test_karmada_dispatcher_ensure_job_applies_job_and_propagation_policy():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi()
    custom = FakeCustomObjectsApi()
    name = await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)
    assert name == job_name_for(spec)
    assert batch.calls == ["create"]
    assert custom.calls == [
        ("create", "propagationpolicies", build_propagation_policy(spec)["metadata"]["name"])
    ]


async def test_karmada_dispatcher_ensure_job_is_idempotent(monkeypatch):
    import src.orchestration.dispatch as dispatch_module

    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    # The aggregated-status lookup is a
    # separate client from `batch` above -- monkeypatch it explicitly here
    # (rather than relying on FED_KARMADA_CONFIG being unset in this
    # environment) so this test's "must not raise" guarantee doesn't
    # silently depend on ambient process state. A still-running status is
    # the least surprising thing to script for a bare "must not raise"
    # idempotency check; the absent/lagging and terminally-failed cases each
    # get their own dedicated test below.
    running = _job_status(conditions=[])
    monkeypatch.setattr(
        dispatch_module, "_karmada_clients", lambda: (FakeKarmadaStatusApi(running), None)
    )
    # Must not raise even though both the Job and the PropagationPolicy
    # already exist from a previous (e.g. retried) attempt.
    await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)
    # ...and must not have deleted the
    # still-running Job either.
    assert "delete:Background" not in batch.calls


# ---------------------------------------------------------------------------
# On a 409 (Job already exists -- exactly what
# a Temporal *activity retry* of a failed worker sees, since job_name_for is
# deterministic), the pre-fix KarmadaJobDispatcher just swallowed the 409 and
# re-attached unconditionally. If the existing Job was terminally Failed
# (real containers crash), every retry attempt re-polled MinIO for an
# artifact a dead Job can never produce, burning the full POD_WATCH_TIMEOUT_S
# per attempt for nothing. FakeKarmadaStatusApi below is a fake for the
# Karmada *aggregated* BatchV1Api (from _karmada_clients, monkeypatched),
# stateful enough to script create -> [409] -> read (classify) ->
# (delete -> poll-until-gone) -> create, mirroring
# test_orchestration_activities.py's FakeBatchApi/PollingFakeBatchApi for
# activities._ensure_job, the single-topology fix for the identical bug.
# ---------------------------------------------------------------------------


def _condition(type_: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(type=type_, status=status)


def _job_status(conditions=None, succeeded=None, failed=None, active=None) -> SimpleNamespace:
    return SimpleNamespace(conditions=conditions, succeeded=succeeded, failed=failed, active=active)


class FakeKarmadaStatusApi:
    """Fake for the Karmada aggregated BatchV1Api returned by
    _karmada_clients(), used only for the read_namespaced_job_status calls
    _karmada_job_status makes -- decoupled from FakeBatchApi's create/delete
    tracking (a real Karmada aggregated-status read goes through a distinct
    client construction from the Job create/delete calls, so this mirrors
    that split rather than collapsing it).

    aggregated_status=None means "absent/lagging" -- Karmada hasn't
    propagated/observed the Job on the member cluster yet, which read_namespaced_
    job_status models as a 404, exactly like an absent Job. reads_until_deleted
    lets a test prove the delete-then-poll loop (_await_job_deleted) actually
    polls instead of assuming instant deletion.
    """

    def __init__(self, aggregated_status, reads_until_deleted=0):
        self.aggregated_status = aggregated_status
        self.reads_until_deleted = reads_until_deleted
        self.read_calls = 0

    def read_namespaced_job_status(self, name, namespace):
        self.read_calls += 1
        if self.aggregated_status is None:
            raise ApiException(status=404)
        if self.reads_until_deleted > 0:
            self.reads_until_deleted -= 1
        elif self.read_calls > 1:
            # Once the delete-wait loop has polled once past the countdown,
            # report gone -- mirrors a real deletion completing.
            self.aggregated_status = None
            raise ApiException(status=404)
        return SimpleNamespace(status=self.aggregated_status)


async def test_karmada_dispatcher_ensure_job_reattaches_to_a_still_active_existing_job(
    monkeypatch,
):
    # A retry against a genuinely live Job: 409 on create, aggregated status
    # is running (no terminal condition) -- must re-attach, never delete.
    import src.orchestration.dispatch as dispatch_module

    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    running = _job_status(conditions=[])
    status_api = FakeKarmadaStatusApi(running)
    monkeypatch.setattr(dispatch_module, "_karmada_clients", lambda: (status_api, None))

    name = await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)

    assert name == job_name_for(spec)
    assert batch.calls == ["create"]  # exactly one create attempt; no delete
    assert status_api.read_calls == 1  # one status read to classify it


async def test_karmada_dispatcher_ensure_job_deletes_and_recreates_a_terminally_failed_job(
    monkeypatch,
):
    # A retry against a terminally-failed Job (a real container crash): 409
    # on create, aggregated status is Failed -- must delete, wait for the
    # delete to complete, and recreate. Must not just re-attach.
    import src.orchestration.dispatch as dispatch_module

    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    failed = _job_status(conditions=[_condition("Failed", "True")])
    status_api = FakeKarmadaStatusApi(failed, reads_until_deleted=0)
    monkeypatch.setattr(dispatch_module, "_karmada_clients", lambda: (status_api, None))

    name = await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)

    assert name == job_name_for(spec)
    # create (409) -> delete -> (poll via _await_job_deleted) -> create again.
    assert batch.calls[0] == "create"
    assert batch.calls[1] == "delete:Background"
    assert batch.calls.count("create") == 2
    assert batch.existing is True  # the recreated Job is left in place
    # The already-existing PropagationPolicy (still 409s) is left untouched
    # -- it selects by the Job's deterministic name, so it stays consistent
    # with whichever Job currently has that name without needing to be
    # deleted/recreated itself.
    assert custom.calls == [
        ("create", "propagationpolicies", build_propagation_policy(spec)["metadata"]["name"])
    ]


async def test_karmada_dispatcher_ensure_job_does_not_delete_on_absent_or_lagging_status(
    monkeypatch,
):
    # THE safety property warned about most
    # explicitly above: the Karmada aggregated status can be absent or lagging
    # right after dispatch (the Job hasn't propagated to the member cluster
    # yet, or the aggregated API hasn't caught up). Absent must mean "not
    # yet", never "failed" -- deleting a healthy, just-propagated Job because
    # its status hasn't appeared would be a worse bug than the one this
    # fixes. A 409 on create (the Job object already exists on the control
    # plane) combined with a 404 on the aggregated status read (Karmada
    # hasn't observed/propagated it yet) is exactly that lagging window.
    import src.orchestration.dispatch as dispatch_module

    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    status_api = FakeKarmadaStatusApi(aggregated_status=None)
    monkeypatch.setattr(dispatch_module, "_karmada_clients", lambda: (status_api, None))

    name = await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)  # must not raise

    assert name == job_name_for(spec)
    assert "delete:Background" not in batch.calls
    assert batch.calls.count("create") == 1  # no delete-and-recreate cycle


async def test_karmada_dispatcher_ensure_job_does_not_delete_when_karmada_unreachable(
    monkeypatch,
):
    # Same safety property, different cause: the Karmada apiserver itself is
    # unreachable (or FED_KARMADA_CONFIG unset). Best-effort status lookup
    # failing must never be misread as "failed" -- MinIO polling (or, here,
    # simply re-attaching) remains the ground truth.
    import src.orchestration.dispatch as dispatch_module

    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    monkeypatch.setattr(
        dispatch_module, "_karmada_clients",
        lambda: (_ for _ in ()).throw(RuntimeError("karmada apiserver unreachable")),
    )

    await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)  # must not raise

    assert "delete:Background" not in batch.calls
    assert batch.calls.count("create") == 1


def test_karmada_dispatcher_delete_job_removes_job_and_policy():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=True)
    custom = FakeCustomObjectsApi(existing=True)
    KarmadaJobDispatcher()._delete_job_with(batch, custom, spec)
    assert batch.calls == ["delete:Background"]
    assert custom.calls == [
        ("delete", "propagationpolicies", build_propagation_policy(spec)["metadata"]["name"])
    ]


def test_karmada_dispatcher_delete_job_tolerates_already_gone():
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi(existing=False)
    custom = FakeCustomObjectsApi(existing=False)
    KarmadaJobDispatcher()._delete_job_with(batch, custom, spec)  # must not raise


async def test_karmada_job_manifest_matches_the_local_dispatchers_manifest():
    # KarmadaJobDispatcher must apply the *same* Job manifest build_job_manifest
    # builds for the local dispatcher -- topology is a dispatch-time concern,
    # not a manifest-shape concern.
    #
    # This test used to assert against a
    # *fresh*, independent build_job_manifest(spec) call rather than what
    # _ensure_job_with actually applied, because FakeBatchApi.create_namespaced_
    # job never recorded its `body` argument -- the reviewer proved it stayed
    # green even when the dispatcher was monkeypatched to apply a deliberately
    # broken manifest (backoffLimit=3, restartPolicy="OnFailure", a different
    # Job name entirely). FakeBatchApi.applied_body now records what was
    # actually passed to create_namespaced_job, so this asserts against that.
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch = FakeBatchApi()
    custom = FakeCustomObjectsApi()
    await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec)

    applied = batch.applied_body
    assert applied is not None
    assert applied["metadata"]["name"] == job_name_for(spec)
    assert applied["spec"]["backoffLimit"] == 0
    assert applied["spec"]["template"]["spec"]["restartPolicy"] == "Never"


class FakeCoreApi:
    """Minimal CoreV1Api double recording namespace creation."""

    def __init__(self, existing=None):
        self.existing = set(existing or [])
        self.created = []

    def read_namespace(self, name):
        if name not in self.existing:
            raise ApiException(status=404)
        return {"metadata": {"name": name}}

    def create_namespace(self, body):
        name = body["metadata"]["name"]
        if name in self.existing:
            raise ApiException(status=409)
        self.existing.add(name)
        self.created.append(name)
        return body


@pytest.mark.asyncio
async def test_karmada_dispatcher_creates_the_namespace_on_the_control_plane():
    """The Karmada apiserver has its own namespaces, separate from the host's.

    Creating the worker Job on Karmada fails with 404 `namespaces "active-fed"
    not found` unless the namespace exists *there* -- the host cluster having
    it is irrelevant. Observed live during the P3 gate: every worker dispatch
    died on this, with no Job and no PropagationPolicy ever created.
    """
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch, custom, core = FakeBatchApi(), FakeCustomObjectsApi(), FakeCoreApi()

    await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec, core=core)

    assert spec.namespace in core.created, (
        "the dispatcher did not create the namespace on the Karmada control "
        "plane, so create_namespaced_job would 404"
    )


@pytest.mark.asyncio
async def test_karmada_dispatcher_tolerates_an_existing_namespace():
    """Re-running against a provisioned control plane must not fail."""
    spec = _spec(topology="multi", member_cluster="active-fed-member1")
    batch, custom = FakeBatchApi(), FakeCustomObjectsApi()
    core = FakeCoreApi(existing={spec.namespace})

    await KarmadaJobDispatcher()._ensure_job_with(batch, custom, spec, core=core)

    assert core.created == [], "an already-present namespace must not be recreated"


@pytest.mark.xfail(
    strict=True,
    reason="fed-twin's k8s/rbac.yaml RBAC swap (nodes get/list for topology='multi') "
    "is Phase 3 work per docs/superpowers/specs/2026-08-23-fed-twin-temporal-retrofit-design.md §3.5 — "
    "not yet applicable while this module is unwired (Phase 1)."
)
def test_rbac_grants_the_node_access_the_multi_endpoint_rewrite_needs():
    """k8s/rbac.yaml must allow listing nodes, or topology='multi' cannot dispatch.

    _resolve_host_node_ip reads the host cluster's node InternalIP to rewrite
    worker MinIO/MLflow endpoints to NodePorts. Without this grant every multi
    dispatch dies with 403 `nodes is forbidden`, which is exactly how the first
    run after the rewrite landed failed -- the code and the RBAC that permits
    it live in different files and nothing tied them together.
    """
    import pathlib

    import yaml as _yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    docs = list(_yaml.safe_load_all((root / "k8s" / "rbac.yaml").read_text()))
    roles = [d for d in docs if d and d.get("kind") == "ClusterRole"]
    assert roles, "no ClusterRole in k8s/rbac.yaml"

    granted = False
    for role in roles:
        for rule in role.get("rules", []):
            if "" in rule.get("apiGroups", []) and "nodes" in rule.get("resources", []):
                if "list" in rule.get("verbs", []):
                    granted = True
    assert granted, (
        "no ClusterRole in k8s/rbac.yaml grants list on core/nodes, so "
        "_resolve_host_node_ip will 403 and every topology='multi' dispatch fails"
    )
