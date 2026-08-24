"""
Dispatch abstraction: create/delete a worker Job either in the local cluster
or, via Karmada, pinned to exactly one member cluster.

Split out of activities.py rather than folded into it: `build_propagation_policy`
must stay pure and importable without a kubeconfig, exactly like
`build_job_manifest`/`job_name_for` in activities.py, so it can be unit-tested
directly. Anything that talks to a Kubernetes/Karmada apiserver is imported
lazily inside a function, following the same discipline as
`activities._k8s_batch_and_core`.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from src.orchestration.activities import (
    JOB_DELETE_POLL_S,
    JOB_DELETE_TIMEOUT_S,
    _await_job_deleted,
    _karmada_job_status,
    build_job_manifest,
    job_name_for,
)
from src.orchestration.types import WorkerSpec

log = logging.getLogger(__name__)

# RFC 1123 label: lowercase alphanumerics and '-', not starting/ending with
# '-'. Kubernetes/Karmada cluster names must satisfy this -- matches
# job_name_for's own validation discipline (activities.py's
# _VALID_RUN_ID_FRAGMENT) applied to the other name this module builds.
_VALID_CLUSTER_NAME = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def _require_valid_cluster_name(member_cluster: str, context: str) -> None:
    """Raise ValueError unless member_cluster is a non-blank, RFC-1123-valid
    Kubernetes/Karmada cluster name.

    Both call sites below used to check only
    `if not spec.member_cluster:`, Python truthiness -- so "   " / "\t\n"
    sailed through (a non-empty whitespace string is truthy). That doesn't
    reach the catastrophic empty-`clusterNames` case (an empty string still
    raises; a whitespace-garbage name is merely non-empty), but a
    PropagationPolicy with a `clusterNames` entry that matches no real
    joined member is accepted by the Karmada apiserver and silently selects
    *zero* clusters -- the Job schedules nowhere, and the round stalls for
    the full watch timeout with nothing pointing at a blank/garbage config
    value as the cause.

    Rejects outright rather than stripping-and-accepting: a caller that
    silently trims a value with leading/trailing whitespace would just as
    silently launder whatever upstream bug produced it (a stray newline from
    a shell substitution, a YAML block scalar, ...) instead of surfacing it
    here, next to the offending value. Checking the full RFC 1123 label
    format -- not just blankness -- also catches the same class of
    non-blank-but-still-broken input a bare `.strip()` truthiness check
    would still miss (uppercase, embedded spaces, a leading/trailing '-'):
    any of these selects zero clusters exactly like whitespace does, so they
    get the same treatment.
    """
    if not member_cluster or not _VALID_CLUSTER_NAME.fullmatch(member_cluster):
        raise ValueError(
            f"{context} requires member_cluster to be a valid Kubernetes/Karmada "
            f"cluster name (RFC 1123 label: lowercase alphanumerics and '-', not "
            f"starting/ending with '-'); got {member_cluster!r}"
        )


class JobDispatcher(Protocol):
    async def ensure_job(self, spec: WorkerSpec) -> str:
        """Create the worker Job if absent. Idempotent. Returns the Job name."""
        ...

    def delete_job(self, spec: WorkerSpec) -> None:
        """Delete the worker Job (and anything dispatch created alongside it).

        Safe to call when already gone.
        """
        ...


def propagation_policy_name_for(spec: WorkerSpec) -> str:
    """Deterministic PropagationPolicy name for one worker's Job.

    Derived from job_name_for(spec) rather than validated independently:
    job_name_for already raises ValueError on a kfp_run_id fragment that
    would produce an invalid Kubernetes name. Reusing it
    here means the policy name inherits that validation instead of a second,
    parallel naming rule.
    """
    return f"{job_name_for(spec)}-pp"


def build_propagation_policy(spec: WorkerSpec) -> dict:
    """Render the Karmada PropagationPolicy that pins one worker's Job to one
    member cluster. Pure -- no I/O, directly unit-testable.

    CRITICAL SAFETY PROPERTY: an empty `clusterNames` list in a Karmada
    PropagationPolicy targets *all* clusters, not none. A spec with an empty
    member_cluster must never reach as far as a rendered policy silently --
    it would propagate this worker's Job to every joined member, running N
    times the intended work on the wrong physics seeds. Raise instead of
    defaulting clusterNames to empty.
    """
    _require_valid_cluster_name(
        spec.member_cluster,
        f"build_propagation_policy (worker {spec.worker_id}, round {spec.fl_round})",
    )

    return {
        "apiVersion": "policy.karmada.io/v1alpha1",
        "kind": "PropagationPolicy",
        "metadata": {
            "name": propagation_policy_name_for(spec),
            "namespace": spec.namespace,
        },
        "spec": {
            "resourceSelectors": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": job_name_for(spec),
                    "namespace": spec.namespace,
                }
            ],
            "placement": {
                "clusterAffinity": {
                    "clusterNames": [spec.member_cluster],
                }
            },
        },
    }


def dispatcher_for(spec: WorkerSpec) -> JobDispatcher:
    """Pick the dispatcher matching spec.topology.

    Refuses (raises ValueError) rather than guessing for a "multi" spec with
    no member_cluster, and for any topology value that isn't recognised --
    the same critical safety property build_propagation_policy enforces,
    caught here too so a caller never even gets a working dispatcher for an
    unsafe spec.
    """
    if spec.topology == "single":
        return LocalJobDispatcher()
    if spec.topology == "multi":
        _require_valid_cluster_name(
            spec.member_cluster,
            f"topology='multi' (worker {spec.worker_id}, round {spec.fl_round})",
        )
        return KarmadaJobDispatcher()
    raise ValueError(f"unknown topology {spec.topology!r}; expected 'single' or 'multi'")


class LocalJobDispatcher:
    """Create/delete a batch/v1 Job in the local cluster."""

    async def ensure_job(self, spec: WorkerSpec) -> str:
        # async only to satisfy JobDispatcher's Protocol (KarmadaJobDispatcher
        # needs to await a delete-and-recreate poll on 409; this path never
        # does -- topology="single" doesn't route through this class at all
        # today, launch_and_watch_pod calls activities._ensure_job directly,
        # which already has its own delete-and-recreate handling).
        batch, _ = _k8s_batch_and_core()
        return self._ensure_job_with(batch, spec)

    def delete_job(self, spec: WorkerSpec) -> None:
        batch, _ = _k8s_batch_and_core()
        self._delete_job_with(batch, spec)

    def _ensure_job_with(self, batch, spec: WorkerSpec) -> str:
        from kubernetes.client.exceptions import ApiException

        manifest = build_job_manifest(spec)
        name = manifest["metadata"]["name"]
        try:
            batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
        except ApiException as e:
            if e.status != 409:  # 409 = already exists; re-attach
                raise
        return name

    def _delete_job_with(self, batch, spec: WorkerSpec) -> None:
        from kubernetes.client.exceptions import ApiException

        try:
            batch.delete_namespaced_job(
                name=job_name_for(spec), namespace=spec.namespace,
                propagation_policy="Background",
            )
        except ApiException as e:
            if e.status != 404:
                raise


class KarmadaJobDispatcher:
    """Applies the worker Job and its PropagationPolicy to the Karmada apiserver.

    Uses the same Job manifest build_job_manifest builds for the local
    cluster -- topology is a dispatch-time concern, not a manifest-shape one.
    """

    async def ensure_job(self, spec: WorkerSpec) -> str:
        batch, custom, core = _karmada_clients()
        return await self._ensure_job_with(batch, custom, spec, core=core)

    def delete_job(self, spec: WorkerSpec) -> None:
        batch, custom, _ = _karmada_clients()
        self._delete_job_with(batch, custom, spec)

    async def _ensure_job_with(self, batch, custom, spec: WorkerSpec, core=None) -> str:
        """Create the worker Job (and its PropagationPolicy) if absent,
        repairing a terminally-failed Job left over from a previous attempt
        instead of blindly re-attaching to it.

        Job names are deterministic
        (job_name_for), so a Temporal *activity* retry of a failed worker
        hits create_namespaced_job for a name that already exists on the
        Karmada control plane (409) -- exactly what a previous, real failure
        left behind. Re-attaching unconditionally (the pre-fix behaviour)
        meant the retry polled MinIO for an artifact a dead Job can never
        produce, burning the full POD_WATCH_TIMEOUT_S on every one of
        Temporal's retry attempts for nothing. This mirrors
        activities._ensure_job, the single-topology fix for the identical
        bug: on 409, classify the existing Job and only delete+recreate
        it if it's terminally failed; otherwise leave it alone.

        The one thing single-topology doesn't need and this does: Job status
        for a *propagated* Job lives on the Karmada aggregated API, not a
        local read, so this reuses activities._karmada_job_status (the same
        helper _karmada_failure_reason/_karmada_terminal_failure already
        share) rather than a second status-reading path. Its contract is
        exactly what makes this safe: an absent or lagging aggregated status
        -- Karmada hasn't propagated/observed the Job on the member cluster
        yet, completely normal for the first few seconds after dispatch, and
        also what an unreachable Karmada apiserver or unset
        FED_KARMADA_CONFIG looks like -- returns outcome=None, treated
        exactly like "still running", never as failure. Only a definite
        Failed condition triggers delete-and-recreate. Getting this backwards
        -- deleting a healthy, just-propagated Job because its aggregated
        status hasn't shown up yet -- would be a worse bug than the one this
        fixes: it would kill a live worker instead of merely wasting time on
        a dead one.

        Deleting a propagated Job means deleting it on the Karmada control
        plane (batch here is the Karmada BatchV1Api, from _karmada_clients),
        the same client this method already creates/reads through -- exactly
        like _delete_job_with's teardown path, `propagation_policy=
        "Background"` cascades the delete down to the member cluster's copy.
        _await_job_deleted (shared with the single-topology path, imported
        rather than reimplemented) blocks until the delete is confirmed gone
        before recreating, so the recreate can't race a half-deleted object.

        The PropagationPolicy is deliberately left untouched by the delete/
        recreate branch: build_propagation_policy selects the Job by its
        deterministic name, not by UID or generation, so the
        already-existing policy (still 409s, below, unchanged either way)
        keeps selecting -- and Karmada keeps re-propagating -- whichever Job
        currently has that name, including the one just recreated here.
        """
        from kubernetes.client.exceptions import ApiException

        # The Karmada control plane is a distinct apiserver with its own
        # namespaces: the host cluster having FED_NAMESPACE says nothing about
        # whether Karmada does. Without this, create_namespaced_job below dies
        # with 404 `namespaces "<ns>" not found` and no Job or
        # PropagationPolicy is ever created -- exactly how every worker
        # dispatch failed on the first live multi-cluster run. Karmada
        # propagates the namespace down to the members itself, so only the
        # control-plane copy needs creating here.
        if core is not None:
            _ensure_namespace(core, spec.namespace)

        manifest = build_job_manifest(spec)
        name = manifest["metadata"]["name"]
        try:
            batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
            # Mirrors activities._ensure_job's "created Job {name}" line for the
            # single-topology path. Without this, the Temporal worker's own logs
            # -- the only durable record once cleanup_worker_job deletes the Job
            # and its PropagationPolicy on success -- carry no evidence that a
            # multi-cluster dispatch ever happened at all.
            log.info(f"created Job {name} on Karmada, targeting member cluster {spec.member_cluster!r}")
        except ApiException as e:
            if e.status != 409:
                raise

            outcome, message = await _karmada_job_status(spec, name)
            if outcome == "failed":
                log.info(
                    f"Job {name} exists on Karmada but is terminally failed "
                    f"({message}); deleting and recreating"
                )
                batch.delete_namespaced_job(
                    name=name, namespace=spec.namespace, propagation_policy="Background"
                )
                await _await_job_deleted(
                    batch, spec.namespace, name,
                    timeout_s=JOB_DELETE_TIMEOUT_S, poll_s=JOB_DELETE_POLL_S,
                )
                batch.create_namespaced_job(namespace=spec.namespace, body=manifest)
            else:
                log.info(f"Job {name} already exists on Karmada ({message}); re-attaching")

        policy = build_propagation_policy(spec)
        group, version = policy["apiVersion"].split("/")
        try:
            custom.create_namespaced_custom_object(
                group=group, version=version, namespace=spec.namespace,
                plural="propagationpolicies", body=policy,
            )
        except ApiException as e:
            if e.status != 409:
                raise
        return name

    def _delete_job_with(self, batch, custom, spec: WorkerSpec) -> None:
        from kubernetes.client.exceptions import ApiException

        try:
            batch.delete_namespaced_job(
                name=job_name_for(spec), namespace=spec.namespace,
                propagation_policy="Background",
            )
        except ApiException as e:
            if e.status != 404:
                raise

        group, version = "policy.karmada.io", "v1alpha1"
        try:
            custom.delete_namespaced_custom_object(
                group=group, version=version, namespace=spec.namespace,
                plural="propagationpolicies", name=propagation_policy_name_for(spec),
            )
        except ApiException as e:
            if e.status != 404:
                raise


def _k8s_batch_and_core():
    """Lazily build a BatchV1Api/CoreV1Api pointed at the local cluster.

    Deliberately duplicated (not imported) from activities.py: importing
    activities._k8s_batch_and_core here would be equally correct, but this
    module's dependencies on activities.py are all either pure functions
    (build_job_manifest, job_name_for) or Karmada-aggregated-API helpers
    (_karmada_job_status, _await_job_deleted, the JOB_DELETE_* constants) --
    never the *local*-cluster client, which keeps the "who talks to which
    cluster" boundary in one place per topology.
    """
    from kubernetes import client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()


def _karmada_clients():
    """Lazily build clients pointed at the Karmada apiserver.

    The kubeconfig path comes from FED_KARMADA_CONFIG, mounted into the
    Temporal worker pod (fed-infra Task 5) -- never the in-cluster/local
    config, since the Karmada apiserver is a distinct control plane the
    worker pod reaches over its own kubeconfig secret.
    """
    import os

    from kubernetes import client
    from kubernetes import config as k8s_config

    kubeconfig = os.environ.get("FED_KARMADA_CONFIG")
    if not kubeconfig:
        raise RuntimeError(
            "FED_KARMADA_CONFIG must be set to the Karmada apiserver kubeconfig "
            "path mounted into the Temporal worker pod for topology='multi'"
        )
    api_client = k8s_config.new_client_from_config(kubeconfig)
    return (
        client.BatchV1Api(api_client),
        client.CustomObjectsApi(api_client),
        client.CoreV1Api(api_client),
    )


def _ensure_namespace(core, namespace: str) -> None:
    """Create `namespace` on whichever apiserver `core` points at, if absent.

    Tolerates both races and re-runs: a 409 from create means someone else
    (or a previous attempt) got there first, which is success, not failure.
    """
    from kubernetes.client.exceptions import ApiException

    try:
        core.read_namespace(name=namespace)
        return
    except ApiException as e:
        if e.status != 404:
            raise
    try:
        core.create_namespace(body={"metadata": {"name": namespace}})
    except ApiException as e:
        if e.status != 409:
            raise
