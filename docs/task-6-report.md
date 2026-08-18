# Phase P3, Task 6 — multi-cluster bootstrap onto vendor/fed-infra

## Summary

`setup/install_multi_cluster_local.sh` (313 lines of hand-rolled bash) is
replaced by a ~110-line consumer script that delegates cluster creation,
Karmada init/join/patch, KFP, MLflow, image loading, waits, and NodePorts to
`vendor/fed-infra`. `setup/teardown_multi_cluster_local.sh` now delegates to
`fed-infra-down`. A new `infra.env.multi` is the multi-profile consumer
contract, mirroring `infra.env`. The submodule pin was bumped (see
"Submodule bump" below) because fed-twin's checked-out `vendor/fed-infra`
predated the `karmada` component and `multi` profile entirely — it had no
`lib/karmada.sh` and no `multi-host.yaml.tpl`/`member.yaml.tpl`.

**Status: DONE_WITH_CONCERNS.** Everything asked for is implemented,
shellcheck-clean, dry-run-verified, and the 22 pytest tests still pass. But
tracing the dry-run against `vendor/fed-infra`'s actual Karmada init flags
surfaced what looks like a real, load-bearing gap in the library itself
(see "Concern: Karmada apiserver NodePort not host-reachable" below) that
could block Task 7's live run. I did not patch it — reporting only, per the
constraint.

## Submodule bump

`vendor/fed-infra` was pinned at `329e578` (the commit right before Karmada/
multi work landed upstream — no `lib/karmada.sh`, no multi profile in
`components.sh`/`config.sh`, no `kind/multi-host.yaml.tpl` or
`kind/member.yaml.tpl`). The task's premise ("fed-infra gained a karmada
component and a multi profile") only holds at a newer commit, so I bumped
the pin to `ed0b955` (`origin/main`, the same commit `active-fed` already
consumes) via `git fetch origin && git checkout ed0b955... ` inside the
submodule, then `git add vendor/fed-infra` — the standard "bump the pinned
SHA" flow documented in `vendor/fed-infra/README.md`'s own "Consumers:
pinning and bumping the SHA" section, and the same pattern as this repo's
own prior commits (`096292c chore: bump vendor/fed-infra to pick up the
infra.env docs and set -e fixes`, etc.). No file inside `vendor/fed-infra`
was edited — only the submodule's pinned commit moved forward to one
already published upstream. `ed0b955` also brings in the (unused by
fed-twin) `temporal` component, since Temporal landed upstream before
Karmada and there is no earlier commit with Karmada but not Temporal;
Temporal is opt-in via `FED_COMPONENTS` and fed-twin's `infra.env.multi`
does not list it, so this is inert.

Ran `vendor/fed-infra`'s own `make check` (131 bats tests) at this pin: all
pass, 0 failures.

## Step-by-step classification of the old 313-line script

| Lines | Step | Verdict |
|---|---|---|
| 1–9 | Shebang, header comment, `set -euo pipefail` | Kept (script boilerplate, rewritten) |
| 7–9 | `HOST_CLUSTER`, `MEMBER_PREFIX`, `IMAGE_NAME` vars | Moved to `infra.env.multi` (`FED_CLUSTER_NAME`, `FED_MEMBER_PREFIX`, `FED_IMAGES`) |
| 11–15 | Comment re: MLflow now via fed-infra | Dropped (stale historical comment, superseded) |
| 16–27 | Source `common.sh`/`config.sh`/`render.sh`/`mlflow.sh`, `fed_config_load infra.env` | Replaced: new script sources only `common.sh`/`config.sh`/`nodeport.sh` (mirrors `install_single_cluster_local.sh`) and loads `infra.env.multi` |
| 29–31 | Banner echo | Dropped (cosmetic) |
| 33–35 | `source ~/.zshrc`, `export PATH=...` | Dropped — personal-workstation PATH hack, not present in `install_single_cluster_local.sh` either |
| 37–43 | `uv sync` | Kept (fed-twin-specific dev step, matches single-cluster script) |
| 45–60 | Read `num_workers` from `config/config.json`, compute `NUM_MEMBER_CLUSTERS` | **Dropped per task instruction** — moved to `infra.env.multi` as explicit `FED_MEMBER_COUNT=2` (matches current `config.json`'s `num_workers: 2`). No dynamic derivation reintroduced. Two sources of truth now: if `config.json`'s `num_workers` changes, `FED_MEMBER_COUNT` must be updated by hand (documented in `infra.env.multi`'s header) — `src/automate_run.py` separately reads `config.json`'s `num_workers` directly for its own member-kubeconfig list, so all three must stay in sync manually. |
| 62–69 | Create host kind cluster from `kind-multi-cluster-host.yaml` | Covered by `fed_kind_ensure_cluster` (`fed_up_multi`) — deleted; `setup/kind-multi-cluster-host.yaml` removed as dead code (see concern below re: its port mappings) |
| 71–82 | Create member kind clusters loop + inotify sysctl on each member | Cluster creation covered by `fed_kind_ensure_cluster` — deleted. Inotify sysctl **kept**, moved to run after `fed-infra-up` returns (generic, not fed-twin-specific; not offered by `vendor/fed-infra/lib/kind.sh` for either profile — promotion candidate) |
| 84–85 | Inotify sysctl on host | Same treatment, folded into the same post-`fed-infra-up` loop |
| 87–91 | Install `karmadactl` if missing | **Kept** — generic to any Karmada consumer, not fed-twin-specific, and not handled by fed-infra (`fed_karmada_init`/`fed_karmada_join` `fed_require_cmd karmadactl` and `fed_die` if absent) — promotion candidate |
| 93–104 | Pre-fetch Karmada core images (`docker pull` + `kind load`) to avoid networking timeouts | **Dropped** — see "Karmada image pre-fetch" below for why this can't be faithfully reproduced without duplicating cluster creation or patching the library |
| 106–122 | `karmadactl init` on host | Covered by `fed_karmada_init` — deleted |
| 124–173 | `join_and_patch` function (join + Docker-IP patch of Cluster + Secret) + host/member join loop | Covered by `fed_karmada_join`/`fed_karmada_wait_cluster` — this is literally the function the task says was ported into fed-infra — deleted |
| 175–177 | Build app image + `fed_mlflow_build_image` | App image build **kept** (fed-twin-specific); `fed_mlflow_build_image` call **deleted** — now called automatically by `fed_up_install_components` when the `mlflow` component is enabled |
| 180–188 | `kind load` app image + mlflow image into host/members | Deleted — covered automatically: `fed_up_multi` loads `FED_IMAGES` into host + every member; `fed_up_install_components` loads `FED_MLFLOW_IMAGE` into host only (same asymmetry as the old script, which never loaded mlflow into members either) |
| 190 | `kubectl config use-context` host | Deleted — `fed_up_multi` already leaves context on the host once `fed-infra-up` returns |
| 197 | `kubectl create namespace kubeflow` | Deleted — created by the KFP kustomize manifests `fed_kfp_install` applies (same as `install_single_cluster_local.sh`, which never has this line either) |
| 198 | `pipeline-runner-extend` clusterrolebinding | **Kept** — fed-twin-specific, matches `install_single_cluster_local.sh` verbatim |
| 200–210 | KFP CRDs + core via kustomize | Covered by `fed_kfp_install` (now git-clone based, more robust than the old kustomize-over-git-URL approach, which had a hard ~27s kustomize git-fetch timeout) — deleted |
| 212–224 | "Container Registry Migration Fix" (ghcr.io images for frontend/api-server/visualization-server/launcher + workflow-controller argoexec patch) | **Confirmed covered** by `fed_kfp_patch_arm` — deleted |
| 226–229 | "Minio Fix" (image, console port, args) | **Confirmed covered** by `fed_kfp_patch_minio` — deleted |
| 231–232 | "Workflow Controller Fix" (argoexec image) | Covered — folded into `fed_kfp_patch_arm`'s last patch — deleted |
| 234–235 | `fed_mlflow_install` | Deleted — called automatically by `fed_up_install_components` |
| 236–239 | Manual `minio-setup-mlflow` job (`mc mb`) | Deleted — covered by `fed_minio_ensure_bucket`, called automatically for the `mlflow` component |
| 242 | NodePort `ml-pipeline-ui` | Deleted — covered by `fed_expose_nodeport` inside `fed_up_install_components` (kfp component) |
| 243 | NodePort `minio-service` (KFP's bundled MinIO) | **Kept** — fed-twin-specific, matches `install_single_cluster_local.sh`'s "Exposing KFP MinIO" step verbatim (this is KFP's *bundled* MinIO in `kubeflow`, not the standalone `minio` component fed-twin doesn't use) |
| 244 | NodePort `mlflow-service` | Deleted — the `mlflow-server.yaml.tpl` manifest fed-infra applies already creates this Service as `type: NodePort` with `nodePort: 30500` (confirmed via `vendor/fed-infra/tests/golden/*/mlflow-server.yaml`) |
| 246–249 | Wait for `workflow-controller`/`ml-pipeline`/`ml-pipeline-ui` rollout | Covered by `fed_kfp_wait` — deleted |
| 251–252 | "Karmada Setup complete" echo | Dropped (cosmetic; fed-infra prints its own summary) |
| 254–297 | Karmada Dashboard: apply manifest, kubeconfig Secrets, NodePort 32000 Service, wait for rollout | **Kept per task instruction** — generic Karmada tooling, not yet in fed-infra, and explicitly not to be added there (fed-infra's agnosticism guard). Adapted to use `$FED_KARMADA_CONFIG`/`$FED_CLUSTER_NAME` instead of the old hardcoded local vars. Promotion candidate. |
| 299–313 | Dashboard admin SA + clusterrolebinding + 24h token | Same as above — kept, promotion candidate |

Every line of the old script is accounted for above as covered/kept/dropped.

## What was implemented

- `infra.env.multi` — new file, mirrors `infra.env` with `FED_CLUSTER_NAME=multi-cluster-host`, `FED_PROFILE=multi`, `FED_COMPONENTS=kfp,training,mlflow,karmada`, `FED_MEMBER_COUNT=2`, `FED_MEMBER_PREFIX=multi-cluster-member`, `FED_KARMADA_CONFIG=${HOME}/.karmada/karmada-apiserver.config`; every other value copied verbatim from `infra.env`, with a header comment explaining the deltas and why `FED_CLUSTER_NAME`/`FED_MEMBER_PREFIX` can't be renamed (`src/automate_run.py` hardcodes `multi-cluster-host`/`multi-cluster-member{i}` container names for its Karmada log-streaming feature — Python source, not touched).
- `setup/install_multi_cluster_local.sh` — rewritten, 313 → 139 lines. Sources `common.sh`/`config.sh`/`nodeport.sh`, loads `infra.env.multi`, builds the app image, installs `karmadactl` if missing, calls `fed-infra-up --env infra.env.multi`, then handles fed-twin-specific steps (cluster-admin binding, KFP MinIO NodePort), the promotion-candidate inotify bump, and the promotion-candidate Karmada Dashboard + token generation.
- `setup/teardown_multi_cluster_local.sh` — rewritten, 23 → 7 lines. Delegates to `fed-infra-down --env infra.env.multi`.
- `setup/kind-multi-cluster-host.yaml` — deleted (dead code; `fed_kind_ensure_cluster` uses fed-infra's own `kind/multi-host.yaml.tpl` now, ignoring this file entirely).
- `Makefile` — no change needed. `multi-cluster-setup`/`multi-cluster-teardown` already invoke `bash setup/install_multi_cluster_local.sh` / `bash setup/teardown_multi_cluster_local.sh`, and those paths/invocations are unchanged.
- `README.md` — updated the multi-cluster bullet (mentions `infra.env.multi`, automatic `karmadactl` install, and the Dashboard's port-forward access note) and the Prerequisites/Local Setup sections (mentions `karmadactl`/`python3`/`git` for multi mode, and both contract files).
- `vendor/fed-infra` submodule bumped `329e578` → `ed0b955` (see "Submodule bump" above).

## Verification

**`bash -n`**: both scripts pass with no output (success).

**shellcheck** (`shellcheck setup/install_multi_cluster_local.sh setup/teardown_multi_cluster_local.sh`): clean, no output. Single-cluster scripts re-checked too, still clean.

**Dry run** (`bash vendor/fed-infra/bin/fed-infra-up --env infra.env.multi --dry-run --render-dir /tmp/ft-multi`):

```
[fed-infra] dry-run: would ensure kind cluster 'multi-cluster-host' exists
[fed-infra] dry-run: would ensure kind cluster 'multi-cluster-member1' exists
[fed-infra] dry-run: would ensure kind cluster 'multi-cluster-member2' exists
[fed-infra] dry-run: would initialize the Karmada control plane on multi-cluster-host
[fed-infra] dry-run: would join multi-cluster-host to the Karmada control plane
[fed-infra] dry-run: would wait for multi-cluster-host to report Ready in Karmada
[fed-infra] dry-run: would join multi-cluster-member1 to the Karmada control plane
[fed-infra] dry-run: would wait for multi-cluster-member1 to report Ready in Karmada
[fed-infra] dry-run: would join multi-cluster-member2 to the Karmada control plane
[fed-infra] dry-run: would wait for multi-cluster-member2 to report Ready in Karmada
[fed-infra] dry-run: would load image fed-twin-app:v1 into cluster multi-cluster-host
[fed-infra] dry-run: would load image fed-twin-app:v1 into cluster multi-cluster-member1
[fed-infra] dry-run: would load image fed-twin-app:v1 into cluster multi-cluster-member2
[fed-infra] dry-run: would install KFP 2.4.0
[fed-infra] dry-run: would patch KFP images for ARM/kind stability
[fed-infra] dry-run: would patch KFP MinIO image and console port
[fed-infra] dry-run: would install Training Operator v1.7.0
[fed-infra] dry-run: would build mlflow image fed-mlflow:2.12.2
[fed-infra] dry-run: would load image fed-mlflow:2.12.2 into cluster multi-cluster-host
[fed-infra] dry-run: rendered namespace -> /tmp/ft-multi/namespace.yaml
[fed-infra] dry-run: rendered mlflow-server -> /tmp/ft-multi/mlflow-server.yaml
[fed-infra] dry-run: would load image fed-twin-app:v1 into cluster multi-cluster-host
[fed-infra] dry-run: would wait for KFP core deployments
[fed-infra] dry-run: would ensure bucket 'mlpipeline' at minio-service.kubeflow.svc.cluster.local:9000
[fed-infra] dry-run: would expose ml-pipeline-ui in kubeflow as NodePort
[fed-infra] dry-run: would ensure bucket 'mlflow-artifacts' at minio-service.kubeflow.svc.cluster.local:9000

[fed-infra] Setup complete. Services:
  Kubeflow Pipelines : http://localhost:8080
  MLflow             : http://localhost:5050
```

Host cluster, both members, and all Karmada join/wait steps for all three
clusters are present, as required. `/tmp/ft-multi/` contains
`namespace.yaml` and `mlflow-server.yaml`, both rendered correctly
(`namespace: kubeflow`, `nodePort: 30500`, etc.).

**`make test`**: 22 passed (unchanged — no `src/`/`tests/` files touched).

**`vendor/fed-infra`'s own `make check`**: 131 bats tests, 0 failures, at the bumped pin.

## Promotion candidates for fed-infra

1. **Karmada Dashboard** (deploy + NodePort 32000 + 24h token) — explicitly told not to add this per the task; noted here per instruction.
2. **`karmadactl` auto-install** — generic to any Karmada consumer; `fed_karmada_init`/`fed_karmada_join` currently just `fed_die` if it's missing.
3. **Inotify limit bump** on kind control-plane containers — generic to any multi-node/multi-cluster kind consumer (more clusters = more fsnotify pressure system-wide), not offered by `lib/kind.sh` for either profile today.
4. **`FED_KARMADA_VERSION` is currently cosmetic** — it only appears in a log line (`karmada.sh:41`); `karmadactl init` has no version flag, so the actual Karmada version installed is whatever `karmadactl`'s own `install-cli.sh` (pulled from `master`, unpinned) resolves to. Not a regression (the old script had the same property), but worth fed-infra either wiring the var through or dropping it from the documented contract.

## Concerns

### Concern: Karmada apiserver NodePort not host-reachable (the important one)

`fed_karmada_init` (`vendor/fed-infra/lib/karmada.sh`) calls `karmadactl init` with `--cert-external-ip=127.0.0.1 --karmada-apiserver-advertise-address=127.0.0.1` — the same flags the old hand-rolled script used, which is why fed-twin's own pipeline code (`src/pipelines/fed_twin_multi_cluster_pipeline.py:290`, `single_twin_multi_cluster_pipeline.py:290`) does a literal string replace of `"https://127.0.0.1:32443"` when adapting the generated `~/.karmada/karmada-apiserver.config` for in-cluster use. That `32443` is Karmada's conventional fixed NodePort for its apiserver Service on a kind host cluster — and it's exactly the port the **old** `setup/kind-multi-cluster-host.yaml` mapped (`containerPort: 32443` / `hostPort: 32443`, labeled "NodePort for FL Server").

`fed_karmada_join` and `fed_karmada_wait_cluster` run `kubectl --kubeconfig="$FED_KARMADA_CONFIG" ...` directly on the **host machine** (not inside a pod), so they need `127.0.0.1:32443` to actually route to the Karmada apiserver Service inside the host kind container. But `vendor/fed-infra/kind/multi-host.yaml.tpl` only maps the four (fed-infra-tracked) NodePorts — KFP, MLflow, MinIO API, MinIO console (plus Temporal) — it has **no mapping for the Karmada apiserver's own NodePort at all**. Since kind cluster port mappings are fixed at container-creation time and cluster creation is entirely internal to `fed_kind_ensure_cluster`, there is no way for a consumer script to add this mapping without either duplicating cluster creation (defeating the point of delegating it to fed-infra) or patching the library (forbidden this task).

If my reading of `karmadactl`'s behavior is right, this would affect not just this consumer script's own Dashboard/admin-token steps (which I kept, and which do need it) but potentially `fed_karmada_join`/`fed_karmada_wait_cluster` themselves during a live `fed-infra-up` run — i.e., a possible blocker for Task 7, not just a cosmetic dashboard-reachability issue. I have **not** verified this against a live cluster (out of scope — "do not create a real cluster" — and dry-run can't exercise real networking), so treat this as a strong hypothesis backed by the evidence above, not a confirmed failure. Flagging prominently rather than guessing further or attempting a workaround that would mean either duplicating fed-infra's cluster lifecycle or patching the submodule.

Related, lower-stakes version of the same gap: the old host kind config also mapped `hostPort: 32000` for the Dashboard, which `multi-host.yaml.tpl` likewise doesn't provide — the install script now says as much and points at `kubectl port-forward -n karmada-system svc/karmada-dashboard 32000:80` as the workaround.

### Other notes

- `FED_HOSTPORT_KFP`/`FED_HOSTPORT_MLFLOW`/`FED_HOSTPORT_MINIO_*` are identical between `infra.env` and `infra.env.multi` (per task instruction to mirror `infra.env` exactly) — so the single-cluster and multi-cluster-host kind clusters bind the same host ports and can't run simultaneously. This isn't a new regression: the old `setup/kind-multi-cluster-host.yaml` already used the same host ports as the single-cluster path.
- `FED_MEMBER_COUNT=2` (in `infra.env.multi`) duplicates `config/config.json`'s `num_workers: 2` and `src/automate_run.py`'s own direct read of that same file. Three places must now be kept in sync by hand if worker count ever changes; this is the explicit tradeoff the task asked for (no dynamic derivation in the script), just flagging it's now a manual-sync concern across two repos worth of files.
