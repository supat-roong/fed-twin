# P3 gate step 5 — fed-twin multi-cluster on fed-infra

Run: KFP `federated-twin-multi-cluster-pipeline-c9knh`, **Succeeded** in 95s
Bring-up: `make multi-cluster-setup`, **REAL EXIT: 0**

## What this establishes

fed-twin's multi-cluster path now runs entirely on `vendor/fed-infra`, with
its bootstrap reduced from 313 hand-rolled lines to a thin wrapper. This is
the first time fed-infra's Karmada component and `multi` profile served a
**second, independent consumer** -- different host/member cluster names
(`multi-cluster-*` vs `active-fed-*`), a different namespace (`kubeflow` vs
`active-fed`), and a different component set (no temporal/minio) -- which is
the entire premise of extracting the library in P0.

## Topology

```
$ kubectl --kubeconfig ~/.karmada/... get clusters
NAME                    VERSION   MODE   READY
multi-cluster-host      v1.35.0   Push   True
multi-cluster-member1   v1.35.0   Push   True
multi-cluster-member2   v1.35.0   Push   True
```

## Workers genuinely on member clusters

```
member1   fl-job-1787077008-worker-1   Running
member2   fl-job-1787077008-worker-2   Running

propagationpolicy/fl-job-1787077008-server-propagation
propagationpolicy/fl-job-1787077008-worker-{0,1,2}-propagation
```

The pipeline produced metrics and plots (`comparison_result.png`,
`worker_diversity.png`, `generalization_gap_fed_twin_multi_cluster.png`).

## Defect found on the way

The image build failed before any cluster work:

```
ERROR: No matching distribution found for flit_core<4,>=3.11
```

`--index-url https://download.pytorch.org/whl/cpu` *replaces* PyPI, so a build
dependency served only from PyPI became unresolvable. The image had built
fine 6 days earlier; the dependency entered the resolution path since. Fixed
by using `--extra-index-url`, which consults the CPU wheel index in addition
to PyPI rather than instead of it. Unrelated to the fed-infra conversion --
the setup script never reached cluster work.

## Not re-verified in this pass

fed-twin's **single**-cluster path was not re-run after the fed-infra bump.
The multi bring-up exercises the same library code for kind, KFP, training,
mlflow and nodeports, so the risk is low, but "low risk" is not "verified" --
`make single-cluster-setup` plus `./run_pipeline.sh fed_twin_single_cluster`
remains outstanding.
