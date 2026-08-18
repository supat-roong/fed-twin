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

---

# fed-twin single-cluster re-verification (after the fed-infra bump)

Bring-up: `make single-cluster-setup`, **REAL EXIT: 0**
Run: KFP `federated-twin-single-cluster-pipeline-wl6fw`, **Succeeded** in 70s

```
round,twin_id,mode,reward,loss
1,train-twin-1,TRAIN,11.00,-0.0307
1,train-twin-2,TRAIN,30.50,0.0003
1,eval-twin-global,EVAL,20.30,0.0
... 10 rows total (fl_rounds x num_workers x 2 modes, plus the eval twin)
```

This closes the last outstanding P3 gate criterion. Both consumers, both
topologies, all verified live.

## Defect found: the metrics step reports success having captured nothing

The **first** attempt succeeded structurally but wrote a header-only CSV. The
FL training itself was fine -- workers logged real rewards
(`EVAL Reward: 24.00`, `24.65`) and pushed them to MLflow. What failed was the
metrics collection component, which scrapes `[METRIC]` lines by *following*
pod logs (`kubectl logs -f -l training.kubeflow.org/job-name=...`):

```
Log streaming finished. Total: 26 lines processed, 0 metrics captured
[WARNING] Warning: Log streaming ended before job completion was confirmed
[WARNING] Warning: Only captured 0/10 expected metrics
```

It emitted warnings and **exited successfully**, so KFP reported `Succeeded`
and the run produced no analysable data. A second run captured all 10
metrics, which identifies the stream ending early as a race rather than a
systematic fault -- but the race is not the interesting part.

**The defect is that `0/10 expected metrics` is a warning.** A run that
collected none of its data is a failed run, and reporting it as `Succeeded`
means a flake is indistinguishable from a good run without opening the CSV.
This is the same class as a harness reporting `exit 0` for a `make` that
exited 1, which cost real time on this project today.

Suggested fix (not applied -- it changes when fed-twin runs fail, which is a
product decision rather than an infrastructure one): fail the component when
`metric_count` is 0, or below some floor of `expected_metrics`. Pre-existing
behaviour, untouched by P3 and unrelated to the fed-infra conversion.

A second, smaller fragility in the same area: `run_pipeline.sh`'s readiness
wait matched a `Terminating` `ml-pipeline` replica left over from a rollout
and timed out on a pod that could never become ready. Re-running cleared it.
