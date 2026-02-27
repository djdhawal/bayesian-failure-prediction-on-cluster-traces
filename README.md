# Bayesian Job Failure Prediction in Distributed Systems

---

## Overview

This project models the lifecycle of computational tasks running on Google's Borg cluster manager as a **hierarchical Bayesian Hidden Markov Model (HMM)**. Rather than predicting failure after it happens, the model infers latent resource-usage regimes from raw CPU and memory measurements — detecting when a task is drifting toward a dangerous state *before* it fails.

The key insight is that Borg's eviction mechanism is deterministic: tasks whose actual resource demand exceeds their requested limit get throttled or killed. A rising CPU utilization trajectory, combined with increasing memory pressure and degraded CPU efficiency (CPI), is a measurable precursor to failure that unfolds over minutes. The HMM is designed to detect exactly this pattern from the time series of 5-minute usage windows.

---

## The Problem

Google Borg manages hundreds of thousands of tasks across large machine clusters. Tasks fail for two reasons:

- **Infrastructure failures** — Borg evicts the task (preemption, OOM, machine failure) or the task crashes
- **User/parent kills** — intentional termination by the user or parent job cleanup (not real failures)

Standard classification approaches treat each measurement window independently, ignoring the temporal structure. A task may look fine on average but be trending toward its resource limit — a pattern that only becomes visible when you look at the sequence over time.

---

## Approach

### Latent States

The model learns three latent states from resource data alone, without being told what they mean. We expect them to correspond to:

| State | Interpretation | Signal |
|-------|---------------|--------|
| **Healthy** | Stable, well below resource limits | Low avg_cpu, low burstiness |
| **Stressed** | Rising utilization, approaching limits | Elevated avg_cpu, increasing max_cpu |
| **Critical** | Near or exceeding limits, resource contention | High cpu_util_ratio, high CPI, negative headroom |

The Borg lifecycle events (EVICT, FAIL, KILL, FINISH) are deliberately **excluded from model inputs**. They serve only as ground-truth labels for evaluating whether the inferred latent states are predictive of subsequent failure.

### Hierarchical Structure

A naive HMM would learn a single transition matrix averaged across all tasks — conflating a MapReduce batch job that briefly spikes and finishes with a long-running server that slowly leaks memory. We address this with a three-level hierarchy:

```
Global prior
    └── Tier-level prior (Free / BestEffortBatch / MidTier / Production / Monitoring)
            └── Per-program transition matrix (indexed by collection_logical_name)
```

This allows frequently-run programs to learn their own failure dynamics while rare programs borrow statistical strength from structurally similar workloads in the same tier.

### Inference

We use **Stochastic Variational Inference (SVI)** via NumPyro and JAX, with an AutoNormal/AutoDiagonalNormal guide and Adam optimizer. MCMC is not used — SVI scales to the 500K+ task sequences in the full dataset where MCMC would be computationally infeasible.

**Identifiability** is enforced by imposing an ordering constraint on emission means: the model's states are ordered by increasing avg_cpu, so state 0 is always the low-utilization regime and state 2 is always the high-utilization regime. Without this, the labels of the latent states are arbitrary and change between runs.

---

## Data

**Source:** [Google Borg Cluster Traces v3 (ClusterData2019)](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)

One month (May 2019) of workload and resource-usage logs from eight production Borg cells, each containing ~12,000 machines. Total size: ~2.4 TiB compressed. We use three of the five tables:

| Table | Content | Role |
|-------|---------|------|
| `instance_usage` | Per-task resource metrics at 5-min granularity | HMM observations (model input) |
| `instance_events` | Lifecycle events (EVICT, FAIL, KILL, FINISH...) | Evaluation labels only |
| `collection_events` | Job-level metadata (priority, logical name, parent) | Hierarchical grouping + label filtering |



## References

- Reiss et al. (2011). *Google Cluster Usage Traces*
- Wilkes (2020). *Google Borg Cluster Traces v3*. [ClusterData2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)
- Verma et al. (2015). *Large-scale cluster management at Google with Borg*. EuroSys.

