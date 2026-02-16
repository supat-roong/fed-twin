# 🤖 Federated Digital Twin with Kubeflow

[![Stack: Kubeflow](https://img.shields.io/badge/Stack-Kubeflow%20|%20PyTorch%20|%20Flower-blue)](https://kubeflow.org)

A personal project for learning **Distributed Digital Twin Training** using Federated Learning. This implementation acts as a simulation framework designed for local Kubernetes clusters (e.g., Kind, Minikube) to mock a distributed robotic fleet. While tested extensively on macOS (Colima/Kind), it is compatible with any local Kubernetes environment.

---

## 🏗 Architecture: Digital Twins Meet Federated Learning

### 🤖 What are Digital Twins?

A **Digital Twin** is not just a replica, it's a **living twin** of a physical system that mirrors its behavior, state, and characteristics in real-time. In practice, digital twins continuously sync with their physical counterparts through sensors and data feeds.

```
Physical Robot ⟷ Digital Twin (Real-time Sync)
     🤖      ⟷        💻
```

**For this project**: We use **simulated environments** (CartPole) as stand-ins for physical systems. While not connected to real hardware, they demonstrate the core FL+DT concepts by creating diverse physics variations.

**The Challenge**: Every physical system is unique (wear, manufacturing tolerances, environment)
- Robot A might have heavier arms
- Robot B operates in different environment
- Robot C has more friction in joints

**Traditional Approach**: Train one model on one perfect simulation ❌  
**Our Approach**: Create multiple digital twins, each with different physics ✅

### 🌐 Why Federated Learning?

Instead of collecting all data centrally, each twin learns **locally** and shares only its "intelligence":

```
┌─────────────────────────────────────────────────────────┐
│         TRADITIONAL (Centralized)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Twin 1 ──┐                                             │
│  Twin 2 ──┼── [Raw Data] ──→ Central Server             │
│  Twin 3 ──┘                         ↓                   │
│                            Train One Model              │
│                                                         │
│  ❌ Privacy concerns                                    │
│  ❌ Huge data transfer                                  │
│  ❌ Single point of failure                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         FEDERATED (Decentralized)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Twin 1: Train locally ──┐                              │
│  Twin 2: Train locally ──┼── [Weights Only] ──→ Server  │
│  Twin 3: Train locally ──┘                        ↓     │
│                                              Aggregate  │
│                                                   ↓     │
│                                            Global Model │
│                                                         │
│  ✅ Data stays local (privacy)                          │
│  ✅ Small weight updates only                           │
│  ✅ Learn from diversity                                │
└─────────────────────────────────────────────────────────┘
```

### 🔄 How It Works: The FL Training Cycle

Each round of federated learning follows this pattern:

```
Round N:
    
    1️⃣ BROADCAST
       Global Model → [Twin 1, Twin 2, Twin 3, ...]
    
    2️⃣ LOCAL TRAINING (Parallel)
       Twin 1: CartPole with high mass   → learns controls for heavy system
       Twin 2: CartPole with low gravity → learns for low-g environment  
       Twin 3: CartPole with high friction → compensates for resistance
       ...each collects experience, updates policy...
    
    3️⃣ AGGREGATION
       [Weights from all twins] → FedAvg → New Global Model
    
    4️⃣ EVALUATION
       Test global model on neutral twin → measure generalization
    
    5️⃣ REPEAT
       New Global Model becomes input for Round N+1
```

### 🎯 The Goal: Generalization Through Knowledge Sharing

**Core Objective**: Build a **single global model** that works well across **all physical variations** by sharing knowledge across the fleet.

**How Knowledge Sharing Works**:
- Twin 1 (heavy system) learns: "Apply stronger force to compensate for mass"
- Twin 2 (low gravity) learns: "Use gentler corrections to avoid overshooting"  
- Twin 3 (high friction) learns: "Anticipate resistance and adjust timing"

When these insights are **aggregated**, the global model learns:
- 💡 Robust strategies that work across different conditions
- 💡 Generalized policies that adapt to unseen physics
- 💡 Knowledge from the entire fleet, not just one twin

**The Result**: A model that performs better on **new, unseen variations** than any individual twin could achieve alone. This is the power of federated learning applied to digital twins—**collective intelligence through privacy-preserving knowledge sharing**!

---

## 📈 Visual vs. Functional Pipelines

This project implements two distinct pipeline strategies to explore different aspects of the ML lifecycle:

### 1. Functional Pipelines (The "Workhorse")
*   **Files**: `fl_pipeline.py`, `single_pipeline.py`
*   **Implementation**: Uses a single `PyTorchJob` Custom Resource from the Kubeflow Training Operator.
*   **Why use it**: This is the efficient way to run experiments. Instead of launching individual pods for every round, the entire fleet orchestration is delegated to the Training Operator. It handles distributed synchronization natively, making it much faster.
*   **UI Representation**: Shows as a single, clean "Training" node in the Kubeflow graph.

### 2. Visual Pipelines (The "Narrative")
*   **Files**: `fl_visual_pipeline.py`, `single_visual_pipeline.py`
*   **Implementation**: Creates individual KFP components for every training and evaluation step.
*   **Why use it**: Kubeflow's default representation can be opaque. These pipelines provide **better observability** by mapping each round and worker to a unique component, making it easy to track the flow of weights and parallel training in the Kubeflow UI.

#### **Federated Learning DAG (`fl_visual`)**
![Federated DAG](assets/fl-dag.png)
*This visualization shows multiple training pods running in parallel for each round, followed by a synchronization step where model weights are aggregated before proceeding to the next iteration.*

#### **Single Agent DAG (`single_visual`)**
![Single Agent DAG](assets/single-dag.png)
*In contrast, the single agent DAG shows a linear progression of training and evaluation rounds, where a single pod learns sequentially without the need for aggregation.*


## 📊 Performance & Analytics

The project includes an automated analysis suite that generates insights after every experiment.

### 1. Federated vs. Single Agent Comparison

**Concept**: Compares the learning efficiency and final performance of the global federated model against a single isolated agent. This metric validates whether collaborative learning across diverse environments yields a more robust policy than learning in single environment.

*   **Analysis Script**: `src/analysis/compare_results.py`
*   **Generated Plot**: `plots/comparison_result.png`

![Performance Comparison](plots/comparison_result.png)

**Key Findings:**
- **Federated Learning (Green)** achieves significantly higher rewards by leveraging knowledge from diverse physics
- **Single Agent (Red)** learns from only one environment, limiting its generalization capability
- FL demonstrates **better generalization** and more stable growth through collective learning
- Both models show initial improvement, but single agents stuck at lower performance due to overfitting, whereas **FL reaches higher performance** due to better generalization.

### 2. Worker Training Dynamics (Worker Diversity)

**Concept**: Measures the variance in training rewards across different twins. In a healthy FL system, we expect individual workers to have different learning curves as they adapt to their unique physical environments (e.g., heavy vs. light gravity), while the global model aggregates these diverse insights.

*   **Analysis Script**: `src/analysis/worker_diversity.py`
*   **Generated Plot**: `plots/worker_diversity.png`

### 3. Generalization Gap

**Concept**: Measures the difference between a model's performance on its training environment vs. a neutral evaluation environment. A smaller gap indicates that the model has truly learned robust policies rather than just memorizing a specific condition. Federated learning typically minimizes this gap by forcing the model to solve for multiple physics variations simultaneously.

*   **Analysis Script**: `src/analysis/generalization_gap.py`
*   **Generated Plot**: `plots/generalization_gap_{type}.png`

---

## 🚀 Getting Started
### 1. Prerequisites
- **Kubernetes Cluster**: A local cluster like [Kind](https://kind.sigs.k8s.io/) or [Minikube](https://minikube.sigs.k8s.io/).
- **Container Runtime**: Docker Desktop, Colima, or Podman.
- **Tools**: `kubectl`, `python 3.10+`, and `pip`.

### 2. Automatic Setup (Recommended)
The provided `run_project.sh` automates the entire lifecycle: environment creation, cluster detection, image building, and pipeline execution.

```bash
# General Command
./run_project.sh all
```

### 3. Manual Control
You can run specific pipeline strategies:
```bash
./run_project.sh single         # Run single worker baseline
./run_project.sh fl             # Run full federated fleet
./run_project.sh fl_visual      # Run FL with real-time DAG visualization
./run_project.sh single_visual  # Run single agent with DAG visualization
```

---

## 📂 Repository Structure

*   **/src/core**: The core code of the project, including `engine.py` (physics simulation), `client.py` (RL training), and `server.py` (FL aggregation).
*   **/src/pipelines**: Definitions for Kubeflow Pipelines (KFP).
*   **/src/analysis**: Python scripts for generating professional plots and metrics analysis.
*   **/metrics**: Consolidated CSV results from every cluster run.
*   **/plots**: Generated visualizations showing project performance.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

