# gps-map-matching-evolution-benchmarks
Comparative benchmark &amp; evolution of GPS map matching algorithms, evaluating classic HMMs against modern DiffMM diffusion models.

# Map Matching Benchmarking & Comparative Suite


> **Benchmarking the Evolution of Trajectory Map Matching:** From Hidden Markov Models (HMM) to One-Step Diffusion Models (`DiffMM`) on low-sampling-rate GPS data.

---

## 📌 Project Overview & Extended Introduction

In last-mile delivery, quick-commerce, and ride-hailing services, accurate spatial tracking is foundational to operational efficiency. Smartphone GPS hardware frequently records low-frequency or noisy location traces to conserve battery and cloud data transmission costs. Raw GPS points regularly drift off actual road networks, landing on adjacent buildings, parallel streets, or opposite lanes. In real-world delivery platforms (such as Zepto's *Route IQ* pipeline), uncleaned raw GPS traces introduce distance calculation errors of 30% to 40%. These errors propagate directly into rider payout calculations, Estimated Time of Arrival (ETA) predictions, delivery zone boundary definitions, and customer billing confidence.

**Map matching** is the fundamental pre-processing step that resolves this spatial ambiguity by aligning raw, sequential GPS points to the underlying digital road graph. Over the past two decades, map-matching techniques have evolved across five distinct stages:

```text
+------------------+     +------------------------+     +-----------------------+     +---------------------+     +-----------------------+
| Stage 1          |     | Stage 2                |     | Stage 3               |     | Stage 4             |     | Stage 5               |
| Geometric        | --> | Topological & Global   | --> | Probabilistic (HMM)   | --> | Fast & Scalable HMM | --> | Deep Learning &       |
| Point/Curve Snap |     | Connectivity & Fréchet |     | Hidden States & Viterbi|    | FMM & Precomputation|     | Diffusion (DiffMM)    |
+------------------+     +------------------------+     +-----------------------+     +---------------------+     +-----------------------+
```

1. **Geometric Matching (pre-2000):** Simple point-to-point and point-to-curve snapping using Euclidean distance. Extremely fast, but fails at complex urban intersections and parallel roads.
2. **Topological & Global Curve Matching (2000–2008):** Incorporated road network connectivity, driving heading, and global curve alignment (e.g., Fréchet distance). Prevented unrealistic disconnected hops, but relied heavily on fragile, hand-tuned rule thresholds.
3. **Probabilistic Matching with HMMs (2009):** Modeled true road segments as hidden states and noisy GPS points as observations using Hidden Markov Models (Newson & Krumm, 2009) solved via the Viterbi algorithm. Simultaneously balanced spatial emission distances and shortest-path transition feasibility.
4. **Fast and Scalable HMM Systems (2011–2018):** Industrialized probabilistic matching through graph precomputation (Fast Map Matching / FMM), routing engine integrations (OSRM), and handling missing observations with non-emitting states (Leuven MapMatching).
5. **Deep Learning & One-Step Diffusion (2022–2026):** Replaced manual emission/transition scoring with learned, data-driven spatial-temporal representation models (DeepMM, RNTrajRec, GraphMM, DiffMM). DiffMM (Han et al., 2026) leverages Transformer point encoders, R-tree spatial indexing, candidate segment attention, and a one-step shortcut diffusion model ($M=1$) to evaluate whole trajectories simultaneously without error accumulation.

This repository provides an independent benchmark suite evaluating the transition from classical HMMs (`LeuvenMapMatching`) to modern generative diffusion (`DiffMM`) under varying GPS sparsity levels.

---

## 🛠 Repository Structure

```text
.

```

---

## 📊 Published Baseline Performance Summary

*Data sourced from published benchmark experiments by Han et al. (2026) on the Porto Taxi Trajectory dataset ($N=1,013,437$ trips)*:

### 1. Map-Matching Accuracy Across GPS Sampling Gaps ($r$)

| Model | Type / Era | 75s Gap | 150s Gap | 300s Gap | 600s Gap | Accuracy Drop (75s $\rightarrow$ 600s) |
|---|---|---|---|---|---|---|
| **HMM** (Newson & Krumm, 2009) | Probabilistic HMM | **92.46%** | 83.82% | 66.62% | 40.04% | $-52.42\%$ |
| **DeepMM** (Feng et al., 2022) | Recurrent Seq2Seq | 86.38% | 83.68% | 81.37% | 78.69% | $-7.69\%$ |
| **RNTrajRec** (Chen et al., 2023) | ST-Transformer | 79.56% | 77.57% | 75.81% | 73.76% | $-5.80\%$ |
| **GraphMM** (Liu et al., 2024) | Graph Neural Network | 52.84% | 49.22% | 37.67% | 34.49% | $-18.35\%$ |
| **DiffMM** (Han et al., 2026) | One-Step Diffusion | **93.43%** | **91.47%** | **89.08%** | **86.87%** | **$-6.56\%$** |

> **Key Insight:** When GPS sampling is dense ($\le 75\text{s}$), the 2009 HMM performs within **0.97%** of DiffMM without requiring any GPU training. However, as data sparsity increases to 10-minute intervals (600s), HMM accuracy degrades severely to 40.04%, whereas DiffMM retains **86.87%** accuracy.

---

### 2. Computational Efficiency & Training Overhead (Beijing Dataset, $r=0.1$)

| Metric | HMM (2009) | GraphMM (2024) | DeepMM (2022) | RNTrajRec (2023) | DiffMM (2026) |
|---|---|---|---|---|---|
| **Inference Time** *(Sec / 1,000 Trajectories)* | 20.57 s | 62.79 s | 88.82 s | 627.65 s | **1.18 s** |
| **Training Time** *(Minutes / Epoch)* | **0.00 min** *(Zero-shot)* | 26.28 min | 9.07 min | 868.23 min | 10.66 min |

---

## 🔬 Component Ablation Analysis (DiffMM on Beijing Network)

Ablation experiments reported by Han et al. (2026) demonstrate the importance of each architectural module:

```text
+-------------------------------------------------------------+
| Full DiffMM Model (Shortcut DiT + Segment Attention)       | ---> 85.39% Accuracy (600s gap)
+-------------------------------------------------------------+
                              |
     [- Remove Segment-Level Candidate Attention]
                              v
+-------------------------------------------------------------+
| Replaced candidate segment attention with simple mean       | ---> 82.71% Accuracy (-2.68%)
+-------------------------------------------------------------+
```

* **Candidate Segment Attention:** Removing spatial attention over candidate road segments within the 50m radius causes the largest single drop in performance (dropping from 85.39% to 82.71% at 600s).
* **One-Step Shortcut Denoising:** Replacing the single-step shortcut model ($M=1$) with multi-step standard diffusion drops accuracy to 83.53% while increasing inference latencies significantly.

---

## ⚡ Quick Start & Setup

### Prerequisites
* Python 3.10+
* PyTorch 2.0+ (CUDA GPU acceleration recommended for DiffMM training)
* `LeuvenMapMatching`, `rtree`, `geopandas`, `networkx`, `scikit-learn`

### Installation



---

## 👥 Authors & Academic Context

* **Contributors:**
  * Deshan Senanayake
  * Deshan Costa
  * Anuradha Dissanayake
  * Chathuranga Dayarathne
  * Dhanuji Jayasingha


---

## 📜 References

1. Newson, P., & Krumm, J. (2009). *Hidden Markov map matching through noise and sparseness*. ACM SIGSPATIAL GIS '09.
2. Luxen, D., & Vetter, C. (2011). *Real-time routing with OpenStreetMap data*. ACM SIGSPATIAL GIS '11.
3. Meert, W., & Verbeke, M. (2018). *HMM with non-emitting states for map matching*. ECDA 2018.
4. Feng, J., et al. (2022). *DeepMM: Deep learning based map matching with data augmentation*. IEEE TMC.
5. Chen, Y., et al. (2023). *RNTrajRec: Road network enhanced trajectory recovery with spatial-temporal transformer*. IEEE ICDE.
6. Liu, Y., et al. (2024). *GraphMM: Graph-based vehicular map matching by leveraging trajectory and road correlations*. IEEE TKDE.
7. Frans, K., et al. (2025). *One step diffusion via shortcut models*. ICLR 2025.
8. Han, C., Yang, S. B., & Hu, J. (2026). *DiffMM: Efficient method for accurate noisy and sparse trajectory map matching via one step diffusion*. AAAI-26.
