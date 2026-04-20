# Inter-Record Distance Metrics — Mathematical Formulation

This document defines the two pairwise-distance metrics provided in `extras/inter_record_metrics/`. These metrics are **independent of the TabOversample–HFPS training pipeline** and can be applied to any pair of real / synthetic tabular datasets.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{R} = \{r_1, \ldots, r_n\}$ | Real dataset ($n$ records) |
| $\mathcal{S} = \{s_1, \ldots, s_m\}$ | Synthetic dataset ($m$ records) |
| $d(\cdot, \cdot)$ | Euclidean distance in standardised feature space |
| $\mathrm{NN}_k(x, \mathcal{P})$ | Set of $k$ nearest neighbors of $x$ in pool $\mathcal{P}$ |

### Mixed-type encoding

Before computing distances, all columns are mapped to a common numeric space:

- **Numeric columns** are z-scored (zero mean, unit variance) using statistics fitted on $\mathcal{R}$.
- **Categorical columns** are ordinal-encoded and then z-scored.

This ensures that every feature dimension contributes approximately equally to the Euclidean distance.

---

## 1. SNN Loss (Similarity of Nearest Neighbors)

### Intuition

SNN asks: *if we pool the real and synthetic records together and look at each point's $k$ nearest neighbors, are those neighbors a balanced mix of real and synthetic?*

A perfect generative model produces synthetic records that are indistinguishable from real ones in the local neighborhood, so approximately half the neighbors should come from each set.

### Definition

Let $\mathcal{P} = \mathcal{R} \cup \mathcal{S}$ be the combined pool. For each point $x_i \in \mathcal{P}$, define a label:

$$
\ell(x_i) =
\begin{cases}
0, & x_i \in \mathcal{R} \\
1, & x_i \in \mathcal{S}
\end{cases}
$$

For a given $k$, let $\mathrm{NN}_k(x_i, \mathcal{P} \setminus \{x_i\})$ be the $k$ nearest neighbors of $x_i$ in $\mathcal{P}$ excluding itself. Define the **cross-set fraction** for $x_i$:

$$
f(x_i) = \frac{1}{k} \sum_{x_j \in \mathrm{NN}_k(x_i)} \mathbf{1}\!\left[\ell(x_j) \neq \ell(x_i)\right]
$$

The **SNN loss** is decomposed into two directional scores:

$$
\mathrm{SNN}_{\text{real}} = \frac{1}{n} \sum_{r_i \in \mathcal{R}} f(r_i)
$$

$$
\mathrm{SNN}_{\text{synth}} = \frac{1}{m} \sum_{s_j \in \mathcal{S}} f(s_j)
$$

$$
\mathrm{SNN}_{\text{mean}} = \frac{\mathrm{SNN}_{\text{real}} + \mathrm{SNN}_{\text{synth}}}{2}
$$

### Interpretation

| $\mathrm{SNN}_{\text{mean}}$ | Meaning |
|:---:|---------|
| $\approx 0.5$ | Real and synthetic are well-mixed (ideal) |
| $\to 0$ | Synthetic data clusters **away** from real data |
| $\to 1$ | Should not happen with balanced $n \approx m$; signals labelling error |

We also report:

$$
\text{deviation} = \left|\mathrm{SNN}_{\text{mean}} - 0.5\right|
$$

Lower deviation is better.

### Complexity

- Time: $O\bigl((n + m)^2 \cdot d\bigr)$ for exact $k$-NN in $d$-dimensional space. The implementation uses `sklearn.NearestNeighbors` with auto-selected algorithm (KD-tree or ball-tree when $d$ is moderate).
- The `subsample` parameter (default 5000) reduces both $n$ and $m$ to keep wall time practical.

### Reference

> M. S. Alaa, B. van Breugel, E. Saveliev, M. van der Schaar, "How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models," *ICML 2022*.

---

## 2. DCR (Distance to Closest Record)

### Intuition

DCR directly measures *how far each synthetic record is from its nearest real record*. This serves as both a **privacy** gauge (very small DCR → potential memorisation) and a **fidelity** gauge (very large DCR → poor coverage of the real distribution).

### Definition

For each synthetic record $s_j$, its DCR with respect to the real set is:

$$
\mathrm{DCR}(s_j, \mathcal{R}) = \min_{r_i \in \mathcal{R}} \; d(s_j, r_i)
$$

We report summary statistics over the synthetic set:

$$
\mathrm{DCR}_{\text{median}} = \mathrm{median}\bigl\{\mathrm{DCR}(s_j, \mathcal{R}) : j = 1, \ldots, m\bigr\}
$$

$$
\mathrm{DCR}_{5\text{th}} = P_5\bigl\{\mathrm{DCR}(s_j, \mathcal{R})\bigr\}
$$

### Holdout baseline

To calibrate these numbers, we also compute the **real-to-real** DCR (leave-one-out):

$$
\mathrm{DCR}_{\text{real}}(r_i) = \min_{r_k \in \mathcal{R} \setminus \{r_i\}} \; d(r_i, r_k)
$$

$$
\mathrm{DCR}_{\text{real,median}} = \mathrm{median}\bigl\{\mathrm{DCR}_{\text{real}}(r_i)\bigr\}
$$

The **DCR ratio** normalises the synthetic DCR by the real-to-real baseline:

$$
\rho = \frac{\mathrm{DCR}_{\text{median}}}{\mathrm{DCR}_{\text{real,median}}}
$$

### Interpretation

| $\rho$ | Meaning |
|:---:|---------|
| $\approx 1.0$ | Synthetic records are about as far from their nearest real record as real records are from each other (ideal) |
| $\ll 1$ | Many synthetic records are suspiciously close to real records — possible **memorisation** |
| $\gg 1$ | Synthetic data is far from real data — **underfitting** or mode collapse |

The 5th percentile ($\mathrm{DCR}_{5\text{th}}$) specifically flags potential privacy leaks: if this is near zero, some synthetic records are near-copies of real ones.

### Complexity

- Time: $O(m \cdot n \cdot d)$ for the synthetic-to-real search, plus $O(n^2 \cdot d)$ for real-to-real. With `subsample=5000`, both are bounded.

### Reference

> Y. Zhao, I. Shumailov, R. Mullins, R. Anderson, "Synthetic Data — Anonymisation Groundhog Day," *USENIX Security 2022*.

---

## Quick usage

```bash
cd extras/inter_record_metrics

# Both metrics at once
python run_metrics.py \
    --real  ../../path/to/real.csv \
    --synth ../../path/to/synthetic.csv \
    --k 5 --subsample 5000 --output-dir results/

# SNN only
python snn_loss.py --real real.csv --synth synth.csv --k 5

# DCR only
python dcr.py --real real.csv --synth synth.csv
```

### Python API

```python
import pandas as pd
from extras.inter_record_metrics import snn_loss, dcr

real_df  = pd.read_csv("real.csv", encoding="utf-8-sig")
synth_df = pd.read_csv("synthetic.csv", encoding="utf-8-sig")

num_cols = ["col_a", "col_b", ...]
cat_cols = ["col_x", "col_y", ...]

snn_result = snn_loss(real_df, synth_df, num_cols, cat_cols, k=5)
dcr_result = dcr(real_df, synth_df, num_cols, cat_cols)
```

---

## Disclaimer

These metrics are **utility / analysis tools** for comparing real and synthetic tabular datasets. They are not part of the TabOversample–HFPS training pipeline and have no dependency on the diffusion model code under `src/`.
