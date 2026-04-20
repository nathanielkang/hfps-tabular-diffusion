"""
SNN (Similarity of Nearest Neighbors) loss for synthetic tabular data evaluation.

Measures whether real and synthetic datasets share the same local neighborhood
structure by computing the fraction of k-nearest-neighbor lookups that land in the
"other" dataset.  A perfect generative model yields SNN ≈ 0.5 (random chance);
if the synthetic data clusters away from the real data, SNN → 0.0 or 1.0.

Reference
---------
The SNN metric was popularized for generative-model evaluation in:
    M. S. Alaa, B. van Breugel, E. Saveliev, M. van der Schaar,
    "How Faithful is your Synthetic Data? Sample-level Metrics for
     Evaluating and Auditing Generative Models," ICML 2022.
See also the broader survey on nearest-neighbor evaluation of synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from typing import Optional


def _encode_mixed(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode mixed-type tables into numeric arrays for distance computation.

    Numeric columns are z-scored (fitted on real); categorical columns are
    ordinal-encoded and then z-scored so that every dimension contributes
    roughly equally to Euclidean distance.
    """
    r = real.copy()
    s = synth.copy()

    parts_r, parts_s = [], []

    if num_cols:
        scaler = StandardScaler().fit(r[num_cols].values.astype(float))
        parts_r.append(scaler.transform(r[num_cols].values.astype(float)))
        parts_s.append(scaler.transform(s[num_cols].values.astype(float)))

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(r[cat_cols].astype(str))
        r_cat = enc.transform(r[cat_cols].astype(str))
        s_cat = enc.transform(s[cat_cols].astype(str))
        cat_scaler = StandardScaler().fit(r_cat)
        parts_r.append(cat_scaler.transform(r_cat))
        parts_s.append(cat_scaler.transform(s_cat))

    return np.hstack(parts_r), np.hstack(parts_s)


def snn_loss(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    k: int = 5,
    subsample: Optional[int] = 5000,
    seed: int = 42,
) -> dict:
    """Compute the SNN (Similarity of Nearest Neighbors) loss.

    Parameters
    ----------
    real : pd.DataFrame
        Real (training) data.
    synth : pd.DataFrame
        Synthetic data from the generative model.
    num_cols : list[str]
        Numeric column names.
    cat_cols : list[str]
        Categorical column names.
    k : int
        Number of nearest neighbors (default 5).
    subsample : int or None
        If not None, randomly subsample both sets to this size to keep
        O(n^2) distance computation tractable (recommended for n > 10 000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        snn_real : float
            Fraction of real query points whose nearest neighbor in the
            combined pool is synthetic.  Ideal ≈ 0.5.
        snn_synth : float
            Fraction of synthetic query points whose nearest neighbor
            in the combined pool is real.  Ideal ≈ 0.5.
        snn_mean : float
            (snn_real + snn_synth) / 2.  Ideal ≈ 0.5.
        deviation : float
            |snn_mean − 0.5|.  Lower is better (0 = perfect mixing).
    """
    rng = np.random.RandomState(seed)

    r_df = real[num_cols + cat_cols].copy()
    s_df = synth[num_cols + cat_cols].copy()

    if subsample is not None:
        n_r = min(len(r_df), subsample)
        n_s = min(len(s_df), subsample)
        r_df = r_df.iloc[rng.choice(len(r_df), n_r, replace=False)].reset_index(drop=True)
        s_df = s_df.iloc[rng.choice(len(s_df), n_s, replace=False)].reset_index(drop=True)

    X_r, X_s = _encode_mixed(r_df, s_df, num_cols, cat_cols)

    n_r, n_s = len(X_r), len(X_s)
    X_all = np.vstack([X_r, X_s])
    labels = np.array([0] * n_r + [1] * n_s)  # 0 = real, 1 = synthetic

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(X_all)
    _, indices = nn.kneighbors(X_all)

    snn_real_hits = 0
    snn_synth_hits = 0
    total_real = 0
    total_synth = 0

    for i in range(len(X_all)):
        neighbors = indices[i, 1:]  # exclude self
        neighbor_labels = labels[neighbors]
        frac_other = np.mean(neighbor_labels != labels[i])

        if labels[i] == 0:
            snn_real_hits += frac_other
            total_real += 1
        else:
            snn_synth_hits += frac_other
            total_synth += 1

    snn_real = float(snn_real_hits / max(total_real, 1))
    snn_synth = float(snn_synth_hits / max(total_synth, 1))
    snn_mean = (snn_real + snn_synth) / 2.0
    deviation = abs(snn_mean - 0.5)

    return {
        "snn_real": round(snn_real, 4),
        "snn_synth": round(snn_synth, 4),
        "snn_mean": round(snn_mean, 4),
        "deviation": round(deviation, 4),
        "k": k,
        "n_real": n_r,
        "n_synth": n_s,
    }


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Compute SNN loss between real and synthetic CSV.")
    parser.add_argument("--real", required=True, help="Path to real data CSV")
    parser.add_argument("--synth", required=True, help="Path to synthetic data CSV")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors (default: 5)")
    parser.add_argument("--subsample", type=int, default=5000, help="Subsample size (0 = no subsample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    real_df = pd.read_csv(args.real, encoding=args.encoding)
    synth_df = pd.read_csv(args.synth, encoding=args.encoding)

    num_cols = real_df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in real_df.columns if c not in num_cols]

    sub = args.subsample if args.subsample > 0 else None
    result = snn_loss(real_df, synth_df, num_cols, cat_cols, k=args.k, subsample=sub, seed=args.seed)

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")
