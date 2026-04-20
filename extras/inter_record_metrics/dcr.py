"""
DCR (Distance to Closest Record) for synthetic tabular data evaluation.

Measures the distance between each synthetic record and its closest real
record.  This captures two complementary aspects:
  - **Privacy**: if DCR values are very small, synthetic records are near-copies
    of real ones (memorisation / overfitting).
  - **Fidelity**: if DCR values are very large, synthetic data does not
    resemble the real distribution (underfitting / mode collapse).

A well-calibrated generative model produces a DCR distribution whose median
is comparable to the "holdout DCR" (real test set → real train set).

Reference
---------
    Y. Zhao, I. Shumailov, R. Mullins, R. Anderson,
    "Synthetic Data — Anonymisation Groundhog Day," USENIX Security 2022.
See also broader privacy-utility evaluation surveys.
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
    """Encode mixed-type tables into numeric arrays for distance computation."""
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


def dcr(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    subsample: Optional[int] = 5000,
    seed: int = 42,
) -> dict:
    """Compute DCR (Distance to Closest Record) statistics.

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
    subsample : int or None
        If not None, randomly subsample both sets to this size to keep
        the O(n * m) distance computation tractable.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        dcr_synth_to_real_median : float
            Median Euclidean distance from each synthetic record to its
            nearest real record (in standardised space).
        dcr_synth_to_real_5th : float
            5th percentile — low values flag potential memorisation.
        dcr_synth_to_real_mean : float
        dcr_real_to_real_median : float
            Holdout baseline — median distance among real records
            (leave-one-out nearest neighbor).
        dcr_ratio : float
            dcr_synth_to_real_median / dcr_real_to_real_median.
            Ideal ≈ 1.0; >>1 means underfitting; <<1 means memorisation.
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

    nn_real = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean")
    nn_real.fit(X_r)

    dists_synth, _ = nn_real.kneighbors(X_s)
    dcr_s2r = dists_synth[:, 0]

    dists_real, _ = nn_real.kneighbors(X_r)
    dcr_r2r = dists_real[:, 1]  # second neighbor (first is self)

    s2r_median = float(np.median(dcr_s2r))
    r2r_median = float(np.median(dcr_r2r))
    ratio = s2r_median / r2r_median if r2r_median > 0 else float("inf")

    return {
        "dcr_synth_to_real_median": round(s2r_median, 4),
        "dcr_synth_to_real_5th": round(float(np.percentile(dcr_s2r, 5)), 4),
        "dcr_synth_to_real_mean": round(float(np.mean(dcr_s2r)), 4),
        "dcr_real_to_real_median": round(r2r_median, 4),
        "dcr_ratio": round(ratio, 4),
        "n_real": len(X_r),
        "n_synth": len(X_s),
    }


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Compute DCR between real and synthetic CSV.")
    parser.add_argument("--real", required=True, help="Path to real data CSV")
    parser.add_argument("--synth", required=True, help="Path to synthetic data CSV")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    parser.add_argument("--subsample", type=int, default=5000, help="Subsample size (0 = no subsample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    real_df = pd.read_csv(args.real, encoding=args.encoding)
    synth_df = pd.read_csv(args.synth, encoding=args.encoding)

    num_cols = real_df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in real_df.columns if c not in num_cols]

    sub = args.subsample if args.subsample > 0 else None
    result = dcr(real_df, synth_df, num_cols, cat_cols, subsample=sub, seed=args.seed)

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {args.output}")
