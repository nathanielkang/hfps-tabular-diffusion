"""
Tier A: Statistical fidelity metrics.
KS statistic for numerics, TVD for categoricals, correlation matrix difference.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from config_eval import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, RESULTS_DIR
import os


def ks_per_column(real_df, synth_df, num_cols):
    results = {}
    for col in num_cols:
        stat, pval = stats.ks_2samp(
            real_df[col].dropna().values.astype(float),
            synth_df[col].dropna().values.astype(float),
        )
        results[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(pval), 6)}
    return results


def tvd_per_column(real_df, synth_df, cat_cols):
    results = {}
    for col in cat_cols:
        real_counts = real_df[col].astype(str).value_counts(normalize=True)
        synth_counts = synth_df[col].astype(str).value_counts(normalize=True)
        all_cats = set(real_counts.index) | set(synth_counts.index)
        tvd = 0.5 * sum(
            abs(real_counts.get(c, 0.0) - synth_counts.get(c, 0.0)) for c in all_cats
        )
        results[col] = {"tvd": round(float(tvd), 4)}
    return results


def correlation_diff(real_df, synth_df, num_cols):
    """Frobenius norm of Pearson correlation matrix difference (numerics only)."""
    real_corr = real_df[num_cols].astype(float).corr().values
    synth_corr = synth_df[num_cols].astype(float).corr().values
    real_corr = np.nan_to_num(real_corr, nan=0.0)
    synth_corr = np.nan_to_num(synth_corr, nan=0.0)
    frob = float(np.linalg.norm(real_corr - synth_corr, "fro"))
    return {"frobenius_norm": round(frob, 4)}


def run_fidelity(real_df, synth_df, label="ours"):
    print(f"  [Fidelity] Running for: {label}")
    ks = ks_per_column(real_df, synth_df, NUMERIC_COLUMNS)
    tvd = tvd_per_column(real_df, synth_df, CATEGORICAL_COLUMNS)
    corr = correlation_diff(real_df, synth_df, NUMERIC_COLUMNS)

    avg_ks = float(np.mean([v["ks_stat"] for v in ks.values()]))
    avg_tvd = float(np.mean([v["tvd"] for v in tvd.values()]))

    summary = {
        "method": label,
        "avg_ks": round(avg_ks, 4),
        "avg_tvd": round(avg_tvd, 4),
        "corr_frobenius": corr["frobenius_norm"],
        "per_column_ks": ks,
        "per_column_tvd": tvd,
        "correlation_diff": corr,
    }

    path = os.path.join(RESULTS_DIR, f"fidelity_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [Fidelity] Saved -> {path}")
    return summary
