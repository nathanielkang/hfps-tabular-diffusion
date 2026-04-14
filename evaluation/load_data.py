"""
Data loading utilities for evaluation.
"""
import numpy as np
import pandas as pd
from config_eval import (
    REAL_CSV, ENCODING, COLUMN_ORDER, SPLIT_RATIOS
)


def load_real():
    df = pd.read_csv(REAL_CSV, encoding=ENCODING)
    df = df[COLUMN_ORDER]
    return df


def load_synthetic(path):
    df = pd.read_csv(path, encoding=ENCODING)
    df = df[COLUMN_ORDER]
    return df


def split_indices(n, ratios, seed):
    """Deterministic train/val/test index split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train:n_train + n_val].tolist(),
        "test": idx[n_train + n_val:].tolist(),
    }


def prepare_xy(df, target_col, cat_cols, num_cols):
    """Encode features (one-hot for cats, passthrough for nums) and extract target."""
    feature_cats = [c for c in cat_cols if c != target_col]
    feature_nums = [c for c in num_cols if c != target_col]

    parts = []
    if feature_nums:
        parts.append(df[feature_nums].astype(float))
    if feature_cats:
        parts.append(pd.get_dummies(df[feature_cats].astype(str), drop_first=False))

    X = pd.concat(parts, axis=1).values.astype(np.float32)
    y = df[target_col].values
    return X, y
