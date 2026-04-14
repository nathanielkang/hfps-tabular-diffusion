"""
Tier C: Train-on-Synthetic, Test-on-Real (TSTR) evaluation.
Classification: macro-F1, accuracy
Regression: RMSE, R^2
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from config_eval import (
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, RESULTS_DIR, SEEDS,
    SPLIT_RATIOS, CLF_TARGET, REG_TARGET,
)
from load_data import split_indices, prepare_xy


def _align_features(X_tr, X_val, X_te):
    """Pad feature arrays to the same width."""
    max_f = max(X_tr.shape[1], X_val.shape[1], X_te.shape[1])
    if X_tr.shape[1] < max_f:
        X_tr = np.pad(X_tr, ((0, 0), (0, max_f - X_tr.shape[1])))
    if X_val.shape[1] < max_f:
        X_val = np.pad(X_val, ((0, 0), (0, max_f - X_val.shape[1])))
    if X_te.shape[1] < max_f:
        X_te = np.pad(X_te, ((0, 0), (0, max_f - X_te.shape[1])))
    return X_tr, X_val, X_te


def _run_classification(train_df, val_df, test_df, target, seed):
    cat_cols = list(CATEGORICAL_COLUMNS)
    num_cols = list(NUMERIC_COLUMNS)

    X_tr, y_tr = prepare_xy(train_df, target, cat_cols, num_cols)
    X_val, y_val = prepare_xy(val_df, target, cat_cols, num_cols)
    X_te, y_te = prepare_xy(test_df, target, cat_cols, num_cols)
    X_tr, X_val, X_te = _align_features(X_tr, X_val, X_te)

    le = LabelEncoder()
    all_labels = np.concatenate([y_tr.astype(str), y_val.astype(str), y_te.astype(str)])
    le.fit(all_labels)
    y_tr_enc = le.transform(y_tr.astype(str))
    y_val_enc = le.transform(y_val.astype(str))
    y_te_enc = le.transform(y_te.astype(str))

    n_cls = len(le.classes_)
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=seed, verbose=-1, n_jobs=-1,
        objective="multiclass", num_class=n_cls,
    )
    model.fit(X_tr, y_tr_enc)
    preds = model.predict(X_te)
    acc = float(accuracy_score(y_te_enc, preds))
    f1 = float(f1_score(y_te_enc, preds, average="macro"))
    return acc, f1


def _run_regression(train_df, val_df, test_df, target, seed):
    cat_cols = list(CATEGORICAL_COLUMNS)
    num_cols = list(NUMERIC_COLUMNS)

    X_tr, y_tr = prepare_xy(train_df, target, cat_cols, num_cols)
    X_val, y_val = prepare_xy(val_df, target, cat_cols, num_cols)
    X_te, y_te = prepare_xy(test_df, target, cat_cols, num_cols)
    X_tr, X_val, X_te = _align_features(X_tr, X_val, X_te)

    y_tr = y_tr.astype(float)
    y_val = y_val.astype(float)
    y_te = y_te.astype(float)

    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=seed, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    r2 = float(r2_score(y_te, preds))
    return rmse, r2


def run_tstr(real_df, synth_df, label="ours"):
    """Run TSTR for both classification and regression tasks across seeds."""
    print(f"  [TSTR] Running for: {label}")
    clf_results = {"accs": [], "f1s": []}
    reg_results = {"rmses": [], "r2s": []}

    for seed in SEEDS:
        idx = split_indices(len(real_df), SPLIT_RATIOS, seed)
        real_test = real_df.iloc[idx["test"]].reset_index(drop=True)
        real_val = real_df.iloc[idx["val"]].reset_index(drop=True)

        n_train = len(idx["train"])
        if len(synth_df) >= n_train:
            synth_train = synth_df.sample(n=n_train, random_state=seed).reset_index(drop=True)
        else:
            synth_train = synth_df.reset_index(drop=True)

        acc, f1 = _run_classification(synth_train, real_val, real_test, CLF_TARGET, seed)
        clf_results["accs"].append(acc)
        clf_results["f1s"].append(f1)

        rmse, r2 = _run_regression(synth_train, real_val, real_test, REG_TARGET, seed)
        reg_results["rmses"].append(rmse)
        reg_results["r2s"].append(r2)

    summary = {
        "method": label,
        "classification": {
            "target": CLF_TARGET,
            "accuracy_mean": round(float(np.mean(clf_results["accs"])), 4),
            "accuracy_std": round(float(np.std(clf_results["accs"])), 4),
            "macro_f1_mean": round(float(np.mean(clf_results["f1s"])), 4),
            "macro_f1_std": round(float(np.std(clf_results["f1s"])), 4),
        },
        "regression": {
            "target": REG_TARGET,
            "rmse_mean": round(float(np.mean(reg_results["rmses"])), 2),
            "rmse_std": round(float(np.std(reg_results["rmses"])), 2),
            "r2_mean": round(float(np.mean(reg_results["r2s"])), 4),
            "r2_std": round(float(np.std(reg_results["r2s"])), 4),
        },
    }

    path = os.path.join(RESULTS_DIR, f"tstr_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [TSTR] Saved -> {path}")
    return summary


def run_trtr(real_df):
    """Train-on-Real, Test-on-Real (upper bound reference)."""
    print("  [TRTR] Running upper bound (Real -> Real)")
    clf_results = {"accs": [], "f1s": []}
    reg_results = {"rmses": [], "r2s": []}

    for seed in SEEDS:
        idx = split_indices(len(real_df), SPLIT_RATIOS, seed)
        real_train = real_df.iloc[idx["train"]].reset_index(drop=True)
        real_val = real_df.iloc[idx["val"]].reset_index(drop=True)
        real_test = real_df.iloc[idx["test"]].reset_index(drop=True)

        acc, f1 = _run_classification(real_train, real_val, real_test, CLF_TARGET, seed)
        clf_results["accs"].append(acc)
        clf_results["f1s"].append(f1)

        rmse, r2 = _run_regression(real_train, real_val, real_test, REG_TARGET, seed)
        reg_results["rmses"].append(rmse)
        reg_results["r2s"].append(r2)

    summary = {
        "method": "Real (TRTR)",
        "classification": {
            "target": CLF_TARGET,
            "accuracy_mean": round(float(np.mean(clf_results["accs"])), 4),
            "accuracy_std": round(float(np.std(clf_results["accs"])), 4),
            "macro_f1_mean": round(float(np.mean(clf_results["f1s"])), 4),
            "macro_f1_std": round(float(np.std(clf_results["f1s"])), 4),
        },
        "regression": {
            "target": REG_TARGET,
            "rmse_mean": round(float(np.mean(reg_results["rmses"])), 2),
            "rmse_std": round(float(np.std(reg_results["rmses"])), 2),
            "r2_mean": round(float(np.mean(reg_results["r2s"])), 4),
            "r2_std": round(float(np.std(reg_results["r2s"])), 4),
        },
    }

    path = os.path.join(RESULTS_DIR, "tstr_real_trtr.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [TRTR] Saved -> {path}")
    return summary
