"""
Tier B: Detection metric.
Train a classifier to distinguish real vs synthetic rows.
Good synthesis -> accuracy near 0.5 (AUC near 0.5).
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from config_eval import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, RESULTS_DIR, SEEDS


def _encode_for_detection(df, num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(df[num_cols].astype(float))
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols].astype(str), drop_first=False))
    return pd.concat(parts, axis=1).values.astype(np.float32)


def run_detection(real_df, synth_df, label="ours", seed=42):
    print(f"  [Detection] Running for: {label}")
    n = min(len(real_df), len(synth_df))
    real_sub = real_df.sample(n=n, random_state=seed).reset_index(drop=True)
    synth_sub = synth_df.sample(n=n, random_state=seed).reset_index(drop=True)

    X_real = _encode_for_detection(real_sub, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
    X_synth = _encode_for_detection(synth_sub, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)

    max_feats = max(X_real.shape[1], X_synth.shape[1])
    if X_real.shape[1] < max_feats:
        X_real = np.pad(X_real, ((0, 0), (0, max_feats - X_real.shape[1])))
    if X_synth.shape[1] < max_feats:
        X_synth = np.pad(X_synth, ((0, 0), (0, max_feats - X_synth.shape[1])))

    X = np.vstack([X_real, X_synth])
    y = np.array([0] * n + [1] * n)

    accs, aucs = [], []
    for s in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=s, verbose=-1, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]
        accs.append(float(accuracy_score(y_te, preds)))
        aucs.append(float(roc_auc_score(y_te, probs)))

    summary = {
        "method": label,
        "detection_accuracy_mean": round(float(np.mean(accs)), 4),
        "detection_accuracy_std": round(float(np.std(accs)), 4),
        "detection_auc_mean": round(float(np.mean(aucs)), 4),
        "detection_auc_std": round(float(np.std(aucs)), 4),
    }

    path = os.path.join(RESULTS_DIR, f"detection_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [Detection] Saved -> {path}")
    return summary
