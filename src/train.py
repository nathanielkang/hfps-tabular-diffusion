"""
train.py - Train the tabular diffusion model on the HFPS dataset.

Usage:
    python train.py                       # full training (500 epochs)
    python train.py --epochs 10           # override epochs
    python train.py --smoke               # smoke-test config (2 epochs, tiny model)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from config import (
    DATA_CSV, DATA_ENCODING, CHECKPOINT_DIR,
    COLUMN_ORDER, FULL_HP, SMOKE_HP,
)
from preprocessing import TabularPreprocessor
from model import build_model, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train HFPS diffusion model")
    parser.add_argument("--smoke", action="store_true",
                        help="Use smoke-test hyperparameters")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--data", type=str, default=DATA_CSV,
                        help="Path to training CSV")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows for quick tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hp = dict(SMOKE_HP if args.smoke else FULL_HP)
    if args.epochs is not None:
        hp["epochs"] = args.epochs

    # --- Load data ---
    print(f"[train] Loading data from {os.path.basename(args.data)}")
    df = pd.read_csv(args.data, encoding=DATA_ENCODING)
    assert list(df.columns) == COLUMN_ORDER, "Column mismatch with schema"

    if args.max_rows is not None:
        df = df.head(args.max_rows)
    print(f"[train] Data shape: {df.shape}")

    # --- Preprocess ---
    print("[train] Fitting preprocessor ...")
    prep = TabularPreprocessor()
    prep.fit(df)
    X = prep.transform(df)
    print(f"[train] Transformed shape: {X.shape} (total_dim={prep.total_dim})")

    # --- Build model ---
    model = build_model(input_dim=prep.total_dim, hp=hp)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {n_params:,}")
    print(f"[train] Hyperparameters: {hp}")

    # --- Train ---
    t0 = time.time()
    losses = model.train_model(
        X_train=X,
        epochs=hp["epochs"],
        batch_size=hp["batch_size"],
        lr=hp["lr"],
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"[train] Training done in {elapsed:.1f}s")
    print(f"[train] Final loss: {losses[-1]:.6f}")

    # --- Save checkpoint ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    prep_path = os.path.join(CHECKPOINT_DIR, "preprocessor.pkl")
    hp_path = os.path.join(CHECKPOINT_DIR, "hyperparams.json")

    save_checkpoint(model, ckpt_path)
    prep.save(prep_path)
    with open(hp_path, "w") as f:
        json.dump(hp, f, indent=2)

    print(f"[train] Saved checkpoint -> {ckpt_path}")
    print(f"[train] Saved preprocessor -> {prep_path}")
    print(f"[train] Saved hyperparams -> {hp_path}")

    # --- Summary JSON ---
    summary = {
        "data_rows": len(df),
        "data_cols": len(df.columns),
        "total_dim": prep.total_dim,
        "n_params": n_params,
        "epochs": hp["epochs"],
        "final_loss": float(losses[-1]),
        "elapsed_seconds": round(elapsed, 1),
        "losses": [float(l) for l in losses],
    }
    summary_path = os.path.join(CHECKPOINT_DIR, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Saved summary -> {summary_path}")

    return losses


if __name__ == "__main__":
    main()
