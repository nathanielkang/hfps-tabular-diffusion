"""
sample.py - Generate synthetic CSV from a trained diffusion checkpoint.

Usage:
    python sample.py                          # default: 18314 rows
    python sample.py --n-samples 1000         # override row count
    python sample.py --output my_synth.csv    # custom output path
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from config import (
    CHECKPOINT_DIR, OUTPUT_DIR, COLUMN_ORDER,
    CATEGORICAL_COLUMNS, N_ROWS,
)
from preprocessing import TabularPreprocessor
from model import build_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Sample from trained HFPS model")
    parser.add_argument("--n-samples", type=int, default=N_ROWS,
                        help="Number of synthetic rows to generate")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: output/synthetic.csv)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_dir = args.checkpoint_dir

    # --- Load preprocessor ---
    prep_path = os.path.join(ckpt_dir, "preprocessor.pkl")
    print(f"[sample] Loading preprocessor from {prep_path}")
    prep = TabularPreprocessor.load(prep_path)

    # --- Load hyperparams ---
    hp_path = os.path.join(ckpt_dir, "hyperparams.json")
    with open(hp_path, "r") as f:
        hp = json.load(f)

    # --- Build and load model ---
    model = build_model(input_dim=prep.total_dim, hp=hp)
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    print(f"[sample] Loading model from {ckpt_path}")
    load_checkpoint(model, ckpt_path)

    # --- Sample ---
    n = args.n_samples
    print(f"[sample] Generating {n} synthetic rows ...")
    raw = model.sample(n, verbose=True)
    print(f"[sample] Raw sample shape: {raw.shape}")

    # --- Decode ---
    df_synth = prep.inverse_transform(raw)
    assert list(df_synth.columns) == COLUMN_ORDER, "Column order mismatch"
    assert len(df_synth) == n, f"Expected {n} rows, got {len(df_synth)}"

    # --- Validate categoricals are in-domain ---
    for col in CATEGORICAL_COLUMNS:
        valid = set(enc_val for _, enc_val in prep.cat_decodings[col])
        actual = set(df_synth[col].unique())
        if not actual.issubset(valid):
            oov = actual - valid
            print(f"[sample] WARNING: {col} has out-of-domain values: {oov}")

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = args.output or os.path.join(OUTPUT_DIR, "synthetic.csv")
    df_synth.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[sample] Saved {len(df_synth)} rows x {len(df_synth.columns)} cols -> {out_path}")

    # Quick stats
    print(f"[sample] Shape: {df_synth.shape}")
    print(f"[sample] Columns match: {list(df_synth.columns) == COLUMN_ORDER}")
    has_nan = df_synth.isnull().any().any()
    print(f"[sample] Any NaN: {has_nan}")

    return df_synth


if __name__ == "__main__":
    main()
