"""
Train CTGAN and TVAE baselines on the real data, generate synthetic CSVs.
"""
import os
import time
import pandas as pd
from ctgan import CTGAN, TVAE
from config_eval import (
    CATEGORICAL_COLUMNS, RESULTS_DIR, ENCODING,
    CTGAN_EPOCHS, TVAE_EPOCHS, BASELINE_BATCH_SIZE, N_REAL,
)


def train_ctgan(real_df, n_samples=None, seed=42):
    if n_samples is None:
        n_samples = len(real_df)
    print(f"  [Baseline] Training CTGAN (epochs={CTGAN_EPOCHS}) ...")
    t0 = time.time()
    model = CTGAN(
        epochs=CTGAN_EPOCHS,
        batch_size=BASELINE_BATCH_SIZE,
        cuda=False,
        verbose=True,
    )
    model.fit(real_df, discrete_columns=CATEGORICAL_COLUMNS)
    elapsed = time.time() - t0
    print(f"  [Baseline] CTGAN training done in {elapsed:.0f}s")

    print(f"  [Baseline] Sampling {n_samples} rows from CTGAN ...")
    synth = model.sample(n_samples)
    path = os.path.join(RESULTS_DIR, "ctgan_synthetic.csv")
    synth.to_csv(path, index=False, encoding=ENCODING)
    print(f"  [Baseline] Saved -> {path}")
    return synth


def train_tvae(real_df, n_samples=None, seed=42):
    if n_samples is None:
        n_samples = len(real_df)
    print(f"  [Baseline] Training TVAE (epochs={TVAE_EPOCHS}) ...")
    t0 = time.time()
    model = TVAE(
        epochs=TVAE_EPOCHS,
        batch_size=BASELINE_BATCH_SIZE,
        cuda=False,
        verbose=True,
    )
    model.fit(real_df, discrete_columns=CATEGORICAL_COLUMNS)
    elapsed = time.time() - t0
    print(f"  [Baseline] TVAE training done in {elapsed:.0f}s")

    print(f"  [Baseline] Sampling {n_samples} rows from TVAE ...")
    synth = model.sample(n_samples)
    path = os.path.join(RESULTS_DIR, "tvae_synthetic.csv")
    synth.to_csv(path, index=False, encoding=ENCODING)
    print(f"  [Baseline] Saved -> {path}")
    return synth
