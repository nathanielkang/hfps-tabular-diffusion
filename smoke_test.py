"""
smoke_test.py - Quick end-to-end sanity check (< 60 seconds on CPU).

Tests:
  1. Preprocessor fit / transform / inverse_transform round-trip
  2. Model builds and trains for 2 epochs (loss decreases)
  3. Sampling produces correct shape
  4. Decoded output has valid column names, in-domain categoricals, finite numerics
"""

import os
import sys
import time
import tempfile

import numpy as np
import pandas as pd
import torch

# Ensure src/ is on path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

from config import (
    DATA_CSV, DATA_ENCODING, COLUMN_ORDER,
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, SMOKE_HP,
)
from preprocessing import TabularPreprocessor
from model import build_model, save_checkpoint, load_checkpoint


def run_smoke_test():
    t_start = time.time()
    n_rows = 200

    # ---- 1. Load a small subset ----
    print("[smoke] Loading first %d rows ..." % n_rows)
    df = pd.read_csv(DATA_CSV, encoding=DATA_ENCODING, nrows=n_rows)
    assert list(df.columns) == COLUMN_ORDER, "FAIL: column order mismatch"
    assert len(df) == n_rows, "FAIL: expected %d rows" % n_rows
    print("[smoke] PASS: data loaded, shape=%s" % str(df.shape))

    # ---- 2. Preprocessor ----
    print("[smoke] Fitting preprocessor ...")
    prep = TabularPreprocessor()
    prep.fit(df)
    X = prep.transform(df)
    assert X.shape == (n_rows, prep.total_dim), (
        "FAIL: transform shape %s != (%d, %d)" % (X.shape, n_rows, prep.total_dim)
    )
    assert not np.isnan(X).any(), "FAIL: NaN in transformed data"
    print("[smoke] PASS: transform shape=%s, total_dim=%d" % (X.shape, prep.total_dim))

    # Round-trip check
    df_rt = prep.inverse_transform(X)
    assert list(df_rt.columns) == COLUMN_ORDER, "FAIL: inverse column order"
    assert len(df_rt) == n_rows, "FAIL: inverse row count"
    print("[smoke] PASS: inverse_transform round-trip shape=%s" % str(df_rt.shape))

    # ---- 3. Build model and train ----
    hp = dict(SMOKE_HP)
    model = build_model(input_dim=prep.total_dim, hp=hp)
    n_params = sum(p.numel() for p in model.parameters())
    print("[smoke] Model params: %d" % n_params)

    print("[smoke] Training for %d epochs ..." % hp["epochs"])
    losses = model.train_model(
        X_train=X,
        epochs=hp["epochs"],
        batch_size=hp["batch_size"],
        lr=hp["lr"],
        verbose=True,
    )
    assert len(losses) == hp["epochs"], "FAIL: loss list length"
    assert losses[-1] < losses[0], (
        "FAIL: loss did not decrease (%.4f -> %.4f)" % (losses[0], losses[-1])
    )
    print("[smoke] PASS: loss decreased %.4f -> %.4f" % (losses[0], losses[-1]))

    # ---- 4. Sample ----
    print("[smoke] Sampling %d rows ..." % n_rows)
    raw = model.sample(n_rows, verbose=False)
    assert raw.shape == (n_rows, prep.total_dim), (
        "FAIL: raw sample shape %s" % str(raw.shape)
    )
    assert np.isfinite(raw).all(), "FAIL: non-finite values in raw sample"
    print("[smoke] PASS: raw sample shape=%s, all finite" % str(raw.shape))

    # ---- 5. Decode and validate ----
    df_synth = prep.inverse_transform(raw)
    assert list(df_synth.columns) == COLUMN_ORDER, "FAIL: synth column order"
    assert len(df_synth) == n_rows, "FAIL: synth row count"

    # Check categoricals in domain
    for col in CATEGORICAL_COLUMNS:
        valid = set(v for _, v in prep.cat_decodings[col])
        actual = set(df_synth[col].unique())
        assert actual.issubset(valid), (
            "FAIL: %s out-of-domain: %s" % (col, actual - valid)
        )

    # Check numerics are finite
    for col in NUMERIC_COLUMNS:
        vals = df_synth[col].values
        assert np.isfinite(vals).all(), "FAIL: %s has non-finite values" % col

    print("[smoke] PASS: decoded shape=%s, all categoricals in domain, all numerics finite"
          % str(df_synth.shape))

    # ---- 6. Checkpoint save/load round-trip ----
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        prep_path = os.path.join(tmpdir, "prep.pkl")
        save_checkpoint(model, ckpt_path)
        prep.save(prep_path)

        model2 = build_model(input_dim=prep.total_dim, hp=hp)
        load_checkpoint(model2, ckpt_path)
        prep2 = TabularPreprocessor.load(prep_path)

        assert prep2.total_dim == prep.total_dim, "FAIL: preprocessor reload dim"
        # Verify same output from loaded model
        torch.manual_seed(0)
        s1 = model.sample(5, verbose=False)
        torch.manual_seed(0)
        s2 = model2.sample(5, verbose=False)
        assert np.allclose(s1, s2, atol=1e-5), "FAIL: checkpoint reload mismatch"
        print("[smoke] PASS: checkpoint save/load round-trip")

    elapsed = time.time() - t_start
    print("[smoke] ALL TESTS PASSED in %.1f seconds" % elapsed)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
