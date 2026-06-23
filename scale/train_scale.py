"""
scale/train_scale.py - Train the DDPM core on a Parquet (sub)slice using a
dictionary-driven schema. Saves checkpoint + preprocessor + schema + summary.

This deliberately loads `train_rows` rows over the modeled columns into memory
and trains the unchanged DDPM on them. For large tiers this load + fit + train
is the feasibility event we are measuring (OOM / impractical wall-clock).
"""

from __future__ import annotations

import argparse, json, os, time

import numpy as np
import torch

import _common  # noqa: F401  (sets sys.path)
from _common import peak_rss_gb
import data as datamod
from schema import (Schema, schema_from_dictionary, schema_from_lists,
                    schema_from_json, save_schema_json)
from preprocess import ScalablePreprocessor
from modelio import build_model, save_checkpoint

SCALE_HP = {
    "n_timesteps": 1000, "hidden_dim": 512, "n_layers": 6,
    "schedule": "cosine", "batch_size": 4096, "lr": 1e-3, "epochs": 200,
}
SMOKE_HP = {
    "n_timesteps": 50, "hidden_dim": 64, "n_layers": 2,
    "schedule": "cosine", "batch_size": 512, "lr": 1e-3, "epochs": 2,
}


def load_schema(args) -> Schema:
    if args.schema_json:
        return schema_from_json(args.schema_json)
    if args.dictionary:
        return schema_from_dictionary(args.dictionary, extra_exclude=args.exclude or [])
    raise SystemExit("Provide --dictionary or --schema-json")


def train_scale(data_path: str, schema: Schema, train_rows: int,
                hp: dict, ckpt_dir: str, seed: int = 42,
                fit_cap: int | None = 2_000_000) -> dict:
    np.random.seed(seed); torch.manual_seed(seed)
    cols = schema.modeled_columns

    t_read0 = time.time()
    df = datamod.read_training_slice(data_path, train_rows, columns=cols)
    t_read = time.time() - t_read0
    actual_rows = len(df)

    prep = ScalablePreprocessor(schema, fit_cap=fit_cap)
    t_pre0 = time.time()
    prep.fit(df)
    X = prep.transform(df)
    t_pre = time.time() - t_pre0
    del df

    model = build_model(input_dim=prep.total_dim, hp=hp)
    n_params = sum(p.numel() for p in model.parameters())

    t_tr0 = time.time()
    losses = model.train_model(X_train=X, epochs=hp["epochs"],
                               batch_size=hp["batch_size"], lr=hp["lr"], verbose=True)
    t_train = time.time() - t_tr0

    os.makedirs(ckpt_dir, exist_ok=True)
    save_checkpoint(model, os.path.join(ckpt_dir, "model.pt"))
    prep.save(os.path.join(ckpt_dir, "preprocessor.pkl"))
    save_schema_json(schema, os.path.join(ckpt_dir, "schema.json"))
    with open(os.path.join(ckpt_dir, "hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2)

    summary = {
        "train_rows_requested": train_rows,
        "train_rows_actual": actual_rows,
        "n_modeled_cols": prep.total_dim,
        "n_numeric": prep.num_dim,
        "n_categorical": prep.cat_dim,
        "n_params": int(n_params),
        "epochs": hp["epochs"],
        "final_loss": float(losses[-1]),
        "seconds_read": round(t_read, 2),
        "seconds_preprocess": round(t_pre, 2),
        "seconds_train": round(t_train, 2),
        "peak_rss_gb": peak_rss_gb(),
        "fit_cap": fit_cap,
    }
    with open(os.path.join(ckpt_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Train DDPM core at scale")
    ap.add_argument("--data", required=True, help="Parquet path")
    ap.add_argument("--dictionary", help="column-type dictionary CSV")
    ap.add_argument("--schema-json", help="schema json (numeric/categorical lists)")
    ap.add_argument("--exclude", nargs="*", default=[], help="extra columns to exclude")
    ap.add_argument("--train-rows", type=int, required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints_scale")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit-cap", type=int, default=2_000_000)
    args = ap.parse_args()

    schema = load_schema(args)
    hp = dict(SMOKE_HP if args.smoke else SCALE_HP)
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    s = train_scale(args.data, schema, args.train_rows, hp, args.ckpt_dir,
                    seed=args.seed, fit_cap=args.fit_cap)
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
