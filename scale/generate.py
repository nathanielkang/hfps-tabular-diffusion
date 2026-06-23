"""
scale/generate.py - Public SynPersona-facing synthesis entry point.

    generate(checkpoint_dir, n_syn, seed,
             numeric_vars=None, categorical_vars=None, exclude_vars=None,
             output_path=None, gen_batch=200_000, output_format="csv")

Memory safety: rows are sampled and decoded in batches of `gen_batch` and
streamed to disk, so n_syn=52,000,000 does not allocate one giant tensor.
Only the trained checkpoint's columns are produced; the index/excluded column
(e.g. 媛援ъ씪?⑤쾲?? is never modeled and is not emitted.

seed semantics (agreed with SynPersona): a non-zero integer -> reproducible;
seed == 0 -> fresh randomness each call.
"""

from __future__ import annotations

import argparse, json, os, time

import numpy as np
import pandas as pd
import torch

import _common  # noqa: F401
from _common import peak_rss_gb
from schema import Schema, schema_from_lists
from preprocess import ScalablePreprocessor
from modelio import build_model, load_checkpoint


def _resolve_seed(seed: int) -> int | None:
    return None if seed == 0 else int(seed)


def _sample_batched(model, n: int, gen_batch: int, verbose: bool = False):
    done = 0
    while done < n:
        b = min(gen_batch, n - done)
        yield model.sample(b, verbose=verbose)
        done += b


def generate(checkpoint_dir: str,
             n_syn: int,
             seed: int = 0,
             numeric_vars=None,
             categorical_vars=None,
             exclude_vars=None,
             output_path: str | None = None,
             gen_batch: int = 200_000,
             output_format: str = "csv") -> dict:
    s = _resolve_seed(seed)
    if s is not None:
        np.random.seed(s); torch.manual_seed(s)

    prep = ScalablePreprocessor.load(os.path.join(checkpoint_dir, "preprocessor.pkl"))

    # Optional caller override of the type assignment (SynPersona path). If the
    # caller passes numeric_vars/categorical_vars they must match the trained
    # checkpoint's modeled columns; we validate rather than silently retrain.
    if numeric_vars is not None or categorical_vars is not None:
        req = schema_from_lists(numeric_vars or prep.numeric,
                                categorical_vars or prep.categorical,
                                exclude_vars or [])
        trained = set(prep.schema.modeled_columns)
        asked = set(req.modeled_columns)
        if asked != trained:
            raise ValueError(
                "numeric_vars/categorical_vars do not match the trained checkpoint. "
                f"Missing from checkpoint: {sorted(asked - trained)[:5]}; "
                f"extra in checkpoint: {sorted(trained - asked)[:5]}. "
                "Retrain with this schema to change the modeled columns.")

    with open(os.path.join(checkpoint_dir, "hyperparams.json"), encoding="utf-8") as f:
        hp = json.load(f)
    model = build_model(input_dim=prep.total_dim, hp=hp)
    load_checkpoint(model, os.path.join(checkpoint_dir, "model.pt"))

    if output_path is None:
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"synthetic_{n_syn}.{ 'parquet' if output_format=='parquet' else 'csv'}")

    t0 = time.time()
    written = 0
    header_written = False
    pq_writer = None
    for raw in _sample_batched(model, n_syn, gen_batch, verbose=False):
        df = prep.inverse_transform(raw)
        if output_format == "parquet":
            import pyarrow as pa, pyarrow.parquet as pq
            table = pa.Table.from_pandas(df, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(output_path, table.schema)
            pq_writer.write_table(table)
        else:
            df.to_csv(output_path, index=False, encoding="utf-8-sig",
                      mode="a" if header_written else "w",
                      header=not header_written)
            header_written = True
        written += len(df)
        del df, raw
    if pq_writer is not None:
        pq_writer.close()
    elapsed = time.time() - t0

    return {
        "n_syn": n_syn, "rows_written": written, "seed": seed,
        "reproducible": s is not None,
        "n_modeled_cols": prep.total_dim,
        "n_numeric": prep.num_dim, "n_categorical": prep.cat_dim,
        "gen_batch": gen_batch, "output_path": output_path,
        "seconds_generate": round(elapsed, 2),
        "peak_rss_gb": peak_rss_gb(),
    }


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic rows (SynPersona entry point)")
    ap.add_argument("--checkpoint-dir", default="checkpoints_scale")
    ap.add_argument("--n-syn", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=200_000)
    ap.add_argument("--output", default=None)
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    args = ap.parse_args()
    res = generate(args.checkpoint_dir, args.n_syn, seed=args.seed,
                   output_path=args.output, gen_batch=args.gen_batch,
                   output_format=args.format)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
