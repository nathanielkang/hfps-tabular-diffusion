"""
scale/run_tier.py - Run ONE ladder tier (train + generate) in its own process.

Isolating each tier in a subprocess means an OOM-kill at, say, 16M rows does not
take down the whole benchmark; the parent records the failure and stops.

Prints a single JSON object to stdout on success.
"""
from __future__ import annotations

import argparse, json, os, time

import _common  # noqa: F401
from _common import peak_rss_gb
from schema import schema_from_dictionary, schema_from_json
import train_scale as ts
import generate as gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dictionary")
    ap.add_argument("--schema-json")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--train-rows", type=int, required=True)
    ap.add_argument("--n-syn", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--gen-batch", type=int, default=200_000)
    ap.add_argument("--fit-cap", type=int, default=2_000_000)
    ap.add_argument("--ckpt-dir", default="checkpoints_scale")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.schema_json:
        schema = schema_from_json(args.schema_json)
    else:
        schema = schema_from_dictionary(args.dictionary, extra_exclude=args.exclude)

    hp = dict(ts.SMOKE_HP if args.smoke else ts.SCALE_HP)
    hp["epochs"] = args.epochs

    ckpt = os.path.join(args.ckpt_dir, f"rows_{args.train_rows}")
    train_summary = ts.train_scale(args.data, schema, args.train_rows, hp, ckpt,
                                   seed=args.seed, fit_cap=args.fit_cap)

    out_path = os.path.join(args.out_dir, f"syn_rows{args.train_rows}_n{args.n_syn}.csv")
    gen_summary = gen.generate(ckpt, args.n_syn, seed=args.seed,
                               output_path=out_path, gen_batch=args.gen_batch)

    rec = {
        "train_rows": train_summary["train_rows_actual"],
        "train_rows_requested": train_summary["train_rows_requested"],
        "n_syn": gen_summary["rows_written"],
        "n_syn_requested": args.n_syn,
        "seed": args.seed,
        "n_numeric": train_summary["n_numeric"],
        "n_categorical": train_summary["n_categorical"],
        "epochs": hp["epochs"],
        "seconds_read": train_summary["seconds_read"],
        "seconds_preprocess": train_summary["seconds_preprocess"],
        "seconds_train": train_summary["seconds_train"],
        "seconds_generate": gen_summary["seconds_generate"],
        "peak_rss_gb": peak_rss_gb(),
        "status": "OK",
    }
    print("TIER_RESULT_JSON " + json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()