"""
Convenience script: compute both SNN loss and DCR in one shot.

Usage
-----
    python run_metrics.py --real path/to/real.csv --synth path/to/synth.csv

Optional flags: --k, --subsample, --seed, --encoding, --output-dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snn_loss import snn_loss
from dcr import dcr


def main():
    parser = argparse.ArgumentParser(
        description="Compute SNN loss and DCR between real and synthetic CSV."
    )
    parser.add_argument("--real", required=True, help="Path to real data CSV")
    parser.add_argument("--synth", required=True, help="Path to synthetic data CSV")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--k", type=int, default=5, help="k for SNN (default: 5)")
    parser.add_argument("--subsample", type=int, default=5000, help="0 = no subsample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None, help="Directory to save JSON results")
    args = parser.parse_args()

    real_df = pd.read_csv(args.real, encoding=args.encoding)
    synth_df = pd.read_csv(args.synth, encoding=args.encoding)

    num_cols = real_df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in real_df.columns if c not in num_cols]

    sub = args.subsample if args.subsample > 0 else None

    print("=" * 60)
    print("  Inter-Record Metrics: SNN Loss + DCR")
    print("=" * 60)

    print("\n[1/2] Computing SNN loss ...")
    snn_result = snn_loss(real_df, synth_df, num_cols, cat_cols, k=args.k, subsample=sub, seed=args.seed)
    print(json.dumps(snn_result, indent=2))

    print("\n[2/2] Computing DCR ...")
    dcr_result = dcr(real_df, synth_df, num_cols, cat_cols, subsample=sub, seed=args.seed)
    print(json.dumps(dcr_result, indent=2))

    combined = {"snn": snn_result, "dcr": dcr_result}

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "inter_record_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"\nSaved combined results -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
