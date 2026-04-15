"""
CLI entrypoint: run full evaluation pipeline.

Usage:
    python run_all.py                 # full evaluation
    python run_all.py --smoke         # smoke test on 200 rows
    python run_all.py --skip-baselines  # skip CTGAN/TVAE training
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_eval import RESULTS_DIR, OURS_CSV, ENCODING, COLUMN_ORDER
from load_data import load_real, load_synthetic
from fidelity import run_fidelity
from detection import run_detection
from tstr import run_tstr, run_trtr
from baselines import train_ctgan, train_tvae


def main():
    parser = argparse.ArgumentParser(description="TabOversample–HFPS evaluation pipeline")
    parser.add_argument("--smoke", action="store_true", help="Smoke test with 200 rows")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip CTGAN/TVAE training")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("TabOversample–HFPS Synthetic Data Evaluation Pipeline")
    print("=" * 60)

    real_df = load_real()
    print(f"Real data loaded: {real_df.shape}")

    if args.smoke:
        print("[SMOKE MODE] Subsetting to 200 rows")
        real_df = real_df.head(200).reset_index(drop=True)

    ours_df = load_synthetic(OURS_CSV)
    print(f"Ours synthetic loaded: {ours_df.shape}")
    if args.smoke:
        ours_df = ours_df.head(200).reset_index(drop=True)

    all_fidelity = []
    all_detection = []
    all_tstr = []

    print("\n--- Tier A: Fidelity ---")
    fid_ours = run_fidelity(real_df, ours_df, label="ours")
    all_fidelity.append(fid_ours)

    print("\n--- Tier B: Detection ---")
    det_ours = run_detection(real_df, ours_df, label="ours")
    all_detection.append(det_ours)

    print("\n--- Tier C: TSTR (Ours) ---")
    tstr_ours = run_tstr(real_df, ours_df, label="ours")
    all_tstr.append(tstr_ours)

    print("\n--- Tier C: TRTR (Upper Bound) ---")
    trtr = run_trtr(real_df)
    all_tstr.append(trtr)

    if not args.skip_baselines:
        n_gen = len(real_df)

        print("\n--- Training CTGAN ---")
        ctgan_df = train_ctgan(real_df, n_samples=n_gen)
        ctgan_df = ctgan_df[COLUMN_ORDER]

        print("\n--- Evaluating CTGAN ---")
        fid_ctgan = run_fidelity(real_df, ctgan_df, label="ctgan")
        all_fidelity.append(fid_ctgan)
        det_ctgan = run_detection(real_df, ctgan_df, label="ctgan")
        all_detection.append(det_ctgan)
        tstr_ctgan = run_tstr(real_df, ctgan_df, label="ctgan")
        all_tstr.append(tstr_ctgan)

        print("\n--- Training TVAE ---")
        tvae_df = train_tvae(real_df, n_samples=n_gen)
        tvae_df = tvae_df[COLUMN_ORDER]

        print("\n--- Evaluating TVAE ---")
        fid_tvae = run_fidelity(real_df, tvae_df, label="tvae")
        all_fidelity.append(fid_tvae)
        det_tvae = run_detection(real_df, tvae_df, label="tvae")
        all_detection.append(det_tvae)
        tstr_tvae = run_tstr(real_df, tvae_df, label="tvae")
        all_tstr.append(tstr_tvae)

    combined = {
        "fidelity": all_fidelity,
        "detection": all_detection,
        "tstr": all_tstr,
    }
    combined_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"All evaluation complete in {elapsed:.0f}s")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60)

    print_summary(combined)


def print_summary(results):
    print("\n### FIDELITY SUMMARY ###")
    print(f"{'Method':<12} {'Avg KS':>8} {'Avg TVD':>9} {'Corr Diff':>10}")
    print("-" * 42)
    for r in results["fidelity"]:
        print(f"{r['method']:<12} {r['avg_ks']:>8.4f} {r['avg_tvd']:>9.4f} {r['corr_frobenius']:>10.4f}")

    print("\n### DETECTION SUMMARY ###")
    print(f"{'Method':<12} {'Acc':>12} {'AUC':>12}")
    print("-" * 38)
    for r in results["detection"]:
        acc_str = f"{r['detection_accuracy_mean']:.4f}+/-{r['detection_accuracy_std']:.4f}"
        auc_str = f"{r['detection_auc_mean']:.4f}+/-{r['detection_auc_std']:.4f}"
        print(f"{r['method']:<12} {acc_str:>12} {auc_str:>12}")

    print("\n### TSTR CLASSIFICATION SUMMARY ###")
    print(f"{'Method':<15} {'Accuracy':>16} {'Macro-F1':>16}")
    print("-" * 50)
    for r in results["tstr"]:
        c = r["classification"]
        acc_str = f"{c['accuracy_mean']:.4f}+/-{c['accuracy_std']:.4f}"
        f1_str = f"{c['macro_f1_mean']:.4f}+/-{c['macro_f1_std']:.4f}"
        print(f"{r['method']:<15} {acc_str:>16} {f1_str:>16}")

    print("\n### TSTR REGRESSION SUMMARY ###")
    print(f"{'Method':<15} {'RMSE':>16} {'R2':>16}")
    print("-" * 50)
    for r in results["tstr"]:
        reg = r["regression"]
        rmse_str = f"{reg['rmse_mean']:.2f}+/-{reg['rmse_std']:.2f}"
        r2_str = f"{reg['r2_mean']:.4f}+/-{reg['r2_std']:.4f}"
        print(f"{r['method']:<15} {rmse_str:>16} {r2_str:>16}")


if __name__ == "__main__":
    main()
