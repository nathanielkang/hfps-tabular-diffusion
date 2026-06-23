"""
scale/benchmark.py - Gate-stepwise n_syn ladder.

Climbs the ladder of (train_rows, n_syn) tiers, running each in an isolated
subprocess. Records timings + peak RAM per tier into results/scale_ladder.csv
and .json. STOPS at the first hard failure (non-zero exit, OOM kill, or
timeout) and records the feasibility boundary, because the deliverable is
"at what scale does CPU diffusion training/synthesis stop being feasible".

Default ladder mirrors the agency targets:
  2만 / 10만 / 100만 / 700만 / 1600만 / 2000만 / 5200만
(train_rows == n_syn per tier unless overridden with --train-rows-cap).
"""
from __future__ import annotations

import argparse, csv, json, os, subprocess, sys, time

import _common  # noqa: F401
PROJECT_ROOT = _common.PROJECT_ROOT

DEFAULT_LADDER = [20_000, 100_000, 1_000_000, 7_000_000,
                  16_000_000, 20_000_000, 52_000_000]

FIELDS = ["tier", "train_rows", "train_rows_requested", "n_syn", "n_syn_requested",
          "seed", "n_numeric", "n_categorical", "epochs", "seconds_read",
          "seconds_preprocess", "seconds_train", "seconds_generate",
          "peak_rss_gb", "status", "detail"]


def run_tier(args, rows: int, n_syn: int, timeout_s: int) -> dict:
    cmd = [sys.executable, os.path.join("scale", "run_tier.py"),
           "--data", args.data,
           "--train-rows", str(rows), "--n-syn", str(n_syn),
           "--seed", str(args.seed), "--epochs", str(args.epochs),
           "--gen-batch", str(args.gen_batch), "--fit-cap", str(args.fit_cap),
           "--ckpt-dir", args.ckpt_dir, "--out-dir", args.out_dir]
    if args.schema_json:
        cmd += ["--schema-json", args.schema_json]
    else:
        cmd += ["--dictionary", args.dictionary]
        if args.exclude:
            cmd += ["--exclude", *args.exclude]
    if args.smoke:
        cmd += ["--smoke"]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=PROJECT_ROOT)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "detail": f">{timeout_s}s", "n_syn_requested": n_syn,
                "train_rows_requested": rows}
    wall = time.time() - t0

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        # negative return code on POSIX = killed by signal (e.g. -9 OOM)
        kind = "OOM/KILLED" if proc.returncode < 0 else "ERROR"
        return {"status": kind, "detail": f"rc={proc.returncode}; " + " | ".join(tail),
                "n_syn_requested": n_syn, "train_rows_requested": rows,
                "seconds_total": round(wall, 1)}

    for line in (proc.stdout or "").splitlines():
        if line.startswith("TIER_RESULT_JSON "):
            rec = json.loads(line[len("TIER_RESULT_JSON "):])
            rec["status"] = rec.get("status", "OK")
            rec["detail"] = ""
            return rec
    return {"status": "ERROR", "detail": "no result json", "n_syn_requested": n_syn,
            "train_rows_requested": rows}


def main():
    ap = argparse.ArgumentParser(description="Gate-stepwise n_syn ladder")
    ap.add_argument("--data", required=True)
    ap.add_argument("--dictionary")
    ap.add_argument("--schema-json")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--ladder", type=int, nargs="*", default=DEFAULT_LADDER)
    ap.add_argument("--train-rows-cap", type=int, default=None,
                    help="if set, train_rows = min(tier, cap); n_syn still = tier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--gen-batch", type=int, default=200_000)
    ap.add_argument("--fit-cap", type=int, default=2_000_000)
    ap.add_argument("--timeout-hours", type=float, default=12.0)
    ap.add_argument("--ckpt-dir", default="checkpoints_scale")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(PROJECT_ROOT, args.results_dir), exist_ok=True)
    csv_path = os.path.join(PROJECT_ROOT, args.results_dir, "scale_ladder.csv")
    json_path = os.path.join(PROJECT_ROOT, args.results_dir, "scale_ladder.json")
    timeout_s = int(args.timeout_hours * 3600)

    records = []
    boundary = None
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, tier in enumerate(args.ladder, 1):
            rows = tier if args.train_rows_cap is None else min(tier, args.train_rows_cap)
            print(f"[ladder] tier {i}/{len(args.ladder)}: train_rows={rows:,} n_syn={tier:,}", flush=True)
            rec = run_tier(args, rows, tier, timeout_s)
            rec["tier"] = i
            row = {k: rec.get(k, "") for k in FIELDS}
            w.writerow(row); fh.flush()
            records.append(rec)
            print(f"[ladder]   -> {rec.get('status')}  {rec.get('detail','')}", flush=True)
            if rec.get("status") != "OK":
                boundary = {"first_failure_tier": i, "train_rows": rows, "n_syn": tier,
                            "status": rec.get("status"), "detail": rec.get("detail")}
                print(f"[ladder] STOP at first hard failure: {boundary}", flush=True)
                break

    summary = {
        "ladder": args.ladder,
        "records": records,
        "feasibility_boundary": boundary,
        "max_ok_n_syn": max([r["n_syn"] for r in records if r.get("status") == "OK"], default=None),
        "max_ok_train_rows": max([r.get("train_rows", 0) for r in records if r.get("status") == "OK"], default=None),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("[ladder] wrote", csv_path, "and", json_path, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()