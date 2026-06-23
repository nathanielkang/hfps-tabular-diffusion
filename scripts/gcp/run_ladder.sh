#!/usr/bin/env bash
# run_ladder.sh - download data (if DRIVE_ID set) and run the gate-stepwise ladder in tmux.
# Usage on VM:
#   cd ~/hfps && source ~/venv/bin/activate
#   DRIVE_ID=<google_drive_file_id> bash scripts/gcp/run_ladder.sh        # real data
#   SYNTH_ROWS=52000000 bash scripts/gcp/run_ladder.sh                    # shape-faithful synthetic
set -euo pipefail
cd ~/hfps
DICT="configs/mock_dictionary.csv"
DATA="data/mock.parquet"
mkdir -p data results

if [[ -n "${DRIVE_ID:-}" ]]; then
  echo "[run] downloading real mock from Drive id=$DRIVE_ID"
  gdown "https://drive.google.com/uc?id=${DRIVE_ID}" -O "$DATA"
elif [[ -n "${SYNTH_ROWS:-}" ]]; then
  echo "[run] building shape-faithful synthetic mock with $SYNTH_ROWS rows"
  python - "$DICT" "$DATA" "$SYNTH_ROWS" <<'PY'
import sys
sys.path.insert(0, "scale")
import schema, data
dict_path, out_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = schema.schema_from_dictionary(dict_path)
idx = s.exclude_vars[0] if s.exclude_vars else None
# write in row-group chunks to bound memory
import pyarrow as pa, pyarrow.parquet as pq, numpy as np, pandas as pd
rng = np.random.default_rng(0)
CHUNK = 1_000_000
writer = None
written = 0
while written < n:
    b = min(CHUNK, n - written)
    cols = {}
    if idx: cols[idx] = np.arange(written, written+b, dtype=np.int64)
    for c in s.numeric_vars: cols[c] = rng.normal(0,1,b).astype(np.float32)
    for c in s.categorical_vars: cols[c] = rng.integers(0,6,b).astype(np.int64)
    t = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
    if writer is None: writer = pq.ParquetWriter(out_path, t.schema)
    writer.write_table(t); written += b
    print(f"  wrote {written:,}/{n:,}", flush=True)
writer.close()
print("done", out_path)
PY
else
  echo "ERROR: set DRIVE_ID=<id> (real data) or SYNTH_ROWS=<n> (synthetic shape)"; exit 1
fi

echo "[run] launching ladder in tmux session 'ladder'"
tmux new -d -s ladder "source ~/venv/bin/activate && python scale/benchmark.py \
  --data $DATA --dictionary $DICT \
  --ladder 20000 100000 1000000 7000000 16000000 20000000 52000000 \
  --epochs 200 --gen-batch 200000 --fit-cap 2000000 --timeout-hours 12 \
  2>&1 | tee results/ladder_run.log"
echo "attach with: tmux attach -t ladder    | results in results/scale_ladder.{csv,json}"