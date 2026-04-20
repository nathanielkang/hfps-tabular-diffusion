# Inter-Record Distance Metrics

Pairwise-distance evaluation tools for comparing real and synthetic tabular datasets. **Not coupled to the TabOversample–HFPS diffusion pipeline** — usable with any pair of CSV files that share the same schema.

## Metrics

| Metric | What it measures | Ideal value |
|--------|-----------------|-------------|
| **SNN loss** (Similarity of Nearest Neighbors) | Whether real and synthetic records mix evenly in k-NN space | SNN_mean ≈ 0.5 |
| **DCR** (Distance to Closest Record) | How far each synthetic record is from its nearest real record | DCR ratio ≈ 1.0 |

## Full math

See **[docs/METRICS.md](docs/METRICS.md)** for complete definitions, equations, interpretation tables, and references.

## Quick start

```bash
# Both metrics at once
python run_metrics.py --real path/to/real.csv --synth path/to/synth.csv

# Individual
python snn_loss.py --real real.csv --synth synth.csv --k 5
python dcr.py      --real real.csv --synth synth.csv
```

## Requirements

Only standard scientific Python: `numpy`, `pandas`, `scikit-learn` (already in the project `requirements.txt`).
