"""
Evaluation configuration: seeds, splits, task definitions, paths.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from config import COLUMN_ORDER, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, DATA_CSV, DATA_ENCODING

REAL_CSV = DATA_CSV
ENCODING = DATA_ENCODING
OURS_CSV = os.path.join(PROJECT_ROOT, "output", "synthetic.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
SPLIT_RATIOS = (0.70, 0.15, 0.15)

CLF_TARGET = "소득계층구간코드(보도용)(보완)"
REG_TARGET = "경상소득(보완)"

N_REAL = 18314

CTGAN_EPOCHS = 300
TVAE_EPOCHS = 300
BASELINE_BATCH_SIZE = 500
