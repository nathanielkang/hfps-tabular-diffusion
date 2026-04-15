"""
config.py - Schema definition, hyperparameters, and paths for TabOversample–HFPS
(HFPS 2024 survey extract).
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CSV = os.path.join(PROJECT_ROOT, "2024_가계금융복지조사_final.csv")
DATA_ENCODING = "utf-8-sig"

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

UAI_CODE_DIR = os.path.join(PROJECT_ROOT, "UAI", "supplementary", "code")

# ---------------------------------------------------------------------------
# Schema: 27 columns in exact delivery order
# ---------------------------------------------------------------------------

COLUMN_ORDER = [
    "수도권여부",
    "가구주_성별코드",
    "가구원수",
    "노인가구여부",
    "한부모가구여부",
    "가구주_교육정도_학력코드",
    "가구주_동거여부",
    "가구주_만연령",
    "가구주_종사상지위(보도용)",
    "가구주_혼인상태코드",
    "입주형태통합코드",
    "주택종류통합코드",
    "소득계층구간코드(보도용)(보완)",
    "자산총액5분위코드",
    "순자산5분위코드",
    "자산",
    "자산_금융자산",
    "자산_실물자산",
    "부채",
    "원리금상환금액",
    "순자산",
    "경상소득(보완)",
    "지출_소비지출비",
    "여유자금운용계획코드",
    "거주지주택가격전망코드",
    "가구주_은퇴여부",
    "가구주_미은퇴_적정생활비",
]

NUMERIC_COLUMNS = [
    "가구원수",
    "가구주_만연령",
    "자산",
    "자산_금융자산",
    "자산_실물자산",
    "부채",
    "원리금상환금액",
    "순자산",
    "경상소득(보완)",
    "지출_소비지출비",
    "가구주_미은퇴_적정생활비",
]

CATEGORICAL_COLUMNS = [
    c for c in COLUMN_ORDER if c not in NUMERIC_COLUMNS
]

assert len(COLUMN_ORDER) == 27
assert len(NUMERIC_COLUMNS) == 11
assert len(CATEGORICAL_COLUMNS) == 16

# ---------------------------------------------------------------------------
# Model hyperparameters (full training)
# ---------------------------------------------------------------------------

FULL_HP = {
    "n_timesteps": 1000,
    "hidden_dim": 512,
    "n_layers": 6,
    "schedule": "cosine",
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 1500,
    "dropout": 0.0,
}

# ---------------------------------------------------------------------------
# Smoke-test hyperparameters (tiny run)
# ---------------------------------------------------------------------------

SMOKE_HP = {
    "n_timesteps": 50,
    "hidden_dim": 64,
    "n_layers": 2,
    "schedule": "cosine",
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 2,
    "dropout": 0.0,
}

N_ROWS = 18314
N_COLS = 27
