"""
preprocessing.py - TabularPreprocessor for mixed numeric / categorical data.

Numeric columns  -> QuantileTransformer -> StandardScaler (force mean=0, std=1)
Categorical cols -> Gaussian quantile encoding (each category -> a N(0,1) quantile)

All 27 columns become 27 continuous dimensions in ~N(0,1) space,
which is ideal for DDPM.
"""

import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from config import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, COLUMN_ORDER


class TabularPreprocessor:
    """Fit on real data, transform to diffusion-ready float array, invert back."""

    def __init__(self):
        self.qt = None
        self.scaler = None
        self.cat_encodings = {}   # col -> {value: float}
        self.cat_decodings = {}   # col -> sorted list of (encoded_val, original_val)
        self.num_dim = len(NUMERIC_COLUMNS)
        self.cat_dim = len(CATEGORICAL_COLUMNS)
        self.total_dim = self.num_dim + self.cat_dim  # 27
        self.num_min = {}
        self.num_max = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        assert list(df.columns) == COLUMN_ORDER, "Column mismatch"

        # --- Numeric: QuantileTransformer + StandardScaler ---
        num_data = df[NUMERIC_COLUMNS].values.astype(np.float64)
        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(len(df), 2000),
            random_state=42,
            subsample=len(df),
        )
        self.qt.fit(num_data)

        qt_out = self.qt.transform(num_data)
        qt_out = np.clip(qt_out, -4.5, 4.5)

        self.scaler = StandardScaler()
        self.scaler.fit(qt_out)

        self.num_min = {col: float(df[col].min()) for col in NUMERIC_COLUMNS}
        self.num_max = {col: float(df[col].max()) for col in NUMERIC_COLUMNS}

        # --- Categorical: Gaussian quantile encoding ---
        for col in CATEGORICAL_COLUMNS:
            unique_vals = sorted(df[col].unique(), key=str)
            K = len(unique_vals)
            quantile_vals = stats.norm.ppf((np.arange(K) + 0.5) / K)
            encoding = {v: float(quantile_vals[i]) for i, v in enumerate(unique_vals)}
            self.cat_encodings[col] = encoding
            self.cat_decodings[col] = sorted(
                [(float(quantile_vals[i]), v) for i, v in enumerate(unique_vals)]
            )

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        n = len(df)
        out = np.zeros((n, self.total_dim), dtype=np.float32)

        num_data = df[NUMERIC_COLUMNS].values.astype(np.float64)
        qt_out = self.qt.transform(num_data)
        qt_out = np.clip(qt_out, -4.5, 4.5)
        out[:, :self.num_dim] = self.scaler.transform(qt_out).astype(np.float32)

        for j, col in enumerate(CATEGORICAL_COLUMNS):
            enc = self.cat_encodings[col]
            out[:, self.num_dim + j] = df[col].map(enc).values.astype(np.float32)

        return out

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        assert self._fitted, "Call fit() first"
        result = {}

        # Numeric: inverse StandardScaler -> clip -> inverse QuantileTransformer -> clip to data range
        num_arr = arr[:, :self.num_dim].astype(np.float64)
        num_arr = self.scaler.inverse_transform(num_arr)
        num_arr = np.clip(num_arr, -4.5, 4.5)
        num_inv = self.qt.inverse_transform(num_arr)

        for i, col in enumerate(NUMERIC_COLUMNS):
            vals = num_inv[:, i]
            vals = np.clip(vals, self.num_min[col], self.num_max[col])
            result[col] = vals

        # Categorical: snap to nearest quantile value
        for j, col in enumerate(CATEGORICAL_COLUMNS):
            raw_vals = arr[:, self.num_dim + j]
            dec_list = self.cat_decodings[col]
            centers = np.array([d[0] for d in dec_list])
            originals = [d[1] for d in dec_list]
            # Vectorized nearest-neighbor decode
            dists = np.abs(raw_vals[:, None] - centers[None, :])
            indices = np.argmin(dists, axis=1)
            result[col] = [originals[idx] for idx in indices]

        df_out = pd.DataFrame(result)
        df_out = df_out[COLUMN_ORDER]

        for col in NUMERIC_COLUMNS:
            vals = df_out[col].values.astype(np.float64)
            vals = np.round(vals).astype(np.int64)
            df_out[col] = vals

        return df_out

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TabularPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
