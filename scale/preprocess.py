"""
scale/preprocess.py - ScalablePreprocessor (dictionary-driven, memory-safe).

Same encoding scheme as the released TabularPreprocessor:
  numeric     -> QuantileTransformer(normal) -> StandardScaler  (~N(0,1))
  categorical -> Gaussian quantile encoding (each level -> a N(0,1) quantile)

Differences for scale:
  - column lists come from a Schema (any number of columns), not hard-coded 27;
  - the fit can be capped to a subsample (fit_cap) so quantile estimation does
    not itself blow up memory while the DDPM still trains on the full slice;
  - inverse_transform decodes categoricals in row-batches to avoid an
    (n_rows x n_levels) distance matrix.
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from schema import Schema


class ScalablePreprocessor:
    def __init__(self, schema: Schema, fit_cap: int | None = 2_000_000):
        self.schema = schema
        self.fit_cap = fit_cap
        self.numeric = list(schema.numeric_vars)
        self.categorical = list(schema.categorical_vars)
        self.num_dim = len(self.numeric)
        self.cat_dim = len(self.categorical)
        self.total_dim = self.num_dim + self.cat_dim
        self.qt = None
        self.scaler = None
        self.cat_centers: dict[str, np.ndarray] = {}
        self.cat_levels: dict[str, list] = {}
        self.cat_enc: dict[str, dict] = {}
        self.num_min: dict[str, float] = {}
        self.num_max: dict[str, float] = {}
        self.num_is_int: dict[str, bool] = {}
        self._fitted = False

    # -- fit -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "ScalablePreprocessor":
        fit_df = df
        if self.fit_cap is not None and len(df) > self.fit_cap:
            fit_df = df.iloc[: self.fit_cap]

        if self.num_dim:
            num_data = fit_df[self.numeric].to_numpy(dtype=np.float64, copy=False)
            self.qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(len(fit_df), 2000),
                subsample=min(len(fit_df), 1_000_000),
                random_state=42,
            )
            self.qt.fit(num_data)
            qt_out = np.clip(self.qt.transform(num_data), -4.5, 4.5)
            self.scaler = StandardScaler().fit(qt_out)
            del num_data, qt_out
            for col in self.numeric:
                s = df[col]
                self.num_min[col] = float(s.min())
                self.num_max[col] = float(s.max())
                self.num_is_int[col] = bool(pd.api.types.is_integer_dtype(s.dtype))

        for col in self.categorical:
            levels = sorted(df[col].dropna().unique(), key=str)
            K = max(len(levels), 1)
            q = stats.norm.ppf((np.arange(K) + 0.5) / K)
            self.cat_levels[col] = list(levels)
            self.cat_centers[col] = q.astype(np.float64)
            self.cat_enc[col] = {v: float(q[i]) for i, v in enumerate(levels)}

        self._fitted = True
        return self

    # -- transform -----------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "call fit() first"
        n = len(df)
        out = np.zeros((n, self.total_dim), dtype=np.float32)
        if self.num_dim:
            num = df[self.numeric].to_numpy(dtype=np.float64, copy=False)
            qt_out = np.clip(self.qt.transform(num), -4.5, 4.5)
            out[:, : self.num_dim] = self.scaler.transform(qt_out).astype(np.float32)
            del num, qt_out
        for j, col in enumerate(self.categorical):
            out[:, self.num_dim + j] = df[col].map(self.cat_enc[col]).fillna(0.0).to_numpy(np.float32)
        return out

    # -- inverse -------------------------------------------------------------
    def inverse_transform(self, arr: np.ndarray, batch: int = 200_000) -> pd.DataFrame:
        assert self._fitted, "call fit() first"
        result: dict[str, np.ndarray | list] = {}

        if self.num_dim:
            num = arr[:, : self.num_dim].astype(np.float64)
            num = np.clip(self.scaler.inverse_transform(num), -4.5, 4.5)
            num = self.qt.inverse_transform(num)
            for i, col in enumerate(self.numeric):
                v = np.clip(num[:, i], self.num_min[col], self.num_max[col])
                if self.num_is_int.get(col, False):
                    v = np.round(v).astype(np.int64)
                result[col] = v
            del num

        n = arr.shape[0]
        for j, col in enumerate(self.categorical):
            centers = self.cat_centers[col]
            levels = self.cat_levels[col]
            raw = arr[:, self.num_dim + j]
            idx = np.empty(n, dtype=np.int64)
            for s in range(0, n, batch):
                e = min(s + batch, n)
                d = np.abs(raw[s:e, None] - centers[None, :])
                idx[s:e] = np.argmin(d, axis=1)
            result[col] = [levels[k] for k in idx]

        return pd.DataFrame(result, columns=self.schema.modeled_columns)

    # -- io ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ScalablePreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)