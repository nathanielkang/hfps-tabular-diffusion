"""
scale/data.py - Parquet streaming + bounded training subsample.

Goals:
  - Never load the full 52M-row table into RAM just to count or sample it.
  - Provide a chunked iterator (Phase A feasibility: prove the data path works).
  - Provide a bounded "training slice" of N rows over the modeled columns only.

Row counting uses Parquet metadata (O(1) in data size). Chunk iteration uses
pyarrow's batched reader so peak memory is ~one batch, not the whole file.
"""

from __future__ import annotations

import os
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
    _HAVE_PARQUET = True
except Exception:  # pragma: no cover
    _HAVE_PARQUET = False


def have_parquet() -> bool:
    return _HAVE_PARQUET


def count_rows(path: str) -> int:
    """Number of rows via Parquet footer metadata (no data scan)."""
    if not _HAVE_PARQUET:
        raise RuntimeError("pyarrow is required for Parquet support")
    return pq.ParquetFile(path).metadata.num_rows


def file_columns(path: str) -> list[str]:
    if not _HAVE_PARQUET:
        raise RuntimeError("pyarrow is required for Parquet support")
    return list(pq.ParquetFile(path).schema.names)


def iter_chunks(path: str,
                columns: Sequence[str] | None = None,
                batch_size: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield the file in row batches as pandas DataFrames (bounded memory)."""
    if not _HAVE_PARQUET:
        raise RuntimeError("pyarrow is required for Parquet support")
    pf = pq.ParquetFile(path)
    cols = list(columns) if columns is not None else None
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield batch.to_pandas()


def streaming_read_probe(path: str,
                         columns: Sequence[str] | None = None,
                         batch_size: int = 200_000,
                         max_batches: int | None = None) -> dict:
    """Phase A feasibility probe: stream the whole file in batches, touching
    each batch, and report rows seen + peak batch shape without holding the
    whole table. Returns a small dict; raises on read error (= infeasible)."""
    rows_seen = 0
    n_batches = 0
    ncols = None
    for df in iter_chunks(path, columns=columns, batch_size=batch_size):
        rows_seen += len(df)
        ncols = df.shape[1]
        n_batches += 1
        # touch the data so a lazy reader actually materializes it
        _ = df.to_numpy(copy=False) if df.shape[1] else None
        if max_batches is not None and n_batches >= max_batches:
            break
    return {"rows_seen": rows_seen, "n_batches": n_batches, "n_cols": ncols,
            "batch_size": batch_size}


def read_training_slice(path: str,
                        n_rows: int,
                        columns: Sequence[str] | None = None,
                        batch_size: int = 200_000) -> pd.DataFrame:
    """Load the first n_rows rows over the requested columns into a DataFrame.

    Loading n_rows x len(columns) is THE memory event for large tiers; this is
    intentional - it is the feasibility boundary we are measuring. Use only the
    modeled columns to avoid wasting memory on excluded/index columns.
    """
    if not _HAVE_PARQUET:
        raise RuntimeError("pyarrow is required for Parquet support")
    parts: list[pd.DataFrame] = []
    got = 0
    for df in iter_chunks(path, columns=columns, batch_size=batch_size):
        if got + len(df) > n_rows:
            df = df.iloc[: n_rows - got]
        parts.append(df)
        got += len(df)
        if got >= n_rows:
            break
    if not parts:
        return pd.DataFrame(columns=list(columns) if columns else None)
    out = pd.concat(parts, ignore_index=True)
    return out


def write_mock_parquet(path: str,
                       n_rows: int,
                       numeric_vars: Sequence[str],
                       categorical_vars: Sequence[str],
                       index_var: str | None = None,
                       n_cat_levels: int = 6,
                       seed: int = 0) -> str:
    """Create a small synthetic Parquet that mimics the national mock's shape,
    for local smoke tests (no access to the real 11 GB file needed)."""
    if not _HAVE_PARQUET:
        raise RuntimeError("pyarrow is required for Parquet support")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    if index_var:
        data[index_var] = np.arange(n_rows, dtype=np.int64)
    for c in numeric_vars:
        data[c] = rng.normal(loc=rng.uniform(-2, 2), scale=rng.uniform(0.5, 3.0),
                             size=n_rows).astype(np.float32)
    for c in categorical_vars:
        k = rng.integers(2, n_cat_levels + 1)
        data[c] = rng.integers(0, k, size=n_rows).astype(np.int64)
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    return path