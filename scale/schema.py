"""
scale/schema.py - Dictionary-driven variable schema.

A "schema" is the partition of columns into numeric vs categorical, plus the
columns to exclude (e.g. an index/serial column). It can be built from:
  - a column-type dictionary CSV   (col_name, type) with labels in
    {numeric / categorical / index}; Korean labels 수치형 / 범주형 / 인덱스 are
    recognized, English ones too.
  - an explicit pair of lists (numeric_vars, categorical_vars) coming from the
    caller (this is what SynPersona passes per user selection).

The schema never assumes a fixed column count, so it works for the released
27-column extract as well as the 250-column national mock.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Iterable

NUMERIC_LABELS = {"numeric", "수치형", "num", "continuous"}
CATEGORICAL_LABELS = {"categorical", "범주형", "cat", "category", "discrete"}
INDEX_LABELS = {"index", "인덱스", "id", "key"}


@dataclass
class Schema:
    numeric_vars: list[str]
    categorical_vars: list[str]
    exclude_vars: list[str] = field(default_factory=list)

    @property
    def modeled_columns(self) -> list[str]:
        """Columns actually fed to the model: numeric first, then categorical."""
        return list(self.numeric_vars) + list(self.categorical_vars)

    def validate(self, available: Iterable[str] | None = None) -> None:
        num = set(self.numeric_vars)
        cat = set(self.categorical_vars)
        overlap = num & cat
        if overlap:
            raise ValueError(f"Columns typed as BOTH numeric and categorical: {sorted(overlap)}")
        dup_n = len(self.numeric_vars) - len(num)
        dup_c = len(self.categorical_vars) - len(cat)
        if dup_n or dup_c:
            raise ValueError("Duplicate column names within numeric/categorical lists")
        if available is not None:
            avail = set(available)
            missing = (num | cat) - avail
            if missing:
                raise ValueError(f"Schema references columns absent from data: {sorted(missing)[:10]}")

    def summary(self) -> dict:
        return {
            "n_numeric": len(self.numeric_vars),
            "n_categorical": len(self.categorical_vars),
            "n_modeled": len(self.modeled_columns),
            "n_excluded": len(self.exclude_vars),
            "excluded": list(self.exclude_vars),
        }


def _norm(label: str) -> str:
    return (label or "").strip().lower()


def schema_from_dictionary(path: str,
                           name_col: int | str = 0,
                           type_col: int | str = 1,
                           extra_exclude: Iterable[str] | None = None) -> Schema:
    """Build a Schema from a (col_name, type) dictionary CSV.

    Index-typed columns are placed in exclude_vars automatically.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    header = rows[0]
    # Resolve column positions
    def resolve(col, default):
        if isinstance(col, int):
            return col
        try:
            return header.index(col)
        except ValueError:
            return default
    ni = resolve(name_col, 0)
    ti = resolve(type_col, 1)
    for r in rows[1:]:
        if len(r) <= max(ni, ti):
            continue
        name = r[ni].strip()
        label = _norm(r[ti])
        if not name:
            continue
        if label in NUMERIC_LABELS:
            numeric.append(name)
        elif label in CATEGORICAL_LABELS:
            categorical.append(name)
        elif label in INDEX_LABELS:
            excluded.append(name)
        else:
            raise ValueError(f"Unrecognized type label '{r[ti]}' for column '{name}'")
    if extra_exclude:
        for c in extra_exclude:
            if c not in excluded:
                excluded.append(c)
    # Drop any explicitly-excluded names from the modeled lists
    ex = set(excluded)
    numeric = [c for c in numeric if c not in ex]
    categorical = [c for c in categorical if c not in ex]
    sch = Schema(numeric_vars=numeric, categorical_vars=categorical, exclude_vars=excluded)
    sch.validate()
    return sch


def schema_from_lists(numeric_vars: Iterable[str],
                      categorical_vars: Iterable[str],
                      exclude_vars: Iterable[str] | None = None) -> Schema:
    """Build a Schema from caller-supplied lists (SynPersona path)."""
    sch = Schema(
        numeric_vars=list(numeric_vars),
        categorical_vars=list(categorical_vars),
        exclude_vars=list(exclude_vars or []),
    )
    sch.validate()
    return sch


def schema_from_json(path: str) -> Schema:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return schema_from_lists(
        d.get("numeric_vars", []),
        d.get("categorical_vars", []),
        d.get("index_vars", []) + d.get("exclude_vars", []),
    )


def save_schema_json(schema: Schema, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "numeric_vars": schema.numeric_vars,
                "categorical_vars": schema.categorical_vars,
                "exclude_vars": schema.exclude_vars,
            },
            fh, ensure_ascii=False, indent=2,
        )