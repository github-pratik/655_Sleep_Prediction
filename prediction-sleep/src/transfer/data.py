from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

TARGET_COL = "fatigue_label"
DATE_COLS = ("label_date", "night_date", "sleep_start", "sleep_end", "date")
METADATA_COLS = set(DATE_COLS) | {TARGET_COL}


def _parse_dates(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def infer_sort_column(df: pd.DataFrame) -> str | None:
    for col in ("label_date", "night_date", "date", "sleep_start"):
        if col in df.columns:
            return col
    return None


def load_model_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Required dataset not found: {path}")

    df = pd.read_csv(path)
    df = _parse_dates(df, DATE_COLS)

    sort_col = infer_sort_column(df)
    if sort_col is not None:
        df = df.sort_values(sort_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def numeric_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    out = df[feature_cols].copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out, feature_cols
