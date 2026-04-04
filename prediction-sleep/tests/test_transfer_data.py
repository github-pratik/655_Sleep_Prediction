"""Tests for src/transfer/data.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "prediction-sleep" / "src"))

from transfer.data import (
    infer_sort_column,
    load_model_data,
    numeric_feature_frame,
)


class TestInferSortColumn:
    def test_prefers_label_date_over_others(self):
        df = pd.DataFrame({"label_date": [], "night_date": [], "date": []})
        assert infer_sort_column(df) == "label_date"

    def test_falls_back_to_night_date(self):
        df = pd.DataFrame({"night_date": [], "date": []})
        assert infer_sort_column(df) == "night_date"

    def test_falls_back_to_date(self):
        df = pd.DataFrame({"date": [], "total_sleep_minutes": []})
        assert infer_sort_column(df) == "date"

    def test_falls_back_to_sleep_start(self):
        df = pd.DataFrame({"sleep_start": [], "total_sleep_minutes": []})
        assert infer_sort_column(df) == "sleep_start"

    def test_returns_none_when_no_date_column(self):
        df = pd.DataFrame({"a": [], "b": []})
        assert infer_sort_column(df) is None


class TestNumericFeatureFrame:
    def test_excludes_fatigue_label(self):
        df = pd.DataFrame({"total_sleep_minutes": [400.0], "fatigue_label": [1]})
        frame, cols = numeric_feature_frame(df)
        assert "fatigue_label" not in cols
        assert "fatigue_label" not in frame.columns

    def test_excludes_date_metadata_columns(self):
        df = pd.DataFrame(
            {
                "night_date": ["2026-01-01"],
                "label_date": ["2026-01-02"],
                "sleep_start": ["2026-01-01 22:00:00"],
                "sleep_end": ["2026-01-02 06:00:00"],
                "date": ["2026-01-02"],
                "total_sleep_minutes": [420.0],
            }
        )
        frame, cols = numeric_feature_frame(df)
        for meta in ("night_date", "label_date", "sleep_start", "sleep_end", "date"):
            assert meta not in cols
            assert meta not in frame.columns

    def test_returns_numeric_columns_as_float(self):
        df = pd.DataFrame(
            {
                "total_sleep_minutes": [420.5, 390.5],
                "rem_minutes": [90.5, 80.5],
                "fatigue_label": [0, 1],
            }
        )
        frame, cols = numeric_feature_frame(df)
        assert list(cols) == ["total_sleep_minutes", "rem_minutes"]
        assert pd.api.types.is_float_dtype(frame["total_sleep_minutes"])

    def test_returns_correct_shape(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0],
                "b": [3.0, 4.0],
                "fatigue_label": [0, 1],
                "night_date": ["2026-01-01", "2026-01-02"],
            }
        )
        frame, cols = numeric_feature_frame(df)
        assert frame.shape == (2, 2)
        assert set(cols) == {"a", "b"}


class TestLoadModelData:
    def test_exits_when_file_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            load_model_data(tmp_path / "nonexistent.csv")

    def test_sorts_by_night_date(self, tmp_path):
        df = pd.DataFrame(
            {
                "night_date": ["2026-01-05", "2026-01-01", "2026-01-03"],
                "total_sleep_minutes": [420.0, 390.0, 410.0],
                "fatigue_label": [0, 1, 0],
            }
        )
        path = tmp_path / "model_data.csv"
        df.to_csv(path, index=False)
        result = load_model_data(path)
        dates = result["night_date"].tolist()
        assert dates == sorted(dates)

    def test_parses_date_columns_as_datetime(self, tmp_path):
        df = pd.DataFrame(
            {
                "night_date": ["2026-01-01", "2026-01-02"],
                "total_sleep_minutes": [420.0, 390.0],
                "fatigue_label": [0, 1],
            }
        )
        path = tmp_path / "model_data.csv"
        df.to_csv(path, index=False)
        result = load_model_data(path)
        assert pd.api.types.is_datetime64_any_dtype(result["night_date"])

    def test_resets_index_after_sort(self, tmp_path):
        df = pd.DataFrame(
            {
                "night_date": ["2026-01-03", "2026-01-01", "2026-01-02"],
                "fatigue_label": [0, 1, 0],
                "x": [1.0, 2.0, 3.0],
            }
        )
        path = tmp_path / "model_data.csv"
        df.to_csv(path, index=False)
        result = load_model_data(path)
        assert list(result.index) == list(range(len(result)))
