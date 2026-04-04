"""Tests for predict.py."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICT_SCRIPT = REPO_ROOT / "prediction-sleep" / "predict.py"
STEP5_SCRIPT = REPO_ROOT / "prediction-sleep" / "scripts" / "step5_train_mobile_models.py"

spec5 = importlib.util.spec_from_file_location("step5_train_mobile_models", STEP5_SCRIPT)
step5 = importlib.util.module_from_spec(spec5)
assert spec5.loader is not None
spec5.loader.exec_module(step5)

spec = importlib.util.spec_from_file_location("predict", PREDICT_SCRIPT)
predict = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(predict)


def _make_fitted_pipeline(feature_cols: list[str]) -> Pipeline:
    rng = np.random.default_rng(42)
    n = 30
    X = pd.DataFrame({col: rng.normal(0, 1, n) for col in feature_cols})
    y = rng.integers(0, 2, n)
    pre = step5.build_preprocessor(feature_cols)
    model = LogisticRegression(max_iter=200, random_state=0)
    pipeline = Pipeline([("preprocess", pre), ("model", model)])
    pipeline.fit(X, y)
    return pipeline


class TestAlignColumns:
    def test_selects_columns_in_correct_order(self):
        df = pd.DataFrame({"b": [1.0], "a": [2.0], "c": [3.0]})
        result = predict.align_columns(df, ["a", "b", "c"])
        assert list(result.columns) == ["a", "b", "c"]

    def test_fills_missing_columns_with_na(self):
        df = pd.DataFrame({"a": [1.0]})
        result = predict.align_columns(df, ["a", "b", "c"])
        assert "b" in result.columns
        assert "c" in result.columns
        assert pd.isna(result["b"].iloc[0])

    def test_handles_extra_columns_gracefully(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "extra": [99.0]})
        result = predict.align_columns(df, ["a", "b"])
        assert list(result.columns) == ["a", "b"]


class TestModelFeatureColumns:
    def test_extracts_feature_columns_from_pipeline(self):
        feature_cols = ["total_sleep_minutes", "rem_minutes", "hr_mean"]
        pipeline = _make_fitted_pipeline(feature_cols)
        result = predict.model_feature_columns(pipeline)
        assert result == feature_cols


class TestResolveModelPath:
    def test_returns_explicit_valid_path(self, tmp_path):
        model_path = tmp_path / "my_model.pkl"
        feature_cols = ["a", "b"]
        pipeline = _make_fitted_pipeline(feature_cols)
        joblib.dump(pipeline, model_path)
        result = predict.resolve_model_path(model_path)
        assert result == model_path

    def test_exits_for_explicit_invalid_path(self, tmp_path):
        with pytest.raises(SystemExit):
            predict.resolve_model_path(tmp_path / "missing_model.pkl")

    def test_exits_when_no_candidate_found(self, tmp_path, monkeypatch):
        # Patch MODEL_CANDIDATES to a list of missing paths
        monkeypatch.setattr(predict, "MODEL_CANDIDATES", [tmp_path / "no_model.pkl"])
        with pytest.raises(SystemExit):
            predict.resolve_model_path(None)

    def test_finds_first_existing_candidate(self, tmp_path, monkeypatch):
        feature_cols = ["a", "b"]
        pipeline = _make_fitted_pipeline(feature_cols)
        first = tmp_path / "first.pkl"
        second = tmp_path / "second.pkl"
        joblib.dump(pipeline, second)
        monkeypatch.setattr(predict, "MODEL_CANDIDATES", [first, second])
        result = predict.resolve_model_path(None)
        assert result == second


class TestLoadFeaturesFromJson:
    def test_loads_dict_json(self, tmp_path):
        feature_cols = ["a", "b"]
        data = {"a": 1.0, "b": 2.0}
        json_path = tmp_path / "features.json"
        json_path.write_text(json.dumps(data))
        result = predict.load_features_from_json(json_path, feature_cols)
        assert list(result.columns) == feature_cols
        assert result["a"].iloc[0] == pytest.approx(1.0)

    def test_loads_list_json(self, tmp_path):
        feature_cols = ["a", "b"]
        data = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        json_path = tmp_path / "features.json"
        json_path.write_text(json.dumps(data))
        result = predict.load_features_from_json(json_path, feature_cols)
        assert len(result) == 2

    def test_fills_missing_features_with_na(self, tmp_path):
        feature_cols = ["a", "b", "c"]
        data = {"a": 1.0}
        json_path = tmp_path / "features.json"
        json_path.write_text(json.dumps(data))
        result = predict.load_features_from_json(json_path, feature_cols)
        assert pd.isna(result["b"].iloc[0])
        assert pd.isna(result["c"].iloc[0])


class TestLoadFeaturesForDate:
    def _make_model_data_csv(self, tmp_path: Path) -> Path:
        df = pd.DataFrame(
            {
                "label_date": ["2026-01-02", "2026-01-03"],
                "night_date": ["2026-01-01", "2026-01-02"],
                "total_sleep_minutes": [420.0, 390.0],
                "fatigue_label": [0, 1],
            }
        )
        path = tmp_path / "model_data.csv"
        df.to_csv(path, index=False)
        return path

    def test_exits_when_date_not_found(self, tmp_path, monkeypatch):
        data_path = self._make_model_data_csv(tmp_path)
        monkeypatch.setattr(predict, "DATA_PATH", data_path)
        with pytest.raises(SystemExit):
            predict.load_features_for_date("2030-01-01", ["total_sleep_minutes"])

    def test_returns_row_for_matching_label_date(self, tmp_path, monkeypatch):
        data_path = self._make_model_data_csv(tmp_path)
        monkeypatch.setattr(predict, "DATA_PATH", data_path)
        result = predict.load_features_for_date("2026-01-02", ["total_sleep_minutes"])
        assert len(result) == 1
        assert result["total_sleep_minutes"].iloc[0] == pytest.approx(420.0)
