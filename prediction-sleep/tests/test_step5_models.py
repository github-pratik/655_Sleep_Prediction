"""Tests for scripts/step5_train_mobile_models.py – pure-function unit tests."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP5_SCRIPT = REPO_ROOT / "prediction-sleep" / "scripts" / "step5_train_mobile_models.py"

spec = importlib.util.spec_from_file_location("step5_train_mobile_models", STEP5_SCRIPT)
step5 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(step5)


def _make_train_test_csvs(tmp_path: Path, n: int = 20):
    feature_cols = ["total_sleep_minutes", "rem_minutes", "hr_mean"]
    rng = np.random.default_rng(42)
    train_df = pd.DataFrame(
        {
            "total_sleep_minutes": rng.normal(420, 30, n),
            "rem_minutes": rng.normal(90, 15, n),
            "hr_mean": rng.normal(58, 5, n),
            "fatigue_label": rng.integers(0, 2, n),
        }
    )
    test_df = pd.DataFrame(
        {
            "total_sleep_minutes": rng.normal(420, 30, 6),
            "rem_minutes": rng.normal(90, 15, 6),
            "hr_mean": rng.normal(58, 5, 6),
            "fatigue_label": rng.integers(0, 2, 6),
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path, feature_cols


class TestLoadData:
    def test_returns_correct_shapes(self, tmp_path):
        train_path, test_path, feature_cols = _make_train_test_csvs(tmp_path)
        cols, X_train, X_test, y_train, y_test = step5.load_data(train_path, test_path)
        assert X_train.shape[1] == len(feature_cols)
        assert len(y_train) == 20
        assert len(y_test) == 6

    def test_feature_cols_excludes_target(self, tmp_path):
        train_path, test_path, _ = _make_train_test_csvs(tmp_path)
        cols, *_ = step5.load_data(train_path, test_path)
        assert step5.TARGET_COL not in cols


class TestEvaluate:
    def test_returns_expected_keys(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 1, 0]
        result = step5.evaluate(y_true, y_pred)
        expected_keys = {
            "accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
            "macro_precision", "macro_recall", "macro_f1", "kappa", "cm",
        }
        assert expected_keys == set(result.keys())

    def test_perfect_prediction_gives_accuracy_1(self):
        y = [0, 1, 0, 1, 1, 0]
        result = step5.evaluate(y, y)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_kappa_is_1_for_perfect_prediction(self):
        y = [0, 1, 0, 1]
        result = step5.evaluate(y, y)
        assert result["kappa"] == pytest.approx(1.0)

    def test_confusion_matrix_correct_size(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 1, 1]
        result = step5.evaluate(y_true, y_pred)
        cm = result["cm"]
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)

    def test_accuracy_between_0_and_1(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 0, 0]
        result = step5.evaluate(y_true, y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0


class TestMobileEfficiencyScore:
    def test_perfect_score_for_small_fast_model(self):
        score = step5.mobile_efficiency_score(size_kb=500.0, single_p95_ms=10.0)
        assert score == pytest.approx(1.0)

    def test_score_below_1_for_large_model(self):
        score = step5.mobile_efficiency_score(size_kb=3000.0, single_p95_ms=10.0)
        assert score < 1.0

    def test_score_below_1_for_slow_model(self):
        score = step5.mobile_efficiency_score(size_kb=500.0, single_p95_ms=100.0)
        assert score < 1.0

    def test_score_non_negative(self):
        # Extremely large and slow model
        score = step5.mobile_efficiency_score(size_kb=100_000.0, single_p95_ms=10_000.0)
        assert score >= 0.0

    def test_size_boundary_exactly_at_1000kb(self):
        score = step5.mobile_efficiency_score(size_kb=1000.0, single_p95_ms=25.0)
        assert score == pytest.approx(1.0)


class TestOverallMobileScore:
    def test_returns_float(self):
        metrics = {"weighted_f1": 0.8, "macro_f1": 0.75, "kappa": 0.6}
        score = step5.overall_mobile_score(metrics, efficiency_score=0.9)
        assert isinstance(score, float)

    def test_score_increases_with_better_f1(self):
        metrics_low = {"weighted_f1": 0.5, "macro_f1": 0.5, "kappa": 0.4}
        metrics_high = {"weighted_f1": 0.9, "macro_f1": 0.85, "kappa": 0.8}
        s_low = step5.overall_mobile_score(metrics_low, efficiency_score=1.0)
        s_high = step5.overall_mobile_score(metrics_high, efficiency_score=1.0)
        assert s_high > s_low


class TestIsMobileEligible:
    def test_eligible_when_under_both_thresholds(self):
        assert step5.is_mobile_eligible(size_kb=500.0, single_p95_ms=10.0) is True

    def test_ineligible_when_size_over_threshold(self):
        assert step5.is_mobile_eligible(size_kb=1001.0, single_p95_ms=10.0) is False

    def test_ineligible_when_latency_over_threshold(self):
        assert step5.is_mobile_eligible(size_kb=500.0, single_p95_ms=26.0) is False

    def test_eligible_exactly_at_boundary(self):
        assert step5.is_mobile_eligible(size_kb=1000.0, single_p95_ms=25.0) is True


class TestBuildPreprocessor:
    def test_transforms_without_error(self):
        feature_cols = ["a", "b", "c"]
        pre = step5.build_preprocessor(feature_cols)
        X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, np.nan, 5.0], "c": [6.0, 7.0, 8.0]})
        result = pre.fit_transform(X)
        assert result.shape == (3, 3)

    def test_imputes_nan_values(self):
        feature_cols = ["x"]
        pre = step5.build_preprocessor(feature_cols)
        X = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = pre.fit_transform(X)
        assert not np.isnan(result).any()


class TestBuildCandidates:
    def test_returns_list_of_tuples(self):
        candidates = step5.build_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_all_tuples_have_name_and_estimator(self):
        candidates = step5.build_candidates()
        for name, estimator in candidates:
            assert isinstance(name, str)
            assert len(name) > 0
            assert hasattr(estimator, "fit")

    def test_includes_logistic_regression_candidates(self):
        candidates = step5.build_candidates()
        names = [name for name, _ in candidates]
        assert any("logreg" in name for name in names)


class TestExportFeatureSchema:
    def test_creates_json_file(self, tmp_path):
        out_path = tmp_path / "feature_schema.json"
        step5.export_feature_schema(
            feature_cols=["a", "b", "c"],
            classes=[0, 1],
            out_path=out_path,
        )
        assert out_path.exists()

    def test_json_has_expected_keys(self, tmp_path):
        out_path = tmp_path / "feature_schema.json"
        step5.export_feature_schema(
            feature_cols=["a", "b"],
            classes=[0, 1],
            out_path=out_path,
        )
        schema = json.loads(out_path.read_text())
        assert "feature_order" in schema
        assert "num_features" in schema
        assert "target_col" in schema
        assert "classes" in schema

    def test_feature_order_matches_input(self, tmp_path):
        out_path = tmp_path / "schema.json"
        step5.export_feature_schema(
            feature_cols=["x", "y", "z"],
            classes=[0, 1],
            out_path=out_path,
        )
        schema = json.loads(out_path.read_text())
        assert schema["feature_order"] == ["x", "y", "z"]
        assert schema["num_features"] == 3


class TestExportLinearContract:
    def _make_fitted_pipeline(self, feature_cols: list[str]) -> Pipeline:
        rng = np.random.default_rng(0)
        n = 40
        X = pd.DataFrame(
            {col: rng.normal(0, 1, n) for col in feature_cols}
        )
        y = rng.integers(0, 2, n)
        pre = step5.build_preprocessor(feature_cols)
        model = LogisticRegression(max_iter=100, random_state=0)
        pipeline = Pipeline([("preprocess", pre), ("model", model)])
        pipeline.fit(X, y)
        return pipeline

    def test_returns_true_for_logistic_regression(self, tmp_path):
        feature_cols = ["a", "b", "c"]
        pipeline = self._make_fitted_pipeline(feature_cols)
        out_path = tmp_path / "contract.json"
        result = step5.export_linear_contract_if_possible(pipeline, feature_cols, out_path)
        assert result is True

    def test_creates_json_file_for_linear_model(self, tmp_path):
        feature_cols = ["a", "b", "c"]
        pipeline = self._make_fitted_pipeline(feature_cols)
        out_path = tmp_path / "contract.json"
        step5.export_linear_contract_if_possible(pipeline, feature_cols, out_path)
        assert out_path.exists()

    def test_json_contains_coef_and_intercept(self, tmp_path):
        feature_cols = ["a", "b"]
        pipeline = self._make_fitted_pipeline(feature_cols)
        out_path = tmp_path / "contract.json"
        step5.export_linear_contract_if_possible(pipeline, feature_cols, out_path)
        contract = json.loads(out_path.read_text())
        assert "coef" in contract
        assert "intercept" in contract

    def test_returns_false_for_non_linear_model(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        feature_cols = ["a", "b"]
        rng = np.random.default_rng(1)
        n = 30
        X = pd.DataFrame({col: rng.normal(0, 1, n) for col in feature_cols})
        y = rng.integers(0, 2, n)
        pre = step5.build_preprocessor(feature_cols)
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        pipeline = Pipeline([("preprocess", pre), ("model", rf)])
        pipeline.fit(X, y)
        out_path = tmp_path / "contract.json"
        result = step5.export_linear_contract_if_possible(pipeline, feature_cols, out_path)
        assert result is False


class TestSerializedSizeBytes:
    def test_returns_positive_integer(self):
        from sklearn.linear_model import LogisticRegression

        feature_cols = ["a", "b"]
        rng = np.random.default_rng(2)
        X = pd.DataFrame({col: rng.normal(0, 1, 20) for col in feature_cols})
        y = rng.integers(0, 2, 20)
        pre = step5.build_preprocessor(feature_cols)
        model = LogisticRegression(max_iter=100)
        pipeline = Pipeline([("preprocess", pre), ("model", model)])
        pipeline.fit(X, y)
        size = step5.serialized_size_bytes(pipeline)
        assert isinstance(size, int)
        assert size > 0
