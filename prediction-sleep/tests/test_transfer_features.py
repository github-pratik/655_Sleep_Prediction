"""Tests for src/transfer/features.py."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "prediction-sleep" / "src"))

from transfer.features import (
    FeatureBundle,
    PublicEncoderInfo,
    _cyclical,
    _missingness_features,
    _ratio_features,
    _rolling_features,
    _safe_div,
    _sanitize_frame,
    _time_features,
    build_baseline_features,
    build_feature_spaces,
    build_transfer_features,
    load_public_encoder_info,
)


def _make_night_df(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "night_date": dates,
            "total_sleep_minutes": rng.normal(420, 30, n),
            "rem_minutes": rng.normal(90, 15, n),
            "deep_minutes": rng.normal(60, 10, n),
            "core_minutes": rng.normal(120, 20, n),
            "in_bed_minutes": rng.normal(480, 30, n),
            "asleep_minutes": rng.normal(420, 30, n),
            "sleep_efficiency": rng.uniform(0.7, 0.95, n),
            "hr_mean": rng.normal(58, 5, n),
            "hr_std": rng.normal(5, 1, n),
            "hrv_mean": rng.normal(45, 8, n),
            "hrv_std": rng.normal(10, 2, n),
            "resp_mean": rng.normal(15, 1, n),
            "resp_std": rng.normal(1, 0.1, n),
            "spo2_mean": rng.uniform(0.95, 0.99, n),
            "spo2_std": rng.uniform(0.001, 0.005, n),
            "fatigue_label": rng.integers(0, 2, n),
        }
    )


class TestSafeDiv:
    def test_zero_denominator_returns_nan(self):
        result = _safe_div(pd.Series([10.0]), pd.Series([0.0]))
        assert pd.isna(result.iloc[0])

    def test_normal_division(self):
        result = _safe_div(pd.Series([10.0]), pd.Series([2.0]))
        assert result.iloc[0] == pytest.approx(5.0)

    def test_inf_replaced_with_nan(self):
        result = _safe_div(pd.Series([np.inf]), pd.Series([1.0]))
        assert pd.isna(result.iloc[0])

    def test_series_length_preserved(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([1.0, 0.0, 2.0])
        result = _safe_div(a, b)
        assert len(result) == 3

    def test_non_numeric_input_handled(self):
        result = _safe_div(pd.Series(["10"]), pd.Series(["2"]))
        assert result.iloc[0] == pytest.approx(5.0)


class TestCyclical:
    def test_returns_two_series(self):
        values = pd.Series([1, 2, 3, 4, 6, 12])
        sin_vals, cos_vals = _cyclical(values, 12)
        assert len(sin_vals) == len(values)
        assert len(cos_vals) == len(values)

    def test_values_within_minus1_to_1(self):
        values = pd.Series(range(1, 13))
        sin_vals, cos_vals = _cyclical(values, 12)
        assert (sin_vals.abs() <= 1.0 + 1e-9).all()
        assert (cos_vals.abs() <= 1.0 + 1e-9).all()

    def test_period_12_january_and_december_are_near_equal_cos(self):
        # Month 1 and month 13 (next Jan) should wrap: cos(2π*1/12) ≈ cos(2π*13/12)
        s1, c1 = _cyclical(pd.Series([1]), 12)
        s2, c2 = _cyclical(pd.Series([13]), 12)
        assert c1.iloc[0] == pytest.approx(c2.iloc[0], abs=1e-6)


class TestSanitizeFrame:
    def test_replaces_positive_inf(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        result = _sanitize_frame(df)
        assert not np.isinf(result["a"]).any()

    def test_replaces_negative_inf(self):
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})
        result = _sanitize_frame(df)
        assert not np.isinf(result["a"]).any()

    def test_converts_non_numeric_strings(self):
        df = pd.DataFrame({"a": ["1.0", "2.0", "3.0"]})
        result = _sanitize_frame(df)
        assert pd.api.types.is_float_dtype(result["a"])


class TestBuildBaselineFeatures:
    def test_returns_feature_bundle(self):
        df = _make_night_df(4)
        result = build_baseline_features(df)
        assert isinstance(result, FeatureBundle)

    def test_excludes_metadata_columns(self):
        df = _make_night_df(4)
        result = build_baseline_features(df)
        for meta in ("night_date", "fatigue_label"):
            assert meta not in result.feature_order

    def test_frame_shape_matches_feature_order(self):
        df = _make_night_df(4)
        result = build_baseline_features(df)
        assert result.frame.shape[1] == len(result.feature_order)

    def test_name_is_baseline(self):
        df = _make_night_df(3)
        result = build_baseline_features(df)
        assert result.name == "baseline"


class TestMissingnessFeatures:
    def test_returns_four_columns(self):
        df = _make_night_df(3)
        base_frame, _ = __import__(
            "transfer.data", fromlist=["numeric_feature_frame"]
        ).numeric_feature_frame(df)
        result = _missingness_features(base_frame)
        assert set(result.keys()) == {
            "observed_numeric_count",
            "missing_numeric_count",
            "missing_numeric_rate",
            "observed_numeric_rate",
        }

    def test_missing_rate_is_zero_for_complete_data(self):
        df = _make_night_df(3)
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _missingness_features(base_frame)
        missing_rate = result["missing_numeric_rate"]
        assert (missing_rate == 0.0).all()


class TestRatioFeatures:
    def test_returns_expected_keys(self):
        df = _make_night_df(3)
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _ratio_features(base_frame)
        assert "sleep_gap_minutes" in result
        assert "rem_to_deep_ratio" in result
        assert "hrv_to_hr_ratio" in result

    def test_sleep_gap_is_in_bed_minus_asleep(self):
        df = pd.DataFrame(
            {
                "night_date": pd.to_datetime(["2026-01-01"]),
                "in_bed_minutes": [480.0],
                "asleep_minutes": [420.0],
                "total_sleep_minutes": [420.0],
                "rem_minutes": [90.0],
                "deep_minutes": [60.0],
                "core_minutes": [120.0],
                "hr_mean": [58.0],
                "hrv_mean": [45.0],
                "resp_mean": [15.0],
                "spo2_mean": [0.97],
                "fatigue_label": [0],
            }
        )
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _ratio_features(base_frame)
        assert result["sleep_gap_minutes"].iloc[0] == pytest.approx(60.0)


class TestRollingFeatures:
    def test_rolling_columns_generated_for_total_sleep(self):
        df = _make_night_df(10)
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _rolling_features(base_frame)
        # Should have at least one rolling mean for total_sleep_minutes
        assert any("total_sleep_minutes_past3_mean" in k for k in result)

    def test_rolling_features_not_empty(self):
        df = _make_night_df(5)
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _rolling_features(base_frame)
        assert len(result) > 0


class TestTimeFeatures:
    def test_returns_day_of_week(self):
        df = _make_night_df(3)
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _time_features(df, base_frame.index)
        assert "day_of_week" in result
        assert "is_weekend" in result
        assert "month_sin" in result
        assert "month_cos" in result

    def test_returns_nan_when_no_date_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _time_features(df, base_frame.index)
        assert pd.isna(result["day_of_week"]).all()

    def test_is_weekend_binary(self):
        df = _make_night_df(7)
        from transfer.data import numeric_feature_frame
        base_frame, _ = numeric_feature_frame(df)
        result = _time_features(df, base_frame.index)
        is_weekend_vals = result["is_weekend"].dropna().unique()
        assert set(is_weekend_vals).issubset({0.0, 1.0})


class TestBuildTransferFeatures:
    def test_returns_feature_bundle_named_transfer(self):
        df = _make_night_df(5)
        result = build_transfer_features(df)
        assert isinstance(result, FeatureBundle)
        assert result.name == "transfer"

    def test_frame_shape_matches_feature_order(self):
        df = _make_night_df(5)
        result = build_transfer_features(df)
        assert result.frame.shape[1] == len(result.feature_order)

    def test_has_rolling_features(self):
        df = _make_night_df(5)
        result = build_transfer_features(df)
        rolling_cols = [c for c in result.feature_order if "past3" in c or "past7" in c]
        assert len(rolling_cols) > 0

    def test_has_ratio_features(self):
        df = _make_night_df(5)
        result = build_transfer_features(df)
        assert "sleep_gap_minutes" in result.feature_order

    def test_has_time_features(self):
        df = _make_night_df(5)
        result = build_transfer_features(df)
        assert "day_of_week" in result.feature_order

    def test_no_inf_values_in_frame(self):
        df = _make_night_df(5)
        result = build_transfer_features(df)
        assert not np.isinf(result.frame.to_numpy(dtype=float, na_value=0.0)).any()


class TestBuildFeatureSpaces:
    def test_returns_three_bundles(self):
        df = _make_night_df(5)
        spaces = build_feature_spaces(df)
        assert set(spaces.keys()) == {"baseline", "transfer", "combined"}

    def test_combined_has_more_features_than_baseline(self):
        df = _make_night_df(5)
        spaces = build_feature_spaces(df)
        assert len(spaces["combined"].feature_order) > len(spaces["baseline"].feature_order)

    def test_all_bundles_have_same_number_of_rows(self):
        df = _make_night_df(6)
        spaces = build_feature_spaces(df)
        for bundle in spaces.values():
            assert len(bundle.frame) == 6


class TestLoadPublicEncoderInfo:
    def test_returns_unavailable_when_path_is_none(self):
        result = load_public_encoder_info(None)
        assert result.available is False

    def test_returns_unavailable_when_path_missing(self, tmp_path):
        result = load_public_encoder_info(tmp_path / "nonexistent.pkl")
        assert result.available is False
        assert "not found" in result.note

    def test_returns_available_for_valid_sklearn_artifact(self, tmp_path):
        # Save a PCA as a simple artifact
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (30, 5))
        pca = PCA(n_components=3)
        pca.fit(X)
        path = tmp_path / "encoder.pkl"
        joblib.dump(pca, path)

        result = load_public_encoder_info(path)
        assert result.available is True
        assert result.artifact is not None

    def test_returns_available_for_dict_artifact(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (20, 4))
        pca = PCA(n_components=2)
        pca.fit(X)
        artifact_dict = {
            "feature_order": ["a", "b", "c", "d"],
            "output_feature_names": ["enc_0", "enc_1"],
            "encoder": pca,
        }
        path = tmp_path / "encoder_dict.pkl"
        joblib.dump(artifact_dict, path)

        result = load_public_encoder_info(path)
        assert result.available is True
        assert result.input_feature_order == ["a", "b", "c", "d"]
        assert result.output_feature_names == ["enc_0", "enc_1"]

    def test_returns_unavailable_for_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"this is not a valid pickle file")
        result = load_public_encoder_info(path)
        assert result.available is False
        assert "failed to load" in result.note


class TestPublicEncoderIntegration:
    def test_build_transfer_features_with_pca_encoder(self, tmp_path):
        df = _make_night_df(8)
        # Build a PCA encoder on the baseline features
        baseline = build_baseline_features(df)
        from transfer.data import numeric_feature_frame
        base_frame, base_cols = numeric_feature_frame(df)
        X = base_frame.fillna(base_frame.median())
        pca = PCA(n_components=2)
        pca.fit(X)
        path = tmp_path / "pca_encoder.pkl"
        artifact = {
            "feature_order": base_cols,
            "output_feature_names": ["pca_0", "pca_1"],
            "encoder": pca,
        }
        joblib.dump(artifact, path)

        result = build_transfer_features(df, encoder_path=path)
        assert result.metadata["encoder_available"] is True
        assert "pca_0" in result.feature_order
        assert "pca_1" in result.feature_order
