"""Tests for scripts/step2_build_features.py."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP2_SCRIPT = REPO_ROOT / "prediction-sleep" / "scripts" / "step2_build_features.py"

spec = importlib.util.spec_from_file_location("step2_build_features", STEP2_SCRIPT)
step2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(step2)


def _make_sleep_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": ["a", "b", "c"],
            "sourceName": ["Apple Watch"] * 3,
            "startDate": [
                "2026-01-01 22:00:00 +0000",
                "2026-01-02 00:00:00 +0000",
                "2026-01-02 01:00:00 +0000",
            ],
            "endDate": [
                "2026-01-02 00:00:00 +0000",
                "2026-01-02 01:00:00 +0000",
                "2026-01-02 06:00:00 +0000",
            ],
            "value": [
                "HKCategoryValueSleepAnalysisAsleepCore",
                "HKCategoryValueSleepAnalysisAsleepREM",
                "HKCategoryValueSleepAnalysisAsleepDeep",
            ],
        }
    )


class TestParseDt:
    def test_converts_timezone_aware_strings(self):
        s = pd.Series(["2026-01-01 22:00:00 +0000", "2026-01-02 06:00:00 +0000"])
        result = step2.parse_dt(s)
        assert result.dtype.tz is not None
        assert result.iloc[0].year == 2026

    def test_returns_series_with_datetime_dtype(self):
        s = pd.Series(["2026-03-15 12:00:00 +0000"])
        result = step2.parse_dt(s)
        assert pd.api.types.is_datetime64_any_dtype(result)


class TestAssignNight:
    def test_before_midnight_maps_to_same_date(self):
        dt = pd.to_datetime(pd.Series(["2026-01-01 22:00:00+00:00"]))
        result = step2.assign_night(dt)
        assert str(result.iloc[0]) == "2026-01-01"

    def test_after_midnight_but_before_6am_maps_to_previous_date(self):
        # 2026-01-02 03:00 UTC shifted back 6h = 2026-01-01 21:00 -> date 2026-01-01
        dt = pd.to_datetime(pd.Series(["2026-01-02 03:00:00+00:00"]))
        result = step2.assign_night(dt)
        assert str(result.iloc[0]) == "2026-01-01"

    def test_after_6am_maps_to_current_date(self):
        # 2026-01-02 08:00 UTC shifted back 6h = 2026-01-02 02:00 -> date 2026-01-02
        dt = pd.to_datetime(pd.Series(["2026-01-02 08:00:00+00:00"]))
        result = step2.assign_night(dt)
        assert str(result.iloc[0]) == "2026-01-02"


class TestLoadCsv:
    def test_returns_empty_df_when_file_missing(self, tmp_path):
        result = step2.load_csv(tmp_path / "nonexistent.csv")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_reads_existing_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(path, index=False)
        result = step2.load_csv(path)
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]


class TestBuildSleepFeatures:
    def test_returns_empty_for_empty_input(self):
        result = step2.build_sleep_features(pd.DataFrame())
        assert result.empty

    def test_output_has_expected_columns(self):
        result = step2.build_sleep_features(_make_sleep_df())
        expected_cols = {
            "night_date", "sleep_start", "sleep_end",
            "in_bed_minutes", "rem_minutes", "deep_minutes", "core_minutes", "asleep_minutes",
            "total_sleep_minutes", "sleep_efficiency", "rem_pct", "deep_pct", "core_pct",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_groups_by_night(self):
        result = step2.build_sleep_features(_make_sleep_df())
        assert len(result) == 1
        assert str(result["night_date"].iloc[0]) == "2026-01-01"

    def test_core_minutes_correctly_summed(self):
        result = step2.build_sleep_features(_make_sleep_df())
        # Core segment: 22:00 -> 00:00 = 120 min
        assert result["core_minutes"].iloc[0] == pytest.approx(120.0)

    def test_rem_minutes_correctly_summed(self):
        result = step2.build_sleep_features(_make_sleep_df())
        # REM segment: 00:00 -> 01:00 = 60 min
        assert result["rem_minutes"].iloc[0] == pytest.approx(60.0)

    def test_deep_minutes_correctly_summed(self):
        result = step2.build_sleep_features(_make_sleep_df())
        # Deep segment: 01:00 -> 06:00 = 300 min
        assert result["deep_minutes"].iloc[0] == pytest.approx(300.0)

    def test_sleep_efficiency_between_0_and_1_when_in_bed(self):
        df = pd.DataFrame(
            {
                "uuid": ["a", "b"],
                "sourceName": ["Watch", "Watch"],
                "startDate": ["2026-01-01 22:00:00 +0000", "2026-01-01 23:00:00 +0000"],
                "endDate": ["2026-01-02 06:00:00 +0000", "2026-01-02 05:00:00 +0000"],
                "value": [
                    "HKCategoryValueSleepAnalysisInBed",
                    "HKCategoryValueSleepAnalysisAsleep",
                ],
            }
        )
        result = step2.build_sleep_features(df)
        eff = result["sleep_efficiency"].iloc[0]
        assert 0.0 <= eff <= 1.0

    def test_two_nights_produce_two_rows(self):
        df = pd.DataFrame(
            {
                "uuid": ["a", "b"],
                "sourceName": ["Watch", "Watch"],
                "startDate": [
                    "2026-01-01 22:00:00 +0000",
                    "2026-01-02 22:00:00 +0000",
                ],
                "endDate": [
                    "2026-01-02 06:00:00 +0000",
                    "2026-01-03 06:00:00 +0000",
                ],
                "value": [
                    "HKCategoryValueSleepAnalysisAsleep",
                    "HKCategoryValueSleepAnalysisAsleep",
                ],
            }
        )
        result = step2.build_sleep_features(df)
        assert len(result) == 2


class TestSummarizeMetric:
    def _make_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    ["2026-01-01 23:00:00+00:00", "2026-01-02 01:00:00+00:00", "2026-01-02 03:00:00+00:00"]
                ),
                "value": [55.0, 58.0, 62.0],
            }
        )
        return df

    def test_returns_none_for_empty_df(self):
        result = step2.summarize_metric(
            pd.DataFrame(), "ts", "value",
            pd.Timestamp("2026-01-01 22:00:00+00:00"),
            pd.Timestamp("2026-01-02 06:00:00+00:00"),
        )
        assert result is None

    def test_returns_none_when_no_rows_in_window(self):
        df = self._make_df()
        result = step2.summarize_metric(
            df, "ts", "value",
            pd.Timestamp("2026-01-03 00:00:00+00:00"),
            pd.Timestamp("2026-01-03 06:00:00+00:00"),
        )
        assert result is None

    def test_returns_dict_with_expected_keys(self):
        df = self._make_df()
        result = step2.summarize_metric(
            df, "ts", "value",
            pd.Timestamp("2026-01-01 22:00:00+00:00"),
            pd.Timestamp("2026-01-02 06:00:00+00:00"),
        )
        assert result is not None
        assert set(result.keys()) == {"mean", "min", "max", "median", "std"}

    def test_mean_is_correct(self):
        df = self._make_df()
        result = step2.summarize_metric(
            df, "ts", "value",
            pd.Timestamp("2026-01-01 22:00:00+00:00"),
            pd.Timestamp("2026-01-02 06:00:00+00:00"),
        )
        assert result is not None
        assert result["mean"] == pytest.approx((55.0 + 58.0 + 62.0) / 3)

    def test_min_max_correct(self):
        df = self._make_df()
        result = step2.summarize_metric(
            df, "ts", "value",
            pd.Timestamp("2026-01-01 22:00:00+00:00"),
            pd.Timestamp("2026-01-02 06:00:00+00:00"),
        )
        assert result is not None
        assert result["min"] == pytest.approx(55.0)
        assert result["max"] == pytest.approx(62.0)

    def test_window_filtering_excludes_outside_rows(self):
        df = self._make_df()
        # Only the first row (23:00) is in window 22:00–00:00
        result = step2.summarize_metric(
            df, "ts", "value",
            pd.Timestamp("2026-01-01 22:00:00+00:00"),
            pd.Timestamp("2026-01-02 00:00:00+00:00"),
        )
        assert result is not None
        assert result["mean"] == pytest.approx(55.0)


class TestAttachPhysioFeatures:
    def test_returns_night_df_unchanged_when_all_physio_empty(self):
        night_df = step2.build_sleep_features(_make_sleep_df())
        result = step2.attach_physio_features(
            night_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        # Should have the same night_date rows
        assert len(result) == len(night_df)

    def test_returns_empty_when_night_df_empty(self):
        result = step2.attach_physio_features(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert result.empty

    def test_attaches_hr_stats(self):
        night_df = step2.build_sleep_features(_make_sleep_df())
        hr_df = pd.DataFrame(
            {
                "startDate": ["2026-01-02 00:30:00 +0000", "2026-01-02 02:00:00 +0000"],
                "value": ["55", "60"],
                "unit": ["count/min", "count/min"],
            }
        )
        result = step2.attach_physio_features(
            night_df, hr_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert "hr_mean" in result.columns
        assert "hr_min" in result.columns
        assert "hr_max" in result.columns

    def test_attaches_hrv_stats(self):
        night_df = step2.build_sleep_features(_make_sleep_df())
        hrv_df = pd.DataFrame(
            {
                "startDate": ["2026-01-02 01:00:00 +0000"],
                "value": ["42"],
                "unit": ["ms"],
            }
        )
        result = step2.attach_physio_features(
            night_df, pd.DataFrame(), hrv_df, pd.DataFrame(), pd.DataFrame()
        )
        assert "hrv_mean" in result.columns

    def test_attaches_resp_stats(self):
        night_df = step2.build_sleep_features(_make_sleep_df())
        resp_df = pd.DataFrame(
            {
                "startDate": ["2026-01-02 02:00:00 +0000"],
                "value": ["15"],
                "unit": ["count/min"],
            }
        )
        result = step2.attach_physio_features(
            night_df, pd.DataFrame(), pd.DataFrame(), resp_df, pd.DataFrame()
        )
        assert "resp_mean" in result.columns

    def test_attaches_spo2_stats(self):
        night_df = step2.build_sleep_features(_make_sleep_df())
        spo2_df = pd.DataFrame(
            {
                "startDate": ["2026-01-02 03:00:00 +0000"],
                "value": ["0.97"],
                "unit": ["%"],
            }
        )
        result = step2.attach_physio_features(
            night_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), spo2_df
        )
        assert "spo2_mean" in result.columns
