"""Tests for scripts/step4_time_split.py."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP4_SCRIPT = REPO_ROOT / "prediction-sleep" / "scripts" / "step4_time_split.py"


def run_step4(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STEP4_SCRIPT), *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _make_model_data(tmp_path: Path, n_rows: int = 20) -> Path:
    dates = [f"2026-0{1 + i // 30:01d}-{1 + (i % 30):02d}" for i in range(n_rows)]
    df = pd.DataFrame(
        {
            "night_date": dates,
            "sleep_start": [f"{d} 22:00:00" for d in dates],
            "sleep_end": [f"{d} 06:00:00" for d in dates],
            "total_sleep_minutes": [420.0 + i for i in range(n_rows)],
            "rem_minutes": [90.0] * n_rows,
            "fatigue_label": [i % 2 for i in range(n_rows)],
        }
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "model_data.csv"
    df.to_csv(path, index=False)
    return path


class TestStep4TimeSplit:
    def test_creates_train_and_test_files(self, tmp_path):
        data_path = _make_model_data(tmp_path)
        result = run_step4(
            ["--data", str(data_path), "--train-out", str(tmp_path / "train.csv"),
             "--test-out", str(tmp_path / "test.csv")],
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "test.csv").exists()

    def test_70_30_split_ratio(self, tmp_path):
        n = 20
        data_path = _make_model_data(tmp_path, n_rows=n)
        result = run_step4(
            ["--data", str(data_path), "--train-out", str(tmp_path / "train.csv"),
             "--test-out", str(tmp_path / "test.csv")],
            cwd=tmp_path,
        )
        assert result.returncode == 0
        train_df = pd.read_csv(tmp_path / "train.csv")
        test_df = pd.read_csv(tmp_path / "test.csv")
        assert len(train_df) == int(n * 0.7)
        assert len(test_df) == n - int(n * 0.7)

    def test_train_plus_test_equals_total(self, tmp_path):
        n = 15
        data_path = _make_model_data(tmp_path, n_rows=n)
        result = run_step4(
            ["--data", str(data_path), "--train-out", str(tmp_path / "train.csv"),
             "--test-out", str(tmp_path / "test.csv")],
            cwd=tmp_path,
        )
        assert result.returncode == 0
        train_df = pd.read_csv(tmp_path / "train.csv")
        test_df = pd.read_csv(tmp_path / "test.csv")
        assert len(train_df) + len(test_df) == n

    def test_creates_split_stats_report(self, tmp_path):
        data_path = _make_model_data(tmp_path)
        run_step4(
            ["--data", str(data_path), "--train-out", str(tmp_path / "train.csv"),
             "--test-out", str(tmp_path / "test.csv")],
            cwd=tmp_path,
        )
        report_path = tmp_path / "reports" / "split_stats.json"
        assert report_path.exists()
        stats = json.loads(report_path.read_text())
        assert "total_rows" in stats
        assert "train_rows" in stats
        assert "test_rows" in stats

    def test_fails_when_data_file_missing(self, tmp_path):
        result = run_step4(
            ["--data", str(tmp_path / "nonexistent.csv"),
             "--train-out", str(tmp_path / "train.csv"),
             "--test-out", str(tmp_path / "test.csv")],
            cwd=tmp_path,
        )
        assert result.returncode != 0

    def test_fails_when_fatigue_label_column_missing(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        df = pd.DataFrame(
            {
                "night_date": ["2026-01-01", "2026-01-02"],
                "sleep_start": ["2026-01-01 22:00:00", "2026-01-02 22:00:00"],
                "sleep_end": ["2026-01-02 06:00:00", "2026-01-03 06:00:00"],
                "total_sleep_minutes": [420.0, 390.0],
            }
        )
        data_path = dataset_dir / "model_data.csv"
        df.to_csv(data_path, index=False)
        result = run_step4(
            ["--data", str(data_path), "--train-out", str(tmp_path / "train.csv"),
             "--test-out", str(tmp_path / "test.csv")],
            cwd=tmp_path,
        )
        assert result.returncode != 0

    def test_target_column_excluded_from_features_in_report(self, tmp_path):
        data_path = _make_model_data(tmp_path)
        result = run_step4(
            ["--data", str(data_path), "--train-out", str(tmp_path / "train.csv"),
             "--test-out", str(tmp_path / "test.csv")],
            cwd=tmp_path,
        )
        assert result.returncode == 0
        stats = json.loads((tmp_path / "reports" / "split_stats.json").read_text())
        assert "fatigue_label" not in stats["features"]
