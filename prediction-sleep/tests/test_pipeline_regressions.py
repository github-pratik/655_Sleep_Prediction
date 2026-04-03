import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = REPO_ROOT / "prediction-sleep"
STEP3_SCRIPT = PRED_ROOT / "scripts" / "step3_merge_labels.py"
STEP5_SCRIPT = PRED_ROOT / "scripts" / "step5_train_mobile_models.py"


def run_script(script_path: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


class TestStep3MergeLabels(unittest.TestCase):
    def _write_night_features(self, tmp_path: Path) -> Path:
        night_path = tmp_path / "night_features.csv"
        df = pd.DataFrame(
            {
                "night_date": ["2026-03-01", "2026-03-02"],
                "total_sleep_minutes": [420.0, 390.0],
            }
        )
        df.to_csv(night_path, index=False)
        return night_path

    def test_accepts_night_date_alias_and_writes_integer_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            night_path = self._write_night_features(tmp_path)
            labels_path = tmp_path / "fatigue_labels.csv"
            out_path = tmp_path / "model_data.csv"

            labels_df = pd.DataFrame(
                {
                    "night_date": ["2026-03-02", "2026-03-03"],
                    "fatigue_label": [0, 1],
                }
            )
            labels_df.to_csv(labels_path, index=False)

            result = run_script(
                STEP3_SCRIPT,
                [
                    "--night-features",
                    str(night_path),
                    "--labels",
                    str(labels_path),
                    "--out",
                    str(out_path),
                ],
                cwd=tmp_path,
            )

            if result.returncode != 0:
                self.fail(f"Script failed unexpectedly:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            merged = pd.read_csv(out_path)
            self.assertEqual(len(merged), 2)
            self.assertEqual(merged["fatigue_label"].tolist(), [0, 1])

            report = json.loads((tmp_path / "reports" / "label_checks.json").read_text())
            self.assertIsInstance(report["labels_present"], int)

    def test_fails_when_any_label_date_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            night_path = self._write_night_features(tmp_path)
            labels_path = tmp_path / "fatigue_labels.csv"
            out_path = tmp_path / "model_data.csv"

            labels_df = pd.DataFrame(
                {
                    "date": ["2026-03-02"],
                    "fatigue_label": [1],
                }
            )
            labels_df.to_csv(labels_path, index=False)

            result = run_script(
                STEP3_SCRIPT,
                [
                    "--night-features",
                    str(night_path),
                    "--labels",
                    str(labels_path),
                    "--out",
                    str(out_path),
                ],
                cwd=tmp_path,
            )

            self.assertNotEqual(result.returncode, 0)
            combined_output = f"{result.stdout}\n{result.stderr}"
            self.assertIn("Missing fatigue labels", combined_output)
            self.assertTrue((tmp_path / "reports" / "missing_label_dates.csv").exists())

    def test_fails_for_non_binary_fatigue_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            night_path = self._write_night_features(tmp_path)
            labels_path = tmp_path / "fatigue_labels.csv"
            out_path = tmp_path / "model_data.csv"

            labels_df = pd.DataFrame(
                {
                    "date": ["2026-03-02", "2026-03-03"],
                    "fatigue_label": ["Low", "1"],
                }
            )
            labels_df.to_csv(labels_path, index=False)

            result = run_script(
                STEP3_SCRIPT,
                [
                    "--night-features",
                    str(night_path),
                    "--labels",
                    str(labels_path),
                    "--out",
                    str(out_path),
                ],
                cwd=tmp_path,
            )

            self.assertNotEqual(result.returncode, 0)
            combined_output = f"{result.stdout}\n{result.stderr}"
            self.assertIn("binary 0/1", combined_output)


class TestStep5EncodeLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("step5_train_mobile_models", STEP5_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.module = module

    def test_encode_labels_accepts_seen_classes(self):
        _, y_train_enc, y_test_enc = self.module.encode_labels(
            pd.Series([0, 1, 0, 1]),
            pd.Series([1, 0]),
        )
        self.assertEqual(len(y_train_enc), 4)
        self.assertEqual(len(y_test_enc), 2)

    def test_encode_labels_raises_clear_error_for_unseen_test_class(self):
        with self.assertRaises(SystemExit) as ctx:
            self.module.encode_labels(
                pd.Series([0, 0, 0]),
                pd.Series([0, 1]),
            )
        self.assertIn("unseen fatigue_label classes", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
