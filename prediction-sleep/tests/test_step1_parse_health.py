"""Tests for scripts/step1_parse_health.py."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1_SCRIPT = REPO_ROOT / "prediction-sleep" / "scripts" / "step1_parse_health.py"

spec = importlib.util.spec_from_file_location("step1_parse_health", STEP1_SCRIPT)
step1 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(step1)


def _make_export_xml(records: list[dict]) -> bytes:
    """Build a minimal Apple Health export.xml with the given records."""
    root = etree.Element("HealthData")
    for rec in records:
        elem = etree.SubElement(root, "Record")
        for k, v in rec.items():
            elem.set(k, v)
    return etree.tostring(root, xml_declaration=True, encoding="utf-8")


class TestIterRecords:
    def test_yields_all_record_elements(self, tmp_path):
        xml_bytes = _make_export_xml([
            {"type": "HKCategoryTypeIdentifierSleepAnalysis", "value": "HKCategoryValueSleepAnalysisAsleep",
             "startDate": "2026-01-01 22:00:00 +0000", "endDate": "2026-01-02 06:00:00 +0000"},
            {"type": "HKQuantityTypeIdentifierHeartRate", "value": "55", "unit": "count/min",
             "startDate": "2026-01-01 23:00:00 +0000", "endDate": "2026-01-01 23:00:00 +0000"},
        ])
        xml_path = tmp_path / "export.xml"
        xml_path.write_bytes(xml_bytes)

        records = list(step1.iter_records(xml_path))
        assert len(records) == 2

    def test_yields_nothing_for_empty_export(self, tmp_path):
        xml_bytes = etree.tostring(etree.Element("HealthData"), xml_declaration=True, encoding="utf-8")
        xml_path = tmp_path / "export.xml"
        xml_path.write_bytes(xml_bytes)
        assert list(step1.iter_records(xml_path)) == []


class TestParse:
    def _make_xml_path(self, tmp_path: Path) -> Path:
        xml_bytes = _make_export_xml([
            {
                "type": "HKCategoryTypeIdentifierSleepAnalysis",
                "value": "HKCategoryValueSleepAnalysisAsleep",
                "sourceName": "Apple Watch",
                "creationDate": "2026-01-02 06:00:00 +0000",
                "startDate": "2026-01-01 22:00:00 +0000",
                "endDate": "2026-01-02 06:00:00 +0000",
                "uuid": "sleep-1",
            },
            {
                "type": "HKQuantityTypeIdentifierHeartRate",
                "value": "55",
                "unit": "count/min",
                "sourceName": "Apple Watch",
                "startDate": "2026-01-01 23:00:00 +0000",
                "endDate": "2026-01-01 23:00:00 +0000",
                "uuid": "hr-1",
            },
            {
                "type": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
                "value": "45",
                "unit": "ms",
                "sourceName": "Apple Watch",
                "startDate": "2026-01-01 23:30:00 +0000",
                "endDate": "2026-01-01 23:30:00 +0000",
                "uuid": "hrv-1",
            },
            {
                "type": "HKQuantityTypeIdentifierRespiratoryRate",
                "value": "16",
                "unit": "count/min",
                "sourceName": "Apple Watch",
                "startDate": "2026-01-02 01:00:00 +0000",
                "endDate": "2026-01-02 01:00:00 +0000",
                "uuid": "resp-1",
            },
        ])
        xml_path = tmp_path / "export.xml"
        xml_path.write_bytes(xml_bytes)
        return xml_path

    def test_creates_sleep_csv(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        assert (out_dir / "sleep.csv").exists()

    def test_creates_hr_csv(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        assert (out_dir / "hr.csv").exists()

    def test_creates_hrv_csv(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        assert (out_dir / "hrv.csv").exists()

    def test_creates_resp_csv(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        assert (out_dir / "resp.csv").exists()

    def test_creates_record_type_counts(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        counts_path = tmp_path / "reports" / "record_type_counts.csv"
        assert counts_path.exists()
        counts_df = pd.read_csv(counts_path)
        assert set(counts_df.columns) == {"record_type", "count"}

    def test_sleep_csv_contains_expected_columns(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        sleep_df = pd.read_csv(out_dir / "sleep.csv")
        assert "startDate" in sleep_df.columns
        assert "endDate" in sleep_df.columns
        assert "value" in sleep_df.columns

    def test_sleep_csv_has_correct_row_count(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        sleep_df = pd.read_csv(out_dir / "sleep.csv")
        assert len(sleep_df) == 1

    def test_no_sleep_csv_when_no_sleep_records(self, tmp_path):
        xml_bytes = _make_export_xml([
            {"type": "HKQuantityTypeIdentifierHeartRate", "value": "60", "unit": "count/min",
             "sourceName": "Apple Watch", "startDate": "2026-01-01 22:00:00 +0000",
             "endDate": "2026-01-01 22:00:00 +0000", "uuid": "hr-1"}
        ])
        xml_path = tmp_path / "export.xml"
        xml_path.write_bytes(xml_bytes)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        assert not (out_dir / "sleep.csv").exists()

    def test_spo2_csv_created_when_present(self, tmp_path):
        xml_bytes = _make_export_xml([
            {
                "type": "HKQuantityTypeIdentifierOxygenSaturation",
                "value": "0.97",
                "unit": "%",
                "sourceName": "Apple Watch",
                "startDate": "2026-01-02 02:00:00 +0000",
                "endDate": "2026-01-02 02:00:00 +0000",
                "uuid": "spo2-1",
            }
        ])
        xml_path = tmp_path / "export.xml"
        xml_path.write_bytes(xml_bytes)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        assert (out_dir / "spo2.csv").exists()

    def test_record_type_counts_correct_total(self, tmp_path):
        xml_path = self._make_xml_path(tmp_path)
        out_dir = tmp_path / "parsed_tables"
        step1.parse(xml_path, out_dir)
        counts_df = pd.read_csv(tmp_path / "reports" / "record_type_counts.csv")
        assert counts_df["count"].sum() == 4
