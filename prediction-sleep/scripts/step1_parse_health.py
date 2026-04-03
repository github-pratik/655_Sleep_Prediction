#!/usr/bin/env python3
"""
Step 1: Parse Apple Health export.xml
- Count record types
- Extract sleep, heart rate, HRV, respiratory rate, SpO2
Outputs:
  parsed_tables/sleep.csv
  parsed_tables/hr.csv
  parsed_tables/hrv.csv
  parsed_tables/resp.csv
  parsed_tables/spo2.csv (if any)
  reports/record_type_counts.csv
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable
from lxml import etree
import pandas as pd

RECORD_MAP = {
    "sleep": "HKCategoryTypeIdentifierSleepAnalysis",
    "hr": "HKQuantityTypeIdentifierHeartRate",
    "hrv": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "resp": "HKQuantityTypeIdentifierRespiratoryRate",
    "spo2": "HKQuantityTypeIdentifierOxygenSaturation",
}


def iter_records(xml_path: Path) -> Iterable[etree._Element]:
    # Use iterparse for streaming; only Record elements
    ctx = etree.iterparse(str(xml_path), events=("end",), tag="Record")
    for _, elem in ctx:
        yield elem
        elem.clear()
        # Clean up previous siblings to keep memory low
        while elem.getprevious() is not None:
            del elem.getparent()[0]


def parse(input_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    counters = Counter()
    buckets = {key: [] for key in RECORD_MAP}

    for record in iter_records(input_path):
        rtype = record.get("type")
        counters[rtype] += 1

        if rtype == RECORD_MAP["sleep"]:
            buckets["sleep"].append({
                "uuid": record.get("uuid"),
                "sourceName": record.get("sourceName"),
                "creationDate": record.get("creationDate"),
                "startDate": record.get("startDate"),
                "endDate": record.get("endDate"),
                "value": record.get("value"),
            })
        elif rtype == RECORD_MAP["hr"]:
            buckets["hr"].append({
                "uuid": record.get("uuid"),
                "sourceName": record.get("sourceName"),
                "unit": record.get("unit"),
                "startDate": record.get("startDate"),
                "endDate": record.get("endDate"),
                "value": record.get("value"),
            })
        elif rtype == RECORD_MAP["hrv"]:
            buckets["hrv"].append({
                "uuid": record.get("uuid"),
                "sourceName": record.get("sourceName"),
                "unit": record.get("unit"),
                "startDate": record.get("startDate"),
                "endDate": record.get("endDate"),
                "value": record.get("value"),
            })
        elif rtype == RECORD_MAP["resp"]:
            buckets["resp"].append({
                "uuid": record.get("uuid"),
                "sourceName": record.get("sourceName"),
                "unit": record.get("unit"),
                "startDate": record.get("startDate"),
                "endDate": record.get("endDate"),
                "value": record.get("value"),
            })
        elif rtype == RECORD_MAP["spo2"]:
            buckets["spo2"].append({
                "uuid": record.get("uuid"),
                "sourceName": record.get("sourceName"),
                "unit": record.get("unit"),
                "startDate": record.get("startDate"),
                "endDate": record.get("endDate"),
                "value": record.get("value"),
            })

    # Save counts
    counts_path = reports_dir / "record_type_counts.csv"
    with counts_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_type", "count"])
        for k, v in counters.most_common():
            writer.writerow([k, v])

    # Save each bucket as CSV if non-empty
    for key, rows in buckets.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{key}.csv", index=False)

    # Show quick heads for verification
    for key in buckets:
        path = out_dir / f"{key}.csv"
        if path.exists():
            df = pd.read_csv(path)
            print(f"[{key}] rows={len(df)}")
            print(df.head(20))
            print("-" * 40)
        else:
            print(f"[{key}] no records found")

    print(f"Record type counts saved to {counts_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse Apple Health export.xml")
    parser.add_argument("--input", required=True, type=Path, help="Path to export.xml")
    parser.add_argument("--out", default=Path("parsed_tables"), type=Path, help="Output directory for CSVs")
    args = parser.parse_args()
    parse(args.input, args.out)


if __name__ == "__main__":
    main()
