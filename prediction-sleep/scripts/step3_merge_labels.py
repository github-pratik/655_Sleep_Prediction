#!/usr/bin/env python3
"""
Merge user-provided fatigue labels with night-level features.
If fatigue_labels.csv is missing, create a template for the user to fill.
Inputs:
  dataset/night_features.csv
  dataset/fatigue_labels.csv (user provided)
Outputs:
  dataset/fatigue_labels_template.csv (if labels missing)
  dataset/model_data.csv (when labels present)
Reports:
  reports/label_checks.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--night-features", default=Path("dataset/night_features.csv"), type=Path)
    parser.add_argument("--labels", default=Path("dataset/fatigue_labels.csv"), type=Path)
    parser.add_argument("--out", default=Path("dataset/model_data.csv"), type=Path)
    args = parser.parse_args()

    night_df = pd.read_csv(args.night_features, parse_dates=["night_date"])

    if not args.labels.exists():
        tmpl = night_df[["night_date"]].copy()
        tmpl["date"] = (tmpl["night_date"] + pd.Timedelta(days=1)).dt.date
        tmpl["fatigue_label"] = ""
        tmpl = tmpl[["date", "fatigue_label"]]
        tmpl.to_csv(args.labels.parent / "fatigue_labels_template.csv", index=False)
        print(
            "fatigue_labels.csv not found. Created template at dataset/fatigue_labels_template.csv. "
            "Fill it with date (YYYY-MM-DD) and fatigue_label (0/1)."
        )
        return

    labels_df = pd.read_csv(args.labels)
    # normalize date column name if user used night_date
    if "night_date" in labels_df.columns and "date" not in labels_df.columns:
        labels_df = labels_df.rename(columns={"night_date": "date"})

    if "date" not in labels_df.columns:
        raise ValueError("labels file must have column date (or night_date)")

    if "fatigue_label" not in labels_df.columns:
        raise ValueError("labels file must have column fatigue_label")

    labels_df["date"] = pd.to_datetime(labels_df["date"], errors="coerce").dt.date
    if labels_df["date"].isna().any():
        raise ValueError("labels file has invalid date values; expected YYYY-MM-DD")

    labels_df["fatigue_label"] = labels_df["fatigue_label"].astype(str).str.strip()
    valid_labels = {"0", "1"}
    invalid = labels_df[~labels_df["fatigue_label"].isin(valid_labels)]
    if not invalid.empty:
        bad_values = sorted(invalid["fatigue_label"].dropna().unique().tolist())
        raise ValueError(
            f"fatigue_label must be binary 0/1; found invalid values: {bad_values}"
        )
    labels_df["fatigue_label"] = labels_df["fatigue_label"].astype(int)

    # align: night_date + 1 day -> label date
    night_df["label_date"] = night_df["night_date"].dt.date + pd.Timedelta(days=1)

    # check duplicates
    dupes = labels_df[labels_df.duplicated("date", keep=False)]
    if not dupes.empty:
        print("Warning: duplicate label dates found. Keeping the first occurrence.")
        labels_df = labels_df.drop_duplicates("date", keep="first")

    merged = night_df.merge(labels_df, left_on="label_date", right_on="date", how="left")

    missing = merged[merged["fatigue_label"].isna()]
    if not missing.empty:
        missing_path = Path("reports/missing_label_dates.csv")
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing[["night_date", "label_date"]].to_csv(missing_path, index=False)
        raise SystemExit(
            f"Missing fatigue labels for {len(missing)} nights. "
            f"Fill labels and rerun. Details: {missing_path}"
        )

    stats = {
        "nights_total": len(merged),
        "labels_present": int(merged["fatigue_label"].notna().sum()),
        "labels_missing": len(missing),
        "class_balance": merged["fatigue_label"].value_counts(dropna=True).to_dict(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    report_path = Path("reports/label_checks.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2, default=str))

    print(f"Merged dataset saved to {args.out}")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
