#!/usr/bin/env python3
"""
Time-aware train/test split (70/30 by chronological order).
Input: dataset/model_data.csv (must include fatigue_label)
Outputs:
  dataset/train.csv
  dataset/test.csv
  reports/split_stats.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=Path("dataset/model_data.csv"), type=Path)
    parser.add_argument("--train-out", default=Path("dataset/train.csv"), type=Path)
    parser.add_argument("--test-out", default=Path("dataset/test.csv"), type=Path)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit("dataset/model_data.csv not found. Run step3_merge_labels after adding fatigue_labels.csv.")

    df = pd.read_csv(args.data, parse_dates=["night_date", "sleep_start", "sleep_end"], infer_datetime_format=True)

    if "fatigue_label" not in df.columns:
        raise SystemExit("fatigue_label column missing in model_data.csv")

    # Sort chronologically by night_date (or label_date if present)
    sort_col = "label_date" if "label_date" in df.columns else "night_date"
    df = df.sort_values(sort_col).reset_index(drop=True)

    # Identify numeric feature columns (exclude non-numeric and target)
    target = "fatigue_label"
    non_feature_cols = {target, "night_date", "sleep_start", "sleep_end", "label_date", "date"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Move target to end for readability
    df = df[feature_cols + [target]]

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    stats = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "split_index": split_idx,
        "features": feature_cols,
        "class_balance_train": train_df[target].value_counts(dropna=False).to_dict(),
        "class_balance_test": test_df[target].value_counts(dropna=False).to_dict(),
    }
    report_path = Path("reports/split_stats.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2, default=str))

    print(json.dumps(stats, indent=2, default=str))
    print(f"Saved {args.train_out} and {args.test_out}")


if __name__ == "__main__":
    main()
