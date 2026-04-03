#!/usr/bin/env python3
"""
Predict fatigue label for a given date or JSON feature input.
Usage:
  ./predict.py --date YYYY-MM-DD           # uses dataset/model_data.csv features for that date
  ./predict.py --json path/to/features.json # json with feature keys matching model_data columns (except target)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import joblib

MODEL_PATH = Path("models/rf.pkl")  # best F1 in step5
DATA_PATH = Path("dataset/model_data.csv")
TARGET_COL = "fatigue_label"


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("Model file not found. Run scripts/step5_train_models.py first.")
    return joblib.load(MODEL_PATH)


def load_features_for_date(date_str: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "label_date" in df.columns:
        df["label_date"] = pd.to_datetime(df["label_date"]).dt.date
        row = df[df["label_date"] == pd.to_datetime(date_str).date()]
    else:
        row = df[df["night_date"] == date_str]
    if row.empty:
        raise SystemExit(f"No features found for date {date_str} in {DATA_PATH}")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return row[feature_cols]


def load_features_from_json(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text())
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Predict fatigue")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Date (YYYY-MM-DD) corresponding to label day")
    group.add_argument("--json", type=Path, help="Path to JSON with feature values")
    args = parser.parse_args()

    model = load_model()

    if args.date:
        X = load_features_for_date(args.date)
    else:
        X = load_features_from_json(args.json)

    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)

    for i, label in enumerate(preds):
        prob_str = ""
        if probs is not None:
            prob = probs[i].max()
            prob_str = f" (confidence={prob:.2f})"
        print(f"prediction[{i}]: {label}{prob_str}")


if __name__ == "__main__":
    main()
