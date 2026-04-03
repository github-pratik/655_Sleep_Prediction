#!/usr/bin/env python3
"""
Predict fatigue label for a given date or JSON feature input.
Usage:
  ./predict.py --date YYYY-MM-DD            # uses dataset/model_data.csv features for that date
  ./predict.py --json path/to/features.json # json with feature keys matching model_data columns (except target)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import joblib

DATA_PATH = Path("dataset/model_data.csv")
TARGET_COL = "fatigue_label"
MODEL_CANDIDATES = [
    Path("models/mobile_champion.pkl"),
    Path("models/rf.pkl"),
    Path("models/logreg.pkl"),
]


def resolve_model_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise SystemExit(f"Model file not found: {explicit}")

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "No model file found. Run scripts/step5_train_mobile_models.py (or step5_train_models.py) first."
    )


def load_model(path: Path):
    return joblib.load(path)


def model_feature_columns(model) -> list[str]:
    try:
        cols = model.named_steps["preprocess"].transformers_[0][2]
        return list(cols)
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unable to read feature columns from model: {exc}") from exc


def align_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = pd.NA
    return df[feature_cols]


def load_features_for_date(date_str: str, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "label_date" in df.columns:
        df["label_date"] = pd.to_datetime(df["label_date"]).dt.date
        row = df[df["label_date"] == pd.to_datetime(date_str).date()]
    else:
        row = df[df["night_date"] == date_str]
    if row.empty:
        raise SystemExit(f"No features found for date {date_str} in {DATA_PATH}")
    return align_columns(row, feature_cols)


def load_features_from_json(json_path: Path, feature_cols: list[str]) -> pd.DataFrame:
    data = json.loads(json_path.read_text())
    if isinstance(data, dict):
        data = [data]
    return align_columns(pd.DataFrame(data), feature_cols)


def main():
    parser = argparse.ArgumentParser(description="Predict fatigue")
    parser.add_argument("--model", type=Path, help="Optional model path (defaults to mobile champion when present)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Date (YYYY-MM-DD) corresponding to label day")
    group.add_argument("--json", type=Path, help="Path to JSON with feature values")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model = load_model(model_path)
    feature_cols = model_feature_columns(model)

    if args.date:
        X = load_features_for_date(args.date, feature_cols)
    else:
        X = load_features_from_json(args.json, feature_cols)

    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)

    print(f"model: {model_path}")
    for i, label in enumerate(preds):
        prob_str = ""
        if probs is not None:
            prob = probs[i].max()
            prob_str = f" (confidence={prob:.2f})"
        print(f"prediction[{i}]: {label}{prob_str}")


if __name__ == "__main__":
    main()
