#!/usr/bin/env python3
"""
Time-aware transfer finetuning for next-day fatigue prediction.

Usage:
  python scripts/step7_transfer_finetune.py
  python scripts/step7_transfer_finetune.py --model-data dataset/model_data.csv
  python scripts/step7_transfer_finetune.py --encoder artifacts/public_pretrain/public_encoder.pkl

Outputs:
  reports/public_transfer_ablation.csv
  reports/public_transfer_report.json
  models/transfer_champion.pkl

The script evaluates three feature ablations:
  - baseline features only
  - transfer features only
  - baseline + transfer

It uses TimeSeriesSplit to avoid temporal leakage and benchmarks the final fitted
pipeline with the same lightweight desktop proxy approach used elsewhere in the repo.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transfer import FeatureBundle, build_feature_spaces, load_model_data
from src.transfer.data import TARGET_COL

RANDOM_STATE = 42
DEFAULT_DATA = ROOT / "dataset/model_data.csv"
DEFAULT_ENCODER = ROOT / "artifacts/public_pretrain/public_encoder.pkl"


def encode_labels(y: pd.Series) -> tuple[LabelEncoder, np.ndarray]:
    le = LabelEncoder()
    encoded = le.fit_transform(y.astype(str))
    return le, encoded


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def mobile_efficiency_score(size_kb: float, p95_ms: float) -> float:
    size_score = 1.0 if size_kb <= 1000 else max(0.0, 1.0 - ((size_kb - 1000.0) / 3000.0))
    lat_score = 1.0 if p95_ms <= 25 else max(0.0, 1.0 - ((p95_ms - 25.0) / 100.0))
    return float(0.65 * size_score + 0.35 * lat_score)


def overall_mobile_score(metrics: dict[str, float], efficiency_score: float) -> float:
    kappa_norm = (metrics["kappa"] + 1.0) / 2.0
    return float(
        0.55 * metrics["weighted_f1"]
        + 0.15 * metrics["macro_f1"]
        + 0.15 * kappa_norm
        + 0.15 * efficiency_score
    )


def serialized_size_bytes(model_pipeline) -> int:
    buf = io.BytesIO()
    joblib.dump(model_pipeline, buf)
    return int(buf.tell())


def benchmark_latency_ms(model_pipeline, X: pd.DataFrame) -> dict[str, dict[str, float]]:
    sample = X.iloc[[0]]

    for _ in range(20):
        _ = model_pipeline.predict(sample)

    single_times: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = model_pipeline.predict(sample)
        single_times.append((time.perf_counter() - t0) * 1000.0)

    batch_times: list[float] = []
    for _ in range(60):
        t0 = time.perf_counter()
        _ = model_pipeline.predict(X)
        batch_times.append((time.perf_counter() - t0) * 1000.0)

    def stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }

    return {"single_pred_ms": stats(single_times), "batch_pred_ms": stats(batch_times)}


def choose_valid_time_splits(y: np.ndarray, max_splits: int = 5) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    for n_splits in range(max_splits, 2, -1):
        splitter = TimeSeriesSplit(n_splits=n_splits)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        valid = True
        for train_idx, test_idx in splitter.split(np.zeros(len(y))):
            y_train = y[train_idx]
            y_test = y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                valid = False
                break
            folds.append((train_idx, test_idx))
        if valid and folds:
            return folds, n_splits
    raise SystemExit(
        "Unable to build a valid TimeSeriesSplit with both classes in every fold. "
        "Check label order or reduce temporal sparsity."
    )


def build_pipeline(model_name: str, feature_cols: list[str]):
    if model_name == "logreg":
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_name == "rf_small":
        return Pipeline(
            steps=[
                ("preprocess", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=96,
                        max_depth=8,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "hist_gb":
        return Pipeline(
            steps=[
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=500,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def select_trainable_columns(X: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    ordered = [c for c in feature_cols if c in X.columns]
    keep = [c for c in ordered if X[c].notna().any()]
    if not keep:
        raise ValueError("No trainable columns after dropping all-missing features.")
    return keep


def fit_pipeline(model_name: str, feature_cols: list[str], X: pd.DataFrame, y: np.ndarray):
    trainable_cols = select_trainable_columns(X, feature_cols)
    pipeline = build_pipeline(model_name, feature_cols)
    pipeline.fit(X[trainable_cols], y)
    return pipeline, trainable_cols


def cv_evaluate(model_name: str, bundle: FeatureBundle, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    fold_rows = []
    max_dropped = 0
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train = bundle.frame.iloc[train_idx][bundle.feature_order]
        X_test = bundle.frame.iloc[test_idx][bundle.feature_order]
        y_train = y[train_idx]
        y_test = y[test_idx]

        trainable_cols = select_trainable_columns(X_train, bundle.feature_order)
        dropped_cols = len(bundle.feature_order) - len(trainable_cols)
        max_dropped = max(max_dropped, dropped_cols)
        pipeline = build_pipeline(model_name, bundle.feature_order)
        fit_start = time.perf_counter()
        pipeline.fit(X_train[trainable_cols], y_train)
        fit_ms = (time.perf_counter() - fit_start) * 1000.0

        pred_start = time.perf_counter()
        y_pred = pipeline.predict(X_test[trainable_cols])
        pred_ms = (time.perf_counter() - pred_start) * 1000.0

        metrics = evaluate_metrics(y_test, y_pred)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "trained_feature_count": int(len(trainable_cols)),
                "dropped_all_missing_feature_count": int(dropped_cols),
                "fit_ms": float(fit_ms),
                "predict_ms": float(pred_ms),
                **metrics,
            }
        )

    agg = {}
    for key in ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_precision", "macro_recall", "macro_f1", "kappa"):
        values = np.asarray([row[key] for row in fold_rows], dtype=float)
        agg[f"mean_{key}"] = float(values.mean())
        agg[f"std_{key}"] = float(values.std(ddof=0))

    return {"fold_rows": fold_rows, "aggregate": agg, "max_dropped_all_missing_features": int(max_dropped)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-aware transfer finetuning for fatigue prediction")
    parser.add_argument("--model-data", default=DEFAULT_DATA, type=Path, help="Path to dataset/model_data.csv")
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        type=Path,
        help="Optional public encoder artifact at artifacts/public_pretrain/public_encoder.pkl",
    )
    parser.add_argument("--max-splits", default=5, type=int, help="Maximum TimeSeriesSplit folds to try")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmarking")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    df = load_model_data(args.model_data)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column {TARGET_COL!r} not found in {args.model_data}")

    y_le, y = encode_labels(df[TARGET_COL])
    feature_spaces = build_feature_spaces(df, encoder_path=args.encoder)
    folds, splits_used = choose_valid_time_splits(y, max_splits=args.max_splits)

    results: list[dict[str, Any]] = []
    fold_details: dict[str, dict[str, Any]] = {}

    for ablation_name, bundle in feature_spaces.items():
        for model_name in ("logreg", "rf_small", "hist_gb"):
            cv_out = cv_evaluate(model_name, bundle, y, folds)
            full_pipeline, trained_cols = fit_pipeline(model_name, bundle.feature_order, bundle.frame, y)

            size_kb = serialized_size_bytes(full_pipeline) / 1024.0
            if args.skip_latency:
                latency = {
                    "single_pred_ms": {"mean_ms": None, "p50_ms": None, "p95_ms": None},
                    "batch_pred_ms": {"mean_ms": None, "p50_ms": None, "p95_ms": None},
                }
                single_p95 = None
            else:
                latency = benchmark_latency_ms(full_pipeline, bundle.frame[trained_cols])
                single_p95 = latency["single_pred_ms"]["p95_ms"]

            metrics = {
                "mean_accuracy": cv_out["aggregate"]["mean_accuracy"],
                "std_accuracy": cv_out["aggregate"]["std_accuracy"],
                "mean_weighted_f1": cv_out["aggregate"]["mean_weighted_f1"],
                "std_weighted_f1": cv_out["aggregate"]["std_weighted_f1"],
                "mean_macro_f1": cv_out["aggregate"]["mean_macro_f1"],
                "std_macro_f1": cv_out["aggregate"]["std_macro_f1"],
                "mean_kappa": cv_out["aggregate"]["mean_kappa"],
                "std_kappa": cv_out["aggregate"]["std_kappa"],
            }
            efficiency = mobile_efficiency_score(
                size_kb=size_kb,
                p95_ms=single_p95 if single_p95 is not None else 9999.0,
            )
            mobile_score = overall_mobile_score(
                {
                    "weighted_f1": metrics["mean_weighted_f1"],
                    "macro_f1": metrics["mean_macro_f1"],
                    "kappa": metrics["mean_kappa"],
                },
                efficiency,
            )
            eligible_mobile = bool(size_kb <= 1000 and (single_p95 is None or single_p95 <= 25))

            model_path = ROOT / f"models/{ablation_name}_{model_name}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(full_pipeline, model_path)

            row = {
                "ablation": ablation_name,
                "model_name": model_name,
                "feature_count": len(bundle.feature_order),
                "trained_feature_count": int(len(trained_cols)),
                "dropped_all_missing_feature_count": int(len(bundle.feature_order) - len(trained_cols)),
                "eligible_mobile": eligible_mobile,
                "size_kb": float(size_kb),
                "single_pred_p50_ms": latency["single_pred_ms"]["p50_ms"],
                "single_pred_p95_ms": latency["single_pred_ms"]["p95_ms"],
                "batch_pred_p50_ms": latency["batch_pred_ms"]["p50_ms"],
                "batch_pred_p95_ms": latency["batch_pred_ms"]["p95_ms"],
                "efficiency_score": float(efficiency),
                "mobile_score": float(mobile_score),
                "model_path": str(model_path),
                **metrics,
            }
            results.append(row)
            fold_details[f"{ablation_name}::{model_name}"] = {
                "bundle_metadata": bundle.metadata,
                "trained_feature_order": trained_cols,
                "max_dropped_all_missing_features": cv_out["max_dropped_all_missing_features"],
                "folds": cv_out["fold_rows"],
            }

    results_df = pd.DataFrame(results).sort_values(
        ["eligible_mobile", "mean_weighted_f1", "mobile_score"],
        ascending=[False, False, False],
    )
    results_df.to_csv(ROOT / "reports/public_transfer_ablation.csv", index=False)

    eligible_df = results_df[results_df["eligible_mobile"] == True]  # noqa: E712
    if len(eligible_df) > 0:
        champion_row = eligible_df.iloc[0]
    else:
        champion_row = results_df.iloc[0]

    champion_bundle = feature_spaces[str(champion_row["ablation"])]
    champion_model_name = str(champion_row["model_name"])
    champion_pipeline, champion_cols = fit_pipeline(champion_model_name, champion_bundle.feature_order, champion_bundle.frame, y)
    champion_path = ROOT / "models/transfer_champion.pkl"
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, champion_path)

    report = {
        "summary": {
            "rows": int(len(df)),
            "labels": [str(c) for c in y_le.classes_.tolist()],
            "cv_splits_requested": int(args.max_splits),
            "cv_splits_used": int(splits_used),
            "encoder_requested": str(args.encoder),
            "encoder_available": bool(feature_spaces["transfer"].metadata.get("encoder_available", False)),
            "champion_ablation": str(champion_row["ablation"]),
            "champion_model": champion_model_name,
            "champion_feature_count": int(len(champion_cols)),
            "champion_feature_order": champion_cols,
            "champion_model_path": str(champion_path),
        },
        "feature_spaces": {
            name: {
                "feature_count": int(bundle.metadata.get("feature_count", len(bundle.feature_order))),
                "feature_order": bundle.feature_order,
                "metadata": bundle.metadata,
            }
            for name, bundle in feature_spaces.items()
        },
        "all_results": results,
        "fold_details": fold_details,
    }
    (ROOT / "reports/public_transfer_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report["summary"], indent=2))
    print("Saved reports/public_transfer_ablation.csv")
    print("Saved reports/public_transfer_report.json")
    print("Saved models/transfer_champion.pkl")


if __name__ == "__main__":
    main()
