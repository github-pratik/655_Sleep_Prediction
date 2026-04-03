#!/usr/bin/env python3
"""
Train and benchmark mobile-friendly models for next-day fatigue prediction.

Inputs:
  dataset/train.csv
  dataset/test.csv

Outputs:
  models/<candidate>.pkl
  models/mobile_champion.pkl
  reports/mobile_model_report.json
  reports/mobile_model_scores.csv
  reports/mobile_confusion_matrix.png
  artifacts/feature_schema.json
  artifacts/mobile_linear_contract.json (when champion is linear)
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "fatigue_label"
RANDOM_STATE = 42


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]
    return feature_cols, X_train, X_test, y_train, y_test


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return le, y_train_enc, y_test_enc


def build_preprocessor(feature_cols):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )


def evaluate(y_true, y_pred):
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(w_prec),
        "weighted_recall": float(w_rec),
        "weighted_f1": float(w_f1),
        "macro_precision": float(m_prec),
        "macro_recall": float(m_rec),
        "macro_f1": float(m_f1),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def benchmark_latency_ms(model_pipeline, X_test: pd.DataFrame):
    sample = X_test.iloc[[0]]

    # Warm up model/runtime state.
    for _ in range(20):
        _ = model_pipeline.predict(sample)

    single_times_ms = []
    for _ in range(250):
        t0 = time.perf_counter()
        _ = model_pipeline.predict(sample)
        single_times_ms.append((time.perf_counter() - t0) * 1000)

    batch_times_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model_pipeline.predict(X_test)
        batch_times_ms.append((time.perf_counter() - t0) * 1000)

    def _stats(arr):
        a = np.array(arr, dtype=float)
        return {
            "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
        }

    return {
        "single_pred_ms": _stats(single_times_ms),
        "batch_pred_ms": _stats(batch_times_ms),
    }


def serialized_size_bytes(model_pipeline) -> int:
    buf = io.BytesIO()
    joblib.dump(model_pipeline, buf)
    return int(buf.tell())


def mobile_efficiency_score(size_kb: float, single_p95_ms: float) -> float:
    # Prefer <= 1 MB and <= 25 ms p95 single prediction.
    if size_kb <= 1000:
        size_score = 1.0
    else:
        size_score = max(0.0, 1.0 - ((size_kb - 1000) / 3000.0))

    if single_p95_ms <= 25:
        lat_score = 1.0
    else:
        lat_score = max(0.0, 1.0 - ((single_p95_ms - 25) / 100.0))

    return float(0.65 * size_score + 0.35 * lat_score)


def overall_mobile_score(metrics: dict[str, Any], efficiency_score: float) -> float:
    kappa_norm = (metrics["kappa"] + 1.0) / 2.0
    return float(
        0.55 * metrics["weighted_f1"]
        + 0.15 * metrics["macro_f1"]
        + 0.15 * kappa_norm
        + 0.15 * efficiency_score
    )


def is_mobile_eligible(size_kb: float, single_p95_ms: float) -> bool:
    return size_kb <= 1000 and single_p95_ms <= 25


def plot_confusion(cm, labels, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable, skipping confusion plot: {exc}")
        return

    try:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to render confusion plot, skipping: {exc}")


def export_feature_schema(feature_cols, classes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_order": feature_cols,
        "num_features": len(feature_cols),
        "target_col": TARGET_COL,
        "classes": classes,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def export_linear_contract_if_possible(model_pipeline, feature_cols, out_path: Path):
    model = model_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return False

    preprocess = model_pipeline.named_steps["preprocess"]
    num_pipe = preprocess.named_transformers_["num"]
    imputer = num_pipe.named_steps["imputer"]
    scaler = num_pipe.named_steps["scaler"]

    if len(getattr(model, "coef_", [])) != 1:
        # Keep export focused on binary logistic-style inference.
        return False

    payload = {
        "contract_type": "binary_linear_classifier",
        "feature_order": feature_cols,
        "imputer_median": {
            col: float(imputer.statistics_[idx]) for idx, col in enumerate(feature_cols)
        },
        "scaler_mean": {
            col: float(scaler.mean_[idx]) for idx, col in enumerate(feature_cols)
        },
        "scaler_scale": {
            col: float(scaler.scale_[idx]) for idx, col in enumerate(feature_cols)
        },
        "coef": {col: float(model.coef_[0][idx]) for idx, col in enumerate(feature_cols)},
        "intercept": float(model.intercept_[0]),
        "classes": [int(c) for c in model.classes_.tolist()],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return True


def build_candidates():
    candidates = []

    for c in [0.1, 0.5, 1.0, 2.0]:
        candidates.append(
            (
                f"logreg_c{str(c).replace('.', '_')}",
                LogisticRegression(
                    max_iter=2000,
                    C=c,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        )

    for alpha in [1e-4, 5e-4, 1e-3]:
        alpha_tag = str(alpha).replace(".", "_")
        candidates.append(
            (
                f"sgd_log_alpha{alpha_tag}",
                SGDClassifier(
                    loss="log_loss",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=5000,
                    tol=1e-3,
                    random_state=RANDOM_STATE,
                ),
            )
        )

    # Small forests kept as reference points only.
    for n_estimators, max_depth in [(64, 6), (96, 8)]:
        candidates.append(
            (
                f"rf_small_n{n_estimators}_d{max_depth}",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=Path("dataset/train.csv"), type=Path)
    parser.add_argument("--test", default=Path("dataset/test.csv"), type=Path)
    parser.add_argument("--skip-plots", action="store_true", help="Skip matplotlib plot generation")
    args = parser.parse_args()

    feature_cols, X_train, X_test, y_train_raw, y_test_raw = load_data(args.train, args.test)
    label_encoder, y_train, y_test = encode_labels(y_train_raw, y_test_raw)

    candidates = build_candidates()
    model_store: dict[str, Pipeline] = {}
    rows = []

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    for model_name, estimator in candidates:
        preprocessor = build_preprocessor(feature_cols)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        latency = benchmark_latency_ms(pipeline, X_test)
        size_bytes = serialized_size_bytes(pipeline)
        size_kb = size_bytes / 1024.0

        single_p95 = latency["single_pred_ms"]["p95_ms"]
        eff_score = mobile_efficiency_score(size_kb=size_kb, single_p95_ms=single_p95)
        mobile_score = overall_mobile_score(metrics, eff_score)
        eligible = is_mobile_eligible(size_kb=size_kb, single_p95_ms=single_p95)

        model_path = Path(f"models/{model_name}.pkl")
        joblib.dump(pipeline, model_path)
        model_store[model_name] = pipeline

        row = {
            "model_name": model_name,
            "eligible_mobile": bool(eligible),
            "size_kb": float(size_kb),
            "single_pred_p50_ms": float(latency["single_pred_ms"]["p50_ms"]),
            "single_pred_p95_ms": float(single_p95),
            "batch_pred_p50_ms": float(latency["batch_pred_ms"]["p50_ms"]),
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_f1": metrics["macro_f1"],
            "kappa": metrics["kappa"],
            "efficiency_score": eff_score,
            "mobile_score": mobile_score,
            "cm": metrics["cm"],
            "model_path": str(model_path),
        }
        rows.append(row)
        print(
            f"{model_name}: f1={metrics['weighted_f1']:.3f}, "
            f"size={size_kb:.1f}KB, p95={single_p95:.2f}ms, eligible={eligible}"
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["eligible_mobile", "mobile_score", "weighted_f1"], ascending=[False, False, False]
    )
    results_df.to_csv("reports/mobile_model_scores.csv", index=False)

    eligible_df = results_df[results_df["eligible_mobile"] == True]  # noqa: E712
    if len(eligible_df) > 0:
        champion_row = eligible_df.sort_values(
            ["weighted_f1", "mobile_score"], ascending=[False, False]
        ).iloc[0]
    else:
        champion_row = results_df.iloc[0]

    champion_name = champion_row["model_name"]
    champion_model = model_store[champion_name]
    champion_metrics = next(r for r in rows if r["model_name"] == champion_name)

    joblib.dump(champion_model, Path("models/mobile_champion.pkl"))
    Path("models/mobile_champion_name.txt").write_text(str(champion_name))

    export_feature_schema(
        feature_cols=feature_cols,
        classes=[int(c) for c in label_encoder.classes_.tolist()],
        out_path=Path("artifacts/feature_schema.json"),
    )

    # Export linear contract from the best linear candidate, even when champion is non-linear.
    linear_rows = [
        r for r in rows if r["model_name"].startswith("logreg_") or r["model_name"].startswith("sgd_log_")
    ]
    linear_contract_model_name = None
    has_contract = False
    if linear_rows:
        best_linear = sorted(linear_rows, key=lambda r: r["weighted_f1"], reverse=True)[0]
        linear_contract_model_name = best_linear["model_name"]
        has_contract = export_linear_contract_if_possible(
            model_pipeline=model_store[linear_contract_model_name],
            feature_cols=feature_cols,
            out_path=Path("artifacts/mobile_linear_contract.json"),
        )

    report = {
        "summary": {
            "num_candidates": len(rows),
            "num_mobile_eligible": int((results_df["eligible_mobile"] == True).sum()),  # noqa: E712
            "champion": champion_name,
            "champion_model_path": "models/mobile_champion.pkl",
            "best_linear_model": linear_contract_model_name,
            "linear_contract_exported": bool(has_contract),
        },
        "champion_metrics": champion_metrics,
        "all_models": rows,
    }
    Path("reports/mobile_model_report.json").write_text(json.dumps(report, indent=2))

    if not args.skip_plots:
        plot_confusion(
            np.array(champion_metrics["cm"]),
            labels=[str(c) for c in label_encoder.classes_.tolist()],
            out_path=Path("reports/mobile_confusion_matrix.png"),
        )

    print(json.dumps(report["summary"], indent=2))
    print("Saved reports/mobile_model_report.json")
    print("Saved reports/mobile_model_scores.csv")


if __name__ == "__main__":
    main()
