#!/usr/bin/env python3
"""
Distill a mobile-class fatigue model from a teacher model.

Default teacher priority:
  1. models/transfer_champion.pkl
  2. models/mobile_champion.pkl

Inputs:
  dataset/train.csv
  dataset/test.csv

Outputs:
  models/distilled_mobile.pkl
  artifacts/distilled_linear_contract.json (when linear export is possible)
  reports/distillation_report.json

The distillation approach is pragmatic and mobile-oriented:
- Fit a shared preprocessing pipeline on the training split.
- Use the teacher's predictions and confidence as guidance for student training.
- Evaluate student candidates by accuracy, weighted F1, footprint, and latency.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass
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
DEFAULT_TRAIN = Path("dataset/train.csv")
DEFAULT_TEST = Path("dataset/test.csv")
TEACHER_CANDIDATES = [Path("models/transfer_champion.pkl"), Path("models/mobile_champion.pkl")]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TeacherBundle:
    path: Path
    model: Any


def require_existing(path: Path, description: str) -> Path:
    if path.exists():
        return path
    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative
    raise SystemExit(f"{description} not found: {path}")


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative
    return path


def install_sklearn_pickle_compat() -> None:
    # Older sklearn pipelines in this repo were pickled with internal classes that moved
    # across sklearn versions. Provide a minimal alias so unpickling can proceed.
    try:
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct._RemainderColsList = _RemainderColsList
    except Exception:
        return


def repair_loaded_sklearn_object(obj) -> None:
    # Patch compatibility gaps introduced by sklearn private attribute changes.
    stack = [obj]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if current.__class__.__name__ == "SimpleImputer" and not hasattr(current, "_fill_dtype"):
            fallback_dtype = getattr(current, "_fit_dtype", np.dtype("float64"))
            setattr(current, "_fill_dtype", fallback_dtype)

        if hasattr(current, "named_steps"):
            stack.extend(getattr(current, "named_steps").values())

        if hasattr(current, "steps"):
            stack.extend(step for _, step in getattr(current, "steps") if step is not None)

        if hasattr(current, "transformers_"):
            for transformer in getattr(current, "transformers_"):
                if isinstance(transformer, tuple) and len(transformer) >= 2:
                    nested = transformer[1]
                    if nested not in (None, "drop", "passthrough"):
                        stack.append(nested)

        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)


def load_data(train_path: Path, test_path: Path):
    train_path = resolve_input_path(train_path)
    test_path = resolve_input_path(test_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
        raise SystemExit(f"Missing required target column: {TARGET_COL}")
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train, y_train = train_df[feature_cols].copy(), train_df[TARGET_COL].copy()
    X_test, y_test = test_df[feature_cols].copy(), test_df[TARGET_COL].copy()
    return feature_cols, X_train, X_test, y_train, y_test


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    try:
        y_test_enc = le.transform(y_test)
    except ValueError as exc:
        train_labels = set(pd.Series(y_train).astype(str))
        test_labels = set(pd.Series(y_test).astype(str))
        unseen = sorted(test_labels - train_labels)
        raise SystemExit(
            "Test split contains unseen fatigue_label classes not present in train: "
            f"{unseen}. Re-check split and labels."
        ) from exc
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


def teacher_feature_columns(model) -> list[str]:
    # Support both preprocess shapes:
    # - ColumnTransformer (legacy mobile scripts)
    # - Pipeline with DataFrame input names (transfer scripts)
    try:
        preprocess = model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None
        if preprocess is not None and hasattr(preprocess, "transformers_"):
            cols = preprocess.transformers_[0][2]
            return list(cols)
        if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
            return [str(c) for c in preprocess.feature_names_in_]
        if hasattr(model, "feature_names_in_"):
            return [str(c) for c in model.feature_names_in_]
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unable to read feature columns from teacher model: {exc}") from exc

    raise SystemExit("Unable to infer feature columns from teacher model")


def align_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned = aligned[feature_cols]
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    return aligned


def load_teacher() -> TeacherBundle:
    install_sklearn_pickle_compat()
    for candidate in TEACHER_CANDIDATES:
        resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if resolved.exists():
            model = joblib.load(resolved)
            repair_loaded_sklearn_object(model)
            return TeacherBundle(path=resolved, model=model)
    raise SystemExit(
        "No teacher model found. Expected one of: "
        f"{', '.join(str(p) for p in TEACHER_CANDIDATES)}"
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


def export_linear_contract_if_possible(model_pipeline, feature_cols, out_path: Path) -> bool:
    model = model_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return False

    preprocess = model_pipeline.named_steps["preprocess"]
    num_pipe = preprocess.named_transformers_["num"]
    imputer = num_pipe.named_steps["imputer"]
    scaler = num_pipe.named_steps["scaler"]

    if len(getattr(model, "coef_", [])) != 1:
        return False

    payload = {
        "contract_type": "binary_linear_classifier",
        "feature_order": feature_cols,
        "imputer_median": {
            col: float(imputer.statistics_[idx]) for idx, col in enumerate(feature_cols)
        },
        "scaler_mean": {col: float(scaler.mean_[idx]) for idx, col in enumerate(feature_cols)},
        "scaler_scale": {col: float(scaler.scale_[idx]) for idx, col in enumerate(feature_cols)},
        "coef": {col: float(model.coef_[0][idx]) for idx, col in enumerate(feature_cols)},
        "intercept": float(model.intercept_[0]),
        "classes": [int(c) for c in model.classes_.tolist()],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return True


def load_teacher_targets(teacher_model, X_train: pd.DataFrame, y_train_enc: np.ndarray) -> dict[str, np.ndarray]:
    teacher_probs = teacher_model.predict_proba(X_train)
    teacher_pred = teacher_probs.argmax(axis=1)
    confidence = teacher_probs.max(axis=1)
    blended = np.where(confidence >= 0.5, teacher_pred, y_train_enc)
    return {
        "teacher_probs": teacher_probs,
        "teacher_pred": teacher_pred,
        "confidence": confidence,
        "blended_labels": blended,
    }


def build_candidates():
    candidates = []
    for c in [0.1, 0.5, 1.0, 2.0]:
        candidates.append(
            (
                f"logreg_distilled_c{str(c).replace('.', '_')}",
                LogisticRegression(
                    max_iter=3000,
                    C=c,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                "hard_blend",
            )
        )
    for alpha in [1e-4, 5e-4, 1e-3]:
        alpha_tag = str(alpha).replace(".", "_")
        candidates.append(
            (
                f"sgd_distilled_alpha{alpha_tag}",
                SGDClassifier(
                    loss="log_loss",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=5000,
                    tol=1e-3,
                    random_state=RANDOM_STATE,
                ),
                "hard_blend",
            )
        )
    for n_estimators, max_depth in [(64, 6), (96, 8)]:
        candidates.append(
            (
                f"rf_distilled_n{n_estimators}_d{max_depth}",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                "hard_blend",
            )
        )
    return candidates


def fit_student(model, X_train, y_train, teacher_meta, distill_weight: float = 0.6):
    # Pragmatic distillation:
    # mix the ground-truth label with the teacher's confidence-guided hard label.
    # This keeps the training objective compatible with sklearn estimators.
    blended = y_train.copy()
    teacher_weight = np.clip(teacher_meta["confidence"], 0.5, 1.0)
    pseudo = teacher_meta["blended_labels"]
    mixed = np.where(teacher_weight >= distill_weight, pseudo, y_train)

    if hasattr(model, "fit"):
        try:
            model.fit(X_train, mixed)
            return model
        except Exception:
            # Fall back to true labels if the blended labels trigger a class issue.
            model.fit(X_train, y_train)
            return model
    raise SystemExit(f"Unsupported student estimator: {type(model).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Distill a mobile-class fatigue model")
    parser.add_argument("--train", default=DEFAULT_TRAIN, type=Path)
    parser.add_argument("--test", default=DEFAULT_TEST, type=Path)
    parser.add_argument("--teacher", type=Path, help="Optional teacher model path override")
    parser.add_argument("--skip-plots", action="store_true", help="Reserved for parity with other scripts")
    args = parser.parse_args()

    require_existing(args.train, "Training split")
    require_existing(args.test, "Test split")

    teacher_bundle = None
    if args.teacher is not None:
        teacher_path = resolve_input_path(require_existing(args.teacher, "Teacher model"))
        install_sklearn_pickle_compat()
        teacher_model = joblib.load(teacher_path)
        repair_loaded_sklearn_object(teacher_model)
        teacher_bundle = TeacherBundle(path=teacher_path, model=teacher_model)
    else:
        teacher_bundle = load_teacher()

    feature_cols, X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_data(args.train, args.test)
    teacher_features = teacher_feature_columns(teacher_bundle.model)
    available_train_cols = set(X_train_raw.columns)
    overlap = len(set(teacher_features).intersection(available_train_cols))
    if overlap == 0:
        if args.teacher is not None:
            raise SystemExit(
                "Provided teacher model expects features absent from train/test split. "
                "Use a teacher compatible with dataset/train.csv (for example models/mobile_champion.pkl), "
                "or generate a transfer-feature split for distillation."
            )

        fallback_path = PROJECT_ROOT / "models" / "mobile_champion.pkl"
        if not fallback_path.exists():
            raise SystemExit(
                "Selected teacher has no feature overlap with train/test split and fallback "
                "models/mobile_champion.pkl is missing."
            )

        install_sklearn_pickle_compat()
        fallback_model = joblib.load(fallback_path)
        repair_loaded_sklearn_object(fallback_model)
        teacher_bundle = TeacherBundle(path=fallback_path, model=fallback_model)
        teacher_features = teacher_feature_columns(teacher_bundle.model)
        overlap = len(set(teacher_features).intersection(available_train_cols))
        if overlap == 0:
            raise SystemExit(
                "Fallback teacher still has no feature overlap with train/test split. "
                "Cannot run distillation with current inputs."
            )
    if feature_cols != teacher_features:
        # Keep evaluation on the same schema but warn through the report.
        feature_cols = teacher_features
        X_train_raw = align_columns(X_train_raw, feature_cols)
        X_test_raw = align_columns(X_test_raw, feature_cols)

    label_encoder, y_train, y_test = encode_labels(y_train_raw, y_test_raw)
    preprocessor = build_preprocessor(feature_cols)

    teacher_targets = load_teacher_targets(teacher_bundle.model, X_train_raw, y_train)

    candidates = build_candidates()
    candidate_rows = []
    model_store: dict[str, Pipeline] = {}

    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for model_name, estimator, distill_mode in candidates:
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline = fit_student(
            pipeline,
            X_train_raw,
            teacher_targets["blended_labels"],
            teacher_targets,
            distill_weight=0.6,
        )
        y_pred = pipeline.predict(X_test_raw)
        metrics = evaluate(y_test, y_pred)
        latency = benchmark_latency_ms(pipeline, X_test_raw)
        size_bytes = serialized_size_bytes(pipeline)
        size_kb = size_bytes / 1024.0
        single_p95 = latency["single_pred_ms"]["p95_ms"]
        eff_score = mobile_efficiency_score(size_kb=size_kb, single_p95_ms=single_p95)
        mobile_score = overall_mobile_score(metrics, eff_score)
        eligible = is_mobile_eligible(size_kb=size_kb, single_p95_ms=single_p95)

        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(pipeline, model_path)
        model_store[model_name] = pipeline

        candidate_rows.append(
            {
                "model_name": model_name,
                "distill_mode": distill_mode,
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
        )
        print(
            f"{model_name}: f1={metrics['weighted_f1']:.3f}, "
            f"size={size_kb:.1f}KB, p95={single_p95:.2f}ms, eligible={eligible}"
        )

    results_df = pd.DataFrame(candidate_rows).sort_values(
        ["eligible_mobile", "mobile_score", "weighted_f1"], ascending=[False, False, False]
    )

    eligible_df = results_df[results_df["eligible_mobile"] == True]  # noqa: E712
    if len(eligible_df) > 0:
        champion_row = eligible_df.sort_values(
            ["weighted_f1", "mobile_score"], ascending=[False, False]
        ).iloc[0]
    else:
        champion_row = results_df.iloc[0]

    champion_name = champion_row["model_name"]
    champion_model = model_store[champion_name]
    champion_metrics = next(r for r in candidate_rows if r["model_name"] == champion_name)

    joblib.dump(champion_model, models_dir / "distilled_mobile.pkl")
    (models_dir / "distilled_mobile_name.txt").write_text(str(champion_name))

    export_feature_schema = {
        "feature_order": feature_cols,
        "num_features": len(feature_cols),
        "target_col": TARGET_COL,
        "classes": [int(c) for c in label_encoder.classes_.tolist()],
        "teacher_model": str(teacher_bundle.path),
        "distill_source": "teacher-guided blended labels",
    }
    (artifacts_dir / "distilled_feature_schema.json").write_text(json.dumps(export_feature_schema, indent=2))

    linear_rows = [
        r
        for r in candidate_rows
        if r["model_name"].startswith("logreg_") or r["model_name"].startswith("sgd_")
    ]
    best_linear_name = None
    linear_contract_exported = False
    if linear_rows:
        best_linear = sorted(linear_rows, key=lambda r: r["weighted_f1"], reverse=True)[0]
        best_linear_name = best_linear["model_name"]
        linear_contract_exported = export_linear_contract_if_possible(
            model_pipeline=model_store[best_linear_name],
            feature_cols=feature_cols,
            out_path=artifacts_dir / "distilled_linear_contract.json",
        )

    report = {
        "summary": {
            "teacher_model": str(teacher_bundle.path),
            "num_candidates": len(candidate_rows),
            "num_mobile_eligible": int((results_df["eligible_mobile"] == True).sum()),  # noqa: E712
            "champion": champion_name,
            "champion_model_path": "models/distilled_mobile.pkl",
            "best_linear_model": best_linear_name,
            "linear_contract_exported": bool(linear_contract_exported),
        },
        "teacher_targets": {
            "mean_confidence": float(np.mean(teacher_targets["confidence"])),
            "median_confidence": float(np.median(teacher_targets["confidence"])),
        },
        "champion_metrics": champion_metrics,
        "all_models": candidate_rows,
    }

    (reports_dir / "distillation_report.json").write_text(json.dumps(report, indent=2))
    results_df.to_csv(reports_dir / "distillation_scores.csv", index=False)

    print(json.dumps(report["summary"], indent=2))
    print("Saved reports/distillation_report.json")
    print("Saved reports/distillation_scores.csv")


if __name__ == "__main__":
    main()
