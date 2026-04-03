#!/usr/bin/env python3
"""
Run research-grade evaluation for fatigue prediction.

Outputs:
  reports/robustness_metrics.json
  reports/research_summary.md

Evaluations:
  - bootstrap confidence intervals for weighted F1
  - synthetic missingness stress test
  - synthetic noise stress test

Default model priority:
  1. models/distilled_mobile.pkl
  2. models/transfer_champion.pkl
  3. models/mobile_champion.pkl
  4. models/rf.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

TARGET_COL = "fatigue_label"
MODEL_CANDIDATES = [
    Path("models/distilled_mobile.pkl"),
    Path("models/transfer_champion.pkl"),
    Path("models/mobile_champion.pkl"),
    Path("models/rf.pkl"),
]
DATASET_PATH = Path("dataset/test.csv")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    try:
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct._RemainderColsList = _RemainderColsList
    except Exception:
        return


def repair_loaded_sklearn_object(obj) -> None:
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


def pick_model_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return require_existing(explicit, "Model file")
    for candidate in MODEL_CANDIDATES:
        resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if resolved.exists():
            return resolved
    raise SystemExit(
        "No model file found. Expected one of: "
        f"{', '.join(str(p) for p in MODEL_CANDIDATES)}"
    )


def load_dataset(path: Path):
    path = resolve_input_path(path)
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Missing required target column: {TARGET_COL}")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df[TARGET_COL].copy()
    return feature_cols, X, y


def positive_class_index(classes: Any) -> int:
    classes_arr = np.asarray(classes)
    classes_list = classes_arr.tolist()
    if 1 in classes_list:
        return int(classes_list.index(1))
    return int(np.argmax(classes_arr))


def infer_threshold_from_distilled_contract() -> float | None:
    contract_path = PROJECT_ROOT / "artifacts" / "distilled_linear_contract.json"
    if not contract_path.exists():
        return None
    try:
        payload = json.loads(contract_path.read_text())
        value = payload.get("decision_threshold")
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def predict_labels(model, X: pd.DataFrame, decision_threshold: float | None = None) -> np.ndarray:
    if decision_threshold is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_", np.array([0, 1]))
        pos_idx = positive_class_index(classes)
        preds = (proba[:, pos_idx] >= float(decision_threshold)).astype(int)
        return np.asarray(preds)
    preds = model.predict(X)
    return np.asarray(preds)


def weighted_f1(y_true, y_pred) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return float(f1)


def bootstrap_ci(y_true, y_pred, n_bootstrap: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(weighted_f1(y_true[idx], y_pred[idx]))
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


def corrupt_missingness(X: pd.DataFrame, level: float, seed: int = 42) -> pd.DataFrame:
    if level <= 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    out = X.copy()
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_cols:
        return out
    mask = rng.random((len(out), len(numeric_cols))) < level
    arr = out[numeric_cols].to_numpy(dtype=float, copy=True)
    arr[mask] = np.nan
    out[numeric_cols] = arr
    return out


def corrupt_noise(X: pd.DataFrame, level: float, seed: int = 42) -> pd.DataFrame:
    if level <= 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    out = X.copy()
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_cols:
        return out
    for c in numeric_cols:
        series = pd.to_numeric(out[c], errors="coerce")
        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        noise = rng.normal(loc=0.0, scale=std * level, size=len(series))
        out[c] = series.to_numpy(dtype=float) + noise
    return out


def corruption_sweep(model, X: pd.DataFrame, y: pd.Series, decision_threshold: float | None = None):
    missing_levels = [0.0, 0.1, 0.3, 0.5]
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    results = {"missingness": [], "noise": []}

    for lvl in missing_levels:
        corrupted = corrupt_missingness(X, lvl)
        preds = predict_labels(model, corrupted, decision_threshold=decision_threshold)
        results["missingness"].append(
            {
                "level": lvl,
                "weighted_f1": weighted_f1(y, preds),
            }
        )

    for lvl in noise_levels:
        corrupted = corrupt_noise(X, lvl)
        preds = predict_labels(model, corrupted, decision_threshold=decision_threshold)
        results["noise"].append(
            {
                "level": lvl,
                "weighted_f1": weighted_f1(y, preds),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run research-grade evaluation")
    parser.add_argument("--model", type=Path, help="Optional model path override")
    parser.add_argument("--data", default=DATASET_PATH, type=Path, help="Test split CSV")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        help="Optional probability threshold override for binary models with predict_proba",
    )
    args = parser.parse_args()

    model_path = pick_model_path(args.model)
    require_existing(args.data, "Test split")
    install_sklearn_pickle_compat()
    model = joblib.load(model_path)
    repair_loaded_sklearn_object(model)
    feature_cols, X, y = load_dataset(args.data)

    decision_threshold = args.decision_threshold
    threshold_source = "default_model_predict"
    if decision_threshold is None and str(model_path).endswith("models/distilled_mobile.pkl"):
        inferred = infer_threshold_from_distilled_contract()
        if inferred is not None:
            decision_threshold = inferred
            threshold_source = "artifacts/distilled_linear_contract.json"
    elif decision_threshold is not None:
        threshold_source = "cli_override"

    y_pred = predict_labels(model, X, decision_threshold=decision_threshold)
    clean_f1 = weighted_f1(y, y_pred)
    ci = bootstrap_ci(y, y_pred, n_bootstrap=args.bootstrap)
    robustness = corruption_sweep(model, X, y, decision_threshold=decision_threshold)

    payload = {
        "model_path": str(model_path),
        "dataset_path": str(args.data),
        "n_rows": int(len(X)),
        "n_features": int(len(feature_cols)),
        "decision_threshold": decision_threshold,
        "decision_threshold_source": threshold_source,
        "clean_weighted_f1": clean_f1,
        "bootstrap": {
            "n_bootstrap": int(args.bootstrap),
            **ci,
        },
        "robustness": robustness,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "robustness_metrics.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# Research Summary",
        "",
        f"- Model: `{model_path}`",
        f"- Test rows: `{len(X)}`",
        f"- Features: `{len(feature_cols)}`",
        f"- Decision threshold: `{decision_threshold if decision_threshold is not None else 'model default'}` ({threshold_source})",
        f"- Clean weighted F1: `{clean_f1:.3f}`",
        f"- Bootstrap 95% CI: `{ci['ci95_low']:.3f}` to `{ci['ci95_high']:.3f}`",
        "",
        "## Missingness Stress Test",
    ]
    for row in robustness["missingness"]:
        md.append(f"- {int(row['level'] * 100)}% missingness: weighted F1 `{row['weighted_f1']:.3f}`")
    md.append("")
    md.append("## Noise Stress Test")
    for row in robustness["noise"]:
        md.append(f"- {int(row['level'] * 100)}% noise: weighted F1 `{row['weighted_f1']:.3f}`")

    (reports_dir / "research_summary.md").write_text("\n".join(md) + "\n")

    print(json.dumps(payload, indent=2))
    print("Saved reports/robustness_metrics.json")
    print("Saved reports/research_summary.md")


if __name__ == "__main__":
    main()
