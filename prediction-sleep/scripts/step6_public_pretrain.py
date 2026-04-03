#!/usr/bin/env python3
"""Pretrain a compact public-data encoder for mobile fatigue modeling."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.public_data import build_public_window_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan local public wearable datasets and fit a compact encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sleepaccel-root",
        action="append",
        type=Path,
        default=[],
        help="Path to a local SleepAccel archive root. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--ppg-dalia-root",
        action="append",
        type=Path,
        default=[],
        help="Path to a local PPG-DaLiA archive root. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=ROOT,
        help="Directory to scan when explicit roots are not supplied.",
    )
    parser.add_argument(
        "--window-sec",
        type=int,
        default=30,
        help="Window size in seconds for standardized feature aggregation.",
    )
    parser.add_argument(
        "--ppg-max-subjects",
        type=int,
        default=3,
        help="Maximum number of PPG-DaLiA subjects to process (keeps runtime bounded).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Target PCA latent dimensionality for the public encoder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "public_pretrain",
        help="Directory for public pretraining artifacts.",
    )
    parser.add_argument(
        "--save-window-table",
        action="store_true",
        help="Persist the combined standardized training table as CSV for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    combined, summary = build_public_window_table(
        sleepaccel_roots=args.sleepaccel_root,
        ppg_dalia_roots=args.ppg_dalia_root,
        search_root=args.search_root,
        window_sec=args.window_sec,
        ppg_max_subjects=args.ppg_max_subjects,
    )

    if combined.empty:
        message = [
            "No public datasets were loaded.",
            "Provide a local dataset root with --sleepaccel-root or --ppg-dalia-root.",
            f"Scanned search root: {args.search_root}",
        ]
        raise SystemExit("\n".join(message))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = _select_feature_columns(combined)
    if not feature_cols:
        raise SystemExit(
            "No numeric feature columns were discovered in the loaded public datasets. "
            "Check the archive layout and file formats."
        )
    feature_frame = combined[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.loc[:, feature_frame.notna().any(axis=0)]
    feature_cols = list(feature_frame.columns)
    if len(feature_frame) < 2:
        raise SystemExit("Need at least two public windows to fit the encoder.")

    max_components = max(1, min(len(feature_cols), len(feature_frame) - 1))
    latent_dim = max(1, min(args.latent_dim, max_components))

    encoder = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=latent_dim, random_state=42)),
        ]
    )
    encoder.fit(feature_frame)

    transformed = encoder.transform(feature_frame)
    transformed = pd.DataFrame(
        transformed,
        columns=[f"latent_{i+1}" for i in range(transformed.shape[1])],
    )

    feature_space = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_sec": int(args.window_sec),
        "feature_columns": feature_cols,
        "num_feature_columns": len(feature_cols),
        "latent_columns": list(transformed.columns),
        "latent_dim": int(transformed.shape[1]),
        "sources": summary.get("datasets", []),
        "row_count": int(len(combined)),
        "subject_count": int(combined["subject_id"].nunique(dropna=True)) if "subject_id" in combined.columns else 0,
        "dataset_names": sorted(set(combined["dataset_name"].dropna().astype(str))) if "dataset_name" in combined.columns else [],
    }

    report = {
        "generated_at": feature_space["generated_at"],
        "inputs": {
            "sleepaccel_roots": [str(p) for p in args.sleepaccel_root],
            "ppg_dalia_roots": [str(p) for p in args.ppg_dalia_root],
            "search_root": str(args.search_root),
            "ppg_max_subjects": int(args.ppg_max_subjects),
        },
        "summary": summary,
        "encoder": {
            "type": "pipeline(imputer -> scaler -> pca)",
            "latent_dim": int(transformed.shape[1]),
            "explained_variance_ratio": _float_list(encoder.named_steps["pca"].explained_variance_ratio_),
            "cumulative_explained_variance": float(encoder.named_steps["pca"].explained_variance_ratio_.sum()),
        },
        "table": {
            "rows": int(len(combined)),
            "subjects": int(combined["subject_id"].nunique(dropna=True)) if "subject_id" in combined.columns else 0,
            "dataset_names": feature_space["dataset_names"],
            "label_cardinality": int(combined["label"].nunique(dropna=True)) if "label" in combined.columns else 0,
        },
        "artifacts": {
            "feature_space": str(output_dir / "public_feature_space.json"),
            "encoder": str(output_dir / "public_encoder.pkl"),
            "report": str(output_dir / "public_pretrain_report.json"),
        },
    }

    (output_dir / "public_feature_space.json").write_text(json.dumps(feature_space, indent=2))
    (output_dir / "public_pretrain_report.json").write_text(json.dumps(report, indent=2))
    joblib.dump(encoder, output_dir / "public_encoder.pkl")

    if args.save_window_table:
        combined.to_csv(output_dir / "public_window_table.csv", index=False)

    print(json.dumps(report["table"], indent=2))
    print(f"Saved encoder to {output_dir / 'public_encoder.pkl'}")
    print(f"Saved report to {output_dir / 'public_pretrain_report.json'}")


def _select_feature_columns(table: pd.DataFrame) -> list[str]:
    excluded = {
        "dataset_name",
        "subject_id",
        "window_index",
        "window_start_s",
        "window_end_s",
        "label",
        "window_mid_s",
    }
    cols = []
    for col in table.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(table[col]):
            cols.append(col)
    return cols


def _float_list(values) -> list[float]:
    return [float(v) for v in values]


if __name__ == "__main__":
    main()
