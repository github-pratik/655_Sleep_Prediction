from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import numeric_feature_frame, infer_sort_column


@dataclass(frozen=True)
class FeatureBundle:
    name: str
    frame: pd.DataFrame
    feature_order: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PublicEncoderInfo:
    available: bool
    path: str | None
    artifact: Any | None
    input_feature_order: list[str]
    output_feature_names: list[str]
    note: str


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denom, errors="coerce")
    numer = pd.to_numeric(numer, errors="coerce")
    out = numer / denom.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2.0 * np.pi * (values / float(period))
    return pd.Series(np.sin(radians), index=values.index), pd.Series(np.cos(radians), index=values.index)


def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def load_public_encoder_info(path: Path | None, fallback_feature_order: list[str] | None = None) -> PublicEncoderInfo:
    if path is None or not path.exists():
        return PublicEncoderInfo(
            available=False,
            path=str(path) if path is not None else None,
            artifact=None,
            input_feature_order=fallback_feature_order or [],
            output_feature_names=[],
            note="public encoder artifact not found",
        )

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        return PublicEncoderInfo(
            available=False,
            path=str(path),
            artifact=None,
            input_feature_order=fallback_feature_order or [],
            output_feature_names=[],
            note=f"failed to load encoder artifact: {exc}",
        )

    input_feature_order: list[str] = []
    output_feature_names: list[str] = []
    note = "loaded encoder artifact"

    if isinstance(artifact, dict):
        input_feature_order = list(
            artifact.get("feature_order")
            or artifact.get("input_feature_order")
            or artifact.get("feature_names")
            or []
        )
        output_feature_names = list(artifact.get("output_feature_names") or [])
        if "notes" in artifact:
            note = str(artifact["notes"])
        if "encoder" in artifact:
            artifact = artifact["encoder"]
        elif "transformer" in artifact:
            artifact = artifact["transformer"]
    else:
        input_feature_order = list(getattr(artifact, "feature_order", []) or [])

    if not input_feature_order:
        if fallback_feature_order is not None:
            input_feature_order = list(fallback_feature_order)
        elif hasattr(artifact, "feature_names_in_"):
            input_feature_order = [str(c) for c in artifact.feature_names_in_]

    return PublicEncoderInfo(
        available=True,
        path=str(path),
        artifact=artifact,
        input_feature_order=input_feature_order,
        output_feature_names=output_feature_names,
        note=note,
    )


def build_baseline_features(df: pd.DataFrame) -> FeatureBundle:
    base_frame, feature_order = numeric_feature_frame(df)
    base_frame = _sanitize_frame(base_frame)
    return FeatureBundle(
        name="baseline",
        frame=base_frame,
        feature_order=feature_order,
        metadata={
            "kind": "baseline",
            "feature_count": len(feature_order),
            "sort_column": infer_sort_column(df),
        },
    )


def _rolling_features(base_frame: pd.DataFrame) -> dict[str, pd.Series]:
    rolling_targets = [
        "total_sleep_minutes",
        "sleep_efficiency",
        "rem_minutes",
        "deep_minutes",
        "core_minutes",
        "hr_mean",
        "hr_std",
        "hrv_mean",
        "hrv_std",
        "resp_mean",
        "resp_std",
        "spo2_mean",
        "spo2_std",
    ]
    rolling_targets = [c for c in rolling_targets if c in base_frame.columns]

    updates: dict[str, pd.Series] = {}
    for col in rolling_targets:
        series = pd.to_numeric(base_frame[col], errors="coerce")
        past = series.shift(1)
        for window in (3, 7):
            roll = past.rolling(window=window, min_periods=1)
            mean_col = f"{col}_past{window}_mean"
            std_col = f"{col}_past{window}_std"
            delta_col = f"{col}_delta"
            trend_col = f"{col}_past{window}_trend"
            updates[mean_col] = roll.mean()
            updates[std_col] = roll.std(ddof=0)
            updates[delta_col] = series - past
            updates[trend_col] = series - roll.mean()
    return updates


def _time_features(df: pd.DataFrame, index: pd.Index) -> dict[str, pd.Series]:
    sort_col = infer_sort_column(df)
    if sort_col is None:
        na = pd.Series(np.nan, index=index, dtype=float)
        return {
            "day_of_week": na.copy(),
            "is_weekend": na.copy(),
            "month_sin": na.copy(),
            "month_cos": na.copy(),
        }

    dates = pd.to_datetime(df[sort_col], errors="coerce")
    dow = dates.dt.dayofweek
    month = dates.dt.month.fillna(1)
    month_sin, month_cos = _cyclical(month, 12)
    return {
        "day_of_week": dow,
        "is_weekend": (dow >= 5).astype(float),
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def _ratio_features(base_frame: pd.DataFrame) -> dict[str, pd.Series]:
    def col(name: str) -> pd.Series:
        if name in base_frame.columns:
            return pd.to_numeric(base_frame[name], errors="coerce")
        return pd.Series(np.nan, index=base_frame.index, dtype=float)

    in_bed = col("in_bed_minutes")
    asleep = col("asleep_minutes")
    total_sleep = col("total_sleep_minutes")
    rem = col("rem_minutes")
    deep = col("deep_minutes")
    core = col("core_minutes")
    hr_mean = col("hr_mean")
    hrv_mean = col("hrv_mean")
    resp_mean = col("resp_mean")
    spo2_mean = col("spo2_mean")

    return {
        "sleep_gap_minutes": in_bed - asleep,
        "sleep_to_bed_ratio": _safe_div(total_sleep, in_bed),
        "sleep_efficiency_proxy": _safe_div(asleep, in_bed),
        "rem_to_deep_ratio": _safe_div(rem, deep),
        "deep_to_core_ratio": _safe_div(deep, core),
        "rem_share": _safe_div(rem, total_sleep),
        "deep_share": _safe_div(deep, total_sleep),
        "core_share": _safe_div(core, total_sleep),
        "sleep_stage_balance": rem + deep - core,
        "hrv_to_hr_ratio": _safe_div(hrv_mean, hr_mean),
        "resp_to_hr_ratio": _safe_div(resp_mean, hr_mean),
        "spo2_deficit": 1.0 - spo2_mean,
    }


def _missingness_features(base_frame: pd.DataFrame) -> dict[str, pd.Series]:
    observed = base_frame.notna().sum(axis=1)
    missing = base_frame.isna().sum(axis=1)
    total = max(len(base_frame.columns), 1)
    return {
        "observed_numeric_count": observed,
        "missing_numeric_count": missing,
        "missing_numeric_rate": missing / float(total),
        "observed_numeric_rate": observed / float(total),
    }


def _public_encoder_features(base_frame: pd.DataFrame, encoder_info: PublicEncoderInfo) -> FeatureBundle | None:
    if not encoder_info.available or encoder_info.artifact is None:
        return None

    input_order = encoder_info.input_feature_order or list(base_frame.columns)
    encoder_input = base_frame.reindex(columns=input_order).copy()

    for col in encoder_input.columns:
        series = pd.to_numeric(encoder_input[col], errors="coerce")
        fill = float(series.median()) if series.notna().any() else 0.0
        encoder_input[col] = series.fillna(fill)

    artifact = encoder_info.artifact
    transformed: Any
    try:
        if hasattr(artifact, "transform"):
            transformed = artifact.transform(encoder_input)
        elif hasattr(artifact, "components_") and hasattr(artifact, "mean_"):
            matrix = encoder_input.to_numpy(dtype=float)
            centered = matrix - np.asarray(artifact.mean_, dtype=float)
            transformed = centered @ np.asarray(artifact.components_, dtype=float).T
        else:
            raise TypeError("encoder artifact does not expose transform or PCA-like parameters")
    except Exception as exc:
        return FeatureBundle(
            name="public_encoder",
            frame=pd.DataFrame(index=base_frame.index),
            feature_order=[],
            metadata={"encoder_available": False, "note": f"encoder transform failed: {exc}"},
        )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    if transformed.ndim == 1:
        transformed = transformed.reshape(-1, 1)

    if encoder_info.output_feature_names and len(encoder_info.output_feature_names) == transformed.shape[1]:
        columns = list(encoder_info.output_feature_names)
    else:
        columns = [f"public_enc_{i:02d}" for i in range(transformed.shape[1])]

    enc_frame = pd.DataFrame(transformed, columns=columns, index=base_frame.index)
    enc_frame = _sanitize_frame(enc_frame)
    return FeatureBundle(
        name="public_encoder",
        frame=enc_frame,
        feature_order=columns,
        metadata={
            "encoder_available": True,
            "encoder_output_dim": len(columns),
            "encoder_path": encoder_info.path,
        },
    )


def build_transfer_features(df: pd.DataFrame, encoder_path: Path | None = None) -> FeatureBundle:
    base_bundle = build_baseline_features(df)
    transfer_map: dict[str, pd.Series] = {}
    transfer_map.update(_missingness_features(base_bundle.frame))
    transfer_map.update(_ratio_features(base_bundle.frame))
    transfer_map.update(_time_features(df, base_bundle.frame.index))
    transfer_map.update(_rolling_features(base_bundle.frame))
    transfer = pd.DataFrame(transfer_map, index=base_bundle.frame.index)

    encoder_info = load_public_encoder_info(encoder_path, fallback_feature_order=base_bundle.feature_order)
    encoder_bundle = _public_encoder_features(base_bundle.frame, encoder_info)
    if encoder_bundle is not None and not encoder_bundle.frame.empty:
        transfer = pd.concat([transfer, encoder_bundle.frame], axis=1)
        transfer_order = list(transfer.columns)
    else:
        transfer_order = list(transfer.columns)

    transfer = _sanitize_frame(transfer).copy()
    metadata = {
        "kind": "transfer",
        "feature_count": len(transfer_order),
        "encoder_available": encoder_info.available and encoder_bundle is not None and not encoder_bundle.frame.empty,
        "encoder_path": encoder_info.path,
        "encoder_note": encoder_info.note,
    }
    return FeatureBundle(name="transfer", frame=transfer, feature_order=transfer_order, metadata=metadata)


def build_feature_spaces(df: pd.DataFrame, encoder_path: Path | None = None) -> dict[str, FeatureBundle]:
    baseline = build_baseline_features(df)
    transfer = build_transfer_features(df, encoder_path=encoder_path)
    combined_frame = pd.concat([baseline.frame, transfer.frame], axis=1)
    combined_frame = _sanitize_frame(combined_frame)
    combined_order = list(baseline.feature_order) + list(transfer.feature_order)
    combined = FeatureBundle(
        name="combined",
        frame=combined_frame,
        feature_order=combined_order,
        metadata={
            "kind": "combined",
            "feature_count": len(combined_order),
            "baseline_feature_count": len(baseline.feature_order),
            "transfer_feature_count": len(transfer.feature_order),
            "encoder_available": transfer.metadata.get("encoder_available", False),
            "encoder_path": transfer.metadata.get("encoder_path"),
        },
    )
    return {
        "baseline": baseline,
        "transfer": transfer,
        "combined": combined,
    }
