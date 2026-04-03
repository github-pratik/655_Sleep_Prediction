"""Dataset loaders for public wearable archives.

The project uses these loaders as a local-data bridge:
- SleepAccel: Apple Watch accelerometer, heart rate, steps, and sleep labels
- PPG-DaLiA: optional auxiliary wearable dataset for robustness/pretraining

The loaders are intentionally conservative:
- they only read from local paths
- they degrade gracefully when a dataset is missing or an archive is unfamiliar
- they emit a standardized window-level table suitable for downstream training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat


TIME_COL = "time_s"
SUBJECT_COL = "subject_id"
WINDOW_COL = "window_index"
DATASET_COL = "dataset_name"
LABEL_COL = "label"


@dataclass
class PublicDatasetResult:
    """Standard result wrapper for a loaded public dataset."""

    name: str
    root: Path
    table: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return int(len(self.table))

    @property
    def subjects(self) -> int:
        if self.table.empty or SUBJECT_COL not in self.table.columns:
            return 0
        return int(self.table[SUBJECT_COL].nunique(dropna=True))


def discover_dataset_roots(
    candidate_roots: Sequence[Path | str] | None = None,
    search_root: Path | str | None = None,
    dataset_kind: str | None = None,
) -> list[Path]:
    """Resolve dataset root directories from explicit paths and common folder names."""

    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except FileNotFoundError:
            return
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)

    if candidate_roots:
        for item in candidate_roots:
            add(Path(item))

    base = Path(search_root or Path.cwd())
    kind_map = {
        "sleepaccel": ("sleepaccel", "sleep_accel", "SleepAccel", "Sleep-Accel"),
        "ppg_dalia": ("ppg-dalia", "ppg_dalia", "PPG-DaLiA", "PPGDalia", "PPGDaLiA"),
    }
    common_names = kind_map.get(dataset_kind or "", tuple(v for values in kind_map.values() for v in values))
    if base.exists():
        for name in common_names:
            for path in base.rglob(name):
                if path.is_dir():
                    add(path)
        for path in base.iterdir():
            if path.is_dir():
                lowered = path.name.lower()
                if dataset_kind == "sleepaccel" and "sleepaccel" in lowered:
                    add(path)
                elif dataset_kind == "ppg_dalia" and "ppg" in lowered and "dalia" in lowered:
                    add(path)
                elif dataset_kind is None and (
                    "sleepaccel" in lowered or ("ppg" in lowered and "dalia" in lowered)
                ):
                    add(path)

    return roots


def load_sleepaccel(root: Path, window_sec: int = 30) -> PublicDatasetResult:
    """Load SleepAccel archives into a standardized window-level table.

    Expected local layout is typically a folder containing subfolders like:
    - motion/
    - heart_rate/ or heartrate/
    - steps/
    - labels/ or labeled_sleep/

    The parser is tolerant to flat directories and to missing optional streams.
    """

    root = Path(root)
    if not root.exists():
        return PublicDatasetResult("sleepaccel", root, pd.DataFrame(), ["root missing"])

    kind_map = {
        "motion": ("motion", "acceler", "accel"),
        "heart_rate": ("heart_rate", "heartrate", "hr"),
        "steps": ("steps",),
        "labels": ("labels", "labeled_sleep", "sleep_label", "stage"),
    }

    files: dict[str, list[Path]] = {kind: [] for kind in kind_map}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        parent = path.parent.name.lower()
        parent_is_dataset_root = path.parent.resolve() == root.resolve()
        for kind, tokens in kind_map.items():
            if any(
                token in lowered
                or (not parent_is_dataset_root and token in parent)
                for token in tokens
            ):
                files[kind].append(path)
                break

    subject_ids = set()
    for paths in files.values():
        subject_ids.update(_infer_subject_id(p) for p in paths)
    subject_ids = {sid for sid in subject_ids if sid}

    if not subject_ids:
        return PublicDatasetResult(
            "sleepaccel",
            root,
            pd.DataFrame(),
            ["no recognizable SleepAccel files found"],
        )

    windows: list[pd.DataFrame] = []
    notes: list[str] = []

    for subject_id in sorted(subject_ids):
        motion_path = _pick_best_file(files["motion"], subject_id)
        hr_path = _pick_best_file(files["heart_rate"], subject_id)
        steps_path = _pick_best_file(files["steps"], subject_id)
        labels_path = _pick_best_file(files["labels"], subject_id)

        if motion_path is None or labels_path is None:
            notes.append(f"skipped {subject_id}: missing motion or labels file")
            continue

        motion = _read_numeric_table(motion_path, subject_id, stream_name="motion")
        if motion.empty or TIME_COL not in motion.columns:
            notes.append(f"skipped {subject_id}: unreadable motion file")
            continue

        motion["acc_mag"] = np.sqrt(
            motion.get("x", 0.0) ** 2 + motion.get("y", 0.0) ** 2 + motion.get("z", 0.0) ** 2
        )
        motion_window = _window_aggregate(
            motion,
            subject_id=subject_id,
            dataset_name="sleepaccel",
            window_sec=window_sec,
            numeric_cols=[c for c in ["x", "y", "z", "acc_mag"] if c in motion.columns],
            prefix="acc_",
            extra_aggs=("mean", "std", "min", "max", "median", "count"),
        )

        hr_window = pd.DataFrame()
        if hr_path is not None:
            hr = _read_numeric_table(hr_path, subject_id, stream_name="heart_rate")
            if not hr.empty and TIME_COL in hr.columns:
                hr_window = _window_aggregate(
                    hr,
                    subject_id=subject_id,
                    dataset_name="sleepaccel",
                    window_sec=window_sec,
                    numeric_cols=_numeric_data_columns(hr, exclude={TIME_COL, SUBJECT_COL, DATASET_COL}),
                    prefix="hr_",
                    extra_aggs=("mean", "std", "min", "max", "median", "count"),
                )

        steps_window = pd.DataFrame()
        if steps_path is not None:
            steps = _read_numeric_table(steps_path, subject_id, stream_name="steps")
            if not steps.empty and TIME_COL in steps.columns:
                steps_window = _window_aggregate(
                    steps,
                    subject_id=subject_id,
                    dataset_name="sleepaccel",
                    window_sec=window_sec,
                    numeric_cols=_numeric_data_columns(steps, exclude={TIME_COL, SUBJECT_COL, DATASET_COL}),
                    prefix="steps_",
                    extra_aggs=("mean", "std", "min", "max", "sum", "count"),
                )

        labels = _read_label_table(labels_path, subject_id)
        label_window = _window_label_table(labels, subject_id=subject_id, dataset_name="sleepaccel", window_sec=window_sec)
        if label_window.empty:
            notes.append(f"skipped {subject_id}: no usable labels")
            continue

        merged = motion_window
        if not hr_window.empty:
            merged = merged.merge(hr_window, on=[SUBJECT_COL, WINDOW_COL, DATASET_COL], how="left")
        if not steps_window.empty:
            merged = merged.merge(steps_window, on=[SUBJECT_COL, WINDOW_COL, DATASET_COL], how="left")
        merged = merged.merge(label_window, on=[SUBJECT_COL, WINDOW_COL, DATASET_COL], how="inner")
        merged["window_start_s"] = merged[WINDOW_COL] * window_sec
        merged["window_end_s"] = merged["window_start_s"] + window_sec
        windows.append(merged)

    table = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    if not table.empty:
        table = table.sort_values([SUBJECT_COL, WINDOW_COL]).reset_index(drop=True)

    return PublicDatasetResult("sleepaccel", root, table, notes)


def load_ppg_dalia(root: Path, window_sec: int = 30) -> PublicDatasetResult:
    """Load PPG-DaLiA-like local archives into a standardized window table.

    The dataset appears in multiple local formats in the wild. This loader tries:
    - CSV/TXT time-series tables
    - MAT files
    - PKL files
    - TS files (generic time-series archive format)

    If the directory layout is unrecognized, the loader returns an empty table
    with a descriptive note instead of raising.
    """

    root = Path(root)
    if not root.exists():
        return PublicDatasetResult("ppg_dalia", root, pd.DataFrame(), ["root missing"])

    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        return PublicDatasetResult("ppg_dalia", root, pd.DataFrame(), ["no files found"])

    windows: list[pd.DataFrame] = []
    notes: list[str] = []

    for path in sorted(files):
        lowered = path.name.lower()
        suffix = path.suffix.lower()

        if suffix not in {".csv", ".txt", ".ts", ".mat", ".pkl", ".pickle"}:
            continue

        subject_id = _infer_subject_id(path)
        if not subject_id:
            subject_id = path.parent.name or path.stem

        sample_frame = _read_generic_timeseries(path, subject_id=subject_id)
        if sample_frame.empty or TIME_COL not in sample_frame.columns:
            continue

        numeric_cols = _numeric_data_columns(sample_frame, exclude={TIME_COL, SUBJECT_COL, DATASET_COL})
        if not numeric_cols:
            continue

        # The dataset is often used for HR estimation. If a heart-rate-like
        # column exists, preserve it as part of the sample frame and aggregate
        # it together with the auxiliary sensor channels.
        window = _window_aggregate(
            sample_frame,
            subject_id=subject_id,
            dataset_name="ppg_dalia",
            window_sec=window_sec,
            numeric_cols=numeric_cols,
            prefix="sig_",
            extra_aggs=("mean", "std", "min", "max", "median", "count"),
        )
        if window.empty:
            continue

        # Preserve a likely target if present.
        label_col = None
        for candidate in ("hr", "heart_rate", "label", "target", "y"):
            if candidate in sample_frame.columns and pd.api.types.is_numeric_dtype(sample_frame[candidate]):
                label_col = candidate
                break
        if label_col is not None:
            label_window = _window_aggregate(
                sample_frame,
                subject_id=subject_id,
                dataset_name="ppg_dalia",
                window_sec=window_sec,
                numeric_cols=[label_col],
                prefix="target_",
                extra_aggs=("mean", "std", "min", "max", "median", "count"),
            )
            window = window.merge(label_window, on=[SUBJECT_COL, WINDOW_COL, DATASET_COL], how="left")

        window["window_start_s"] = window[WINDOW_COL] * window_sec
        window["window_end_s"] = window["window_start_s"] + window_sec
        windows.append(window)

    table = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    if not table.empty:
        table = table.sort_values([SUBJECT_COL, WINDOW_COL]).reset_index(drop=True)
    else:
        notes.append("no recognizable PPG-DaLiA files or columns found")

    return PublicDatasetResult("ppg_dalia", root, table, notes)


def build_public_window_table(
    sleepaccel_roots: Sequence[Path | str] | None = None,
    ppg_dalia_roots: Sequence[Path | str] | None = None,
    search_root: Path | str | None = None,
    window_sec: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Load all available public datasets into one standardized table."""

    search_base = Path(search_root or Path.cwd())
    resolved_sleepaccel = list(sleepaccel_roots) if sleepaccel_roots else discover_dataset_roots(None, search_base, dataset_kind="sleepaccel")
    resolved_ppg = list(ppg_dalia_roots) if ppg_dalia_roots else discover_dataset_roots(None, search_base, dataset_kind="ppg_dalia")

    results: list[PublicDatasetResult] = []
    if resolved_sleepaccel:
        for root in resolved_sleepaccel:
            result = load_sleepaccel(root, window_sec=window_sec)
            if not result.table.empty:
                results.append(result)
    if resolved_ppg:
        for root in resolved_ppg:
            result = load_ppg_dalia(root, window_sec=window_sec)
            if not result.table.empty:
                results.append(result)

    if not results:
        summary = {
            "datasets": [],
            "notes": [
                "No supported public datasets were discovered.",
                "Provide at least one local archive root via --sleepaccel-root or --ppg-dalia-root.",
            ],
        }
        return pd.DataFrame(), summary

    tables = []
    dataset_summaries = []
    for result in results:
        tables.append(result.table)
        dataset_summaries.append(
            {
                "name": result.name,
                "root": str(result.root),
                "rows": result.rows,
                "subjects": result.subjects,
                "notes": result.notes,
            }
        )

    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined = combined.sort_values([DATASET_COL, SUBJECT_COL, WINDOW_COL]).reset_index(drop=True)
    summary = {
        "datasets": dataset_summaries,
        "rows": int(len(combined)),
        "subjects": int(combined[SUBJECT_COL].nunique(dropna=True)) if SUBJECT_COL in combined.columns else 0,
        "window_sec": int(window_sec),
    }
    return combined, summary


def _infer_subject_id(path: Path) -> str:
    stem = path.stem
    parent = path.parent.name
    generic_stems = {"data", "sample", "samples", "recording", "recordings", "subject", "file", "output"}
    patterns = (
        r"(?P<id>.+?)(?:_acceleration|_accelerometer|_heartrate|_heart_rate|_steps|_labeled_sleep|_labels|_label|_motion)$",
        r"(?P<id>.+?)(?:-acceleration|-heartrate|-heart-rate|-steps|-labels|-label)$",
    )
    for pattern in patterns:
        match = re.match(pattern, stem, re.IGNORECASE)
        if match:
            return match.group("id")
    if stem and stem.lower() not in generic_stems:
        return stem
    if parent and parent.lower() not in generic_stems:
        return parent
    return parent


def _pick_best_file(paths: Sequence[Path], subject_id: str) -> Path | None:
    subject_id_l = subject_id.lower()
    ranked = [p for p in paths if subject_id_l in _infer_subject_id(p).lower() or subject_id_l in p.name.lower()]
    if ranked:
        return sorted(ranked)[0]
    return None


def _read_label_table(path: Path, subject_id: str) -> pd.DataFrame:
    frame = _read_numeric_table(path, subject_id=subject_id, stream_name="labels", preserve_non_numeric=True)
    if frame.empty:
        return frame

    label_col = None
    for candidate in ("label", "labels", "stage", "sleep_stage", "value"):
        if candidate in frame.columns:
            label_col = candidate
            break
    if label_col is None and len(frame.columns) >= 2:
        label_col = frame.columns[1]
    if label_col is None:
        return pd.DataFrame()

    label_frame = frame[[TIME_COL, label_col]].copy()
    label_frame.columns = [TIME_COL, LABEL_COL]
    label_frame[TIME_COL] = pd.to_numeric(label_frame[TIME_COL], errors="coerce")
    label_frame = label_frame.dropna(subset=[TIME_COL, LABEL_COL]).sort_values(TIME_COL).reset_index(drop=True)
    return label_frame


def _window_label_table(labels: pd.DataFrame, subject_id: str, dataset_name: str, window_sec: int) -> pd.DataFrame:
    if labels.empty or TIME_COL not in labels.columns:
        return pd.DataFrame()

    labels = labels.sort_values(TIME_COL).reset_index(drop=True)
    labels[TIME_COL] = pd.to_numeric(labels[TIME_COL], errors="coerce").astype(float)
    min_t = float(labels[TIME_COL].min())
    max_t = float(labels[TIME_COL].max())
    if not np.isfinite(min_t) or not np.isfinite(max_t):
        return pd.DataFrame()

    starts = np.arange(
        int(np.floor(min_t / window_sec)),
        int(np.ceil(max_t / window_sec)) + 1,
        dtype=int,
    )
    window_start_s = starts * window_sec
    lookup = pd.DataFrame(
        {
            SUBJECT_COL: subject_id,
            DATASET_COL: dataset_name,
            WINDOW_COL: starts,
            "window_mid_s": window_start_s + (window_sec / 2.0),
        }
    )
    merged = pd.merge_asof(
        lookup.sort_values("window_mid_s"),
        labels[[TIME_COL, LABEL_COL]].sort_values(TIME_COL),
        left_on="window_mid_s",
        right_on=TIME_COL,
        direction="backward",
    )
    merged = merged.dropna(subset=[LABEL_COL]).copy()
    merged = merged[[SUBJECT_COL, DATASET_COL, WINDOW_COL, LABEL_COL]]
    merged[LABEL_COL] = merged[LABEL_COL].astype(str)
    return merged


def _window_aggregate(
    frame: pd.DataFrame,
    subject_id: str,
    dataset_name: str,
    window_sec: int,
    numeric_cols: Sequence[str],
    prefix: str,
    extra_aggs: Sequence[str],
) -> pd.DataFrame:
    if frame.empty or TIME_COL not in frame.columns or not numeric_cols:
        return pd.DataFrame()

    working = frame.copy()
    working[TIME_COL] = pd.to_numeric(working[TIME_COL], errors="coerce")
    working = working.dropna(subset=[TIME_COL])
    if working.empty:
        return pd.DataFrame()

    working[WINDOW_COL] = np.floor(working[TIME_COL] / window_sec).astype(int)
    working[SUBJECT_COL] = subject_id
    working[DATASET_COL] = dataset_name
    group_cols = [SUBJECT_COL, DATASET_COL, WINDOW_COL]

    aggs: dict[str, list[str]] = {col: list(extra_aggs) for col in numeric_cols if col in working.columns}
    if not aggs:
        return pd.DataFrame()

    grouped = working[group_cols + list(aggs.keys())].groupby(group_cols, dropna=False).agg(aggs)
    grouped.columns = [f"{prefix}{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


def _numeric_data_columns(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for col in frame.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _read_generic_timeseries(path: Path, subject_id: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".ts"}:
        frame = _read_delimited_frame(path)
    elif suffix == ".mat":
        frame = _read_mat_frame(path)
    elif suffix in {".pkl", ".pickle"}:
        frame = _read_pickle_frame(path)
    else:
        return pd.DataFrame()

    if frame.empty:
        return frame
    return _standardize_frame(frame, subject_id=subject_id)


def _read_numeric_table(path: Path, subject_id: str, stream_name: str, preserve_non_numeric: bool = False) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".ts"}:
        frame = _read_delimited_frame(path)
    elif suffix == ".mat":
        frame = _read_mat_frame(path)
    elif suffix in {".pkl", ".pickle"}:
        frame = _read_pickle_frame(path)
    else:
        return pd.DataFrame()

    if frame.empty:
        return frame
    return _standardize_frame(frame, subject_id=subject_id, preserve_non_numeric=preserve_non_numeric)


def _standardize_frame(frame: pd.DataFrame, subject_id: str, preserve_non_numeric: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]

    time_col = None
    for candidate in ("time_s", "time", "timestamp", "seconds", "second", "sec", "t"):
        if candidate in frame.columns:
            time_col = candidate
            break
    if time_col is None and len(frame.columns) > 0:
        time_col = frame.columns[0]

    frame[TIME_COL] = _to_seconds(frame[time_col])
    if time_col != TIME_COL:
        frame = frame.drop(columns=[time_col], errors="ignore")

    frame[SUBJECT_COL] = subject_id

    if not preserve_non_numeric:
        keep = {TIME_COL, SUBJECT_COL}
        numeric_cols = [c for c in frame.columns if c in keep or pd.api.types.is_numeric_dtype(frame[c])]
        frame = frame[numeric_cols]

    for col in frame.columns:
        if col in {TIME_COL, SUBJECT_COL}:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            continue
        frame[col] = frame[col].astype(str)

    return frame.reset_index(drop=True)


def _to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        base = parsed.dropna().iloc[0]
        return (parsed - base).dt.total_seconds()

    return pd.to_numeric(series, errors="coerce")


def _read_delimited_frame(path: Path) -> pd.DataFrame:
    headerless = _looks_headerless_numeric(path)
    attempts = []
    if headerless:
        attempts.extend(
            [
                {"sep": r"\s+|,|\t|;", "engine": "python", "header": None},
                {"sep": None, "engine": "python", "header": None},
            ]
        )
    attempts.extend(
        [
            {"sep": None, "engine": "python", "header": "infer"},
            {"sep": r"\s+|,|\t|;", "engine": "python", "header": None},
        ]
    )
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, comment="#", **kwargs)
            if not frame.empty:
                return frame
        except Exception:
            continue
    return pd.DataFrame()


def _looks_headerless_numeric(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                tokens = re.split(r"[\s,;]+", stripped)
                if not tokens:
                    return False
                numeric = 0
                for token in tokens:
                    try:
                        float(token)
                        numeric += 1
                    except Exception:
                        pass
                return numeric >= max(1, len(tokens) - 1)
    except Exception:
        return False
    return False


def _read_mat_frame(path: Path) -> pd.DataFrame:
    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        warnings.warn(f"Failed to read MAT file {path.name}: {exc}")
        return pd.DataFrame()

    mapping = _flatten_mapping(payload)
    return _mapping_to_frame(mapping)


def _read_pickle_frame(path: Path) -> pd.DataFrame:
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        warnings.warn(f"Failed to read PKL file {path.name}: {exc}")
        return pd.DataFrame()

    mapping = _flatten_mapping(payload)
    return _mapping_to_frame(mapping)


def _flatten_mapping(obj) -> dict[str, object]:
    if isinstance(obj, dict):
        flattened: dict[str, object] = {}
        for key, value in obj.items():
            key = str(key)
            if isinstance(value, dict):
                nested = _flatten_mapping(value)
                for nested_key, nested_value in nested.items():
                    flattened[f"{key}_{nested_key}"] = nested_value
            else:
                flattened[key] = value
        return flattened
    if hasattr(obj, "_asdict"):
        return {str(k): v for k, v in obj._asdict().items()}
    if hasattr(obj, "__dict__"):
        return {str(k): v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _mapping_to_frame(mapping: dict[str, object]) -> pd.DataFrame:
    if not mapping:
        return pd.DataFrame()

    arrays: dict[str, np.ndarray] = {}
    for key, value in mapping.items():
        if isinstance(value, (str, bytes)):
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.size == 0 or arr.dtype == object:
            continue
        if arr.ndim == 0:
            continue
        arrays[key] = arr

    if not arrays:
        return pd.DataFrame()

    time_key = None
    for key in ("time", "times", "timestamp", "timestamps", "t", "sec", "seconds"):
        if key in arrays:
            time_key = key
            break

    target_len = None
    if time_key is not None:
        target_len = len(np.asarray(arrays[time_key]).reshape(-1))
    else:
        lengths = [len(np.asarray(arr).reshape(-1)) for arr in arrays.values() if np.asarray(arr).ndim in (1, 2)]
        if not lengths:
            return pd.DataFrame()
        target_len = max(set(lengths), key=lengths.count)

    frame = pd.DataFrame()
    if time_key is not None:
        frame[TIME_COL] = np.asarray(arrays.pop(time_key)).reshape(-1)[:target_len]

    for key, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) == target_len:
            frame[key] = arr
        elif arr.ndim == 2 and arr.shape[0] == target_len:
            for idx in range(arr.shape[1]):
                frame[f"{key}_{idx}"] = arr[:, idx]

    if frame.empty:
        return pd.DataFrame()

    if TIME_COL not in frame.columns:
        frame.insert(0, TIME_COL, np.arange(len(frame), dtype=float))
    return frame
