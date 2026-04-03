#!/usr/bin/env python3
"""
Build night-level features from parsed Apple Health CSVs.
Inputs: parsed_tables/sleep.csv, hr.csv, hrv.csv, resp.csv, spo2.csv (if available)
Output: dataset/night_features.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

SLEEP_COL = "value"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S %z")


def assign_night(dt: pd.Series) -> pd.Series:
    # map timestamps to the "night" date (even if after midnight). Shift back 6 hours.
    return (dt - pd.Timedelta(hours=6)).dt.date


def build_sleep_features(sleep_df: pd.DataFrame) -> pd.DataFrame:
    if sleep_df.empty:
        return pd.DataFrame()

    sleep_df = sleep_df.copy()
    sleep_df["start_dt"] = parse_dt(sleep_df["startDate"])
    sleep_df["end_dt"] = parse_dt(sleep_df["endDate"])
    sleep_df["duration_min"] = (sleep_df["end_dt"] - sleep_df["start_dt"]).dt.total_seconds() / 60
    sleep_df["night"] = assign_night(sleep_df["start_dt"])

    # stage flags
    sleep_df["is_in_bed"] = sleep_df[SLEEP_COL].str.contains("InBed", case=False, na=False)
    sleep_df["is_rem"] = sleep_df[SLEEP_COL].str.contains("REM", case=False, na=False)
    sleep_df["is_deep"] = sleep_df[SLEEP_COL].str.contains("Deep", case=False, na=False)
    sleep_df["is_core"] = sleep_df[SLEEP_COL].str.contains("Core", case=False, na=False)
    sleep_df["is_asleep"] = sleep_df[SLEEP_COL].str.contains("Asleep", case=False, na=False)

    agg = sleep_df.groupby("night").agg(
        sleep_start=("start_dt", "min"),
        sleep_end=("end_dt", "max"),
        in_bed_minutes=("duration_min", lambda x: x[sleep_df.loc[x.index, "is_in_bed"]].sum()),
        rem_minutes=("duration_min", lambda x: x[sleep_df.loc[x.index, "is_rem"]].sum()),
        deep_minutes=("duration_min", lambda x: x[sleep_df.loc[x.index, "is_deep"]].sum()),
        core_minutes=("duration_min", lambda x: x[sleep_df.loc[x.index, "is_core"]].sum()),
        asleep_minutes=("duration_min", lambda x: x[sleep_df.loc[x.index, "is_asleep"]].sum()),
    )

    agg = agg.reset_index().rename(columns={"night": "night_date"})
    agg["total_sleep_minutes"] = agg["asleep_minutes"]
    agg["sleep_efficiency"] = agg["asleep_minutes"] / agg["in_bed_minutes"].replace({0: pd.NA})

    # percentages
    agg["rem_pct"] = agg["rem_minutes"] / agg["total_sleep_minutes"]
    agg["deep_pct"] = agg["deep_minutes"] / agg["total_sleep_minutes"]
    agg["core_pct"] = agg["core_minutes"] / agg["total_sleep_minutes"]

    return agg


def summarize_metric(metric_df: pd.DataFrame, ts_col: str, value_col: str, window_start, window_end):
    if metric_df.empty:
        return None
    mask = (metric_df[ts_col] >= window_start) & (metric_df[ts_col] <= window_end)
    if not mask.any():
        return None
    subset = metric_df.loc[mask, value_col].astype(float)
    return {
        "mean": subset.mean(),
        "min": subset.min(),
        "max": subset.max(),
        "median": subset.median(),
        "std": subset.std(ddof=0),
    }


def attach_physio_features(night_df: pd.DataFrame, hr_df: pd.DataFrame, hrv_df: pd.DataFrame,
                           resp_df: pd.DataFrame, spo2_df: pd.DataFrame) -> pd.DataFrame:
    if night_df.empty:
        return night_df

    # Parse timestamps
    for df in [hr_df, hrv_df, resp_df, spo2_df]:
        if not df.empty:
            df["ts"] = parse_dt(df.get("startDate", df.get("endDate")))
            if "value" in df:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

    rows = []
    for _, row in night_df.iterrows():
        start = row["sleep_start"]
        end = row["sleep_end"]
        features = row.to_dict()

        hr_stats = summarize_metric(hr_df, "ts", "value", start, end) if not hr_df.empty else None
        hrv_stats = summarize_metric(hrv_df, "ts", "value", start, end) if not hrv_df.empty else None
        resp_stats = summarize_metric(resp_df, "ts", "value", start, end) if not resp_df.empty else None
        spo2_stats = summarize_metric(spo2_df, "ts", "value", start, end) if not spo2_df.empty else None

        if hr_stats:
            features.update({
                "hr_mean": hr_stats["mean"],
                "hr_min": hr_stats["min"],
                "hr_max": hr_stats["max"],
                "hr_median": hr_stats["median"],
                "hr_std": hr_stats["std"],
            })
        if hrv_stats:
            features.update({
                "hrv_mean": hrv_stats["mean"],
                "hrv_min": hrv_stats["min"],
                "hrv_max": hrv_stats["max"],
                "hrv_median": hrv_stats["median"],
                "hrv_std": hrv_stats["std"],
            })
        if resp_stats:
            features.update({
                "resp_mean": resp_stats["mean"],
                "resp_min": resp_stats["min"],
                "resp_max": resp_stats["max"],
                "resp_median": resp_stats["median"],
                "resp_std": resp_stats["std"],
            })
        if spo2_stats:
            features.update({
                "spo2_mean": spo2_stats["mean"],
                "spo2_min": spo2_stats["min"],
                "spo2_max": spo2_stats["max"],
                "spo2_median": spo2_stats["median"],
                "spo2_std": spo2_stats["std"],
            })
        rows.append(features)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build night-level features from Apple Health data")
    parser.add_argument("--parsed-dir", default=Path("parsed_tables"), type=Path)
    parser.add_argument("--out", default=Path("dataset/night_features.csv"), type=Path)
    args = parser.parse_args()

    parsed_dir = args.parsed_dir
    sleep_df = load_csv(parsed_dir / "sleep.csv")
    hr_df = load_csv(parsed_dir / "hr.csv")
    hrv_df = load_csv(parsed_dir / "hrv.csv")
    resp_df = load_csv(parsed_dir / "resp.csv")
    spo2_df = load_csv(parsed_dir / "spo2.csv")

    night_df = build_sleep_features(sleep_df)
    night_df = attach_physio_features(night_df, hr_df, hrv_df, resp_df, spo2_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    night_df.to_csv(args.out, index=False)

    print(f"Night rows: {len(night_df)} -> {args.out}")
    if not night_df.empty:
        print(night_df.head())
        print(night_df.describe(include='all'))


if __name__ == "__main__":
    main()
