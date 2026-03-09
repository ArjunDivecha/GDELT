#!/usr/bin/env python3
"""
Build country-level daily sentiment signals from per-day GDELT country aggregates.

Inputs:
  - data/aggregates/*/country_day_all.csv

Outputs:
  - one panel CSV with raw components and composite indicators

Notes:
  - attention_shock and the composite indicators are trailing-window signals.
  - They are intentionally blank until there is enough history for a country.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build country-level daily GDELT signals")
    parser.add_argument(
        "--country-day-dir",
        default="data/country_day",
        help="Directory containing daily country-day parquet files",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests/country_day",
        help="Optional directory containing per-day manifest JSON files for fetch coverage metadata.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/panels/country_signal_daily.csv",
        help="Output CSV path for the combined signal panel",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/panels/country_signal_daily.parquet",
        help="Output parquet path for the combined signal panel",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Trailing lookback window in days for z-scored features",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=10,
        help="Minimum number of prior observations required before a z-score is emitted",
    )
    parser.add_argument(
        "--observation-windows",
        action="store_true",
        help="Use trailing observed rows instead of calendar-day windows. Default is calendar-aware rolling windows.",
    )
    return parser.parse_args()


def load_aggregate_panel(country_day_dir: Path) -> pd.DataFrame:
    files = sorted(country_day_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No daily country parquet files found under {country_day_dir}")

    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    required = {
        "date",
        "country_code_gdelt",
        "country_name",
        "country_iso3",
        "n_articles",
        "tone_mean",
        "tone_wavg_wordcount",
        "negative_mean",
        "tone_dispersion",
        "local_tone",
        "foreign_tone",
        "local_n_articles",
        "local_source_total_articles",
        "local_attention_share",
        "foreign_n_articles",
        "unknown_source_n_articles",
        "source_resolution_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Daily aggregates are missing required columns. Rebuild country_day_all.csv with "
            f"the updated build_country_day.py first. Missing: {', '.join(missing)}"
        )

    frame = frame.dropna(subset=["country_iso3"]).copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["country_iso3", "date"]).drop_duplicates(
        subset=["country_iso3", "date"], keep="last"
    )
    return frame


def load_manifest_coverage(manifest_dir: Path) -> pd.DataFrame:
    if not manifest_dir.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "day_status",
                "gkg_fetch_share",
                "gkg_files_expected",
                "gkg_files_fetched",
                "gkg_files_missing",
            ]
        )

    rows = []
    for path in sorted(manifest_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "date": payload.get("date"),
                "day_status": payload.get("status"),
                "gkg_fetch_share": payload.get("gkg_fetch_share"),
                "gkg_files_expected": payload.get("gkg_files_expected"),
                "gkg_files_fetched": payload.get("gkg_files_fetched"),
                "gkg_files_missing": payload.get("gkg_files_missing"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "day_status",
                "gkg_fetch_share",
                "gkg_files_expected",
                "gkg_files_fetched",
                "gkg_files_missing",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def trailing_zscore(series: pd.Series, window: int, min_history: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    prior = numeric.shift(1)
    mean = prior.rolling(window=window, min_periods=min_history).mean()
    std = prior.rolling(window=window, min_periods=min_history).std(ddof=0)
    std = std.mask(std == 0)
    return (numeric - mean) / std


def expand_country_to_calendar(country_frame: pd.DataFrame) -> pd.DataFrame:
    country_frame = country_frame.sort_values("date").copy()
    country_frame["has_country_day_data"] = True

    full_dates = pd.date_range(
        start=country_frame["date"].min(),
        end=country_frame["date"].max(),
        freq="D",
        name="date",
    )
    expanded = country_frame.set_index("date").reindex(full_dates).reset_index()
    expanded = expanded.rename(columns={"index": "date"})
    expanded["has_country_day_data"] = expanded["has_country_day_data"].notna()

    for column in ("country_code_gdelt", "country_name", "country_iso3"):
        non_null = country_frame[column].dropna()
        if not non_null.empty:
            expanded[column] = expanded[column].fillna(non_null.iloc[0])

    return expanded


def compute_country_signals(
    frame: pd.DataFrame,
    window: int,
    min_history: int,
    observation_windows: bool,
) -> pd.DataFrame:
    groups = []
    for _, country_frame in frame.groupby("country_iso3", sort=False):
        country_frame = country_frame.sort_values("date").copy()
        country_frame["has_country_day_data"] = True

        if not observation_windows:
            country_frame = expand_country_to_calendar(country_frame)

        country_frame["country_news_sentiment_raw"] = pd.to_numeric(
            country_frame["local_tone"].fillna(country_frame["tone_wavg_wordcount"]),
            errors="coerce",
        )
        country_frame["country_news_attention"] = np.log1p(
            pd.to_numeric(country_frame["local_n_articles"], errors="coerce")
        )
        country_frame["sentiment_x_attention_raw"] = (
            country_frame["country_news_sentiment_raw"]
            * pd.to_numeric(country_frame["local_attention_share"], errors="coerce")
        )
        country_frame["country_news_risk_raw"] = (
            -1.0 * country_frame["country_news_sentiment_raw"]
            + 0.5 * pd.to_numeric(country_frame["tone_dispersion"], errors="coerce")
        )

        country_frame["attention_shock"] = trailing_zscore(
            country_frame["country_news_attention"], window=window, min_history=min_history
        )
        country_frame["local_tone_z"] = trailing_zscore(
            country_frame["local_tone"], window=window, min_history=min_history
        )
        country_frame["foreign_tone_z"] = trailing_zscore(
            country_frame["foreign_tone"], window=window, min_history=min_history
        )
        country_frame["tone_dispersion_z"] = trailing_zscore(
            country_frame["tone_dispersion"], window=window, min_history=min_history
        )
        country_frame["local_attention_share_z"] = trailing_zscore(
            country_frame["local_attention_share"], window=window, min_history=min_history
        )
        country_frame["country_news_sentiment"] = trailing_zscore(
            country_frame["country_news_sentiment_raw"], window=window, min_history=min_history
        )
        country_frame["country_news_sentiment_x_attention"] = trailing_zscore(
            country_frame["sentiment_x_attention_raw"], window=window, min_history=min_history
        )
        country_frame["country_news_risk"] = trailing_zscore(
            country_frame["country_news_risk_raw"], window=window, min_history=min_history
        )

        observed_frame = country_frame.loc[country_frame["has_country_day_data"]].copy()
        observed_frame["days_since_prior_observation"] = (
            observed_frame["date"].diff().dt.days.astype("float")
        )
        groups.append(observed_frame)

    result = pd.concat(groups, ignore_index=True)
    return result.sort_values(["date", "country_iso3"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    country_day_dir = Path(args.country_day_dir)
    output_csv_path = Path(args.output_csv)
    output_parquet_path = Path(args.output_parquet)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_aggregate_panel(country_day_dir)
    manifest_frame = load_manifest_coverage(Path(args.manifest_dir))
    if not manifest_frame.empty:
        frame = frame.merge(manifest_frame, on="date", how="left", validate="many_to_one")

    result = compute_country_signals(
        frame,
        window=args.window,
        min_history=args.min_history,
        observation_windows=args.observation_windows,
    )

    keep_columns = [
        "date",
        "country_code_gdelt",
        "country_name",
        "country_iso3",
        "n_articles",
        "local_n_articles",
        "local_source_total_articles",
        "local_attention_share",
        "foreign_n_articles",
        "unknown_source_n_articles",
        "source_resolution_rate",
        "tone_mean",
        "tone_wavg_wordcount",
        "negative_mean",
        "tone_dispersion",
        "local_tone",
        "foreign_tone",
        "country_news_sentiment_raw",
        "country_news_attention",
        "sentiment_x_attention_raw",
        "country_news_risk_raw",
        "attention_shock",
        "local_tone_z",
        "foreign_tone_z",
        "tone_dispersion_z",
        "local_attention_share_z",
        "country_news_sentiment",
        "country_news_sentiment_x_attention",
        "country_news_risk",
        "days_since_prior_observation",
    ]
    optional_columns = [
        "day_status",
        "gkg_fetch_share",
        "gkg_files_expected",
        "gkg_files_fetched",
        "gkg_files_missing",
    ]
    keep_columns.extend(column for column in optional_columns if column in result.columns)
    panel = result[keep_columns]
    panel.to_csv(output_csv_path, index=False)
    panel.to_parquet(output_parquet_path, index=False)

    date_count = result["date"].nunique()
    country_count = result["country_iso3"].nunique()
    print(f"saved {output_csv_path}")
    print(f"saved {output_parquet_path}")
    print(f"dates={date_count} countries={country_count} rows={len(result)}")


if __name__ == "__main__":
    main()
