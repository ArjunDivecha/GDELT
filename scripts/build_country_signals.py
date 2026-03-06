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


def trailing_zscore(series: pd.Series, window: int, min_history: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    prior = numeric.shift(1)
    mean = prior.rolling(window=window, min_periods=min_history).mean()
    std = prior.rolling(window=window, min_periods=min_history).std(ddof=0)
    std = std.mask(std == 0)
    return (numeric - mean) / std


def compute_country_signals(frame: pd.DataFrame, window: int, min_history: int) -> pd.DataFrame:
    groups = []
    for _, country_frame in frame.groupby("country_iso3", sort=False):
        country_frame = country_frame.sort_values("date").copy()

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

        groups.append(country_frame)

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
    result = compute_country_signals(frame, window=args.window, min_history=args.min_history)

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
    ]
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
