#!/usr/bin/env python3
"""
Build a monthly country-level decision layer from the daily GDELT signal panel.

The daily panel remains the source of truth. This script snapshots recency-weighted
daily features at each country-month end and standardizes them within country.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "country_iso3",
    "country_name",
    "country_news_sentiment_raw",
    "country_news_risk_raw",
    "local_attention_share",
    "country_news_attention",
    "local_tone",
    "foreign_tone",
    "tone_dispersion",
}

MONTHLY_Z_COLUMNS = [
    "sentiment_fast_z",
    "sentiment_slow_z",
    "sentiment_trend_z",
    "attention_fast_z",
    "attention_slow_z",
    "attention_trend_z",
    "risk_fast_z",
    "dispersion_fast_z",
    "local_tone_fast_z",
    "foreign_tone_fast_z",
    "local_foreign_gap_z",
]

MONTHLY_RAW_COLUMNS = [
    "sentiment_fast",
    "sentiment_slow",
    "sentiment_trend",
    "attention_fast",
    "attention_slow",
    "attention_trend",
    "risk_fast",
    "dispersion_fast",
    "local_tone_fast",
    "foreign_tone_fast",
    "local_foreign_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the monthly GDELT metronome panel")
    parser.add_argument(
        "--daily-panel-parquet",
        default="data/panels/country_signal_daily.parquet",
        help="Input daily signal parquet produced by build_country_signals.py",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/panels/country_signal_monthly.parquet",
        help="Output parquet path for the monthly signal panel",
    )
    parser.add_argument(
        "--output-csv",
        default="data/panels/country_signal_monthly.csv",
        help="Output CSV path for the monthly signal panel",
    )
    parser.add_argument(
        "--fast-span",
        type=int,
        default=5,
        help="EWMA span for the fast monthly feature block",
    )
    parser.add_argument(
        "--slow-span",
        type=int,
        default=20,
        help="EWMA span for the slow monthly feature block",
    )
    parser.add_argument(
        "--risk-span",
        type=int,
        default=10,
        help="EWMA span for risk and dispersion features",
    )
    parser.add_argument(
        "--z-window-months",
        type=int,
        default=24,
        help="Trailing monthly window for within-country z-scores",
    )
    parser.add_argument(
        "--min-history-months",
        type=int,
        default=6,
        help="Minimum number of prior months required before a monthly z-score is emitted",
    )
    return parser.parse_args()


def load_daily_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Daily signal panel not found: {path}")

    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"Daily signal panel is missing required columns: {', '.join(missing)}"
        )

    frame = frame.dropna(subset=["country_iso3"]).copy()
    frame["country_iso3"] = frame["country_iso3"].astype(str).str.strip()
    frame = frame.loc[frame["country_iso3"] != ""].copy()
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


def worst_day_status(values: pd.Series) -> str | None:
    ordered = ["partial", "ok"]
    statuses = [str(v).strip().lower() for v in values.dropna() if str(v).strip()]
    if not statuses:
        return None
    for candidate in ordered:
        if candidate in statuses:
            return candidate
    return statuses[0]


def build_country_monthly(
    country_frame: pd.DataFrame,
    fast_span: int,
    slow_span: int,
    risk_span: int,
    z_window_months: int,
    min_history_months: int,
) -> pd.DataFrame:
    country_frame = country_frame.sort_values("date").copy()
    country_frame["signal_month"] = country_frame["date"].dt.to_period("M")

    country_frame["sentiment_fast"] = country_frame["country_news_sentiment_raw"].ewm(
        span=fast_span, adjust=False, min_periods=1
    ).mean()
    country_frame["sentiment_slow"] = country_frame["country_news_sentiment_raw"].ewm(
        span=slow_span, adjust=False, min_periods=1
    ).mean()
    country_frame["sentiment_trend"] = (
        country_frame["sentiment_fast"] - country_frame["sentiment_slow"]
    )

    country_frame["attention_fast"] = country_frame["local_attention_share"].ewm(
        span=fast_span, adjust=False, min_periods=1
    ).mean()
    country_frame["attention_slow"] = country_frame["local_attention_share"].ewm(
        span=slow_span, adjust=False, min_periods=1
    ).mean()
    country_frame["attention_trend"] = (
        country_frame["attention_fast"] - country_frame["attention_slow"]
    )

    country_frame["risk_fast"] = country_frame["country_news_risk_raw"].ewm(
        span=risk_span, adjust=False, min_periods=1
    ).mean()
    country_frame["dispersion_fast"] = country_frame["tone_dispersion"].ewm(
        span=risk_span, adjust=False, min_periods=1
    ).mean()
    country_frame["local_tone_fast"] = country_frame["local_tone"].ewm(
        span=fast_span, adjust=False, min_periods=1
    ).mean()
    country_frame["foreign_tone_fast"] = country_frame["foreign_tone"].ewm(
        span=fast_span, adjust=False, min_periods=1
    ).mean()
    country_frame["local_foreign_gap"] = (
        country_frame["local_tone_fast"] - country_frame["foreign_tone_fast"]
    )

    monthly = (
        country_frame.groupby("signal_month", sort=True, group_keys=False).tail(1).copy()
    )
    monthly["month_end_date_used"] = monthly["date"]
    monthly["month_calendar_days"] = monthly["signal_month"].dt.days_in_month.astype(int)

    month_counts = (
        country_frame.groupby("signal_month", sort=True)["date"]
        .size()
        .rename("month_obs_count")
        .reset_index()
    )
    monthly = monthly.merge(month_counts, on="signal_month", how="left", validate="one_to_one")
    monthly["month_obs_share"] = (
        monthly["month_obs_count"] / monthly["month_calendar_days"]
    )

    if "day_status" in country_frame.columns:
        status = (
            country_frame.groupby("signal_month", sort=True)["day_status"]
            .apply(worst_day_status)
            .rename("month_day_status_worst")
            .reset_index()
        )
        monthly = monthly.merge(status, on="signal_month", how="left", validate="one_to_one")

    if "gkg_fetch_share" in country_frame.columns:
        fetch = (
            country_frame.groupby("signal_month", sort=True)["gkg_fetch_share"]
            .agg(
                month_gkg_fetch_share_mean="mean",
                month_gkg_fetch_share_min="min",
            )
            .reset_index()
        )
        monthly = monthly.merge(fetch, on="signal_month", how="left", validate="one_to_one")

    for base in MONTHLY_RAW_COLUMNS:
        monthly[f"{base}_z"] = trailing_zscore(
            monthly[base], window=z_window_months, min_history=min_history_months
        )

    monthly["monthly_metronome"] = (
        0.35 * monthly["sentiment_fast_z"]
        + 0.20 * monthly["sentiment_slow_z"]
        + 0.20 * monthly["sentiment_trend_z"]
        + 0.15 * monthly["attention_fast_z"]
        - 0.10 * monthly["risk_fast_z"]
    )
    monthly["monthly_risk"] = (
        0.45 * monthly["risk_fast_z"]
        + 0.30 * monthly["dispersion_fast_z"]
        - 0.15 * monthly["sentiment_fast_z"]
        - 0.10 * monthly["foreign_tone_fast_z"]
    )
    monthly["monthly_defensive"] = -1.0 * monthly["monthly_risk"]

    monthly["signal_month"] = monthly["signal_month"].astype(str)
    return monthly


def add_cross_sectional_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["monthly_metronome_rank_pct"] = frame.groupby("signal_month")[
        "monthly_metronome"
    ].rank(pct=True)
    frame["monthly_risk_rank_pct"] = frame.groupby("signal_month")["monthly_risk"].rank(
        pct=True
    )
    frame["monthly_defensive_rank_pct"] = frame.groupby("signal_month")[
        "monthly_defensive"
    ].rank(pct=True)
    return frame


def finalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "signal_month",
        "date",
        "month_end_date_used",
        "country_iso3",
        "country_name",
        "month_obs_count",
        "month_calendar_days",
        "month_obs_share",
        *MONTHLY_RAW_COLUMNS,
        *MONTHLY_Z_COLUMNS,
        "monthly_metronome",
        "monthly_risk",
        "monthly_defensive",
        "monthly_metronome_rank_pct",
        "monthly_risk_rank_pct",
        "monthly_defensive_rank_pct",
        "month_day_status_worst",
        "month_gkg_fetch_share_mean",
        "month_gkg_fetch_share_min",
    ]
    present = [column for column in preferred if column in frame.columns]
    remainder = [column for column in frame.columns if column not in present]
    result = frame[present + remainder].copy()
    result = result.sort_values(["date", "country_iso3"]).reset_index(drop=True)
    return result


def main() -> None:
    args = parse_args()
    input_path = Path(args.daily_panel_parquet)
    output_parquet = Path(args.output_parquet)
    output_csv = Path(args.output_csv)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    daily = load_daily_panel(input_path)
    monthly_frames = []
    for _iso3, country_frame in daily.groupby("country_iso3", sort=False):
        monthly_frames.append(
            build_country_monthly(
                country_frame=country_frame,
                fast_span=args.fast_span,
                slow_span=args.slow_span,
                risk_span=args.risk_span,
                z_window_months=args.z_window_months,
                min_history_months=args.min_history_months,
            )
        )

    if not monthly_frames:
        raise ValueError("No country data found in the daily signal panel")

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly = add_cross_sectional_ranks(monthly)
    monthly = finalize_columns(monthly)

    monthly.to_parquet(output_parquet, index=False)
    monthly.to_csv(output_csv, index=False)

    print(f"saved {output_parquet}")
    print(f"saved {output_csv}")
    print(f"months={monthly['signal_month'].nunique()} countries={monthly['country_iso3'].nunique()} rows={len(monthly)}")


if __name__ == "__main__":
    main()
