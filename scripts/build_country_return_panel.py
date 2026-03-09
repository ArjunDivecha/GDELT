#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_BUCKETS = [
    ("Singapore", "SGP"),
    ("Australia", "AUS"),
    ("Canada", "CAN"),
    ("Germany", "DEU"),
    ("Japan", "JPN"),
    ("Switzerland", "CHE"),
    ("U.K.", "GBR"),
    ("NASDAQ", "USA"),
    ("U.S.", "USA"),
    ("France", "FRA"),
    ("Netherlands", "NLD"),
    ("Sweden", "SWE"),
    ("Italy", "ITA"),
    ("ChinaA", "CHN"),
    ("Chile", "CHL"),
    ("Indonesia", "IDN"),
    ("Philippines", "PHL"),
    ("Poland", "POL"),
    ("US SmallCap", "USA"),
    ("Malaysia", "MYS"),
    ("Taiwan", "TWN"),
    ("Mexico", "MEX"),
    ("Korea", "KOR"),
    ("Brazil", "BRA"),
    ("South Africa", "ZAF"),
    ("Denmark", "DNK"),
    ("India", "IND"),
    ("ChinaH", "CHN"),
    ("Hong Kong", "HKG"),
    ("Thailand", "THA"),
    ("Turkey", "TUR"),
    ("Spain", "ESP"),
    ("Vietnam", "VNM"),
    ("Saudi Arabia", "SAU"),
]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_HORIZONS = (1, 5, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a country-bucket price/return panel and merge it with the GDELT signal panel."
    )
    parser.add_argument(
        "--price-xlsx",
        default="Daily Return.xlsx",
        help="Workbook containing the PX_LAST sheet with daily country bucket prices.",
    )
    parser.add_argument(
        "--sheet-name",
        default="PX_LAST",
        help="Sheet name in the price workbook.",
    )
    parser.add_argument(
        "--signal-panel-parquet",
        default="data/panels/country_signal_daily.parquet",
        help="Optional GDELT signal panel parquet used to build the matched backtest sample.",
    )
    parser.add_argument(
        "--output-return-csv",
        default="data/panels/country_price_return_daily.csv",
        help="Output CSV for the price/return panel.",
    )
    parser.add_argument(
        "--output-return-parquet",
        default="data/panels/country_price_return_daily.parquet",
        help="Output parquet for the price/return panel.",
    )
    parser.add_argument(
        "--output-backtest-csv",
        default="data/panels/country_signal_backtest_daily.csv",
        help="Output CSV for the matched signal/return panel.",
    )
    parser.add_argument(
        "--output-backtest-parquet",
        default="data/panels/country_signal_backtest_daily.parquet",
        help="Output parquet for the matched signal/return panel.",
    )
    parser.add_argument(
        "--horizons",
        default="1,5,20",
        help="Comma-separated forward return horizons. Used for both calendar-day and session returns.",
    )
    return parser.parse_args()


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        horizon = int(part)
        if horizon <= 0:
            raise ValueError(f"Forward return horizons must be positive integers. Got: {horizon}")
        horizons.append(horizon)
    if not horizons:
        return DEFAULT_HORIZONS
    return tuple(sorted(set(horizons)))


def bucket_metadata_frame() -> pd.DataFrame:
    rows = []
    for order, (bucket_label, country_iso3) in enumerate(PRICE_BUCKETS, start=1):
        rows.append(
            {
                "bucket_label": bucket_label,
                "country_iso3": country_iso3,
                "bucket_order": order,
            }
        )
    return pd.DataFrame(rows)


def load_price_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price workbook not found: {path}")

    frame = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    expected_columns = ["Country", *[label for label, _iso3 in PRICE_BUCKETS]]
    missing = sorted(set(expected_columns) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected_columns))
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing columns: {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected columns: {', '.join(unexpected)}")
        raise ValueError(
            f"Price workbook columns do not match the expected country bucket layout ({'; '.join(problems)})"
        )

    frame = frame[expected_columns].copy()
    frame = frame.rename(columns={"Country": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Price workbook contains non-date values in column A")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Price workbook dates must be sorted ascending")

    for bucket_label, _iso3 in PRICE_BUCKETS:
        cleaned = frame[bucket_label].where(
            ~frame[bucket_label].astype(str).str.fullmatch(r"\s*"),
            np.nan,
        )
        frame[bucket_label] = pd.to_numeric(
            cleaned,
            errors="coerce",
        )

    return frame


def infer_active_weekdays(dates: pd.Series, series: pd.Series) -> tuple[int, ...]:
    prior = series.shift(1)
    changed = series.notna() & prior.notna() & series.ne(prior)
    counts = changed.groupby(dates.dt.dayofweek).sum()
    active_weekdays = tuple(int(day) for day, count in counts.items() if count > 0)
    if not active_weekdays:
        valid_days = sorted({int(day) for day in dates[series.notna()].dt.dayofweek})
        active_weekdays = tuple(valid_days)
    return tuple(sorted(active_weekdays))


def weekday_labels(active_weekdays: tuple[int, ...]) -> str:
    return ",".join(WEEKDAY_NAMES[day] for day in active_weekdays)


def compute_forward_session_returns(
    dates: pd.Series,
    series: pd.Series,
    active_weekdays: tuple[int, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    values = series.to_numpy(dtype=float)
    weekdays = dates.dt.dayofweek.to_numpy(dtype=int)
    valid_mask = ~np.isnan(values)
    active_mask = valid_mask & np.isin(weekdays, list(active_weekdays))
    active_positions = np.flatnonzero(active_mask)

    outputs: dict[str, np.ndarray] = {}
    for horizon in horizons:
        outputs[f"ret_fwd_{horizon}session"] = np.full(len(values), np.nan, dtype=float)
        outputs[f"ret_fwd_{horizon}session_date"] = np.full(
            len(values), np.datetime64("NaT"), dtype="datetime64[ns]"
        )

    for idx, px in enumerate(values):
        if np.isnan(px):
            continue
        future_start = np.searchsorted(active_positions, idx + 1, side="left")
        for horizon in horizons:
            target_slot = future_start + horizon - 1
            if target_slot >= len(active_positions):
                continue
            target_idx = active_positions[target_slot]
            target_px = values[target_idx]
            if np.isnan(target_px):
                continue
            outputs[f"ret_fwd_{horizon}session"][idx] = target_px / px - 1.0
            outputs[f"ret_fwd_{horizon}session_date"][idx] = dates.iloc[target_idx].to_datetime64()

    return pd.DataFrame(outputs)


def build_price_return_panel(price_wide: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    metadata = bucket_metadata_frame()
    dates = price_wide["date"]
    bucket_frames = []

    for bucket_label, country_iso3 in PRICE_BUCKETS:
        series = price_wide[bucket_label].copy()
        active_weekdays = infer_active_weekdays(dates, series)

        bucket_frame = pd.DataFrame(
            {
                "date": dates,
                "bucket_label": bucket_label,
                "country_iso3": country_iso3,
                "px_last": series,
                "price_available": series.notna(),
                "calendar_weekday": dates.dt.day_name(),
                "is_active_weekday": dates.dt.dayofweek.isin(active_weekdays),
                "active_weekdays": weekday_labels(active_weekdays),
                "ret_1d": series.divide(series.shift(1)) - 1.0,
            }
        )

        for horizon in horizons:
            bucket_frame[f"ret_fwd_{horizon}d"] = series.shift(-horizon).divide(series) - 1.0

        bucket_frame = pd.concat(
            [bucket_frame, compute_forward_session_returns(dates, series, active_weekdays, horizons)],
            axis=1,
        )
        bucket_frames.append(bucket_frame)

    result = pd.concat(bucket_frames, ignore_index=True)
    result = result.merge(metadata, on=["bucket_label", "country_iso3"], how="left")
    result = result.sort_values(["date", "bucket_order"]).reset_index(drop=True)
    return result


def load_signal_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signal panel not found: {path}")

    signal_frame = pd.read_parquet(path)
    signal_frame["date"] = pd.to_datetime(signal_frame["date"]).dt.normalize()
    signal_frame = signal_frame.dropna(subset=["country_iso3"]).copy()
    signal_frame = signal_frame.sort_values(["date", "country_iso3"]).drop_duplicates(
        subset=["date", "country_iso3"], keep="last"
    )
    return signal_frame


def build_backtest_panel(price_return_panel: pd.DataFrame, signal_frame: pd.DataFrame) -> pd.DataFrame:
    merged = price_return_panel.merge(
        signal_frame,
        on=["date", "country_iso3"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_signal"),
    )
    return merged.sort_values(["date", "bucket_order"]).reset_index(drop=True)


def write_panel(frame: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)

    price_xlsx_path = Path(args.price_xlsx)
    output_return_csv_path = Path(args.output_return_csv)
    output_return_parquet_path = Path(args.output_return_parquet)
    output_backtest_csv_path = Path(args.output_backtest_csv)
    output_backtest_parquet_path = Path(args.output_backtest_parquet)

    price_wide = load_price_workbook(price_xlsx_path, args.sheet_name)
    price_return_panel = build_price_return_panel(price_wide, horizons=horizons)
    write_panel(price_return_panel, output_return_csv_path, output_return_parquet_path)

    print(f"saved {output_return_csv_path}")
    print(f"saved {output_return_parquet_path}")
    print(
        "price_panel "
        f"dates={price_return_panel['date'].nunique()} "
        f"buckets={price_return_panel['bucket_label'].nunique()} "
        f"rows={len(price_return_panel)}"
    )

    signal_panel_path = Path(args.signal_panel_parquet)
    if not signal_panel_path.exists():
        print(f"signal panel missing, skipped backtest merge: {signal_panel_path}")
        return

    signal_frame = load_signal_panel(signal_panel_path)
    backtest_panel = build_backtest_panel(price_return_panel, signal_frame)
    if backtest_panel.empty:
        print(
            "warning: no overlapping signal/price dates for the matched backtest panel "
            f"(price range {price_return_panel['date'].min().date()} to "
            f"{price_return_panel['date'].max().date()}, signal range {signal_frame['date'].min().date()} to "
            f"{signal_frame['date'].max().date()})"
        )
        return

    write_panel(backtest_panel, output_backtest_csv_path, output_backtest_parquet_path)
    print(f"saved {output_backtest_csv_path}")
    print(f"saved {output_backtest_parquet_path}")
    print(
        "backtest_panel "
        f"dates={backtest_panel['date'].nunique()} "
        f"buckets={backtest_panel['bucket_label'].nunique()} "
        f"rows={len(backtest_panel)}"
    )


if __name__ == "__main__":
    main()
