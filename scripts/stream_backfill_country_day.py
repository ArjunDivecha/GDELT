#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from gdelt_support import ensure_support_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill a date range of daily country parquet files")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--day-workers",
        type=int,
        default=4,
        help="Number of country-day builds to run in parallel",
    )
    parser.add_argument(
        "--lookups-dir",
        default="data/lookups",
        help="Directory for masterfilelist and lookup files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/country_day",
        help="Directory for daily country parquet files",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests/country_day",
        help="Directory for daily manifest files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing daily parquet files",
    )
    return parser.parse_args()


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def run_day(date_label: str, args: argparse.Namespace) -> str:
    command = [
        "python3",
        "scripts/stream_build_country_day.py",
        "--date",
        date_label,
        "--lookups-dir",
        args.lookups_dir,
        "--output-dir",
        args.output_dir,
        "--manifest-dir",
        args.manifest_dir,
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    return date_label


def main() -> None:
    args = parse_args()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise SystemExit("end-date must be on or after start-date")

    ensure_support_files(Path(args.lookups_dir))
    dates = [dt.isoformat() for dt in iter_dates(start_date, end_date)]
    with ThreadPoolExecutor(max_workers=args.day_workers) as executor:
        futures = {executor.submit(run_day, date_label, args): date_label for date_label in dates}
        for future in as_completed(futures):
            print(f"[done] {future.result()}")

    print(f"Completed {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    main()
