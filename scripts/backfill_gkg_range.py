#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import urlopen


MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill a date range of GDELT GKG data and build daily aggregates."
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--masterfilelist",
        default="data/raw/gkg/masterfilelist.txt",
        help="Path to masterfilelist.txt",
    )
    parser.add_argument(
        "--raw-base-dir",
        default="data/raw/gkg",
        help="Directory for raw GKG ZIP files",
    )
    parser.add_argument(
        "--normalized-base-dir",
        default="data/normalized",
        help="Directory for normalized daily outputs",
    )
    parser.add_argument(
        "--aggregates-base-dir",
        default="data/aggregates",
        help="Directory for daily aggregate outputs",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Concurrent download workers per day",
    )
    parser.add_argument(
        "--day-workers",
        type=int,
        default=4,
        help="Number of days to process in parallel",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Top-N countries to include in the daily top file",
    )
    parser.add_argument(
        "--refresh-masterfilelist",
        action="store_true",
        help="Redownload masterfilelist.txt before processing",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def ensure_masterfilelist(path: Path, refresh: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return
    with urlopen(MASTERFILELIST_URL) as response:
        content = response.read()
    path.write_bytes(content)


def load_gkg_urls_by_date(path: Path, wanted_dates: set[str]) -> dict[str, list[str]]:
    urls_by_date = {date_key: [] for date_key in wanted_dates}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            url = parts[-1]
            if not url.endswith(".gkg.csv.zip"):
                continue
            basename = url.rsplit("/", 1)[-1]
            date_key = basename[:8]
            if date_key in urls_by_date:
                urls_by_date[date_key].append(url)
    return urls_by_date


def download_one(url: str, destination: Path) -> None:
    if destination.exists():
        return
    with urlopen(url) as response:
        destination.write_bytes(response.read())


def download_day(raw_day_dir: Path, urls: list[str], workers: int) -> None:
    raw_day_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for url in urls:
        destination = raw_day_dir / url.rsplit("/", 1)[-1]
        if not destination.exists():
            missing.append((url, destination))
    if not missing:
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, url, destination) for url, destination in missing]
        for future in as_completed(futures):
            future.result()


def run_step(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise SystemExit("end-date must be on or after start-date")

    requested = [dt.strftime("%Y%m%d") for dt in iter_dates(start_date, end_date)]
    masterfilelist_path = Path(args.masterfilelist)
    ensure_masterfilelist(masterfilelist_path, refresh=args.refresh_masterfilelist)
    urls_by_date = load_gkg_urls_by_date(masterfilelist_path, set(requested))

    missing_days = [day for day in requested if not urls_by_date.get(day)]
    if missing_days:
        raise SystemExit(f"No GKG URLs found for: {', '.join(missing_days)}")

    def process_day(date_key: str) -> str:
        urls = sorted(urls_by_date[date_key])
        raw_day_dir = Path(args.raw_base_dir) / date_key
        normalized_dir = Path(args.normalized_base_dir) / f"gkg_{date_key}"
        aggregate_dir = Path(args.aggregates_base_dir) / f"gkg_{date_key}"
        aggregate_file = aggregate_dir / "country_day_all.csv"
        normalized_files = [
            normalized_dir / "article.csv",
            normalized_dir / "article_country.csv",
            normalized_dir / "article_theme.csv",
        ]

        print(f"[{date_key}] urls={len(urls)}")
        download_day(raw_day_dir, urls, args.download_workers)

        if not all(path.exists() for path in normalized_files):
            run_step(
                [
                    "python3",
                    "scripts/normalize_gkg.py",
                    "--input-glob",
                    f"{raw_day_dir}/*.zip",
                    "--output-dir",
                    args.normalized_base_dir,
                ]
            )

        if not aggregate_file.exists():
            run_step(
                [
                    "python3",
                    "scripts/build_country_day.py",
                    "--normalized-dir",
                    str(normalized_dir),
                    "--output-dir",
                    str(aggregate_dir),
                    "--top-n",
                    str(args.top_n),
                ]
            )
        return date_key

    with ThreadPoolExecutor(max_workers=args.day_workers) as executor:
        futures = {executor.submit(process_day, date_key): date_key for date_key in requested}
        for future in as_completed(futures):
            completed_day = future.result()
            print(f"[done] {completed_day}")

    print(f"Completed {requested[0]} to {requested[-1]}")


if __name__ == "__main__":
    main()
