#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from gdelt_support import (
    clean_text,
    enrich_country,
    ensure_support_files,
    fetch_bytes,
    infer_source_country_code,
    iter_gkg_rows_from_zip_bytes,
    load_domain_country_lookup,
    load_gdelt_country_lookup,
    load_geonames_fips_lookup,
    load_gkg_urls_by_date,
    parse_v2location_item,
    parse_v2theme_item,
    quantile,
    split_items,
    stddev,
    tone_parts,
    weighted_avg,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream one GDELT GKG day directly into country-day parquet")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument(
        "--lookups-dir",
        default="data/lookups",
        help="Directory for masterfilelist and lookup files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/country_day",
        help="Directory for daily country aggregate parquet files",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests/country_day",
        help="Directory for per-day manifest JSON files",
    )
    parser.add_argument(
        "--top-k-themes",
        type=int,
        default=0,
        help="Optional number of top themes to retain per country-day",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing daily parquet file",
    )
    return parser.parse_args()


def parse_date(value: str) -> tuple[str, str]:
    dt = datetime.strptime(value, "%Y-%m-%d").date()
    return dt.isoformat(), dt.strftime("%Y%m%d")


def extract_country_mentions(record: dict[str, str]) -> Counter:
    mentions = Counter()
    for item in split_items(record.get("v2locations", "")):
        parsed = parse_v2location_item(item)
        country_code = parsed["country_code"] if parsed else ""
        if country_code:
            mentions[country_code] += 1
    return mentions


def extract_theme_counts(record: dict[str, str]) -> Counter:
    counts = Counter()
    raw_themes = record.get("v2themes") or ""
    for item in split_items(raw_themes):
        parsed = parse_v2theme_item(item)
        if parsed:
            counts[parsed["theme"]] += 1
    return counts


def main() -> None:
    args = parse_args()
    date_label, date_key = parse_date(args.date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{date_label}.parquet"
    manifest_path = manifest_dir / f"{date_label}.json"

    if output_path.exists() and not args.overwrite:
        print(f"exists {output_path}")
        return

    support = ensure_support_files(Path(args.lookups_dir))
    urls_by_date = load_gkg_urls_by_date(support["masterfilelist"], [date_key])
    urls = sorted(urls_by_date.get(date_key, []))
    if not urls:
        raise SystemExit(f"No GKG URLs found for {date_label}")

    gdelt_lookup = load_gdelt_country_lookup(support["gdelt_country_lookup"])
    geonames_lookup = load_geonames_fips_lookup(support["geonames_countryinfo"])
    domain_country_lookup = load_domain_country_lookup(support["domain_country_lookup"])

    country_location_mentions = Counter()
    source_total_articles = Counter()
    stats = defaultdict(
        lambda: {
            "n_articles": 0,
            "sum_tone": 0.0,
            "sum_pos": 0.0,
            "sum_neg": 0.0,
            "sum_polarity": 0.0,
            "sum_word_count": 0,
            "tone_weighted_num": 0.0,
            "tone_weighted_den": 0,
            "tones": [],
            "local_n_articles": 0,
            "foreign_n_articles": 0,
            "unknown_source_n_articles": 0,
            "local_sum_tone": 0.0,
            "foreign_sum_tone": 0.0,
            "local_weighted_num": 0.0,
            "local_weighted_den": 0,
            "foreign_weighted_num": 0.0,
            "foreign_weighted_den": 0,
        }
    )
    theme_counts = defaultdict(Counter)
    seen_documents = set()
    rows_scanned = 0
    rows_kept = 0

    for url in urls:
        payload = fetch_bytes(url)
        for record in iter_gkg_rows_from_zip_bytes(payload):
            rows_scanned += 1
            document_identifier = clean_text(record.get("document_identifier"))
            dedupe_key = document_identifier or clean_text(record.get("gkg_record_id"))
            if not dedupe_key or dedupe_key in seen_documents:
                continue
            seen_documents.add(dedupe_key)

            country_mentions = extract_country_mentions(record)
            if not country_mentions:
                continue

            tone = tone_parts(record.get("v2tone", ""))
            if None in (
                tone["tone"],
                tone["positive_score"],
                tone["negative_score"],
                tone["polarity"],
            ):
                continue

            rows_kept += 1
            source_country_code = infer_source_country_code(
                record.get("source_common_name", ""),
                record.get("document_identifier", ""),
                domain_country_lookup,
            )
            if source_country_code:
                source_total_articles[source_country_code] += 1

            word_count = tone["word_count"] or 0
            theme_counter = extract_theme_counts(record) if args.top_k_themes > 0 else Counter()

            for country_code, mention_count in country_mentions.items():
                country_location_mentions[country_code] += mention_count
                stat = stats[country_code]
                stat["n_articles"] += 1
                stat["sum_tone"] += tone["tone"]
                stat["sum_pos"] += tone["positive_score"]
                stat["sum_neg"] += tone["negative_score"]
                stat["sum_polarity"] += tone["polarity"]
                stat["sum_word_count"] += word_count
                stat["tones"].append(tone["tone"])
                if word_count > 0:
                    stat["tone_weighted_num"] += tone["tone"] * word_count
                    stat["tone_weighted_den"] += word_count

                if not source_country_code:
                    stat["unknown_source_n_articles"] += 1
                elif source_country_code == country_code:
                    stat["local_n_articles"] += 1
                    stat["local_sum_tone"] += tone["tone"]
                    if word_count > 0:
                        stat["local_weighted_num"] += tone["tone"] * word_count
                        stat["local_weighted_den"] += word_count
                else:
                    stat["foreign_n_articles"] += 1
                    stat["foreign_sum_tone"] += tone["tone"]
                    if word_count > 0:
                        stat["foreign_weighted_num"] += tone["tone"] * word_count
                        stat["foreign_weighted_den"] += word_count

                if theme_counter:
                    for theme, count in theme_counter.items():
                        theme_counts[country_code][theme] += count

    rows = []
    for country_code, stat in stats.items():
        n_articles = stat["n_articles"]
        if n_articles == 0:
            continue

        total_theme_mentions = sum(theme_counts[country_code].values())
        top_themes_json = ""
        if args.top_k_themes > 0:
            payload = []
            for theme, count in theme_counts[country_code].most_common(args.top_k_themes):
                share = count / total_theme_mentions if total_theme_mentions else 0.0
                payload.append({"theme": theme, "count": count, "share": round(share, 6)})
            top_themes_json = json.dumps(payload, ensure_ascii=True)

        local_tone = weighted_avg(
            stat["local_weighted_num"],
            stat["local_weighted_den"],
            stat["local_sum_tone"],
            stat["local_n_articles"],
        )
        foreign_tone = weighted_avg(
            stat["foreign_weighted_num"],
            stat["foreign_weighted_den"],
            stat["foreign_sum_tone"],
            stat["foreign_n_articles"],
        )
        resolved_source_n_articles = stat["local_n_articles"] + stat["foreign_n_articles"]
        local_source_total_articles = source_total_articles[country_code]
        rows.append(
            {
                "date": date_label,
                **enrich_country(country_code, gdelt_lookup, geonames_lookup),
                "n_articles": n_articles,
                "country_location_mentions": country_location_mentions[country_code],
                "total_word_count": stat["sum_word_count"],
                "tone_mean": stat["sum_tone"] / n_articles,
                "tone_wavg_wordcount": weighted_avg(
                    stat["tone_weighted_num"],
                    stat["tone_weighted_den"],
                    stat["sum_tone"],
                    stat["n_articles"],
                ),
                "positive_mean": stat["sum_pos"] / n_articles,
                "negative_mean": stat["sum_neg"] / n_articles,
                "polarity_mean": stat["sum_polarity"] / n_articles,
                "tone_p10": quantile(stat["tones"], 0.10),
                "tone_p50": quantile(stat["tones"], 0.50),
                "tone_p90": quantile(stat["tones"], 0.90),
                "tone_dispersion": stddev(stat["tones"]),
                "local_tone": local_tone,
                "foreign_tone": foreign_tone,
                "local_n_articles": stat["local_n_articles"],
                "foreign_n_articles": stat["foreign_n_articles"],
                "unknown_source_n_articles": stat["unknown_source_n_articles"],
                "source_resolution_rate": (
                    resolved_source_n_articles / n_articles if n_articles else None
                ),
                "local_source_total_articles": local_source_total_articles,
                "local_attention_share": (
                    stat["local_n_articles"] / local_source_total_articles
                    if local_source_total_articles > 0
                    else None
                ),
                "total_theme_mentions": total_theme_mentions,
                "top_themes_json": top_themes_json,
            }
        )

    rows.sort(key=lambda row: row["n_articles"], reverse=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False)

    write_json(
        manifest_path,
        {
            "date": date_label,
            "gkg_files": len(urls),
            "rows_scanned": rows_scanned,
            "rows_kept": rows_kept,
            "countries": len(rows),
            "output_path": str(output_path),
        },
    )

    print(f"saved {output_path}")
    print(f"files={len(urls)} rows_scanned={rows_scanned} rows_kept={rows_kept} countries={len(rows)}")


if __name__ == "__main__":
    main()
