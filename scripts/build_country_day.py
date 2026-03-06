#!/usr/bin/env python3
"""
Build country-day sentiment aggregates from normalized GDELT tables.

Inputs required in <normalized-dir>:
  - article.csv
  - article_country.csv
  - article_theme.csv

Outputs in <output-dir>:
  - country_day_all.csv
  - country_day_topN.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

COUNTRY_OVERRIDES = {
    # GDELT occasionally emits non-standard or sub-territory codes that still need a usable ISO3.
    "RB": {"country_name": "Serbia", "country_iso3": "SRB"},
    "GZ": {"country_name": "Palestinian Territory", "country_iso3": "PSE"},
    "JN": {"country_name": "Svalbard and Jan Mayen", "country_iso3": "SJM"},
    # Keep names for non-sovereign or disputed areas, but leave ISO3 blank.
    "OC": {"country_name": "Ocean", "country_iso3": ""},
    "OS": {"country_name": "Oceans", "country_iso3": ""},
    "PF": {"country_name": "Paracel Islands", "country_iso3": ""},
    "PG": {"country_name": "Spratly Islands", "country_iso3": ""},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build country-day aggregates")
    parser.add_argument(
        "--normalized-dir",
        required=True,
        help="Directory containing article.csv, article_country.csv, article_theme.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for aggregate outputs",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of countries to include in topN output",
    )
    parser.add_argument(
        "--top-k-themes",
        type=int,
        default=10,
        help="Number of themes to keep in top theme summary",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Date label (YYYY-MM-DD). If omitted, inferred from normalized dir name when possible.",
    )
    parser.add_argument(
        "--gdelt-country-lookup",
        default="data/lookups/COUNTRY-GEO-LOOKUP.TXT",
        help="Tab-delimited GDELT country lookup: country_name, lat, lon, gdelt/FIPS code",
    )
    parser.add_argument(
        "--geonames-countryinfo",
        default="data/lookups/geonames_countryInfo.txt",
        help="GeoNames countryInfo.txt path used for FIPS -> ISO3 enrichment",
    )
    parser.add_argument(
        "--domain-country-lookup",
        default="data/lookups/gdelt_domains_by_country_2015_2021.csv",
        help="GDELT domain-to-country lookup used for local vs foreign source classification",
    )
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_date(normalized_dir: Path) -> str:
    # e.g. gkg_20260227 -> 2026-02-27
    m = re.search(r"(\d{8})", normalized_dir.name)
    if not m:
        return ""
    d = m.group(1)
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


def quantile(values, q):
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return xs[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stddev(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def load_gdelt_country_lookup(path: Path):
    mapping = {}
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            country_name, _lat, _lon, gdelt_code = parts[:4]
            if gdelt_code:
                mapping[gdelt_code] = country_name
    return mapping


def load_geonames_fips_lookup(path: Path):
    mapping = {}
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            iso2, iso3, _numeric, fips_code, country_name = parts[:5]
            if fips_code:
                mapping[fips_code] = {
                    "country_name": country_name,
                    "country_iso2": iso2,
                    "country_iso3": iso3,
                }
    return mapping


def load_domain_country_lookup(path: Path):
    mapping = {}
    if not path.exists():
        return mapping

    best_counts = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = (row.get("domain") or "").strip().lower()
            gdelt_code = (row.get("countrycode") or "").strip().upper()
            count = safe_int(row.get("cnt")) or 0
            if not domain or not gdelt_code:
                continue
            if domain not in best_counts or count > best_counts[domain]:
                best_counts[domain] = count
                mapping[domain] = gdelt_code
    return mapping


def normalize_domain(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = (parsed.netloc or parsed.path).strip().lower()
    host = host.split("@")[-1].split(":")[0].strip(".")
    host = re.sub(r"^www\d*\.", "", host)
    return host


def domain_candidates(*values: str):
    candidates = []
    seen = set()
    for value in values:
        host = normalize_domain(value)
        if not host:
            continue
        parts = host.split(".")
        for start in range(len(parts) - 1):
            candidate = ".".join(parts[start:])
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    return candidates


def infer_source_country_code(source_common_name: str, document_identifier: str, domain_country_lookup):
    for candidate in domain_candidates(source_common_name, document_identifier):
        gdelt_code = domain_country_lookup.get(candidate)
        if gdelt_code:
            return gdelt_code
    return ""


def weighted_avg(weighted_num, weighted_den, raw_sum, raw_count):
    if weighted_den and weighted_den > 0:
        return weighted_num / weighted_den
    if raw_count and raw_count > 0:
        return raw_sum / raw_count
    return None


def enrich_country(gdelt_code: str, gdelt_lookup, geonames_lookup):
    if gdelt_code in COUNTRY_OVERRIDES:
        override = COUNTRY_OVERRIDES[gdelt_code]
        return {
            "country_code_gdelt": gdelt_code,
            "country_name": override["country_name"],
            "country_iso3": override["country_iso3"],
        }

    geo = geonames_lookup.get(gdelt_code)
    if geo:
        return {
            "country_code_gdelt": gdelt_code,
            "country_name": geo["country_name"],
            "country_iso3": geo["country_iso3"],
        }

    return {
        "country_code_gdelt": gdelt_code,
        "country_name": gdelt_lookup.get(gdelt_code, ""),
        "country_iso3": "",
    }


def main():
    args = parse_args()
    normalized_dir = Path(args.normalized_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    article_path = normalized_dir / "article.csv"
    article_country_path = normalized_dir / "article_country.csv"
    article_theme_path = normalized_dir / "article_theme.csv"

    if not (article_path.exists() and article_country_path.exists() and article_theme_path.exists()):
        raise SystemExit("Missing one or more required input files in normalized dir")

    date_label = args.date if args.date else infer_date(normalized_dir)
    gdelt_lookup = load_gdelt_country_lookup(Path(args.gdelt_country_lookup))
    geonames_lookup = load_geonames_fips_lookup(Path(args.geonames_countryinfo))
    domain_country_lookup = load_domain_country_lookup(Path(args.domain_country_lookup))

    # record_id -> list[gdelt country code]
    record_countries = defaultdict(list)
    # country -> total location mentions (from article_country table)
    country_location_mentions = Counter()
    # source country -> total number of articles from sources in that country
    source_total_articles = Counter()

    with article_country_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("gkg_record_id", "")
            country = row.get("country_iso2", "")
            if not rid or not country:
                continue
            record_countries[rid].append(country)
            country_location_mentions[country] += safe_int(row.get("country_mention_count", 0)) or 0

    # Country-level sentiment accumulators.
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

    with article_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("gkg_record_id", "")
            countries = record_countries.get(rid)
            if not countries:
                continue

            tone = safe_float(row.get("tone"))
            pos = safe_float(row.get("positive_score"))
            neg = safe_float(row.get("negative_score"))
            pol = safe_float(row.get("polarity"))
            wc = safe_int(row.get("word_count")) or 0
            source_country_code = infer_source_country_code(
                row.get("source_common_name", ""),
                row.get("document_identifier", ""),
                domain_country_lookup,
            )
            if source_country_code:
                source_total_articles[source_country_code] += 1

            # Skip rows missing core tone fields.
            if tone is None or pos is None or neg is None or pol is None:
                continue

            for c in countries:
                s = stats[c]
                s["n_articles"] += 1
                s["sum_tone"] += tone
                s["sum_pos"] += pos
                s["sum_neg"] += neg
                s["sum_polarity"] += pol
                s["sum_word_count"] += wc
                s["tones"].append(tone)
                if wc > 0:
                    s["tone_weighted_num"] += tone * wc
                    s["tone_weighted_den"] += wc
                if not source_country_code:
                    s["unknown_source_n_articles"] += 1
                elif source_country_code == c:
                    s["local_n_articles"] += 1
                    s["local_sum_tone"] += tone
                    if wc > 0:
                        s["local_weighted_num"] += tone * wc
                        s["local_weighted_den"] += wc
                else:
                    s["foreign_n_articles"] += 1
                    s["foreign_sum_tone"] += tone
                    if wc > 0:
                        s["foreign_weighted_num"] += tone * wc
                        s["foreign_weighted_den"] += wc

    # Theme counts per country.
    theme_counts = defaultdict(Counter)
    with article_theme_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("gkg_record_id", "")
            countries = record_countries.get(rid)
            if not countries:
                continue
            theme = row.get("theme", "")
            if not theme:
                continue
            mention_count = safe_int(row.get("theme_mention_count")) or 1
            for c in countries:
                theme_counts[c][theme] += mention_count

    rows = []
    for country, s in stats.items():
        n = s["n_articles"]
        if n == 0:
            continue

        total_theme_mentions = sum(theme_counts[country].values())
        top_items = theme_counts[country].most_common(args.top_k_themes)
        top_payload = []
        for theme, cnt in top_items:
            share = (cnt / total_theme_mentions) if total_theme_mentions else 0.0
            top_payload.append({"theme": theme, "count": cnt, "share": round(share, 6)})

        tone_wavg = (
            s["tone_weighted_num"] / s["tone_weighted_den"]
            if s["tone_weighted_den"] > 0
            else None
        )
        local_tone = weighted_avg(
            s["local_weighted_num"],
            s["local_weighted_den"],
            s["local_sum_tone"],
            s["local_n_articles"],
        )
        foreign_tone = weighted_avg(
            s["foreign_weighted_num"],
            s["foreign_weighted_den"],
            s["foreign_sum_tone"],
            s["foreign_n_articles"],
        )
        resolved_source_n_articles = s["local_n_articles"] + s["foreign_n_articles"]
        source_resolution_rate = resolved_source_n_articles / n if n else None
        local_source_total_articles = source_total_articles[country]
        local_attention_share = (
            s["local_n_articles"] / local_source_total_articles
            if local_source_total_articles > 0
            else None
        )

        rows.append(
            {
                "date": date_label,
                **enrich_country(country, gdelt_lookup, geonames_lookup),
                "n_articles": n,
                "country_location_mentions": country_location_mentions[country],
                "total_word_count": s["sum_word_count"],
                "tone_mean": s["sum_tone"] / n,
                "tone_wavg_wordcount": tone_wavg,
                "positive_mean": s["sum_pos"] / n,
                "negative_mean": s["sum_neg"] / n,
                "polarity_mean": s["sum_polarity"] / n,
                "tone_p10": quantile(s["tones"], 0.10),
                "tone_p50": quantile(s["tones"], 0.50),
                "tone_p90": quantile(s["tones"], 0.90),
                "tone_dispersion": stddev(s["tones"]),
                "local_tone": local_tone,
                "foreign_tone": foreign_tone,
                "local_n_articles": s["local_n_articles"],
                "foreign_n_articles": s["foreign_n_articles"],
                "unknown_source_n_articles": s["unknown_source_n_articles"],
                "source_resolution_rate": source_resolution_rate,
                "local_source_total_articles": local_source_total_articles,
                "local_attention_share": local_attention_share,
                "total_theme_mentions": total_theme_mentions,
                "top_themes_json": json.dumps(top_payload, ensure_ascii=True),
            }
        )

    rows.sort(key=lambda r: r["n_articles"], reverse=True)
    top_rows = rows[: args.top_n]

    fieldnames = [
        "date",
        "country_code_gdelt",
        "country_name",
        "country_iso3",
        "n_articles",
        "country_location_mentions",
        "total_word_count",
        "tone_mean",
        "tone_wavg_wordcount",
        "positive_mean",
        "negative_mean",
        "polarity_mean",
        "tone_p10",
        "tone_p50",
        "tone_p90",
        "tone_dispersion",
        "local_tone",
        "foreign_tone",
        "local_n_articles",
        "foreign_n_articles",
        "unknown_source_n_articles",
        "source_resolution_rate",
        "local_source_total_articles",
        "local_attention_share",
        "total_theme_mentions",
        "top_themes_json",
    ]

    def write_csv(path: Path, data):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    all_path = output_dir / "country_day_all.csv"
    top_path = output_dir / f"country_day_top{args.top_n}.csv"
    write_csv(all_path, rows)
    write_csv(top_path, top_rows)

    unresolved = sum(1 for row in rows if not row["country_iso3"])

    print(f"Normalized input: {normalized_dir}")
    print(f"Date label: {date_label}")
    print(f"Countries aggregated: {len(rows)}")
    print(f"Countries with ISO3 mapping: {len(rows) - unresolved}")
    print(f"Countries without ISO3 mapping: {unresolved}")
    print(f"All countries output: {all_path}")
    print(f"Top-{args.top_n} output: {top_path}")


if __name__ == "__main__":
    main()
