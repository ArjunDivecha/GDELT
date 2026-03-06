#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_COUNTRY_LOOKUP_URL = "http://data.gdeltproject.org/api/v2/guides/LOOKUP-COUNTRIES.TXT"
GEONAMES_COUNTRYINFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
GDELT_DOMAIN_COUNTRY_URL = (
    "https://blog.gdeltproject.org/wp-content/uploads/2021-news-outlets-by-countrycode-2015-2021.csv"
)

COUNTRY_OVERRIDES = {
    "RB": {"country_name": "Serbia", "country_iso3": "SRB"},
    "GZ": {"country_name": "Palestinian Territory", "country_iso3": "PSE"},
    "JN": {"country_name": "Svalbard and Jan Mayen", "country_iso3": "SJM"},
    "OC": {"country_name": "Ocean", "country_iso3": ""},
    "OS": {"country_name": "Oceans", "country_iso3": ""},
    "PF": {"country_name": "Paracel Islands", "country_iso3": ""},
    "PG": {"country_name": "Spratly Islands", "country_iso3": ""},
}

GKG_COLUMNS = [
    "gkg_record_id",
    "date",
    "source_collection_identifier",
    "source_common_name",
    "document_identifier",
    "counts",
    "v2counts",
    "themes",
    "v2themes",
    "locations",
    "v2locations",
    "persons",
    "v2persons",
    "organizations",
    "v2organizations",
    "v2tone",
    "dates",
    "gcam",
    "sharing_image",
    "related_images",
    "social_image_embeds",
    "social_video_embeds",
    "quotations",
    "all_names",
    "amounts",
    "translation_info",
    "extras",
]


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


def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def tone_parts(v2tone):
    parts = (v2tone or "").split(",")
    parts += [""] * (7 - len(parts))
    return {
        "tone": safe_float(parts[0]),
        "positive_score": safe_float(parts[1]),
        "negative_score": safe_float(parts[2]),
        "polarity": safe_float(parts[3]),
        "activity_ref_density": safe_float(parts[4]),
        "self_group_ref_density": safe_float(parts[5]),
        "word_count": int(float(parts[6])) if parts[6] not in ("", None) else None,
    }


def split_items(value):
    if not value:
        return []
    return [item for item in value.split(";") if item]


def parse_v2location_item(item):
    parts = item.split("#")
    if len(parts) < 8:
        return None
    parts += [""] * (9 - len(parts))
    return {
        "location_type": parts[0],
        "full_name": clean_text(parts[1]),
        "country_code": clean_text(parts[2]),
        "adm1_code": clean_text(parts[3]),
        "adm2_code": clean_text(parts[4]),
        "latitude": safe_float(parts[5]),
        "longitude": safe_float(parts[6]),
        "feature_id": clean_text(parts[7]),
        "char_offset": int(parts[8]) if re.fullmatch(r"-?\d+", parts[8] or "") else None,
    }


def parse_v2theme_item(item):
    if not item:
        return None
    if "," in item:
        theme, offset = item.split(",", 1)
    else:
        theme, offset = item, ""
    theme = clean_text(theme)
    if not theme:
        return None
    return {
        "theme": theme,
        "char_offset": int(offset) if re.fullmatch(r"-?\d+", offset or "") else None,
    }


def quantile(values, q):
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return xs[0]
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
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
    return variance ** 0.5


def weighted_avg(weighted_num, weighted_den, raw_sum, raw_count):
    if weighted_den and weighted_den > 0:
        return weighted_num / weighted_den
    if raw_count and raw_count > 0:
        return raw_sum / raw_count
    return None


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


def fetch_bytes(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise last_error


def ensure_file(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(fetch_bytes(url))
    return path


def ensure_support_files(base_dir: Path) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "masterfilelist": ensure_file(base_dir / "masterfilelist.txt", MASTERFILELIST_URL),
        "gdelt_country_lookup": ensure_file(base_dir / "COUNTRY-GEO-LOOKUP.TXT", GDELT_COUNTRY_LOOKUP_URL),
        "geonames_countryinfo": ensure_file(base_dir / "geonames_countryInfo.txt", GEONAMES_COUNTRYINFO_URL),
        "domain_country_lookup": ensure_file(
            base_dir / "gdelt_domains_by_country_2015_2021.csv", GDELT_DOMAIN_COUNTRY_URL
        ),
    }


def load_gdelt_country_lookup(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            country_name, _lat, _lon, gdelt_code = parts[:4]
            if gdelt_code:
                mapping[gdelt_code] = country_name
    return mapping


def load_geonames_fips_lookup(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
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
    best_counts = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
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


def load_gkg_urls_by_date(masterfilelist_path: Path, date_keys: Iterable[str]) -> dict[str, list[str]]:
    wanted = set(date_keys)
    urls_by_date = {date_key: [] for date_key in wanted}
    with masterfilelist_path.open("r", encoding="utf-8", errors="replace") as handle:
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


def iter_gkg_rows_from_zip_bytes(payload: bytes):
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0]) as raw:
            wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            reader = csv.reader(wrapper, delimiter="\t")
            for row in reader:
                values = list(row[:27])
                if len(values) < 27:
                    values += [""] * (27 - len(values))
                yield dict(zip(GKG_COLUMNS, values))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
