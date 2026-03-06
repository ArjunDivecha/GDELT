#!/usr/bin/env python3
"""Normalize GDELT GKG TSV into article, article_country, and article_theme tables.

Usage:
  python3 scripts/normalize_gkg.py \
    --input data/samples/gkg_latest_200rows.tsv \
    --output-dir data/normalized
"""

import argparse
import csv
import io
import re
from collections import defaultdict
from pathlib import Path
import sys
import zipfile


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


# GDELT GCAM columns can exceed Python csv default field size.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize GDELT GKG TSV")
    parser.add_argument(
        "--input",
        required=False,
        help="Path to GKG TSV or GKG ZIP file",
    )
    parser.add_argument(
        "--input-glob",
        required=False,
        help="Glob of GKG TSV/ZIP files (e.g., data/raw/gkg/20260227/*.zip)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory; run-specific subdir will be created",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick tests",
    )
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_text(value):
    if value is None:
        return ""
    return value.replace("\r", " ").replace("\n", " ").strip()


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
    # Expected shape:
    # type#full_name#country_code#adm1_code#adm2_code#lat#lon#feature_id#char_offset
    parts = item.split("#")
    if len(parts) < 8:
        return None

    # Some rows can omit adm2 or char offset; pad for consistent indexing.
    parts += [""] * (9 - len(parts))

    return {
        "location_type": parts[0],
        "full_name": clean_text(parts[1]),
        "country_iso2": clean_text(parts[2]),
        "adm1_code": clean_text(parts[3]),
        "adm2_code": clean_text(parts[4]),
        "latitude": safe_float(parts[5]),
        "longitude": safe_float(parts[6]),
        "feature_id": clean_text(parts[7]),
        "char_offset": int(parts[8]) if re.fullmatch(r"-?\d+", parts[8] or "") else None,
    }


def parse_v2theme_item(item):
    # Expected shape typically: THEME,offset
    # Some rows may include only THEME
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


def build_run_id(input_path):
    stem = input_path.stem
    # For names like gkg_latest_200rows.tsv keep full stem.
    return stem


def infer_run_id_from_inputs(paths):
    if len(paths) == 1:
        return build_run_id(paths[0])

    # Try date directory style .../YYYYMMDD/*.zip
    parent = paths[0].parent.name
    if re.fullmatch(r"\d{8}", parent):
        return f"gkg_{parent}"

    return f"gkg_batch_{len(paths)}files"


def iter_rows_from_path(path):
    suffix = path.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if not names:
                return
            with zf.open(names[0]) as raw:
                wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                reader = csv.reader(wrapper, delimiter="\t")
                for row in reader:
                    yield row
        return

    # Default: plain TSV/text file
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            yield row


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    if not args.input and not args.input_glob:
        raise SystemExit("Provide --input or --input-glob")

    input_paths = []
    if args.input:
        input_paths.append(Path(args.input))
    if args.input_glob:
        input_paths.extend(sorted(Path().glob(args.input_glob)))

    if not input_paths:
        raise SystemExit("No input files matched")

    base_output = Path(args.output_dir)
    run_id = infer_run_id_from_inputs(input_paths)
    run_output = base_output / run_id

    article_rows = []
    article_country_rows = []
    article_theme_rows = []

    processed_rows = 0
    for input_path in input_paths:
        for raw_row in iter_rows_from_path(input_path):
            if args.max_rows is not None and processed_rows >= args.max_rows:
                break

            row = list(raw_row[:27])
            if len(row) < 27:
                row += [""] * (27 - len(row))

            record = dict(zip(GKG_COLUMNS, row))
            gkg_record_id = clean_text(record["gkg_record_id"])
            if not gkg_record_id:
                continue

            tone = tone_parts(record["v2tone"])
            article_rows.append(
                {
                    "gkg_record_id": gkg_record_id,
                    "date": clean_text(record["date"]),
                    "source_collection_identifier": clean_text(record["source_collection_identifier"]),
                    "source_common_name": clean_text(record["source_common_name"]),
                    "document_identifier": clean_text(record["document_identifier"]),
                    "tone": tone["tone"],
                    "positive_score": tone["positive_score"],
                    "negative_score": tone["negative_score"],
                    "polarity": tone["polarity"],
                    "activity_ref_density": tone["activity_ref_density"],
                    "self_group_ref_density": tone["self_group_ref_density"],
                    "word_count": tone["word_count"],
                    "v2tone_raw": clean_text(record["v2tone"]),
                    "gcam_raw": clean_text(record["gcam"]),
                    "extras_raw": clean_text(record["extras"]),
                }
            )

            # Country extraction from V2 locations.
            country_buckets = defaultdict(lambda: {"mention_count": 0, "location_samples": []})
            for item in split_items(record["v2locations"]):
                loc = parse_v2location_item(item)
                if not loc:
                    continue
                country = loc["country_iso2"]
                if not country:
                    continue
                bucket = country_buckets[country]
                bucket["mention_count"] += 1
                if len(bucket["location_samples"]) < 3 and loc["full_name"]:
                    bucket["location_samples"].append(loc["full_name"])

            for country_iso2, bucket in country_buckets.items():
                article_country_rows.append(
                    {
                        "gkg_record_id": gkg_record_id,
                        "country_iso2": country_iso2,
                        "country_mention_count": bucket["mention_count"],
                        "location_samples": " | ".join(bucket["location_samples"]),
                    }
                )

            # Theme extraction, prefer V2 themes.
            theme_source = record["v2themes"] if clean_text(record["v2themes"]) else record["themes"]
            theme_buckets = defaultdict(lambda: {"mention_count": 0, "min_char_offset": None})
            for item in split_items(theme_source):
                parsed = parse_v2theme_item(item)
                if not parsed:
                    continue
                theme = parsed["theme"]
                bucket = theme_buckets[theme]
                bucket["mention_count"] += 1
                if parsed["char_offset"] is not None:
                    cur = bucket["min_char_offset"]
                    bucket["min_char_offset"] = (
                        parsed["char_offset"] if cur is None else min(cur, parsed["char_offset"])
                    )

            for theme, bucket in theme_buckets.items():
                article_theme_rows.append(
                    {
                        "gkg_record_id": gkg_record_id,
                        "theme": theme,
                        "theme_mention_count": bucket["mention_count"],
                        "theme_first_char_offset": bucket["min_char_offset"],
                    }
                )

            processed_rows += 1
        if args.max_rows is not None and processed_rows >= args.max_rows:
            break

    write_csv(
        run_output / "article.csv",
        article_rows,
        [
            "gkg_record_id",
            "date",
            "source_collection_identifier",
            "source_common_name",
            "document_identifier",
            "tone",
            "positive_score",
            "negative_score",
            "polarity",
            "activity_ref_density",
            "self_group_ref_density",
            "word_count",
            "v2tone_raw",
            "gcam_raw",
            "extras_raw",
        ],
    )
    write_csv(
        run_output / "article_country.csv",
        article_country_rows,
        [
            "gkg_record_id",
            "country_iso2",
            "country_mention_count",
            "location_samples",
        ],
    )
    write_csv(
        run_output / "article_theme.csv",
        article_theme_rows,
        [
            "gkg_record_id",
            "theme",
            "theme_mention_count",
            "theme_first_char_offset",
        ],
    )

    print(f"Input files: {len(input_paths)}")
    if len(input_paths) <= 5:
        for p in input_paths:
            print(f"  - {p}")
    else:
        print(f"  - first: {input_paths[0]}")
        print(f"  - last: {input_paths[-1]}")
    print(f"Output: {run_output}")
    print(f"article rows: {len(article_rows)}")
    print(f"article_country rows: {len(article_country_rows)}")
    print(f"article_theme rows: {len(article_theme_rows)}")


if __name__ == "__main__":
    main()
