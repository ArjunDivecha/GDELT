# GDELT Country Sentiment Pipeline

This repo builds a daily country-level sentiment panel from GDELT GKG without storing raw ZIP archives or article-level normalized CSVs.

## Design

The long-history pipeline is stream-first:

1. stream each 15-minute GKG ZIP directly from GDELT
2. aggregate rows in memory into one `country_day` file per date
3. compute rolling country signals from the retained daily files
4. export workbook views from the signal panel

Durable outputs:

- `data/country_day/YYYY-MM-DD.parquet`
- `data/manifests/country_day/YYYY-MM-DD.json`
- `data/panels/country_signal_daily.parquet`
- `data/panels/country_signal_daily.csv`

The pipeline auto-downloads the small lookup files it needs into `data/lookups/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One Day

```bash
python3 scripts/stream_build_country_day.py \
  --date 2026-02-27
```

## Backfill A Range

```bash
python3 scripts/stream_backfill_country_day.py \
  --start-date 2026-01-29 \
  --end-date 2026-02-27 \
  --day-workers 4
```

## Build 30-Day Signals

```bash
python3 scripts/build_country_signals.py \
  --country-day-dir data/country_day \
  --window 30 \
  --min-history 10
```

The core GDELT-only signal block is:

- `country_news_sentiment_raw = local_tone`
- `country_news_attention = log(1 + local_n_articles)`
- `local_attention_share = local_n_articles / local_source_total_articles`
- `sentiment_x_attention_raw = local_tone * local_attention_share`
- `country_news_risk_raw = -local_tone + 0.5 * tone_dispersion`
- rolling 30-day z-scores for the above

## Export Workbook

```bash
python3 scripts/export_country_sentiment_workbook.py \
  --panel-parquet data/panels/country_signal_daily.parquet \
  --output output/spreadsheet/gdelt_country_signals.xlsx \
  --template-xlsx "/Users/arjundivecha/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Sample Country .xlsx"
```

## Two-Machine Backfill

Use the repo for code sync, not data sync.

Example split:

- machine 1: `2015-02-18` to `2020-12-31`
- machine 2: `2021-01-01` to `2026-02-27`

After both finish:

1. copy the resulting `data/country_day/*.parquet` files into one machine
2. run `build_country_signals.py`
3. export the workbook

Because outputs are one file per day, merge is just a date-wise union.

## Notes

- The current GKG v2 archive starts in February 2015, not 2000.
- `scripts/normalize_gkg.py`, `scripts/build_country_day.py`, and `scripts/backfill_gkg_range.py` are the older CSV-heavy path and should be treated as legacy.
