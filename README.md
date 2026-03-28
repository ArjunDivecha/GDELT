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

If some quarter-hour GKG ZIPs return `404` or another fetch error, the day build now skips those files, writes a partial day from the files that were available, and records the missing ZIPs in the per-day manifest. No prior-day carry-forward or replacement data is used.

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
  --manifest-dir data/manifests/country_day \
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

Signal build notes:

- rolling windows are calendar-aware by default, so missing country-day files create real gaps instead of compressing the lookback window to the next 30 observed rows
- per-day manifest coverage is merged into the signal panel when available via `day_status`, `gkg_fetch_share`, `gkg_files_expected`, `gkg_files_fetched`, and `gkg_files_missing`
- use `--observation-windows` only if you explicitly want the older behavior of rolling over the last N observed rows regardless of date gaps

## Build Monthly Metronome

The daily panel remains the source of truth. The monthly layer is a month-end decision panel built from recency-weighted daily features rather than a simple monthly average.

Monthly schema:

![Monthly metronome schema](docs/assets/monthly_metronome_schema.png)

```bash
python3 scripts/build_monthly_metronome.py \
  --daily-panel-parquet data/panels/country_signal_daily.parquet \
  --output-parquet data/panels/country_signal_monthly.parquet \
  --output-csv data/panels/country_signal_monthly.csv
```

The monthly feature block includes:

- `sentiment_fast`, `sentiment_slow`, `sentiment_trend`
- `attention_fast`, `attention_slow`, `attention_trend`
- `risk_fast`
- `dispersion_fast`
- `local_tone_fast`, `foreign_tone_fast`, `local_foreign_gap`

The primary monthly composites are:

- `monthly_metronome`
- `monthly_risk`
- `monthly_defensive`

Cross-sectional rank percentiles are emitted for each month so countries can be ranked directly at the rebalance date.

## Build Price Returns And Backtest Panel

Given a `PX_LAST` workbook with one daily price row per country bucket:

```bash
python3 scripts/build_country_return_panel.py \
  --price-xlsx "Daily Return.xlsx" \
  --signal-panel-parquet data/panels/country_signal_daily.parquet
```

Outputs:

- `data/panels/country_price_return_daily.parquet`
- `data/panels/country_price_return_daily.csv`
- `data/panels/country_signal_backtest_daily.parquet` when the price and signal panels overlap in date
- `data/panels/country_signal_backtest_daily.csv` when the price and signal panels overlap in date

Notes:

- blank or whitespace-only price cells are treated as missing
- forward returns are emitted for both calendar-day horizons and next active market-session horizons
- duplicated signal countries are intentionally mapped onto multiple return buckets, e.g. `USA -> NASDAQ, U.S., US SmallCap`
- market-session weekdays are inferred from the price history for each bucket rather than hardcoded

## Analyze Signal Predictiveness

Run first-pass IC, decile-spread, and simple regression tests on the matched panel:

```bash
python3 scripts/analyze_country_signal_predictiveness.py \
  --backtest-panel-parquet data/panels/country_signal_backtest_daily.parquet \
  --output-dir output/analysis/first_pass_session
```

Optional filters:

- `--signals signal_a,signal_b,...`
- `--horizons 1,5,20`
- `--max-gap-days 1`
- `--exclude-partial-days`

Outputs:

- `output/analysis/*/country_signal_ic_summary.csv`
- `output/analysis/*/country_signal_decile_summary.csv`
- `output/analysis/*/country_signal_regression_summary.csv`

## Export Template-Aligned Analysis Workbook

To create an Excel workbook whose dates and bucket columns match `Daily Return.xlsx` exactly on every sheet:

```bash
python3 scripts/export_analysis_template_workbook.py \
  --template-xlsx "Daily Return.xlsx" \
  --backtest-panel-parquet data/panels/country_signal_backtest_daily.parquet \
  --output-xlsx output/spreadsheet/gdelt_analysis_template_workbook.xlsx
```

To create a smaller workbook with only a selected subset of variables:

```bash
python3 scripts/export_analysis_template_workbook.py \
  --template-xlsx "Daily Return.xlsx" \
  --backtest-panel-parquet data/panels/country_signal_backtest_daily.parquet \
  --output-xlsx output/spreadsheet/gdelt_core_analysis_workbook.xlsx \
  --variables "ret_1d,ret_fwd_1session,ret_fwd_5session,ret_fwd_20session,country_news_sentiment,country_news_sentiment_x_attention,country_news_risk,local_tone_z,foreign_tone_z,tone_dispersion_z"
```

Notes:

- every exported sheet uses the exact same date rows and bucket headers as the template workbook
- blank cells are left blank when a variable has no value on a template date
- Excel sheet names are capped at 31 characters, so `country_news_sentiment_x_attention` is exported as `country_news_sent_x_attention`

## Export Workbook

```bash
python3 scripts/export_country_sentiment_workbook.py \
  --panel-parquet data/panels/country_signal_daily.parquet \
  --output output/spreadsheet/gdelt_country_signals.xlsx \
  --template-xlsx "/Users/arjundivecha/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Sample Country .xlsx"
```

To export the monthly metronome workbook instead:

```bash
python3 scripts/export_country_sentiment_workbook.py \
  --panel-parquet data/panels/country_signal_monthly.parquet \
  --output output/spreadsheet/gdelt_country_signals_monthly.xlsx \
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
