# PRD: Monthly Country Sentiment Metronome

## Objective

Build a monthly country-level decision layer from the existing daily GDELT signal panel without replacing the daily dataset as the source of truth.

The monthly layer must preserve:

- freshness at the rebalance date
- trend information within the month
- attention and stress dynamics
- within-country comparability over time
- cross-country comparability at each month-end

## Problem

A simple monthly average of daily sentiment is too blunt for allocation decisions.

It has two main failures:

1. It gives early-month and late-month news nearly equal influence even when the portfolio decision is made at month-end.
2. It discards the path inside the month, including whether a country improved, deteriorated, or experienced a late-month stress event.

The monthly layer should therefore be a month-end snapshot of recency-weighted daily features rather than a raw monthly average.

## Scope

In scope:

- consume the existing daily panel from `data/panels/country_signal_daily.parquet`
- generate one month-end row per country-month
- emit monthly feature columns and final monthly composites
- emit a parquet and CSV output
- allow workbook export using the existing exporter

Out of scope:

- new article-level NLP
- FinBERT or any external text model
- direct return prediction
- portfolio optimization

## Source Of Truth

Daily source of truth remains:

- `data/country_day/YYYY-MM-DD.parquet`
- `data/panels/country_signal_daily.parquet`

The monthly layer is a derived panel only.

## Inputs

Required daily columns:

- `date`
- `country_iso3`
- `country_name`
- `country_news_sentiment_raw`
- `country_news_risk_raw`
- `local_attention_share`
- `country_news_attention`
- `local_tone`
- `foreign_tone`
- `tone_dispersion`

Optional quality columns carried through when present:

- `day_status`
- `gkg_fetch_share`
- `gkg_files_expected`
- `gkg_files_fetched`
- `gkg_files_missing`

## Monthly Construction Logic

### Month-End Observation Rule

For each country and each calendar month:

- use the last available daily observation in that month
- record both the calendar month and the actual `month_end_date_used`

This preserves a clean month bucket while making the actual source date auditable.

### Daily Feature Families

The monthly layer is constructed from these daily feature families:

1. Sentiment
- `country_news_sentiment_raw`

2. Attention
- `local_attention_share`

3. Stress
- `country_news_risk_raw`
- `tone_dispersion`

4. Cross-border tone split
- `local_tone`
- `foreign_tone`

### Recency-Weighted Daily Features

At the daily level, compute exponentially weighted moving averages:

- `sentiment_fast`: EWMA span 5 of `country_news_sentiment_raw`
- `sentiment_slow`: EWMA span 20 of `country_news_sentiment_raw`
- `sentiment_trend`: `sentiment_fast - sentiment_slow`
- `attention_fast`: EWMA span 5 of `local_attention_share`
- `attention_slow`: EWMA span 20 of `local_attention_share`
- `attention_trend`: `attention_fast - attention_slow`
- `risk_fast`: EWMA span 10 of `country_news_risk_raw`
- `dispersion_fast`: EWMA span 10 of `tone_dispersion`
- `local_tone_fast`: EWMA span 5 of `local_tone`
- `foreign_tone_fast`: EWMA span 5 of `foreign_tone`
- `local_foreign_gap`: `local_tone_fast - foreign_tone_fast`

The spans are chosen to separate:

- latest signal state
- slower monthly inertia
- direction of change inside the month

## Standardization

Monthly features are standardized within each country using trailing monthly history:

- default z-score window: 24 months
- default minimum history before emitting a z-score: 6 months

This follows the logic that country news levels are not directly comparable in raw form across countries.

## Final Monthly Outputs

### Monthly Feature Columns

Raw monthly feature columns:

- `sentiment_fast`
- `sentiment_slow`
- `sentiment_trend`
- `attention_fast`
- `attention_slow`
- `attention_trend`
- `risk_fast`
- `dispersion_fast`
- `local_tone_fast`
- `foreign_tone_fast`
- `local_foreign_gap`

Z-scored monthly feature columns:

- `sentiment_fast_z`
- `sentiment_slow_z`
- `sentiment_trend_z`
- `attention_fast_z`
- `attention_slow_z`
- `attention_trend_z`
- `risk_fast_z`
- `dispersion_fast_z`
- `local_tone_fast_z`
- `foreign_tone_fast_z`
- `local_foreign_gap_z`

### Composite Signals

Primary directional composite:

```text
monthly_metronome
= 0.35 * sentiment_fast_z
+ 0.20 * sentiment_slow_z
+ 0.20 * sentiment_trend_z
+ 0.15 * attention_fast_z
- 0.10 * risk_fast_z
```

Primary defensive composite:

```text
monthly_risk
= 0.45 * risk_fast_z
+ 0.30 * dispersion_fast_z
- 0.15 * sentiment_fast_z
- 0.10 * foreign_tone_fast_z
```

These weights are engineered priors, not estimated coefficients.
They are intended as a transparent first version that can be tested and revised after backtesting.

## Ranking Layer

For each month, rank countries cross-sectionally:

- `monthly_metronome_rank_pct`: 1.0 is the strongest positive metronome
- `monthly_risk_rank_pct`: 1.0 is the highest risk
- `monthly_defensive_rank_pct`: 1.0 is the lowest risk

## Quality Columns

Each monthly row should include:

- `signal_month`
- `month_end_date_used`
- `month_obs_count`
- `month_calendar_days`
- `month_obs_share`

If daily fetch metadata exists:

- `month_day_status_worst`
- `month_gkg_fetch_share_mean`
- `month_gkg_fetch_share_min`

These columns make it possible to exclude or downweight low-coverage months later.

## File Outputs

Primary outputs:

- `data/panels/country_signal_monthly.parquet`
- `data/panels/country_signal_monthly.csv`

Workbook output:

- `output/spreadsheet/gdelt_country_signals_monthly.xlsx`

## CLI

New script:

```bash
python3 scripts/build_monthly_metronome.py \
  --daily-panel-parquet data/panels/country_signal_daily.parquet \
  --output-parquet data/panels/country_signal_monthly.parquet \
  --output-csv data/panels/country_signal_monthly.csv
```

Workbook export:

```bash
python3 scripts/export_country_sentiment_workbook.py \
  --panel-parquet data/panels/country_signal_monthly.parquet \
  --output output/spreadsheet/gdelt_country_signals_monthly.xlsx \
  --template-xlsx "/Users/arjundivecha/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Sample Country .xlsx"
```

## Validation

Validation checks:

1. one monthly row per `country_iso3` and `signal_month`
2. `month_end_date_used` is the last observed daily row in the month
3. raw EWMA features are non-null when there is at least one daily row in the month
4. z-scored features remain null until minimum history is met
5. rank percentiles are computed only among countries with non-null composite values

## Success Criteria

The implementation is successful when:

1. it produces a monthly panel from the existing daily panel without new raw storage
2. the monthly panel is auditable back to the exact month-end daily observation used
3. the workbook export works with the existing country bucket layout
4. the resulting monthly composites are ready for cross-country ranking and backtesting
