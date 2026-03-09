#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SIGNALS = [
    "country_news_sentiment",
    "country_news_risk",
    "country_news_sentiment_x_attention",
    "attention_shock",
    "local_tone_z",
    "foreign_tone_z",
    "tone_dispersion_z",
    "local_attention_share_z",
    "country_news_sentiment_raw",
    "country_news_risk_raw",
    "country_news_attention",
    "local_tone",
    "foreign_tone",
    "tone_dispersion",
    "local_attention_share",
    "n_articles",
    "local_n_articles",
]
DEFAULT_HORIZONS = (1, 5, 20)
REQUIRED_ID_COLUMNS = ["date", "bucket_label", "country_iso3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze country-bucket signal predictiveness using ICs, decile spreads, and simple regressions."
    )
    parser.add_argument(
        "--backtest-panel-parquet",
        default="data/panels/country_signal_backtest_daily.parquet",
        help="Matched signal/return panel built by build_country_return_panel.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/analysis",
        help="Directory for summary and daily analysis outputs.",
    )
    parser.add_argument(
        "--signals",
        default="",
        help="Comma-separated signal columns to test. Defaults to the core signal block when present.",
    )
    parser.add_argument(
        "--horizons",
        default="1,5,20",
        help="Comma-separated session return horizons, matching ret_fwd_{horizon}session columns.",
    )
    parser.add_argument(
        "--min-cross-section",
        type=int,
        default=10,
        help="Minimum non-null bucket count required on a date to include it in IC/decile/regression tests.",
    )
    parser.add_argument(
        "--min-fetch-share",
        type=float,
        default=0.0,
        help="Optional minimum gkg_fetch_share. Rows below this threshold are excluded when the column exists.",
    )
    parser.add_argument(
        "--max-gap-days",
        type=float,
        default=None,
        help="Optional maximum days_since_prior_observation. Larger gaps are excluded when the column exists.",
    )
    parser.add_argument(
        "--exclude-partial-days",
        action="store_true",
        help="Exclude rows whose manifest day_status is partial or missing_all_files.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> tuple[int, ...]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        return DEFAULT_HORIZONS
    unique = sorted(set(items))
    for item in unique:
        if item <= 0:
            raise ValueError(f"Horizons must be positive integers. Got: {item}")
    return tuple(unique)


def parse_signal_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def summary_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n == 0:
        return {
            "n_periods": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "t_stat": np.nan,
            "hit_rate": np.nan,
            "mean_abs": np.nan,
            "ir": np.nan,
        }

    mean = float(clean.mean())
    median = float(clean.median())
    mean_abs = float(clean.abs().mean())
    if n == 1:
        return {
            "n_periods": 1,
            "mean": mean,
            "median": median,
            "std": np.nan,
            "t_stat": np.nan,
            "hit_rate": float((clean > 0).mean()),
            "mean_abs": mean_abs,
            "ir": np.nan,
        }

    std = float(clean.std(ddof=1))
    t_stat = mean / (std / sqrt(n)) if std > 0 else np.nan
    ir = mean / std if std > 0 else np.nan
    return {
        "n_periods": n,
        "mean": mean,
        "median": median,
        "std": std,
        "t_stat": t_stat,
        "hit_rate": float((clean > 0).mean()),
        "mean_abs": mean_abs,
        "ir": ir,
    }


def load_backtest_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backtest panel not found: {path}")

    frame = pd.read_parquet(path)
    missing = [column for column in REQUIRED_ID_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Backtest panel missing required columns: {', '.join(missing)}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.sort_values(["date", "bucket_label"]).reset_index(drop=True)


def resolve_signals(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            raise ValueError(f"Requested signals missing from panel: {', '.join(missing)}")
        return requested

    signals = [column for column in DEFAULT_SIGNALS if column in frame.columns]
    if signals:
        return signals

    numeric_columns = []
    excluded_prefixes = ("ret_fwd_",)
    excluded = {
        "px_last",
        "price_available",
        "bucket_order",
        "gkg_fetch_share",
        "gkg_files_expected",
        "gkg_files_fetched",
        "gkg_files_missing",
        "days_since_prior_observation",
    }
    for column in frame.columns:
        if column in REQUIRED_ID_COLUMNS or column in excluded:
            continue
        if column.startswith(excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_columns.append(column)
    return numeric_columns


def filter_panel(
    frame: pd.DataFrame,
    return_column: str,
    min_fetch_share: float,
    max_gap_days: float | None,
    exclude_partial_days: bool,
) -> pd.DataFrame:
    subset = frame.copy()
    subset = subset.dropna(subset=[return_column])

    if "gkg_fetch_share" in subset.columns:
        subset = subset.loc[subset["gkg_fetch_share"].fillna(0.0) >= min_fetch_share]

    if max_gap_days is not None and "days_since_prior_observation" in subset.columns:
        gap_ok = subset["days_since_prior_observation"].isna() | (
            subset["days_since_prior_observation"] <= max_gap_days
        )
        subset = subset.loc[gap_ok]

    if exclude_partial_days and "day_status" in subset.columns:
        subset = subset.loc[~subset["day_status"].isin(["partial", "missing_all_files"])]

    return subset


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - numeric.mean()) / std


def assign_deciles(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.qcut(ranks, 10, labels=False) + 1


def ols_with_intercept(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    df_resid = n - X.shape[1]
    if df_resid > 0:
        sigma2 = rss / df_resid
        cov = sigma2 * np.linalg.inv(X.T @ X)
        stderr = np.sqrt(np.diag(cov))
        intercept_stderr = float(stderr[0])
        beta_stderr = float(stderr[1])
    else:
        intercept_stderr = np.nan
        beta_stderr = np.nan

    return {
        "alpha": float(beta[0]),
        "beta": float(beta[1]),
        "alpha_stderr": intercept_stderr,
        "beta_stderr": beta_stderr,
        "alpha_t_stat": float(beta[0] / intercept_stderr) if intercept_stderr and intercept_stderr > 0 else np.nan,
        "beta_t_stat": float(beta[1] / beta_stderr) if beta_stderr and beta_stderr > 0 else np.nan,
        "r_squared": 1.0 - rss / tss if tss > 0 else np.nan,
        "n_obs": n,
        "df_resid": df_resid,
    }


def ols_no_intercept(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = len(x)
    denom = float(np.sum(x**2))
    if denom == 0:
        return {
            "beta": np.nan,
            "beta_stderr": np.nan,
            "beta_t_stat": np.nan,
            "r_squared": np.nan,
            "n_obs": n,
            "df_resid": n - 1,
        }

    beta = float(np.sum(x * y) / denom)
    resid = y - beta * x
    rss = float(np.sum(resid**2))
    tss = float(np.sum(y**2))
    df_resid = n - 1
    if df_resid > 0:
        sigma2 = rss / df_resid
        beta_stderr = sqrt(sigma2 / denom)
        beta_t_stat = beta / beta_stderr if beta_stderr > 0 else np.nan
    else:
        beta_stderr = np.nan
        beta_t_stat = np.nan

    return {
        "beta": beta,
        "beta_stderr": beta_stderr,
        "beta_t_stat": beta_t_stat,
        "r_squared": 1.0 - rss / tss if tss > 0 else np.nan,
        "n_obs": n,
        "df_resid": df_resid,
    }


def compute_daily_ics(
    frame: pd.DataFrame,
    signal: str,
    return_column: str,
    min_cross_section: int,
) -> pd.DataFrame:
    rows = []
    for date, date_frame in frame.groupby("date", sort=True):
        subset = date_frame[["bucket_label", signal, return_column]].dropna()
        if len(subset) < min_cross_section:
            continue
        x = pd.to_numeric(subset[signal], errors="coerce")
        y = pd.to_numeric(subset[return_column], errors="coerce")
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            continue

        rows.append(
            {
                "date": date,
                "signal": signal,
                "return_column": return_column,
                "universe_size": len(subset),
                "pearson_ic": x.corr(y, method="pearson"),
                "spearman_ic": x.corr(y, method="spearman"),
            }
        )

    return pd.DataFrame(rows)


def summarize_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    if daily_ic.empty:
        return pd.DataFrame(
            columns=[
                "signal",
                "return_column",
                "correlation_type",
                "n_dates",
                "avg_universe_size",
                "mean_ic",
                "median_ic",
                "std_ic",
                "t_stat",
                "hit_rate",
                "mean_abs_ic",
                "ic_ir",
            ]
        )

    rows = []
    for correlation_type in ("pearson_ic", "spearman_ic"):
        stats = summary_stats(daily_ic[correlation_type])
        rows.append(
            {
                "signal": daily_ic["signal"].iloc[0],
                "return_column": daily_ic["return_column"].iloc[0],
                "correlation_type": correlation_type.replace("_ic", ""),
                "n_dates": stats["n_periods"],
                "avg_universe_size": float(daily_ic["universe_size"].mean()),
                "mean_ic": stats["mean"],
                "median_ic": stats["median"],
                "std_ic": stats["std"],
                "t_stat": stats["t_stat"],
                "hit_rate": stats["hit_rate"],
                "mean_abs_ic": stats["mean_abs"],
                "ic_ir": stats["ir"],
            }
        )
    return pd.DataFrame(rows)


def compute_daily_deciles(
    frame: pd.DataFrame,
    signal: str,
    return_column: str,
    min_cross_section: int,
) -> pd.DataFrame:
    rows = []
    for date, date_frame in frame.groupby("date", sort=True):
        subset = date_frame[["bucket_label", signal, return_column]].dropna()
        if len(subset) < max(min_cross_section, 10):
            continue
        x = pd.to_numeric(subset[signal], errors="coerce")
        y = pd.to_numeric(subset[return_column], errors="coerce")
        if x.nunique(dropna=True) < 2:
            continue

        deciles = assign_deciles(x)
        top_mask = deciles == 10
        bottom_mask = deciles == 1
        if top_mask.sum() == 0 or bottom_mask.sum() == 0:
            continue

        top_return = float(y[top_mask].mean())
        bottom_return = float(y[bottom_mask].mean())
        rows.append(
            {
                "date": date,
                "signal": signal,
                "return_column": return_column,
                "universe_size": len(subset),
                "top_bucket_size": int(top_mask.sum()),
                "bottom_bucket_size": int(bottom_mask.sum()),
                "top_return": top_return,
                "bottom_return": bottom_return,
                "spread_return": top_return - bottom_return,
            }
        )

    return pd.DataFrame(rows)


def summarize_deciles(daily_deciles: pd.DataFrame) -> pd.DataFrame:
    if daily_deciles.empty:
        return pd.DataFrame(
            columns=[
                "signal",
                "return_column",
                "n_dates",
                "avg_universe_size",
                "avg_top_bucket_size",
                "avg_bottom_bucket_size",
                "mean_top_return",
                "mean_bottom_return",
                "mean_spread_return",
                "spread_std",
                "spread_t_stat",
                "spread_hit_rate",
                "spread_ir",
            ]
        )

    spread_stats = summary_stats(daily_deciles["spread_return"])
    return pd.DataFrame(
        [
            {
                "signal": daily_deciles["signal"].iloc[0],
                "return_column": daily_deciles["return_column"].iloc[0],
                "n_dates": spread_stats["n_periods"],
                "avg_universe_size": float(daily_deciles["universe_size"].mean()),
                "avg_top_bucket_size": float(daily_deciles["top_bucket_size"].mean()),
                "avg_bottom_bucket_size": float(daily_deciles["bottom_bucket_size"].mean()),
                "mean_top_return": float(daily_deciles["top_return"].mean()),
                "mean_bottom_return": float(daily_deciles["bottom_return"].mean()),
                "mean_spread_return": spread_stats["mean"],
                "spread_std": spread_stats["std"],
                "spread_t_stat": spread_stats["t_stat"],
                "spread_hit_rate": spread_stats["hit_rate"],
                "spread_ir": spread_stats["ir"],
            }
        ]
    )


def compute_regressions(
    frame: pd.DataFrame,
    signal: str,
    return_column: str,
    min_cross_section: int,
) -> pd.DataFrame:
    kept_groups = []
    for date, date_frame in frame.groupby("date", sort=True):
        subset = date_frame[["date", signal, return_column]].dropna().copy()
        if len(subset) < min_cross_section:
            continue
        signal_z = cross_sectional_zscore(subset[signal])
        if signal_z.notna().sum() < min_cross_section:
            continue
        subset["signal_z"] = signal_z
        subset["return_within_date"] = subset[return_column] - subset[return_column].mean()
        kept_groups.append(subset)

    if not kept_groups:
        return pd.DataFrame(
            columns=[
                "signal",
                "return_column",
                "model",
                "n_obs",
                "n_dates",
                "avg_universe_size",
                "alpha",
                "alpha_stderr",
                "alpha_t_stat",
                "beta",
                "beta_stderr",
                "beta_t_stat",
                "r_squared",
            ]
        )

    sample = pd.concat(kept_groups, ignore_index=True)
    x = sample["signal_z"].to_numpy(dtype=float)
    y = sample[return_column].to_numpy(dtype=float)
    y_within = sample["return_within_date"].to_numpy(dtype=float)

    pooled = ols_with_intercept(x, y)
    date_fe = ols_no_intercept(x, y_within)

    n_dates = int(sample["date"].nunique())
    avg_universe_size = float(len(sample) / n_dates) if n_dates else np.nan
    rows = [
        {
            "signal": signal,
            "return_column": return_column,
            "model": "pooled_z",
            "n_obs": pooled["n_obs"],
            "n_dates": n_dates,
            "avg_universe_size": avg_universe_size,
            "alpha": pooled["alpha"],
            "alpha_stderr": pooled["alpha_stderr"],
            "alpha_t_stat": pooled["alpha_t_stat"],
            "beta": pooled["beta"],
            "beta_stderr": pooled["beta_stderr"],
            "beta_t_stat": pooled["beta_t_stat"],
            "r_squared": pooled["r_squared"],
        },
        {
            "signal": signal,
            "return_column": return_column,
            "model": "date_fe_z",
            "n_obs": date_fe["n_obs"],
            "n_dates": n_dates,
            "avg_universe_size": avg_universe_size,
            "alpha": 0.0,
            "alpha_stderr": np.nan,
            "alpha_t_stat": np.nan,
            "beta": date_fe["beta"],
            "beta_stderr": date_fe["beta_stderr"],
            "beta_t_stat": date_fe["beta_t_stat"],
            "r_squared": date_fe["r_squared"],
        },
    ]
    return pd.DataFrame(rows)


def write_frame(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    parquet_path = output_dir / f"{stem}.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)


def main() -> None:
    args = parse_args()
    horizons = parse_int_list(args.horizons)
    requested_signals = parse_signal_list(args.signals)

    panel = load_backtest_panel(Path(args.backtest_panel_parquet))
    signals = resolve_signals(panel, requested_signals)
    if not signals:
        raise SystemExit("No usable signal columns found in the backtest panel.")

    daily_ic_frames = []
    ic_summary_frames = []
    daily_decile_frames = []
    decile_summary_frames = []
    regression_frames = []

    for horizon in horizons:
        return_column = f"ret_fwd_{horizon}session"
        if return_column not in panel.columns:
            print(f"skip missing return column {return_column}")
            continue

        filtered_panel = filter_panel(
            panel,
            return_column=return_column,
            min_fetch_share=args.min_fetch_share,
            max_gap_days=args.max_gap_days,
            exclude_partial_days=args.exclude_partial_days,
        )
        if filtered_panel.empty:
            print(f"skip {return_column}: no rows after filtering")
            continue

        for signal in signals:
            if signal not in filtered_panel.columns:
                continue

            daily_ic = compute_daily_ics(
                filtered_panel,
                signal=signal,
                return_column=return_column,
                min_cross_section=args.min_cross_section,
            )
            if not daily_ic.empty:
                daily_ic_frames.append(daily_ic)
                ic_summary_frames.append(summarize_ic(daily_ic))

            daily_deciles = compute_daily_deciles(
                filtered_panel,
                signal=signal,
                return_column=return_column,
                min_cross_section=args.min_cross_section,
            )
            if not daily_deciles.empty:
                daily_decile_frames.append(daily_deciles)
                decile_summary_frames.append(summarize_deciles(daily_deciles))

            regressions = compute_regressions(
                filtered_panel,
                signal=signal,
                return_column=return_column,
                min_cross_section=args.min_cross_section,
            )
            if not regressions.empty:
                regression_frames.append(regressions)

    output_dir = Path(args.output_dir)

    daily_ic_panel = pd.concat(daily_ic_frames, ignore_index=True) if daily_ic_frames else pd.DataFrame()
    ic_summary_panel = pd.concat(ic_summary_frames, ignore_index=True) if ic_summary_frames else pd.DataFrame()
    daily_decile_panel = (
        pd.concat(daily_decile_frames, ignore_index=True) if daily_decile_frames else pd.DataFrame()
    )
    decile_summary_panel = (
        pd.concat(decile_summary_frames, ignore_index=True) if decile_summary_frames else pd.DataFrame()
    )
    regression_summary_panel = (
        pd.concat(regression_frames, ignore_index=True) if regression_frames else pd.DataFrame()
    )

    write_frame(daily_ic_panel, output_dir, "country_signal_ic_daily")
    write_frame(ic_summary_panel, output_dir, "country_signal_ic_summary")
    write_frame(daily_decile_panel, output_dir, "country_signal_decile_daily")
    write_frame(decile_summary_panel, output_dir, "country_signal_decile_summary")
    write_frame(regression_summary_panel, output_dir, "country_signal_regression_summary")

    print(f"saved {output_dir / 'country_signal_ic_daily.csv'}")
    print(f"saved {output_dir / 'country_signal_ic_summary.csv'}")
    print(f"saved {output_dir / 'country_signal_decile_daily.csv'}")
    print(f"saved {output_dir / 'country_signal_decile_summary.csv'}")
    print(f"saved {output_dir / 'country_signal_regression_summary.csv'}")
    print(
        f"signals_tested={len(signals)} horizons_tested={len(horizons)} "
        f"ic_rows={len(daily_ic_panel)} decile_rows={len(daily_decile_panel)} "
        f"regression_rows={len(regression_summary_panel)}"
    )


if __name__ == "__main__":
    main()
