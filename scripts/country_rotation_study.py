#!/usr/bin/env python3
"""
Country rotation study: GDELT news signals for next-month equity return prediction.

================================================================================
INPUT FILES:
  1. T2 Master.xlsx (returns + trailing momentum)
     /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/T2-GDELT/T2 Master.xlsx
     - Sheet "1MRet": forward 1-month return. Stamp M = return realized in month M+1.
     - Sheet "12MTR": TRAILING 12-month total return. Known at decision time.
     - Sheet "Earnings Yield": monthly earnings yield per country bucket.
     NOTE: "Ret" sheets = FORWARD returns (future). "TR" sheets = TRAILING (past).
           We ONLY use TR sheets as predictors to avoid look-ahead bias.

  2. GDELT monthly panel
     <project>/data/panels/country_signal_monthly_fullhistory.csv
     Key columns: foreign_tone, tone_mean, country_news_risk_raw

OUTPUT FILES:
  - output/spreadsheet/country_rotation_study.xlsx
  - output/charts/country_rotation_study.pdf

METHODOLOGY:
  - 30 country equity buckets (1:1 ISO mapping, excludes NASDAQ/US SmallCap/ChinaH)
  - Each month, cross-sectionally rank countries on each signal
  - Pick top 3 by rank (or composite rank), equally weight, hold for 1 month
  - Compare to equal-weight all 30 countries
  - Expanding window: start predictions from month 13 onward to maximize study length
  - No ML models. Pure rank-based selection.

SIGNAL RANKING (from cross-sectional Spearman vs next-month return, no look-ahead):
  foreign_tone_3m_chg  t=3.58  (strongest single signal)
  foreign_tone         t=3.19
  tone_mean            t=2.85
  tone_mean_1m_chg     t=2.76
  12MTR (trailing mom) t=-0.47 (NO predictive power for next month)

VERSION: v2 2026-04-03
================================================================================
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import Workbook

warnings.filterwarnings("ignore", category=UserWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_T2 = "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/T2-GDELT/T2 Master.xlsx"
DEFAULT_GDELT = PROJECT_ROOT / "data/panels/country_signal_monthly_fullhistory.csv"
DEFAULT_XLSX = PROJECT_ROOT / "output/spreadsheet/country_rotation_study.xlsx"
DEFAULT_PDF = PROJECT_ROOT / "output/charts/country_rotation_study.pdf"

BUCKETS = [
    ("Singapore", "SGP"), ("Australia", "AUS"), ("Canada", "CAN"), ("Germany", "DEU"),
    ("Japan", "JPN"), ("Switzerland", "CHE"), ("U.K.", "GBR"), ("U.S.", "USA"),
    ("France", "FRA"), ("Netherlands", "NLD"), ("Sweden", "SWE"), ("Italy", "ITA"),
    ("Chile", "CHL"), ("Indonesia", "IDN"), ("Philippines", "PHL"), ("Poland", "POL"),
    ("Malaysia", "MYS"), ("Taiwan", "TWN"), ("Mexico", "MEX"), ("Korea", "KOR"),
    ("Brazil", "BRA"), ("South Africa", "ZAF"), ("Denmark", "DNK"), ("India", "IND"),
    ("Hong Kong", "HKG"), ("Thailand", "THA"), ("Turkey", "TUR"), ("Spain", "ESP"),
    ("Vietnam", "VNM"), ("Saudi Arabia", "SAU"),
]
ISO_TO_BUCKET = {iso: bl for bl, iso in BUCKETS}
BUCKET_LABELS = [bl for bl, _ in BUCKETS]


def parse_args():
    p = argparse.ArgumentParser(description="GDELT country rotation study.")
    p.add_argument("--t2-xlsx", default=DEFAULT_T2)
    p.add_argument("--gdelt-monthly", default=str(DEFAULT_GDELT))
    p.add_argument("--out-xlsx", default=str(DEFAULT_XLSX))
    p.add_argument("--out-pdf", default=str(DEFAULT_PDF))
    p.add_argument("--warmup", type=int, default=12,
                   help="Skip first N months before starting predictions.")
    p.add_argument("--top-n", type=int, default=3,
                   help="Number of countries to pick each month.")
    return p.parse_args()


def load_data(t2_path: Path, gdelt_path: Path) -> pd.DataFrame:
    """Load T2 + GDELT, merge into one long panel with one row per (month, bucket)."""
    ret = pd.read_excel(t2_path, sheet_name="1MRet")
    ret["Country"] = pd.to_datetime(ret["Country"]).dt.normalize()
    mom12 = pd.read_excel(t2_path, sheet_name="12MTR")
    mom12["Country"] = pd.to_datetime(mom12["Country"]).dt.normalize()
    ey = pd.read_excel(t2_path, sheet_name="Earnings Yield")
    ey["Country"] = pd.to_datetime(ey["Country"]).dt.normalize()

    g = pd.read_parquet(gdelt_path) if gdelt_path.suffix == ".parquet" else pd.read_csv(gdelt_path)
    g["signal_month"] = g["signal_month"].astype(str)
    g["country_iso3"] = g["country_iso3"].astype(str).str.strip().str.upper()

    # Pre-compute per-country time-series diffs for GDELT
    gdelt_derived = {}
    for iso in ISO_TO_BUCKET:
        sub = g[g["country_iso3"] == iso].sort_values("signal_month").copy()
        for col in ["foreign_tone", "tone_mean"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub[f"{col}_1m_chg"] = sub[col].diff(1)
            sub[f"{col}_3m_chg"] = sub[col].diff(3)
        sub["risk_raw"] = pd.to_numeric(sub["country_news_risk_raw"], errors="coerce")
        gdelt_derived[iso] = sub

    rows = []
    for _, rrow in ret.iterrows():
        dt = rrow["Country"]
        m_str = dt.strftime("%Y-%m")
        mrow = mom12[mom12["Country"] == dt]
        eyrow = ey[ey["Country"] == dt]

        for iso, bl in ISO_TO_BUCKET.items():
            if bl not in ret.columns:
                continue
            rv = pd.to_numeric(rrow[bl], errors="coerce")
            mv = pd.to_numeric(mrow[bl].iloc[0], errors="coerce") if (not mrow.empty and bl in mrow.columns) else np.nan
            eyv = pd.to_numeric(eyrow[bl].iloc[0], errors="coerce") if (not eyrow.empty and bl in eyrow.columns) else np.nan

            dsub = gdelt_derived.get(iso)
            grow = dsub[dsub["signal_month"] == m_str] if dsub is not None else pd.DataFrame()

            def gval(col):
                return pd.to_numeric(grow[col].iloc[0], errors="coerce") if not grow.empty and col in grow.columns else np.nan

            rows.append({
                "month": dt, "bucket": bl, "iso": iso,
                "next_ret": rv,
                "mom12_trailing": mv,
                "earnings_yield": eyv,
                "foreign_tone": gval("foreign_tone"),
                "tone_mean": gval("tone_mean"),
                "ft_1m_chg": gval("foreign_tone_1m_chg"),
                "ft_3m_chg": gval("foreign_tone_3m_chg"),
                "tm_1m_chg": gval("tone_mean_1m_chg"),
                "tm_3m_chg": gval("tone_mean_3m_chg"),
                "risk_raw": gval("risk_raw"),
            })

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["month", "bucket"]).reset_index(drop=True)
    return panel


def run_strategies(panel: pd.DataFrame, warmup: int, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all strategies month by month. Return (monthly_returns_df, monthly_picks_df)."""
    months = sorted(panel["month"].unique())

    strategy_names = [
        "EW_all",
        "top3_tone_mean",
        "top3_foreign_tone",
        "top3_ft_3m_chg",
        "top3_combo_tm_ftchg",
        "top3_mom12_trailing",
        "top3_combo_tm_mom",
        "filtered_risk",
    ]
    monthly_rets = []
    monthly_picks = []

    for i, dt in enumerate(months):
        if i < warmup:
            continue

        df = panel[panel["month"] == dt].copy()
        have_ft = df["foreign_tone"].notna().sum()
        have_ret = df["next_ret"].notna().sum()
        if have_ft < 15 or have_ret < 15:
            continue

        # Cross-sectional percentile ranks (higher = better)
        for col in ["tone_mean", "foreign_tone", "ft_3m_chg", "tm_1m_chg",
                     "mom12_trailing", "earnings_yield"]:
            df[f"r_{col}"] = df[col].rank(pct=True)
        # Risk: lower is better, so invert
        df["r_risk_inv"] = (1 - df["risk_raw"].rank(pct=True))

        # Composites
        df["combo_tm_ftchg"] = 0.5 * df["r_tone_mean"] + 0.5 * df["r_ft_3m_chg"]
        df["combo_tm_mom"] = 0.6 * df["r_tone_mean"] + 0.4 * df["r_mom12_trailing"]

        row = {"month": dt}
        picks_row = {"month": dt}

        # EW all
        ew_ret = df.dropna(subset=["next_ret"])["next_ret"].mean()
        row["EW_all"] = ew_ret

        def pick_top(score_col, label):
            valid = df.dropna(subset=[score_col, "next_ret"])
            if len(valid) < top_n:
                return np.nan, ""
            top = valid.nlargest(top_n, score_col)
            return top["next_ret"].mean(), ", ".join(top["bucket"].tolist())

        for score_col, label in [
            ("r_tone_mean", "top3_tone_mean"),
            ("r_foreign_tone", "top3_foreign_tone"),
            ("r_ft_3m_chg", "top3_ft_3m_chg"),
            ("combo_tm_ftchg", "top3_combo_tm_ftchg"),
            ("r_mom12_trailing", "top3_mom12_trailing"),
            ("combo_tm_mom", "top3_combo_tm_mom"),
        ]:
            ret_val, names = pick_top(score_col, label)
            row[label] = ret_val
            picks_row[label] = names

        # Filtered: exclude top quartile risk, then top 3 by tone_mean
        valid = df.dropna(subset=["risk_raw", "tone_mean", "next_ret"])
        if len(valid) >= top_n + 5:
            risk_cut = valid["risk_raw"].quantile(0.75)
            filt = valid[valid["risk_raw"] <= risk_cut]
            if len(filt) >= top_n:
                top = filt.nlargest(top_n, "r_tone_mean")
                row["filtered_risk"] = top["next_ret"].mean()
                picks_row["filtered_risk"] = ", ".join(top["bucket"].tolist())
            else:
                row["filtered_risk"] = np.nan
                picks_row["filtered_risk"] = ""
        else:
            row["filtered_risk"] = np.nan
            picks_row["filtered_risk"] = ""

        monthly_rets.append(row)
        monthly_picks.append(picks_row)

    return pd.DataFrame(monthly_rets), pd.DataFrame(monthly_picks)


def compute_metrics(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute portfolio metrics for each strategy column."""
    strat_cols = [c for c in monthly_df.columns if c != "month"]
    rows = []
    for col in strat_cols:
        r = monthly_df[col].dropna().to_numpy()
        if len(r) < 6:
            continue
        cum = (1 + r).prod() - 1
        n = len(r)
        ann = (1 + r).prod() ** (12 / n) - 1
        vol = r.std() * np.sqrt(12)
        sr = (r.mean() * 12) / vol if vol > 0 else 0

        # Max drawdown
        wealth = np.cumprod(1 + r)
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak
        max_dd = dd.min()

        # Hit rate vs EW
        if col != "EW_all":
            ew = monthly_df["EW_all"].dropna().to_numpy()
            min_len = min(len(r), len(ew))
            hit = (r[:min_len] > ew[:min_len]).mean() * 100
        else:
            hit = 0

        rows.append({
            "strategy": col,
            "months": n,
            "cumulative_return": cum,
            "annualized_return": ann,
            "annualized_vol": vol,
            "sharpe_ratio": sr,
            "max_drawdown": max_dd,
            "hit_rate_vs_ew": hit,
        })
    return pd.DataFrame(rows)


def xcell(v):
    """Convert value to Excel-safe type."""
    if isinstance(v, pd.Period):
        return str(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def write_xlsx(path: Path, summary: pd.DataFrame, monthly_rets: pd.DataFrame,
               monthly_picks: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    def dump(title, df):
        ws = wb.create_sheet(title)
        ws.append([str(c) for c in df.columns])
        for row in df.itertuples(index=False):
            ws.append([xcell(x) for x in row])

    ws0 = wb.active
    ws0.title = "README"
    ws0.append(["GDELT Country Rotation Study"])
    ws0.append(["Top N countries by signal rank vs equal-weight all 30"])
    ws0.append(["1MRet stamp M = return realized in M+1"])
    ws0.append(["12MTR = TRAILING 12m return (known at decision time)"])
    ws0.append(["12MRet = FORWARD 12m return (NOT used - future data)"])

    dump("summary", summary)
    dump("monthly_returns", monthly_rets)
    dump("monthly_picks", monthly_picks)
    wb.save(path)


def write_pdf(path: Path, monthly_rets: pd.DataFrame, summary: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    strat_cols = [c for c in monthly_rets.columns if c != "month"]
    months_str = [d.strftime("%Y-%m") for d in monthly_rets["month"]]

    with PdfPages(path) as pdf:
        # Page 1: Cumulative return (log scale)
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in strat_cols:
            r = monthly_rets[col].fillna(0).to_numpy()
            wealth = np.cumprod(1 + r)
            ax.plot(months_str, wealth, label=col, linewidth=1.5 if "combo" in col or col == "EW_all" else 1)
        ax.set_yscale("log")
        ax.set_title("Cumulative wealth (log scale): top 3 by signal vs EW all")
        ax.set_ylabel("Growth of $1")
        step = max(1, len(months_str) // 15)
        ax.set_xticks(months_str[::step])
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Rolling 12m excess return of best combo vs EW
        best_combo = "top3_combo_tm_ftchg"
        if best_combo in monthly_rets.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            combo_r = monthly_rets[best_combo].fillna(0).to_numpy()
            ew_r = monthly_rets["EW_all"].fillna(0).to_numpy()
            excess = combo_r - ew_r
            if len(excess) >= 12:
                roll_12 = pd.Series(excess).rolling(12).sum().to_numpy()
                ax.bar(months_str, roll_12, width=1.0, alpha=0.7)
                ax.axhline(0, color="black", lw=0.8)
                ax.set_title(f"Rolling 12-month excess return: {best_combo} minus EW_all")
                ax.set_ylabel("12m excess return")
                ax.set_xticks(months_str[::step])
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Page 3: Sharpe ratio bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        sr = summary.set_index("strategy")["sharpe_ratio"]
        bars = ax.barh(sr.index, sr.values)
        ax.set_xlabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio by Strategy")
        ax.axvline(0, color="black", lw=0.5)
        for bar, val in zip(bars, sr.values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                    va="center", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    args = parse_args()
    t2_path = Path(args.t2_xlsx)
    gdelt_path = Path(args.gdelt_monthly)

    print("Loading data...")
    panel = load_data(t2_path, gdelt_path)
    months = sorted(panel["month"].unique())
    print(f"  Panel: {len(panel)} rows, {len(months)} months "
          f"({months[0].date()} to {months[-1].date()}), "
          f"{panel['bucket'].nunique()} buckets")

    print(f"Running strategies (warmup={args.warmup}, top_n={args.top_n})...")
    monthly_rets, monthly_picks = run_strategies(panel, args.warmup, args.top_n)
    print(f"  {len(monthly_rets)} scored months "
          f"({monthly_rets['month'].min().date()} to {monthly_rets['month'].max().date()})")

    summary = compute_metrics(monthly_rets)
    print("\n" + summary.to_string(index=False))

    out_xlsx = Path(args.out_xlsx)
    out_pdf = Path(args.out_pdf)
    write_xlsx(out_xlsx, summary, monthly_rets, monthly_picks)
    write_pdf(out_pdf, monthly_rets, summary)
    print(f"\nWrote {out_xlsx}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
