#!/usr/bin/env python3
"""GDELT monthly features + T2 1MRet: rolling 5y train window, 5y+5y split. INPUT: T2 Master.xlsx 1MRet; GDELT country_signal_monthly_fullhistory.csv. OUTPUT: output/spreadsheet/monthly_return_forecast_eval.xlsx, output/charts/monthly_return_forecast_eval.pdf. Last update: 2026-04-02."""
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore", category=UserWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_T2 = "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/T2-GDELT/T2 Master.xlsx"
DEFAULT_GDELT = PROJECT_ROOT / "data/panels/country_signal_monthly_fullhistory.csv"
DEFAULT_XLSX = PROJECT_ROOT / "output/spreadsheet/monthly_return_forecast_eval.xlsx"
DEFAULT_PDF = PROJECT_ROOT / "output/charts/monthly_return_forecast_eval.pdf"

from build_country_return_panel import PRICE_BUCKETS  # noqa: E402

FEATURE_COLUMNS = [
    "monthly_metronome", "monthly_risk", "monthly_defensive",
    "monthly_metronome_rank_pct", "monthly_risk_rank_pct", "monthly_defensive_rank_pct",
    "sentiment_fast_z", "sentiment_slow_z", "sentiment_trend_z",
    "attention_fast_z", "attention_slow_z", "attention_trend_z",
    "risk_fast_z", "dispersion_fast_z", "local_tone_fast_z", "foreign_tone_fast_z", "local_foreign_gap_z",
]


def parse_args():
    a = argparse.ArgumentParser(description="Monthly return forecasts: GDELT + T2 1MRet.")
    a.add_argument("--t2-xlsx", default=DEFAULT_T2)
    a.add_argument("--gdelt-monthly", default=str(DEFAULT_GDELT))
    a.add_argument("--out-xlsx", default=str(DEFAULT_XLSX))
    a.add_argument("--out-pdf", default=str(DEFAULT_PDF))
    a.add_argument("--tune-months", type=int, default=48)
    a.add_argument("--val-months", type=int, default=12)
    a.add_argument("--random-state", type=int, default=42)
    a.add_argument("--n-jobs", type=int, default=-1)
    return a.parse_args()


def load_gdelt(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    df["signal_month"] = pd.PeriodIndex(df["signal_month"].astype(str), freq="M")
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip().str.upper()
    return df


def load_t2_long(path: Path) -> pd.DataFrame:
    w = pd.read_excel(path, sheet_name="1MRet")
    w["Country"] = pd.to_datetime(w["Country"]).dt.normalize()
    cols = [lbl for lbl, _ in PRICE_BUCKETS]
    miss = [c for c in cols if c not in w.columns]
    if miss:
        raise ValueError("1MRet missing columns: " + ", ".join(miss))
    L = w.melt(id_vars=["Country"], value_vars=cols, var_name="bucket_label", value_name="y_ret")
    L["signal_month"] = L["Country"].dt.to_period("M")
    return L.drop(columns=["Country"])


def build_frame(g: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    isos = {iso for _, iso in PRICE_BUCKETS}
    g = g.loc[g["country_iso3"].isin(isos)]
    parts = []
    for bl, iso in PRICE_BUCKETS:
        sub = g.loc[g["country_iso3"] == iso].copy()
        if sub.empty:
            continue
        sub["bucket_label"] = bl
        parts.append(sub)
    p = pd.concat(parts, ignore_index=True)
    return p.merge(r, on=["signal_month", "bucket_label"], how="inner", validate="one_to_one").sort_values(
        ["signal_month", "bucket_label"]
    ).reset_index(drop=True)


def validate_t2(path: Path) -> dict:
    px = pd.read_excel(path, sheet_name="PX_LAST")
    rt = pd.read_excel(path, sheet_name="1MRet")
    px["Country"] = pd.to_datetime(px["Country"])
    rt["Country"] = pd.to_datetime(rt["Country"])
    col = "Singapore"
    lp = px.sort_values("Country").groupby(px["Country"].dt.to_period("M"))[col].last()
    mom = lp.pct_change()
    out = []
    for s in ["2016-03-01", "2018-06-01"]:
        ts = pd.Timestamp(s)
        row = rt.loc[rt["Country"] == ts, col]
        if row.empty:
            continue
        y = float(row.iloc[0])
        n = ts.to_period("M") + 1
        c = float(mom.loc[n]) if n in mom.index else float("nan")
        out.append({"stamp": s, "1MRet": y, "px_mom": c, "abs_diff": abs(y - c)})
    return {"checks": out, "join_rule": "T2 row first-of-month M labels return in calendar month M+1"}


def pick_features(df: pd.DataFrame):
    use = [c for c in FEATURE_COLUMNS if c in df.columns]
    miss = set(FEATURE_COLUMNS) - set(use)
    if miss:
        raise ValueError("Missing GDELT columns: " + ", ".join(sorted(miss)))
    return use


def tune(tr: pd.DataFrame, cols, tune_m, val_m, rs, nj):
    tf = tr[tr["signal_month"].isin(tune_m)].dropna(subset=cols + ["y_ret"])
    vf = tr[tr["signal_month"].isin(val_m)].dropna(subset=cols + ["y_ret"])
    if len(tf) < 50 or len(vf) < 20:
        m = ExtraTreesRegressor(n_estimators=400, max_depth=12, min_samples_leaf=5, random_state=rs, n_jobs=nj)
        m.fit(tf[cols].to_numpy(), tf["y_ret"].to_numpy())
        return "ExtraTrees(default)", m
    Xt, yt = tf[cols].to_numpy(), tf["y_ret"].to_numpy()
    Xv, yv = vf[cols].to_numpy(), vf["y_ret"].to_numpy()
    best = (np.inf, None, None)
    for g in ParameterGrid({"n_estimators": [300, 500], "max_depth": [8, 12, None], "min_samples_leaf": [2, 5]}):
        m = ExtraTreesRegressor(random_state=rs, n_jobs=nj, **g)
        m.fit(Xt, yt)
        e = root_mean_squared_error(yv, m.predict(Xv))
        if e < best[0]:
            best = (e, f"ET{g}", m)
    for g in ParameterGrid({"n_estimators": [300, 500], "max_depth": [8, 12], "min_samples_leaf": [2, 5]}):
        m = RandomForestRegressor(random_state=rs, n_jobs=nj, **g)
        m.fit(Xt, yt)
        e = root_mean_squared_error(yv, m.predict(Xv))
        if e < best[0]:
            best = (e, f"RF{g}", m)
    for g in ParameterGrid({"alpha": [0.1, 1.0, 10.0, 100.0]}):
        m = Ridge(**g)
        m.fit(Xt, yt)
        e = root_mean_squared_error(yv, m.predict(Xv))
        if e < best[0]:
            best = (e, f"Ridge{g}", m)
    return best[1], best[2]


def ctor_from(m, rs, nj):
    if isinstance(m, ExtraTreesRegressor):
        p = m.get_params()
        return lambda: ExtraTreesRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            random_state=rs, n_jobs=nj,
        )
    if isinstance(m, RandomForestRegressor):
        p = m.get_params()
        return lambda: RandomForestRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            random_state=rs, n_jobs=nj,
        )
    return lambda: Ridge(alpha=m.get_params()["alpha"])


def roll_oos(merged, cols, months, i0, mk, name):
    pr = []
    for ti in range(i0 + 60, min(i0 + 120, len(months))):
        t = months[ti]
        si = ti - 60
        if si < 0:
            continue
        win = months[si:ti]
        tr = merged[merged["signal_month"].isin(win)].dropna(subset=cols + ["y_ret"])
        te = merged[merged["signal_month"] == t].dropna(subset=cols)
        if tr.empty or te.empty:
            continue
        model = mk()
        model.fit(tr[cols].to_numpy(), tr["y_ret"].to_numpy())
        te = te.copy()
        te["y_pred"] = model.predict(te[cols].to_numpy())
        te["model"] = name
        pr.append(te[["signal_month", "bucket_label", "country_iso3", "y_ret", "y_pred", "model"]])
    return pd.concat(pr, ignore_index=True) if pr else pd.DataFrame()


def q_ls(df):
    rows = []
    for sm, g in df.groupby("signal_month", sort=True):
        if len(g) < 10:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g["y_pred"], 5, labels=False, duplicates="drop")
        if g["q"].nunique() < 5:
            continue
        hi, lo = g["q"].max(), g["q"].min()
        rows.append({"signal_month": sm, "long_short": g.loc[g["q"] == hi, "y_ret"].mean() - g.loc[g["q"] == lo, "y_ret"].mean()})
    return pd.DataFrame(rows)


def single_oos(merged, cols, months, i0, mk):
    tr_m, te_m = months[i0 : i0 + 60], months[i0 + 60 : i0 + 120]
    tr = merged[merged["signal_month"].isin(tr_m)].dropna(subset=cols + ["y_ret"])
    te = merged[merged["signal_month"].isin(te_m)].dropna(subset=cols)
    if tr.empty or te.empty:
        return pd.DataFrame()
    m = mk()
    m.fit(tr[cols].to_numpy(), tr["y_ret"].to_numpy())
    te = te.copy()
    te["y_pred"] = m.predict(te[cols].to_numpy())
    te["model"] = "single_fit"
    return te[["signal_month", "bucket_label", "country_iso3", "y_ret", "y_pred", "model"]]


def save_xlsx(path, val, name, pr, ps, qr, qs):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    w0 = wb.active
    w0.title = "validation"
    w0.append([json.dumps(val, indent=2, default=str)])
    w0.append(["model", name])

    def _xcell(v):
        if hasattr(v, "strftime"):
            return v.strftime("%Y-%m-%d") if hasattr(v, "hour") else str(v)
        if isinstance(v, (np.integer, np.floating)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    def sh(nm, df):
        w = wb.create_sheet(nm)
        if df.empty:
            w.append(["empty"])
            return
        w.append([str(c) for c in df.columns])
        for row in df.itertuples(index=False):
            w.append([_xcell(x) for x in row])

    if not pr.empty:
        sh("sum_roll", pd.DataFrame([{"RMSE": root_mean_squared_error(pr["y_ret"], pr["y_pred"]), "MAE": mean_absolute_error(pr["y_ret"], pr["y_pred"])}]))
    sh("pred_roll", pr)
    sh("q_roll", qr)
    if not ps.empty:
        sh("sum_single", pd.DataFrame([{"RMSE": root_mean_squared_error(ps["y_ret"], ps["y_pred"])}]))
    sh("pred_single", ps)
    sh("q_single", qs)
    wb.save(path)


def save_pdf(path, qr, qs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        if not qr.empty:
            ax.plot(np.arange(len(qr)), qr["long_short"].cumsum(), label="rolling")
        if not qs.empty:
            ax.plot(np.arange(len(qs)), qs["long_short"].cumsum(), label="single")
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend()
        ax.set_title("Cumulative Q5-Q1 realized")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    args = parse_args()
    gp = Path(args.gdelt_monthly)
    if not gp.exists():
        raise FileNotFoundError(gp)
    val = validate_t2(Path(args.t2_xlsx))
    g = load_gdelt(gp)
    r = load_t2_long(Path(args.t2_xlsx))
    merged = build_frame(g, r)
    cols = pick_features(merged)
    merged[cols] = merged[cols].apply(pd.to_numeric, errors="coerce")

    months = sorted(merged["signal_month"].unique())
    buckets = {b for b, _ in PRICE_BUCKETS}
    i0 = None
    for i, m in enumerate(months):
        sub = merged.loc[merged["signal_month"] == m]
        if set(sub["bucket_label"]) >= buckets and sub["y_ret"].notna().all():
            i0 = i
            break
    if i0 is None:
        raise RuntimeError("No full cross-section month")
    if i0 + 120 > len(months):
        raise RuntimeError("Not enough months for 5y+5y after M0")
    if args.tune_months + args.val_months > 60:
        raise ValueError("tune + val > 60")
    tb = months[i0 : i0 + 60]
    tune_m = tb[: args.tune_months]
    val_m = tb[args.tune_months : args.tune_months + args.val_months]
    tr_sub = merged[merged["signal_month"].isin(tb)]
    nm, tm = tune(tr_sub, cols, tune_m, val_m, args.random_state, args.n_jobs)
    mk = ctor_from(tm, args.random_state, args.n_jobs)
    pr = roll_oos(merged, cols, months, i0, mk, nm)
    ps = single_oos(merged, cols, months, i0, mk)
    qr, qs = q_ls(pr), q_ls(ps)
    save_xlsx(Path(args.out_xlsx), val, nm, pr, ps, qr, qs)
    save_pdf(Path(args.out_pdf), qr, qs)
    print("Wrote", args.out_xlsx, args.out_pdf)
    if not pr.empty:
        print("RMSE roll", root_mean_squared_error(pr["y_ret"], pr["y_pred"]))


if __name__ == "__main__":
    main()
