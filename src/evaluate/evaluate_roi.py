"""
ROI grading for daily prediction files.

Replaces evaluate_roi_v4_2.py and evaluate_all_with_results.py, both of which
applied decimal-odds formulas to American odds:

    df["roi"] = np.where(df["won"], df["predicted_odds"] - 1, -1)
    df["implied_prob"] = 1 / df["predicted_odds"]
    "Heavy Fav" if odds < 1.83 ...

A winning bet at -150 was scored as -151 units, and every negative price fell
into the "Heavy Fav (<1.83)" bucket. That is the origin of the impossible
values in performance/rolling_roi.csv (-376%, -547%) which made a roughly
break-even model look catastrophic.

All arithmetic here routes through src/backtest/odds.py, which is tested.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.odds import implied_prob_array, settle
from src.utils.config import PERFORMANCE_DIR, PREDICTIONS_DIR


def _odds_bucket(american: float) -> str:
    """Group by price, in American terms."""
    o = float(american)
    if o <= -200:
        return "Heavy favourite (<= -200)"
    if o < 0:
        return "Favourite (-200 to -100)"
    if o < 200:
        return "Underdog (+100 to +200)"
    return "Long shot (>= +200)"


def grade_file(path: Path) -> pd.DataFrame | None:
    """Grade one prediction file. Returns per-bucket summary, or None."""
    df = pd.read_csv(path)
    if "actual_winner" not in df.columns or "prediction" not in df.columns:
        return None

    df = df[df["actual_winner"].notna() & (df["actual_winner"] != "UNKNOWN")]
    if df.empty or not {"home_odds", "away_odds"}.issubset(df.columns):
        return None

    on_home = df["prediction"] == "HOME"
    df = df.assign(
        bet_odds=np.where(on_home, df["home_odds"], df["away_odds"]),
        won=df["prediction"] == df["actual_winner"],
    )
    df = df[df["bet_odds"].notna() & (df["bet_odds"] != 0)]
    if df.empty:
        return None

    df["profit"] = settle(df["bet_odds"].to_numpy(), df["won"].to_numpy())
    df["implied_prob"] = implied_prob_array(df["bet_odds"].to_numpy())
    df["bucket"] = df["bet_odds"].map(_odds_bucket)
    if "model_win_prob" in df.columns:
        df["edge"] = df["model_win_prob"] - df["implied_prob"]
    else:
        df["edge"] = np.nan

    out = (df.groupby("bucket", as_index=False)
             .agg(bets=("profit", "size"),
                  win_rate=("won", "mean"),
                  roi_pct=("profit", lambda s: s.mean() * 100),
                  profit_units=("profit", "sum"),
                  avg_edge=("edge", "mean")))
    return out


def grade_all(pattern: str = "predictions_*.csv", verbose: bool = True) -> pd.DataFrame:
    """Grade every prediction file with results, and report the total."""
    rows = []
    for p in sorted(PREDICTIONS_DIR.glob(pattern)):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if not m:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "actual_winner" not in df.columns or "prediction" not in df.columns:
            continue
        df = df[df["actual_winner"].notna() & (df["actual_winner"] != "UNKNOWN")]
        if df.empty or not {"home_odds", "away_odds"}.issubset(df.columns):
            continue

        on_home = df["prediction"] == "HOME"
        odds = np.where(on_home, df["home_odds"], df["away_odds"])
        won = (df["prediction"] == df["actual_winner"]).to_numpy()
        ok = pd.notna(odds) & (odds != 0)
        if not ok.any():
            continue
        profit = settle(np.asarray(odds)[ok], won[ok])
        rows.append({"date": m.group(1), "file": p.name, "bets": int(ok.sum()),
                     "accuracy": float(won[ok].mean()),
                     "profit_units": float(profit.sum()),
                     "roi_pct": float(profit.mean() * 100)})

    res = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
    if verbose and len(res):
        tot_bets = int(res["bets"].sum())
        tot_profit = float(res["profit_units"].sum())
        acc = float((res["accuracy"] * res["bets"]).sum() / tot_bets)
        se = None
        print(res.to_string(index=False,
              formatters={"accuracy": "{:.1%}".format,
                          "profit_units": "{:+.2f}".format,
                          "roi_pct": "{:+.2f}".format}))
        print("-" * 62)
        print(f"  bets {tot_bets:,}   accuracy {acc:.1%}   "
              f"profit {tot_profit:+.2f}u   ROI {tot_profit/tot_bets*100:+.2f}%")
        if tot_bets < 500:
            print(f"  NOTE: {tot_bets} bets is a small sample. At typical variance "
                  f"the standard error\n        on that ROI is several percentage "
                  f"points -- treat it as indicative, not proven.")
    elif verbose:
        print("No graded prediction files found.")
    return res


if __name__ == "__main__":
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    res = grade_all()
    if len(res):
        out = PERFORMANCE_DIR / "roi_by_date.csv"
        res.to_csv(out, index=False)
        print(f"\nSaved -> {out.relative_to(PERFORMANCE_DIR.parent)}")
