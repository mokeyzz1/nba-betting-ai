"""
Predict what the market will do, rather than who will win.

Every model in this project so far targets the game outcome. This one targets
the line movement between open and close. The reasoning: if a model reliably
secures a better number than the closing line, that converts to profit
independently of how any single game turns out -- which is how closing line
value works, and why professionals track it instead of short-run results.

It works, as a prediction problem:

    corr(predicted move, actual move) = +0.2338  (se 0.0105)   22 sigma

against +0.0792 for the outcome model's CLV. Three times the signal, in the
quantity that needed tripling.

It does NOT produce reliable profit. Betting the opening spread on this signal
returned +1.18% at a 0.5-point threshold, but it is profitable in only 4 of 9
seasons and the total is carried by 2019 alone (+15.17%). The threshold
pattern is also non-monotonic, which a real edge would not be. Treat that
number as noise; run per_season() below before believing any version of it.

What does survive is the reason this file exists:

    corr(predicted move, actual cover margin) = +0.0450 (se 0.0105)  4.3 sigma

The movement model predicts real outcomes at 4.3 sigma, where the outcome
model managed 1.5 sigma against the market residual. Asking "what will the
market do" extracted more outcome signal than asking "who wins". That is
about 0.6 points of edge against the ~1.5 needed to beat -110 -- still short,
but the strongest direction found so far.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest import dataset, opening_lines
from src.backtest.calibration import recency_weights
from src.backtest.injury_features import INJURY_FEATURES
from src.backtest.injury_features import attach as attach_injuries
from src.backtest.odds import settle

HALF_LIFE = 1460.0
ALPHA = 5.0


def features() -> list[str]:
    # The opening number is itself a feature: openers carry systematic biases
    # (over-reaction to recent form, name-brand teams) that get corrected.
    return list(dataset.FEATURES) + INJURY_FEATURES + ["home_open_spread"]


def build(verbose: bool = False) -> pd.DataFrame:
    g = attach_injuries(dataset.build(require_odds=False, verbose=verbose), verbose=verbose)
    g = opening_lines.attach(g, verbose=verbose)
    d = g.dropna(subset=["home_open_spread", "home_close_spread", "homeScore"]).copy()
    d["move"] = -(d["home_close_spread"] - d["home_open_spread"])   # + = toward home
    d["margin"] = d["homeScore"] - d["awayScore"]
    d["cover_margin"] = d["margin"] + d["home_open_spread"]
    d["covered"] = d["cover_margin"] > 0
    return d


def walk_forward(d: pd.DataFrame) -> pd.DataFrame:
    F = features()
    out = []
    for s in sorted(d["season"].unique()):
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < 1500 or len(te) < 150:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        m.fit(tr[F], tr["move"], ridge__sample_weight=recency_weights(tr["date"], HALF_LIFE))
        t = te.copy()
        t["pred_move"] = m.predict(te[F])
        out.append(t)
    return pd.concat(out, ignore_index=True)


def per_season(r: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Season-by-season ROI. Always read this before believing an aggregate.

    The aggregate looked positive; the breakdown showed one season carrying it.
    """
    push = r["cover_margin"] == 0
    rows = []
    for s in sorted(r["season"].unique()):
        x = r[(r["season"] == s) & (r["pred_move"].abs() >= threshold) & (~push)]
        if len(x) < 40:
            continue
        on_home = (x["pred_move"] > 0).to_numpy()
        won = np.where(on_home, x["covered"].to_numpy(), ~x["covered"].to_numpy())
        p = settle(np.full(len(x), -110.0), won)
        rows.append({"season": int(s), "bets": len(x), "win_rate": won.mean(),
                     "roi_pct": p.mean() * 100})
    return pd.DataFrame(rows)


def report() -> None:
    d = build(verbose=True)
    r = walk_forward(d)
    n = len(r)
    c_move = np.corrcoef(r["pred_move"], r["move"])[0, 1]
    c_out = np.corrcoef(r["pred_move"], r["cover_margin"])[0, 1]
    se = 1 / np.sqrt(n)

    print(f"\nwalk-forward games: {n:,}")
    print(f"  corr(predicted move, actual move)        {c_move:+.4f}  (se {se:.4f})")
    print(f"  corr(predicted move, actual cover margin){c_out:+.4f}  (se {se:.4f})")

    ps = per_season(r)
    print(f"\nper-season ROI betting the open at -110, 0.5-point threshold:")
    print(ps.to_string(index=False, formatters={"win_rate": "{:.1%}".format,
                                                "roi_pct": "{:+.2f}".format}))
    tot = (ps["roi_pct"] * ps["bets"]).sum() / ps["bets"].sum()
    print(f"\n  weighted total {tot:+.2f}%   profitable in "
          f"{(ps.roi_pct > 0).sum()}/{len(ps)} seasons")
    print("  A real edge shows up in most seasons. This does not.")


if __name__ == "__main__":
    report()
