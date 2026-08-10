"""
Roster strength: not just who is missing, but how good they were.

injury_features.py weights an absent player by his recent minutes, which is a
crude proxy for importance -- it cannot tell a star from a rotation filler who
happens to play the same twenty minutes. This module weights by production
instead, using per-player plus-minus and scoring rate from PlayerStatistics,
and builds two views the model has never had:

  available_value  the quality actually suiting up tonight
  missing_value    the quality that is not

The hypothesis being tested is that the market prices the *fact* of an
absence quickly (it moves lines within minutes of a scratch) but prices the
*magnitude* less precisely, especially for second-order cases: a team missing
its third and fourth best players rather than its first.

The prior is not strong. The injury result already showed this class of signal
is real but small, and the residual test showed the market has our other
information. What matters is the measured correlation against the market's
error, and whether it clears roughly 0.045 -- see ROADMAP.md.

Leakage discipline as everywhere else: .shift(1) before every rolling window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.injuries import load_cached

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"

WINDOW = 15          # games of history used to value a player
MIN_GAMES = 4


def _player_value() -> pd.DataFrame:
    """Per player-date: recent production, computed from prior games only."""
    cols = ["firstName", "lastName", "gameDate", "numMinutes", "points",
            "plusMinusPoints", "playerteamCity", "playerteamName"]
    p = pd.read_csv(DATA / "PlayerStatistics.csv", usecols=cols, low_memory=False)
    p["gameDate"] = pd.to_datetime(p["gameDate"], errors="coerce", format="mixed")

    recent = DATA / "player_stats_recent.csv"
    if recent.exists():
        r = pd.read_csv(recent)
        r["gameDate"] = pd.to_datetime(r["gameDate"], errors="coerce", format="mixed")
        # The recent backfill has no plus-minus; scoring rate carries it.
        for c in cols:
            if c not in r.columns:
                r[c] = np.nan
        p = pd.concat([p, r[cols]], ignore_index=True)

    p = p.dropna(subset=["gameDate"])
    p["player"] = (p["firstName"].astype(str).str.strip() + " " +
                   p["lastName"].astype(str).str.strip()).str.replace(r"\s+", " ", regex=True)
    p["team"] = (p["playerteamCity"].astype(str).str.strip() + " " +
                 p["playerteamName"].astype(str).str.strip()).str.strip()
    for c in ("numMinutes", "points", "plusMinusPoints"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["date"] = p["gameDate"].dt.normalize()
    p = p[p["date"].dt.year >= 2015].sort_values(["player", "date"])

    g = p.groupby("player", sort=False)
    def roll(col):
        return g[col].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())

    p["mpg"] = roll("numMinutes")
    p["ppg"] = roll("points")
    p["pm"] = roll("plusMinusPoints")

    # Value combines court time with impact while on it. Plus-minus is noisy
    # per game but stable enough over fifteen; scoring rate backs it up where
    # plus-minus is absent (the API backfill does not carry it).
    p["value"] = p["mpg"].fillna(0) * (
        p["pm"].fillna(0) / 36.0 + p["ppg"].fillna(0) / 36.0 * 0.5
    )
    return p[["player", "team", "date", "mpg", "value"]].dropna(subset=["mpg"])


def build() -> pd.DataFrame:
    """Per team-date: value available and value missing."""
    inj = load_cached()
    inj = inj[~inj["reason"].astype(str).str.contains("NOT YET SUBMITTED", case=False, na=False)]
    inj = inj.dropna(subset=["game_date"]).copy()
    inj["date"] = inj["game_date"].dt.normalize()
    inj["team"] = inj["team"].astype(str).str.strip()
    inj = inj.sort_values("date")

    pv = _player_value().sort_values("date")

    # As-of join for the same reason as injury_features: an absent player has
    # no box-score row that day, so an equality join drops exactly the players
    # we care about.
    merged = pd.merge_asof(
        inj, pv[["player", "date", "value", "mpg"]].rename(columns={"date": "vdate"}),
        left_on="date", right_on="vdate", by="player", direction="backward")
    stale = (merged["date"] - merged["vdate"]).dt.days > 45
    merged.loc[stale, ["value", "mpg"]] = np.nan
    merged[["value", "mpg"]] = merged[["value", "mpg"]].fillna(0.0)

    merged["missing_value"] = merged["weight"] * merged["value"]
    out_only = merged["status"].eq("out")
    merged["missing_value_out"] = np.where(out_only, merged["value"], 0.0)

    agg = (merged.groupby(["team", "date"], as_index=False)
           .agg(missing_value=("missing_value", "sum"),
                missing_value_out=("missing_value_out", "sum")))

    # Total roster value on that date, so "missing" can be expressed as a share.
    roster = (pv.groupby(["team", "date"], as_index=False)["value"].sum()
                .rename(columns={"value": "roster_value"}))
    roster = roster.sort_values("date")
    agg = agg.sort_values("date")
    agg = pd.merge_asof(agg, roster, on="date", by="team", direction="backward")
    agg["roster_value"] = agg["roster_value"].replace(0, np.nan)
    agg["missing_share"] = agg["missing_value"] / agg["roster_value"]
    agg["missing_share_out"] = agg["missing_value_out"] / agg["roster_value"]
    return agg.fillna({"missing_share": 0.0, "missing_share_out": 0.0,
                       "roster_value": 0.0})


ROSTER_FEATURES = ["missing_value_diff", "missing_share_diff", "roster_value_diff"]


def attach(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    r = build()
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    for side in ("home", "away"):
        b = r.rename(columns={"team": f"{side}_team",
                              "missing_value": f"{side}_missing_value",
                              "missing_value_out": f"{side}_missing_value_out",
                              "missing_share": f"{side}_missing_share",
                              "missing_share_out": f"{side}_missing_share_out",
                              "roster_value": f"{side}_roster_value"})
        g = g.merge(b, on=[f"{side}_team", "date"], how="left")
    for c in [c for c in g.columns if "missing_" in c or "roster_value" in c]:
        g[c] = g[c].fillna(0.0)

    # Positive means the away side is more depleted / weaker: good for home.
    g["missing_value_diff"] = g["away_missing_value"] - g["home_missing_value"]
    g["missing_share_diff"] = g["away_missing_share"] - g["home_missing_share"]
    g["roster_value_diff"] = g["home_roster_value"] - g["away_roster_value"]
    if verbose:
        cov = (g[["home_roster_value", "away_roster_value"]] != 0).all(axis=1).mean()
        print(f"  roster features attached, non-zero on {cov:.1%} of games")
    return g


if __name__ == "__main__":
    from src.backtest import dataset
    g = attach(dataset.build(require_odds=False, verbose=False))
    sub = g[(g.home_roster_value != 0) & (g.away_roster_value != 0)]
    print(f"\ngames with roster values: {len(sub):,}  "
          f"{sub.date.min().date()} -> {sub.date.max().date()}")
    for c in ROSTER_FEATURES:
        print(f"  {c:<24} corr w/ home_win {sub[c].corr(sub['home_win']):+.4f}")
