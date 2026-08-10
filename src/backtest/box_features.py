"""
Features from team box scores: how a team has been playing, not just winning.

data/TeamStatistics.csv holds 143,172 team-games of full box scores and had
never been used. It matters because every feature the model had until now --
Elo, win percentage, point margin -- describes *results*. The market has those
too, and prices them (see the -0.0057 residual test in ROADMAP.md).

What box scores add is whether the results were deserved. Basketball outcomes
are noisy in specific, well-understood ways: three-point percentage swings
wildly game to game, and close games are close to coin flips. A team that has
won six of eight on 42% from three has a record that overstates it, and will
regress. If the market anchors on the record and underweights the manner of
it, that gap is exploitable.

Whether it actually is remains to be measured. The point is that this is
different information, not more of the same -- which the previous feature set
could not claim.

Everything is computed on prior games only: .shift(1) before every rolling
window, exactly as in dataset.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"

USE_COLS = [
    "gameId", "gameDate", "teamId", "opponentTeamId", "home", "win",
    "teamScore", "opponentScore",
    "fieldGoalsAttempted", "fieldGoalsMade",
    "threePointersAttempted", "threePointersMade",
    "freeThrowsAttempted", "freeThrowsMade",
    "reboundsOffensive", "reboundsDefensive",
    "turnovers", "assists",
]
# benchPoints and the points-breakdown columns (paint, fast break, second
# chance) are 98.5% null in this file -- present in only ~3% of even the
# 2010-2025 rows -- so they are not requested. The four factors are complete
# from 1997 onward, which sets the usable span.


def _load() -> pd.DataFrame:
    t = pd.read_csv(DATA / "TeamStatistics.csv", usecols=USE_COLS, low_memory=False)
    t["gameDate"] = pd.to_datetime(t["gameDate"], errors="coerce", format="mixed")
    t = t.dropna(subset=["gameDate", "teamId", "teamScore", "fieldGoalsAttempted"])
    t = t[t["fieldGoalsAttempted"] > 0]
    t["date"] = t["gameDate"].dt.normalize()
    return t.sort_values(["teamId", "date", "gameId"])


def _per_game_rates(t: pd.DataFrame) -> pd.DataFrame:
    """Four factors and efficiency, per game. Possessions via the standard estimate."""
    fga, fta = t["fieldGoalsAttempted"], t["freeThrowsAttempted"]
    oreb, tov = t["reboundsOffensive"], t["turnovers"]

    t["poss"] = fga - oreb + tov + 0.44 * fta
    t["poss"] = t["poss"].replace(0, np.nan)

    # Effective field goal % credits the extra point a three is worth.
    t["efg"] = (t["fieldGoalsMade"] + 0.5 * t["threePointersMade"]) / fga
    t["tov_rate"] = tov / t["poss"]
    t["ft_rate"] = fta / fga
    t["three_rate"] = t["threePointersAttempted"] / fga
    t["three_pct"] = np.where(t["threePointersAttempted"] > 0,
                              t["threePointersMade"] / t["threePointersAttempted"], np.nan)
    t["off_rtg"] = 100.0 * t["teamScore"] / t["poss"]
    t["def_rtg"] = 100.0 * t["opponentScore"] / t["poss"]
    t["margin"] = t["teamScore"] - t["opponentScore"]
    t["close_game"] = (t["margin"].abs() <= 5).astype(float)
    t["close_win"] = ((t["margin"] > 0) & (t["margin"].abs() <= 5)).astype(float)
    return t


# Rolling windows: 10 games for form, 30 for the baseline a team regresses to.
SHORT, LONG = 10, 30


def _rolling(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("teamId", sort=False)

    def roll(col, w, minp=5):
        return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())

    for c in ("efg", "tov_rate", "ft_rate", "three_rate", "off_rtg", "def_rtg",
              "poss"):
        t[f"{c}_{SHORT}"] = roll(c, SHORT)

    # Long baselines, used to judge how far short-term form has strayed.
    t[f"three_pct_{LONG}"] = roll("three_pct", LONG, 10)
    t[f"three_pct_{SHORT}"] = roll("three_pct", SHORT)
    t[f"efg_{LONG}"] = roll("efg", LONG, 10)
    t[f"margin_{SHORT}"] = roll("margin", SHORT)

    # --- luck terms -------------------------------------------------------
    # Three-point shooting is the noisiest major component of an NBA result.
    # Recent minus baseline is how much of current form is likely to evaporate.
    t["three_luck"] = t[f"three_pct_{SHORT}"] - t[f"three_pct_{LONG}"]
    t["efg_luck"] = t[f"efg_{SHORT}"] - t[f"efg_{LONG}"]

    # Close games are near coin flips, so a strong close-game record is mostly
    # variance and should be discounted rather than extrapolated.
    t["close_rate"] = roll("close_game", 20, 8)
    t["close_win_rate"] = g["close_win"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=8).sum()
    ) / g["close_game"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=8).sum()
    ).replace(0, np.nan)

    # Net rating built from efficiency rather than raw points: less exposed to
    # blowouts and garbage time than point margin is.
    t["net_rtg_10"] = t[f"off_rtg_{SHORT}"] - t[f"def_rtg_{SHORT}"]
    return t


FEATURE_BASES = [
    "efg_10", "tov_rate_10", "ft_rate_10", "three_rate_10",
    "off_rtg_10", "def_rtg_10", "poss_10",
    "net_rtg_10", "three_luck", "efg_luck", "close_win_rate",
]

BOX_FEATURES = [f"{b}_diff" for b in FEATURE_BASES]


def build() -> pd.DataFrame:
    """Per team-date table of rolling box-score form."""
    t = _rolling(_per_game_rates(_load()))
    keep = ["teamId", "date"] + FEATURE_BASES
    return t[keep].dropna(subset=["teamId", "date"])


def attach(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Join home/away box form onto a game table and form differentials."""
    box = build()
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()

    for side, idcol in (("home", "hometeamId"), ("away", "awayteamId")):
        b = box.rename(columns={"teamId": idcol,
                                **{c: f"{side}_{c}" for c in FEATURE_BASES}})
        g = g.merge(b, on=[idcol, "date"], how="left")

    for base in FEATURE_BASES:
        g[f"{base}_diff"] = g[f"home_{base}"] - g[f"away_{base}"]

    if verbose:
        cov = g[BOX_FEATURES].notna().all(axis=1).mean()
        print(f"  box-score features attached, complete on {cov:.1%} of games")
    return g


if __name__ == "__main__":
    from src.backtest import dataset
    g = attach(dataset.build(require_odds=False, verbose=False))
    sub = g.dropna(subset=BOX_FEATURES)
    print(f"\ngames with complete box features: {len(sub):,}")
    print(f"span: {sub.date.min().date()} -> {sub.date.max().date()}")
    print("\ncorrelation with home_win:")
    for c in BOX_FEATURES:
        print(f"  {c:<22} {sub[c].corr(sub['home_win']):+.4f}")
