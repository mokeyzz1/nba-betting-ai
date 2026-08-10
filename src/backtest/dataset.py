"""
Builds a game-level dataset where every feature is knowable before tipoff.

The rule this module exists to enforce: a feature for a game played on date D
may only use information from games played strictly before D. That rule was
violated in the previous pipeline by src/features/merge_advanced_stats.py,
which joined team advanced stats on (team, season) -- attaching end-of-season
ratings to games played in November, so each game's own result was baked into
its own features. That is the main reason 68% validation accuracy never
survived contact with live games.

Everything here is derived sequentially from raw box scores. No season
aggregates, no np.random, no placeholder constants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_EDGE = 100.0        # ~65% home win rate historically
ELO_SEASON_CARRY = 0.75      # regress toward mean between seasons


def _load_games() -> pd.DataFrame:
    """One row per completed game, with real final scores."""
    g = pd.read_csv(
        DATA / "Games.csv",
        usecols=[
            "gameId", "gameDate", "gameType",
            "hometeamCity", "hometeamName", "hometeamId",
            "awayteamCity", "awayteamName", "awayteamId",
            "homeScore", "awayScore",
        ],
        low_memory=False,
    )
    g["gameDate"] = pd.to_datetime(g["gameDate"], errors="coerce")
    g = g.dropna(subset=["gameDate", "homeScore", "awayScore", "hometeamId", "awayteamId"])
    g = g[(g["homeScore"] > 0) & (g["awayScore"] > 0)]

    # Keep competitive basketball only; preseason and exhibitions distort form.
    keep = g["gameType"].astype(str).str.contains("Regular Season|Playoff", case=False, na=False)
    g = g[keep]

    g["date"] = g["gameDate"].dt.normalize()
    g["home_team"] = (g["hometeamCity"].astype(str) + " " + g["hometeamName"].astype(str)).str.strip()
    g["away_team"] = (g["awayteamCity"].astype(str) + " " + g["awayteamName"].astype(str)).str.strip()
    g["home_win"] = (g["homeScore"] > g["awayScore"]).astype(int)
    g["season"] = np.where(g["date"].dt.month >= 10, g["date"].dt.year + 1, g["date"].dt.year)

    g = g.sort_values(["date", "gameId"]).reset_index(drop=True)
    return g


def _add_elo(g: pd.DataFrame) -> pd.DataFrame:
    """Sequential Elo. Ratings are recorded BEFORE each game is played."""
    elo: dict[int, float] = {}
    last_season: dict[int, int] = {}
    home_elo = np.empty(len(g))
    away_elo = np.empty(len(g))

    for i, (h, a, hw, season) in enumerate(
        zip(g["hometeamId"].values, g["awayteamId"].values,
            g["home_win"].values, g["season"].values)
    ):
        for t in (h, a):
            if t not in elo:
                elo[t] = ELO_START
                last_season[t] = season
            elif last_season[t] != season:
                elo[t] = ELO_START + ELO_SEASON_CARRY * (elo[t] - ELO_START)
                last_season[t] = season

        rh, ra = elo[h], elo[a]
        home_elo[i], away_elo[i] = rh, ra          # pre-game snapshot

        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ELO_HOME_EDGE) - ra) / 400.0))
        delta = ELO_K * (hw - exp_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    g["home_elo"] = home_elo
    g["away_elo"] = away_elo
    return g


def _team_rolling(g: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """Rolling form per team, computed from prior games only.

    Built on a long (one row per team-game) view, then folded back. The
    .shift(1) before every .rolling() is what makes these pre-game: without
    it, a team's last-10 average would include the game being predicted.
    """
    home = pd.DataFrame({
        "gameId": g["gameId"], "date": g["date"], "team": g["hometeamId"],
        "pts": g["homeScore"], "opp_pts": g["awayScore"], "won": g["home_win"],
        "side": "home",
    })
    away = pd.DataFrame({
        "gameId": g["gameId"], "date": g["date"], "team": g["awayteamId"],
        "pts": g["awayScore"], "opp_pts": g["homeScore"], "won": 1 - g["home_win"],
        "side": "away",
    })
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "gameId"])

    grp = long.groupby("team", sort=False)
    long["rest_days"] = grp["date"].diff().dt.days
    long["margin"] = long["pts"] - long["opp_pts"]

    for w in windows:
        long[f"win_pct_{w}"] = grp["won"].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
        long[f"margin_{w}"] = grp["margin"].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
        long[f"pts_{w}"] = grp["pts"].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
        long[f"opp_pts_{w}"] = grp["opp_pts"].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())

    # Season-to-date record, prior games only.
    long["season"] = np.where(long["date"].dt.month >= 10,
                              long["date"].dt.year + 1, long["date"].dt.year)
    long["season_win_pct"] = (
        long.groupby(["team", "season"], sort=False)["won"]
        .transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    )

    long["rest_days"] = long["rest_days"].clip(upper=10)
    long["b2b"] = (long["rest_days"] <= 1).astype(float)

    feat_cols = ["rest_days", "b2b", "season_win_pct"] + [
        c for w in windows for c in
        (f"win_pct_{w}", f"margin_{w}", f"pts_{w}", f"opp_pts_{w}")
    ]

    h = long[long["side"] == "home"].set_index("gameId")[feat_cols].add_prefix("home_")
    a = long[long["side"] == "away"].set_index("gameId")[feat_cols].add_prefix("away_")
    return g.set_index("gameId").join(h).join(a).reset_index()


def _attach_odds(g: pd.DataFrame) -> pd.DataFrame:
    """Join real historical moneyline prices.

    Joined on (date, home score, away score) rather than team names. The two
    sources spell teams differently ("LA Lakers" vs "Los Angeles Lakers"), and
    a date-plus-final-score key is effectively unique and needs no mapping.
    """
    o = pd.read_csv(DATA / "nba_historical_odds.csv")
    o["date"] = pd.to_datetime(o["date"], errors="coerce").dt.normalize()
    o = o[o["home/visitor"].astype(str).str.strip().str.lower() == "vs"]  # home perspective
    o = o.dropna(subset=["date", "score", "opponentScore", "moneyLine", "opponentMoneyLine"])

    o = o.rename(columns={
        "score": "homeScore", "opponentScore": "awayScore",
        "moneyLine": "home_odds", "opponentMoneyLine": "away_odds",
        "total": "market_total", "spread": "market_spread",
    })
    o = o[["date", "homeScore", "awayScore", "home_odds", "away_odds",
           "market_total", "market_spread"]]

    o["homeScore"] = o["homeScore"].astype(int)
    o["awayScore"] = o["awayScore"].astype(int)
    o = o.drop_duplicates(subset=["date", "homeScore", "awayScore"])

    g["homeScore"] = g["homeScore"].astype(int)
    g["awayScore"] = g["awayScore"].astype(int)
    merged = g.merge(o, on=["date", "homeScore", "awayScore"], how="left")

    # An odds row of 0 is a missing quote, not an even-money price.
    bad = (merged["home_odds"] == 0) | (merged["away_odds"] == 0)
    merged.loc[bad, ["home_odds", "away_odds"]] = np.nan
    return merged


FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "win_pct_5_diff", "win_pct_10_diff",
    "margin_5_diff", "margin_10_diff",
    "off_form_diff", "def_form_diff",
    "season_win_pct_diff",
    "rest_diff", "b2b_diff",
]


def _add_differentials(g: pd.DataFrame) -> pd.DataFrame:
    g["elo_diff"] = g["home_elo"] - g["away_elo"]
    g["win_pct_5_diff"] = g["home_win_pct_5"] - g["away_win_pct_5"]
    g["win_pct_10_diff"] = g["home_win_pct_10"] - g["away_win_pct_10"]
    g["margin_5_diff"] = g["home_margin_5"] - g["away_margin_5"]
    g["margin_10_diff"] = g["home_margin_10"] - g["away_margin_10"]
    # Home offence against away defence, and vice versa.
    g["off_form_diff"] = g["home_pts_10"] - g["away_opp_pts_10"]
    g["def_form_diff"] = g["away_pts_10"] - g["home_opp_pts_10"]
    g["season_win_pct_diff"] = g["home_season_win_pct"] - g["away_season_win_pct"]
    g["rest_diff"] = g["home_rest_days"] - g["away_rest_days"]
    g["b2b_diff"] = g["home_b2b"] - g["away_b2b"]
    return g


def build(require_odds: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Build the modelling table. Set require_odds=False to keep unpriced games."""
    g = _load_games()
    if verbose:
        print(f"  games loaded          : {len(g):,}  ({g['date'].min().date()} -> {g['date'].max().date()})")

    g = _add_elo(g)
    g = _team_rolling(g)
    g = _attach_odds(g)
    g = _add_differentials(g)

    before = len(g)
    g = g.dropna(subset=FEATURES)
    if verbose:
        print(f"  with complete features: {len(g):,}  (dropped {before - len(g):,} early-season games)")

    if require_odds:
        before = len(g)
        g = g.dropna(subset=["home_odds", "away_odds"])
        if verbose:
            print(f"  with real odds        : {len(g):,}  (dropped {before - len(g):,} unpriced)")

    g = g.sort_values("date").reset_index(drop=True)
    assert_no_leakage(g, verbose=verbose)
    return g


def assert_no_leakage(g: pd.DataFrame, verbose: bool = True) -> None:
    """Fail loudly if a feature looks fabricated or post-hoc.

    This is the check the old build_enhanced_* scripts never had. Those
    generated their own market odds and, for totals and spreads, their own
    target -- so their backtests measured nothing.
    """
    problems = []

    for col in FEATURES:
        s = g[col]
        if s.nunique(dropna=True) <= 1:
            problems.append(f"{col}: constant -- likely a frozen placeholder")
        if not np.isfinite(s.to_numpy(dtype=float)).all():
            problems.append(f"{col}: contains inf/nan")

    # Any feature that correlates near-perfectly with the outcome is a leak.
    for col in FEATURES:
        r = abs(np.corrcoef(g[col].to_numpy(float), g["home_win"].to_numpy(float))[0, 1])
        if r > 0.60:
            problems.append(f"{col}: |corr| with target = {r:.3f} -- suspiciously high")

    banned = [c for c in g.columns if c in
              ("homeScore", "awayScore", "winner", "margin", "pointDifference")]
    for col in banned:
        if col in FEATURES:
            problems.append(f"{col}: post-game quantity used as a feature")

    if problems:
        raise AssertionError("Leakage check failed:\n  - " + "\n  - ".join(problems))
    if verbose:
        print(f"  leakage check         : passed ({len(FEATURES)} features)")


if __name__ == "__main__":
    print("Building honest dataset...")
    df = build()
    print(f"\nRows: {len(df):,}   Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"Home win rate: {df['home_win'].mean():.1%}")
