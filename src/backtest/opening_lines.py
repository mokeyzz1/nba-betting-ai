"""
Opening and closing lines, for measuring closing line value.

Every profitability test in this project so far has been graded against a
single price per game -- effectively the closing number, the sharpest figure a
market produces. Nobody profitable targets that. The professional approach is
to beat the *opening* number and let the market move toward you; the resulting
metric, closing line value, reads a real edge in roughly 50 bets where ROI
needs on the order of 1,000.

Source: github.com/flancast90/sportsbookreview-scraper (MIT), which publishes
a pre-scraped SportsbookReview archive with opening and closing spreads and
totals plus closing moneyline. 13,903 games, seasons 2011-2021, every field
100% populated. This is strictly more than the $99 Odds Warehouse product,
which lacks moneyline; the only thing that would buy is 2022-2025 coverage.

Data quality note: raw mean spread movement reads +5.34 against a median
absolute movement of 1.0, which is a sign-convention or outlier problem rather
than a real drift. _clean_movement() below drops the implausible tail rather
than trusting it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
CACHE = DATA / "sbr_open_close.csv"

URL = ("https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper"
       "/main/data/nba_archive_10Y.json")

# SBR uses team nicknames; the box-score data uses "City Nickname".
NICK_TO_FULL = {
    "Hawks": "Atlanta Hawks", "Celtics": "Boston Celtics", "Nets": "Brooklyn Nets",
    "Hornets": "Charlotte Hornets", "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers", "Mavericks": "Dallas Mavericks",
    "Nuggets": "Denver Nuggets", "Pistons": "Detroit Pistons",
    "Warriors": "Golden State Warriors", "Rockets": "Houston Rockets",
    "Pacers": "Indiana Pacers", "Clippers": "Los Angeles Clippers",
    "Lakers": "Los Angeles Lakers", "Grizzlies": "Memphis Grizzlies",
    "Heat": "Miami Heat", "Bucks": "Milwaukee Bucks",
    "Timberwolves": "Minnesota Timberwolves", "Pelicans": "New Orleans Pelicans",
    "Knicks": "New York Knicks", "Thunder": "Oklahoma City Thunder",
    "Magic": "Orlando Magic", "76ers": "Philadelphia 76ers",
    "Sixers": "Philadelphia 76ers", "Suns": "Phoenix Suns",
    "Trailblazers": "Portland Trail Blazers", "Blazers": "Portland Trail Blazers",
    "Kings": "Sacramento Kings", "Spurs": "San Antonio Spurs",
    "Raptors": "Toronto Raptors", "Jazz": "Utah Jazz", "Wizards": "Washington Wizards",
}


def _clean_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records whose open/close pair is not credible.

    NBA spreads rarely move more than a few points between open and close.
    Anything beyond 10 is a data error, not a market event, and left in it
    dominates every average.
    """
    mv = df["home_close_spread"] - df["home_open_spread"]
    bad = mv.abs() > 10
    if bad.any():
        df = df[~bad].copy()
    return df


def download(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE, parse_dates=["date"])

    raw = pd.DataFrame(requests.get(URL, timeout=180).json())
    raw["date"] = pd.to_datetime(raw["date"].astype(int).astype(str),
                                 format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["date"])

    out = pd.DataFrame({
        "date": raw["date"].dt.normalize(),
        "home_team": raw["home_team"].map(NICK_TO_FULL),
        "away_team": raw["away_team"].map(NICK_TO_FULL),
        "homeScore": pd.to_numeric(raw["home_final"], errors="coerce"),
        "awayScore": pd.to_numeric(raw["away_final"], errors="coerce"),
        "home_open_spread": pd.to_numeric(raw["home_open_spread"], errors="coerce"),
        "home_close_spread": pd.to_numeric(raw["home_close_spread"], errors="coerce"),
        "open_total": pd.to_numeric(raw["open_over_under"], errors="coerce"),
        "close_total": pd.to_numeric(raw["close_over_under"], errors="coerce"),
        "home_close_ml": pd.to_numeric(raw["home_close_ml"], errors="coerce"),
        "away_close_ml": pd.to_numeric(raw["away_close_ml"], errors="coerce"),
    }).dropna(subset=["home_team", "away_team", "home_open_spread",
                      "home_close_spread", "homeScore"])

    out = _clean_movement(out)
    out["spread_move"] = out["home_close_spread"] - out["home_open_spread"]
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"])
    out.to_csv(CACHE, index=False)
    if verbose:
        print(f"  {len(out):,} games  {out.date.min().date()} -> {out.date.max().date()}")
    return out


def attach(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    o = download(verbose=verbose)
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    merged = g.merge(o.drop(columns=["homeScore", "awayScore"]),
                     on=["date", "home_team", "away_team"], how="left")
    if verbose:
        n = merged["home_open_spread"].notna().sum()
        print(f"  games with open+close lines: {n:,} / {len(merged):,}")
    return merged


if __name__ == "__main__":
    d = download(force=True)
    mv = d["spread_move"]
    print(f"\nafter cleaning: {len(d):,} games")
    print(f"spread movement open->close: mean {mv.mean():+.3f}  "
          f"median |move| {mv.abs().median():.1f}  sd {mv.std():.2f}")
    print(f"  moved at all : {(mv.abs() > 0).mean():.1%}")
    print(f"  moved 1+ pt  : {(mv.abs() >= 1).mean():.1%}")
    print(f"  moved 2+ pts : {(mv.abs() >= 2).mean():.1%}")
