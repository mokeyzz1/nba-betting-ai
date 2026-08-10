"""
Recent-season moneyline odds (2021-22 through 2024-25).

data/nba_historical_odds.csv stops after the 2022-23 season, which left the
seasons with injury coverage almost untestable for profit: the overlap was
two seasons, one of which had no injury data in its training window.

Source: github.com/csdurfee/scrape_yahoo_odds, which publishes BetMGM lines
scraped from Yahoo. Note it declares no license -- fine for private research,
but do not redistribute the raw files.

Beyond closing prices it carries two fields nothing else here has:
stake_percentage (share of money on a side) and wager_percentage (share of
tickets). When they diverge, a few large bets sit opposite many small ones,
which is the conventional sharp-versus-public signal.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
CACHE = DATA / "odds_recent.csv"

RAW = "https://raw.githubusercontent.com/csdurfee/scrape_yahoo_odds/main/yahoo_scrapes"
FOLDERS = (2021, 2022, 2023, 2024)

# Feed uses short names; the box-score data uses "City Nickname".
TEAM_MAP = {
    "Atlanta": "Atlanta Hawks", "Boston": "Boston Celtics", "Brooklyn": "Brooklyn Nets",
    "Charlotte": "Charlotte Hornets", "Chicago": "Chicago Bulls",
    "Cleveland": "Cleveland Cavaliers", "Dallas": "Dallas Mavericks",
    "Denver": "Denver Nuggets", "Detroit": "Detroit Pistons",
    "Golden State": "Golden State Warriors", "Houston": "Houston Rockets",
    "Indiana": "Indiana Pacers", "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers", "Memphis": "Memphis Grizzlies",
    "Miami": "Miami Heat", "Milwaukee": "Milwaukee Bucks",
    "Minnesota": "Minnesota Timberwolves", "New Orleans": "New Orleans Pelicans",
    "New York": "New York Knicks", "Oklahoma City": "Oklahoma City Thunder",
    "Orlando": "Orlando Magic", "Philadelphia": "Philadelphia 76ers",
    "Phoenix": "Phoenix Suns", "Portland": "Portland Trail Blazers",
    "Sacramento": "Sacramento Kings", "San Antonio": "San Antonio Spurs",
    "Toronto": "Toronto Raptors", "Utah": "Utah Jazz", "Washington": "Washington Wizards",
}


def download(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Fetch and normalise all seasons. Cached after the first call."""
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE, parse_dates=["date"])

    frames = []
    for folder in FOLDERS:
        url = f"{RAW}/{folder}/odds.csv"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d["src_folder"] = folder
        frames.append(d)
        if verbose:
            print(f"  fetched {folder}: {len(d):,} games")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["game_date"].str[:10], errors="coerce")
    df = df.dropna(subset=["date", "money_home_odds", "money_away_odds"])

    out = pd.DataFrame({
        "date": df["date"].dt.normalize(),
        "home_team": df["home_team"].map(TEAM_MAP).fillna(df["home_team"]),
        "away_team": df["away_team"].map(TEAM_MAP).fillna(df["away_team"]),
        "home_odds": pd.to_numeric(df["money_home_odds"], errors="coerce"),
        "away_odds": pd.to_numeric(df["money_away_odds"], errors="coerce"),
        "home_stake_pct": pd.to_numeric(df.get("money_home_stake_percentage"), errors="coerce"),
        "home_wager_pct": pd.to_numeric(df.get("money_home_wager_percentage"), errors="coerce"),
        "market_total": pd.to_numeric(df.get("total_over_points"), errors="coerce"),
        "market_spread": pd.to_numeric(df.get("spread_home_points"), errors="coerce"),
    }).dropna(subset=["home_odds", "away_odds"])

    out = out[(out["home_odds"] != 0) & (out["away_odds"] != 0)]
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"])
    out.to_csv(CACHE, index=False)
    if verbose:
        print(f"  normalised {len(out):,} games -> {CACHE.relative_to(BASE)}")
    return out


def attach(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fill missing odds on a game table from this source.

    Joined on (date, home team, away team). Existing prices from
    nba_historical_odds.csv win, so the older seasons keep the source the
    backtest was originally validated against.
    """
    o = download(verbose=verbose)
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()

    merged = g.merge(o.rename(columns={
        "home_odds": "home_odds_new", "away_odds": "away_odds_new",
        "market_total": "total_new", "market_spread": "spread_new"}),
        on=["date", "home_team", "away_team"], how="left")

    before = merged["home_odds"].notna().sum() if "home_odds" in merged else 0
    for col, new in (("home_odds", "home_odds_new"), ("away_odds", "away_odds_new")):
        if col in merged:
            merged[col] = merged[col].fillna(merged[new])
        else:
            merged[col] = merged[new]
    after = merged["home_odds"].notna().sum()

    if verbose:
        print(f"  games with odds: {before:,} -> {after:,} (+{after - before:,})")
    return merged.drop(columns=[c for c in
                                ("home_odds_new", "away_odds_new", "total_new", "spread_new")
                                if c in merged])


if __name__ == "__main__":
    df = download(force=True)
    print(f"\n{len(df):,} games   {df.date.min().date()} -> {df.date.max().date()}")
    print(f"teams mapped: {df.home_team.nunique()}")
    unmapped = set(df.home_team) - set(TEAM_MAP.values())
    print(f"unmapped names: {unmapped or 'none'}")
