"""
Backfill recent seasons from the NBA API.

data/Games.csv stops at 2025-03-18. Everything after that -- the end of the
2024-25 season and the whole of 2025-26 -- is missing, which is why the two
seasons the project most wants to test on could not be tested at all.

Game results are free from nba_api. Historical closing odds are not: The Odds
API returns HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN for the /historical
endpoint. So these seasons can validate the model's accuracy and calibration,
but not its ROI, until prices for them are sourced separately.

Writes data/games_recent.csv in the same shape as Games.csv so the dataset
builder can concatenate the two without special-casing.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
OUT = DATA / "games_recent.csv"

SEASONS = ["2024-25", "2025-26"]


def _fetch_season(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder

    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable=season_type,
    ).get_data_frames()[0]
    df["season_label"] = season
    df["season_type"] = season_type
    return df


def _fold_to_games(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse two team-rows per game into one game-row.

    MATCHUP encodes the venue: 'LAL vs. BOS' is the home side, 'BOS @ LAL'
    the away side. Splitting on that is more reliable than guessing from
    team ordering.
    """
    df = team_rows.copy()
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)

    home = df[df["is_home"]].set_index("GAME_ID")
    away = df[~df["is_home"]].set_index("GAME_ID")
    common = home.index.intersection(away.index)
    home, away = home.loc[common], away.loc[common]

    out = pd.DataFrame({
        "gameId": common,
        "gameDate": pd.to_datetime(home["GAME_DATE"].values),
        "gameType": home["season_type"].values,
        "hometeamId": home["TEAM_ID"].values,
        "awayteamId": away["TEAM_ID"].values,
        "homeScore": home["PTS"].values,
        "awayScore": away["PTS"].values,
        "home_team_full": home["TEAM_NAME"].values,
        "away_team_full": away["TEAM_NAME"].values,
        "season_label": home["season_label"].values,
    })

    # Games.csv splits team names into city + nickname; mirror that so the two
    # sources concatenate cleanly. Nickname is the last token.
    for side in ("home", "away"):
        full = out[f"{side}_team_full"].astype(str)
        out[f"{side}teamName"] = full.str.rsplit(" ", n=1).str[-1]
        out[f"{side}teamCity"] = full.str.rsplit(" ", n=1).str[0]

    out = out.drop(columns=["home_team_full", "away_team_full"])
    return out.sort_values("gameDate").reset_index(drop=True)


def run(seasons: list[str] | None = None, include_playoffs: bool = True) -> pd.DataFrame:
    seasons = seasons or SEASONS
    frames = []
    for season in seasons:
        types = ["Regular Season"] + (["Playoffs"] if include_playoffs else [])
        for st in types:
            try:
                raw = _fetch_season(season, st)
                if len(raw):
                    frames.append(raw)
                    print(f"  {season} {st:<14}: {len(raw):>5,} team-rows")
                else:
                    print(f"  {season} {st:<14}: none")
            except Exception as e:
                print(f"  {season} {st:<14}: FAILED {type(e).__name__}: {str(e)[:80]}")
            time.sleep(1.5)  # be polite to stats.nba.com

    if not frames:
        raise RuntimeError("No data returned from the NBA API")

    games = _fold_to_games(pd.concat(frames, ignore_index=True))
    games.to_csv(OUT, index=False)
    print(f"\n  wrote {len(games):,} games -> {OUT.relative_to(BASE)}")
    print(f"  range: {games['gameDate'].min().date()} -> {games['gameDate'].max().date()}")
    return games


PLAYERS_OUT = DATA / "player_stats_recent.csv"


def run_players(seasons: list[str] | None = None) -> pd.DataFrame:
    """Player box scores for recent seasons.

    data/PlayerStatistics.csv ends 2025-03-18. Without this, injury features
    for 2025-26 have no minutes to weight by -- burden collapses to zero for
    74% of games and the feature becomes noise that dilutes a real signal.
    Written in PlayerStatistics.csv's column names so the two concatenate.
    """
    from nba_api.stats.endpoints import leaguegamelog

    seasons = seasons or SEASONS
    frames = []
    for season in seasons:
        for st in ("Regular Season", "Playoffs"):
            try:
                df = leaguegamelog.LeagueGameLog(
                    season=season, player_or_team_abbreviation="P",
                    season_type_all_star=st).get_data_frames()[0]
                if len(df):
                    df["season_type"] = st
                    frames.append(df)
                    print(f"  {season} {st:<14}: {len(df):>6,} player-games")
            except Exception as e:
                print(f"  {season} {st:<14}: FAILED {type(e).__name__}")
            time.sleep(2.0)

    if not frames:
        raise RuntimeError("No player data returned")

    df = pd.concat(frames, ignore_index=True)
    name = df["PLAYER_NAME"].astype(str).str.strip()
    out = pd.DataFrame({
        "firstName": name.str.split(" ", n=1).str[0],
        "lastName": name.str.split(" ", n=1).str[-1],
        "gameId": df["GAME_ID"],
        "gameDate": pd.to_datetime(df["GAME_DATE"]),
        "numMinutes": pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0),
        "points": pd.to_numeric(df["PTS"], errors="coerce").fillna(0.0),
    })
    team = df["TEAM_NAME"].astype(str).str.strip()
    out["playerteamCity"] = team.str.rsplit(" ", n=1).str[0]
    out["playerteamName"] = team.str.rsplit(" ", n=1).str[-1]
    # Preserve the exact full name; splitting on the first space mangles
    # multi-word surnames ("Shai Gilgeous-Alexander" is fine, "Nickeil
    # Alexander-Walker" is fine, but "Karl-Anthony Towns" needs the original).
    out["fullName"] = name

    out.to_csv(PLAYERS_OUT, index=False)
    print(f"\n  wrote {len(out):,} player-games -> {PLAYERS_OUT.relative_to(BASE)}")
    print(f"  range: {out['gameDate'].min().date()} -> {out['gameDate'].max().date()}")
    return out


if __name__ == "__main__":
    import sys
    if "--players" in sys.argv:
        print("Backfilling player box scores from the NBA API...")
        run_players()
    else:
        print("Backfilling recent seasons from the NBA API...")
        g = run()
        print("\nPer season:")
        for s, grp in g.groupby("season_label"):
            print(f"  {s}: {len(grp):>5,} games   home win rate {(grp.homeScore > grp.awayScore).mean():.1%}")
