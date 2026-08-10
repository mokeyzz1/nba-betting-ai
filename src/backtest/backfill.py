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


if __name__ == "__main__":
    print("Backfilling recent seasons from the NBA API...")
    g = run()
    print("\nPer season:")
    for s, grp in g.groupby("season_label"):
        print(f"  {s}: {len(grp):>5,} games   home win rate {(grp.homeScore > grp.awayScore).mean():.1%}")
