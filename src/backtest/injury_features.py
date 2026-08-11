"""
Turn injury reports into features the model can use.

A raw count of listed players is close to worthless: a report listing eight
names is usually seven two-way call-ups and one rotation player, and a team
missing its starting point guard looks identical to one resting a G League
signing. What matters is how much of a team's actual production is missing.

So each listed player is weighted by the minutes they had been playing
BEFORE this game -- a rolling average, shifted by one game so the current
game never informs its own feature. Weighting by minutes also disposes of the
"G League - Two-Way" entries automatically: those players average near zero
minutes and therefore contribute near zero burden, which is more robust than
trying to pattern-match the reason text.

Status is graded rather than binary, since Questionable resolves to playing
more often than not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.injuries import load_cached

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"

TEAM_MINUTES_PER_GAME = 240.0      # 5 players x 48 minutes


def _player_minutes(min_season: int = 2021) -> pd.DataFrame:
    """Rolling pre-game minutes per player. Never includes the current game."""
    cols = ["firstName", "lastName", "gameId", "gameDate",
            "playerteamCity", "playerteamName", "numMinutes"]
    p = pd.read_csv(DATA / "PlayerStatistics.csv", usecols=cols, low_memory=False)
    p["gameDate"] = pd.to_datetime(p["gameDate"], errors="coerce", format="mixed")

    # PlayerStatistics.csv ends 2025-03-18; without this the 2025-26 season
    # has no minutes to weight injuries by and the feature becomes noise.
    recent = DATA / "player_stats_recent.csv"
    if recent.exists():
        r = pd.read_csv(recent)
        r["gameDate"] = pd.to_datetime(r["gameDate"], errors="coerce", format="mixed")
        keep = [c for c in cols if c in r.columns] + (["fullName"] if "fullName" in r.columns else [])
        p = pd.concat([p, r[keep]], ignore_index=True)

    p = p.dropna(subset=["gameDate"])
    p = p[p["gameDate"].dt.year >= min_season]

    built = (p["firstName"].astype(str).str.strip() + " " +
             p["lastName"].astype(str).str.strip())
    # fullName, where present, is the unsplit original -- safer for hyphenated
    # and multi-word names than reassembling first + last.
    p["player"] = (p["fullName"] if "fullName" in p.columns else built).fillna(built)
    p["player"] = p["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    p["team"] = (p["playerteamCity"].astype(str).str.strip() + " " +
                 p["playerteamName"].astype(str).str.strip()).str.strip()
    p["numMinutes"] = pd.to_numeric(p["numMinutes"], errors="coerce").fillna(0.0)

    p = p.sort_values(["player", "gameDate"])
    # shift(1) before rolling: strictly prior games only.
    p["mpg_recent"] = (
        p.groupby("player", sort=False)["numMinutes"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=2).mean())
    )
    p["date"] = p["gameDate"].dt.normalize()
    return p[["player", "team", "date", "mpg_recent"]].dropna(subset=["mpg_recent"])


def build_injury_burden() -> pd.DataFrame:
    """Per team-date: the share of recent minutes that is unavailable.

    Teams that had not filed by the 08:00 report appear with the reason
    "NOT YET SUBMITTED" and no player rows -- about 16% of team-days. Those
    must not be read as "nobody is hurt", which is what a naive zero would
    mean. They are flagged via inj_known=0 and imputed to the league-average
    burden, so a zero in this table genuinely means "reported, no absences".
    Raising the report time to 13:30 would only cut this to 12%, and would
    publish after afternoon tipoffs -- not worth the leak.
    """
    inj = load_cached()
    if inj.empty:
        raise RuntimeError("No cached injury reports. Run src.backtest.injuries first.")

    inj = inj.dropna(subset=["game_date"]).copy()
    inj["date"] = inj["game_date"].dt.normalize()
    inj["team"] = inj["team"].astype(str).str.strip()

    not_sub = inj["reason"].astype(str).str.contains("NOT YET SUBMITTED", case=False, na=False)
    unknown = (inj[not_sub][["team", "date"]].drop_duplicates().assign(inj_known=0))
    inj = inj[~not_sub]

    mins = _player_minutes()

    # As-of join, NOT an equality join on date. PlayerStatistics.csv only has
    # a row for a game the player actually appeared in -- so a player listed
    # as Out has no row that day, and matching on (player, date) fails for
    # exactly the players that matter. An 18% match rate was the tell.
    # Instead take each player's most recent minutes average strictly before
    # the report date, which is both correct and still pre-game.
    inj = inj.sort_values("date")
    mins = mins.sort_values("date")
    merged = pd.merge_asof(
        inj, mins[["player", "date", "mpg_recent"]].rename(columns={"date": "mins_date"}),
        left_on="date", right_on="mins_date", by="player",
        direction="backward", allow_exact_matches=True,
    )
    # Stale entries are retired or long-term-absent players; drop after 45 days.
    age = (merged["date"] - merged["mins_date"]).dt.days
    merged.loc[age > 45, "mpg_recent"] = np.nan

    match_rate = merged["mpg_recent"].notna().mean()
    merged["mpg_recent"] = merged["mpg_recent"].fillna(0.0)

    merged["burden"] = merged["weight"] * merged["mpg_recent"]
    merged["burden_out"] = np.where(merged["status"].eq("out"), merged["mpg_recent"], 0.0)

    agg = (merged.groupby(["team", "date"], as_index=False)
           .agg(inj_burden=("burden", "sum"),
                inj_burden_out=("burden_out", "sum"),
                inj_listed=("player", "count")))

    agg["inj_burden"] /= TEAM_MINUTES_PER_GAME
    agg["inj_burden_out"] /= TEAM_MINUTES_PER_GAME
    agg["inj_known"] = 1

    # Append the non-filing team-days, imputed to the league average.
    unknown = unknown.merge(agg[["team", "date"]].assign(_seen=1),
                            on=["team", "date"], how="left")
    unknown = unknown[unknown["_seen"].isna()].drop(columns=["_seen"])
    if len(unknown):
        unknown["inj_burden"] = agg["inj_burden"].mean()
        unknown["inj_burden_out"] = agg["inj_burden_out"].mean()
        unknown["inj_listed"] = agg["inj_listed"].median()
        agg = pd.concat([agg, unknown], ignore_index=True)

    agg.attrs["match_rate"] = match_rate
    agg.attrs["unknown_share"] = len(unknown) / max(len(agg), 1)
    return agg


INJURY_FEATURES = ["inj_burden_diff", "inj_burden_out_diff", "inj_listed_diff"]


def attach(games: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Join injury burden onto a game table and form home-minus-away diffs.

    Games outside injury coverage (before 2021-22) get 0.0, which is the
    neutral value -- it says "no known absences", the same as a clean report.
    """
    burden = build_injury_burden()
    if verbose:
        print(f"  injury player match rate: {burden.attrs['match_rate']:.1%}")

    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()

    for side in ("home", "away"):
        b = burden.rename(columns={
            "team": f"{side}_team",
            "inj_burden": f"{side}_inj_burden",
            "inj_burden_out": f"{side}_inj_burden_out",
            "inj_listed": f"{side}_inj_listed",
        })
        g = g.merge(b, on=[f"{side}_team", "date"], how="left")

    for c in ["home_inj_burden", "away_inj_burden", "home_inj_burden_out",
              "away_inj_burden_out", "home_inj_listed", "away_inj_listed"]:
        g[c] = g[c].fillna(0.0)

    # Positive means the AWAY side is more depleted, i.e. good for home.
    g["inj_burden_diff"] = g["away_inj_burden"] - g["home_inj_burden"]
    g["inj_burden_out_diff"] = g["away_inj_burden_out"] - g["home_inj_burden_out"]
    g["inj_listed_diff"] = g["away_inj_listed"] - g["home_inj_listed"]

    if verbose:
        covered = (g["home_inj_listed"] + g["away_inj_listed"]) > 0
        print(f"  games with injury data  : {covered.sum():,} / {len(g):,}")
    return g


def live_burden(day, verbose: bool = True) -> dict[str, dict[str, float]]:
    """Today's injury burden per team, for live prediction.

    Fetches the day's report rather than reading the historical cache, so the
    live path uses the same minutes-weighted definition as the backtest. If no
    report exists yet, returns {} and the caller must decide -- predicting with
    silent zeros is what this project spent the day removing.
    """
    import pandas as pd

    from src.backtest.injuries import STATUS_WEIGHT, fetch_day

    raw = fetch_day(pd.Timestamp(day).to_pydatetime())
    if raw is None or len(raw) == 0:
        if verbose:
            print(f"  no injury report published for {pd.Timestamp(day).date()}")
        return {}

    inj = raw.copy()
    inj.columns = [c.strip().lower().replace(" ", "_") for c in inj.columns]
    inj = inj[~inj["reason"].astype(str).str.contains("NOT YET SUBMITTED", case=False, na=False)]
    if inj.empty:
        return {}

    inj["status"] = inj["current_status"].astype(str).str.strip().str.lower()
    inj["weight"] = inj["status"].map(STATUS_WEIGHT).fillna(0.0)
    name = inj["player_name"].astype(str).str.strip()
    inj["player"] = name.str.split(",", n=1).apply(
        lambda p: f"{p[1].strip()} {p[0].strip()}" if len(p) == 2 else p[0]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    inj["team"] = inj["team"].astype(str).str.strip()

    mins = _player_minutes()
    latest = (mins.sort_values("date").groupby("player", as_index=False).last()
              [["player", "mpg_recent", "date"]])
    cutoff = pd.Timestamp(day) - pd.Timedelta(days=45)
    latest.loc[latest["date"] < cutoff, "mpg_recent"] = np.nan

    merged = inj.merge(latest[["player", "mpg_recent"]], on="player", how="left")
    merged["mpg_recent"] = merged["mpg_recent"].fillna(0.0)
    merged["burden"] = merged["weight"] * merged["mpg_recent"]
    merged["burden_out"] = np.where(merged["status"].eq("out"), merged["mpg_recent"], 0.0)

    agg = merged.groupby("team").agg(
        inj_burden=("burden", "sum"),
        inj_burden_out=("burden_out", "sum"),
        inj_listed=("player", "count"))
    agg["inj_burden"] /= TEAM_MINUTES_PER_GAME
    agg["inj_burden_out"] /= TEAM_MINUTES_PER_GAME

    if verbose:
        print(f"  injury report: {len(agg)} teams, {int(agg.inj_listed.sum())} players listed")
    return agg.to_dict("index")


if __name__ == "__main__":
    from src.backtest import dataset
    print("Building injury features...")
    games = dataset.build(require_odds=False, verbose=False)
    g = attach(games)
    cov = g[g["home_inj_listed"] + g["away_inj_listed"] > 0]
    print(f"\ncoverage span: {cov['date'].min().date()} -> {cov['date'].max().date()}")
    print(f"\n{cov[['inj_burden_diff','inj_burden_out_diff','inj_listed_diff']].describe().to_string()}")
    print(f"\ncorrelation with home_win (covered games only):")
    for c in INJURY_FEATURES:
        print(f"  {c:<24} {cov[c].corr(cov['home_win']):+.4f}")
