"""
Player points projection, and a live comparison against real book lines.

WHY PROPS AT ALL. Moneyline is closed -- see ROADMAP.md; nine approaches, and
the decisive blend test showed the model adds 0.00004 nats to the market
price. The structural argument for props is different and stronger: a book
sets one moneyline per game with its best model and 200+ props a night with
far less attention each. That asymmetry is where inefficiency has to live if
it lives anywhere.

WHAT CANNOT BE DONE. There is no free archive of historical prop lines. We
have what players scored (PlayerStatistics.csv, 250k player-games) but not
what was offered on them. So props cannot be backtested the way moneyline
was, and any "backtest" against a self-invented line is measuring our guess
against our other guess.

TWO ARTIFACTS THIS MODULE EXISTS TO AVOID. Both were hit while exploring:

  1. Points are right-skewed. A rolling MEAN sits above the typical game, so
     a mean-based line has a ~45% over-rate and always-UNDER "wins" 54.7%.
     Any projection built on a mean must be calibrated to the median before
     it is compared to a book line, or the apparent edge is pure skew.

  2. A naive line that ignores minutes is trivially beaten by a projection
     that uses them. Real books project minutes. Beating a minutes-blind
     line proves nothing.

WHAT SURVIVED BOTH, on NBA data against a calibrated minutes-aware line:
57.4% at a 1-point edge over 68,798 player-games. The mechanism looks like
minutes mean-reversion -- a player whose last three games ran below his
ten-game average tends to revert up. Unverified against real prices.

This module tests the same idea against real WNBA lines, because the WNBA is
in season and NBA is not. The sport differs; the mechanism is meant to be
general.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
LOG = DATA / "prop_predictions.csv"
load_dotenv(BASE / ".env")
API_KEY = os.getenv("ODDS_API_KEY")

SHORT_W, LONG_W = 3, 10
MIN_MINUTES = 10.0        # below this a player is not reliably in the rotation


def fetch_player_logs(season: str, league_id: str = "10") -> pd.DataFrame:
    """Player game logs. league_id 10 = WNBA, 00 = NBA."""
    from nba_api.stats.endpoints import leaguegamelog

    df = leaguegamelog.LeagueGameLog(
        season=season, league_id=league_id,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season").get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["PLAYER_NAME", "GAME_DATE"])


def build_projections(logs: pd.DataFrame, stat: str = "PTS") -> pd.DataFrame:
    """Per player: a calibrated projection of the next game's stat.

    Every window is .shift(1)-ed so a game never informs its own projection.
    The projection is scoring RATE times projected MINUTES, where projected
    minutes blends the 3-game and 10-game averages -- the blend is what
    encodes mean-reversion, since a short-run dip gets pulled back toward the
    longer-run level.

    The calibration offset is the median of (actual - raw), which converts a
    mean-like estimate into a median-like one. Without it the projection sits
    systematically above any book line and every disagreement points the same
    way, which reads as edge and is only skew.
    """
    d = logs.copy()
    g = d.groupby("PLAYER_NAME", sort=False)

    d["min_short"] = g["MIN"].transform(
        lambda s: s.shift(1).rolling(SHORT_W, min_periods=2).mean())
    d["min_long"] = g["MIN"].transform(
        lambda s: s.shift(1).rolling(LONG_W, min_periods=4).mean())
    d["rate"] = g.apply(
        lambda x: (x[stat].shift(1).rolling(LONG_W, min_periods=4).sum()
                   / x["MIN"].shift(1).rolling(LONG_W, min_periods=4).sum())
    ).reset_index(level=0, drop=True)

    d = d.dropna(subset=["rate", "min_short", "min_long"])
    d["proj_min"] = 0.5 * d["min_short"] + 0.5 * d["min_long"]
    d["raw"] = d["rate"] * d["proj_min"]

    offset = float(np.median(d[stat] - d["raw"]))
    d["proj"] = d["raw"] + offset
    d.attrs["offset"] = offset
    return d


def latest_per_player(proj: pd.DataFrame) -> pd.DataFrame:
    last = proj.groupby("PLAYER_NAME").last().reset_index()
    return last[last["min_long"] >= MIN_MINUTES]


def fetch_prop_lines(sport: str = "basketball_wnba",
                     market: str = "player_points") -> pd.DataFrame:
    """Consensus line per player across books, for games not yet started."""
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    root = f"https://api.the-odds-api.com/v4/sports/{sport}"
    events = requests.get(f"{root}/events", params={"apiKey": API_KEY},
                          timeout=30).json()
    rows = []
    for e in events:
        r = requests.get(f"{root}/events/{e['id']}/odds",
                         params={"apiKey": API_KEY, "regions": "us",
                                 "markets": market, "oddsFormat": "american"},
                         timeout=30)
        if r.status_code != 200:
            continue
        for b in r.json().get("bookmakers", []):
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    if o.get("point") is None:
                        continue
                    rows.append({"player": o.get("description"),
                                 "side": o["name"], "line": o["point"],
                                 "odds": o["price"], "book": b["key"],
                                 "event": f"{e['away_team']} @ {e['home_team']}",
                                 "commence": e["commence_time"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    over = df[df["side"] == "Over"]
    return (over.groupby(["player", "event", "commence"], as_index=False)
            .agg(line=("line", "median"), books=("book", "nunique"),
                 best_over=("odds", "max")))


def compare(season: str = "2026", sport: str = "basketball_wnba",
            league_id: str = "10", verbose: bool = True) -> pd.DataFrame:
    logs = fetch_player_logs(season, league_id)
    proj = build_projections(logs)
    last = latest_per_player(proj)
    lines = fetch_prop_lines(sport)
    if lines.empty:
        if verbose:
            print("No prop lines available right now.")
        return lines

    m = lines.merge(last[["PLAYER_NAME", "proj", "min_short", "min_long", "rate"]],
                    left_on="player", right_on="PLAYER_NAME", how="inner")
    m["diff"] = m["proj"] - m["line"]
    m["side"] = np.where(m["diff"] > 0, "OVER", "UNDER")
    m["logged_at"] = datetime.now().isoformat(timespec="seconds")

    if verbose:
        print(f"calibration offset: {proj.attrs['offset']:+.2f} pts")
        print(f"matched {len(m)} players with both a line and a projection")
        print(f"  corr(projection, line) = {m['proj'].corr(m['line']):.4f}")
        print(f"  mean diff {m['diff'].mean():+.2f}   MAE {m['diff'].abs().mean():.2f}")
        print(f"  OVER {int((m['diff']>0).sum())}   UNDER {int((m['diff']<0).sum())}"
              "   (balanced means the skew is calibrated out)")
    return m


def log_predictions(m: pd.DataFrame) -> None:
    """Append to the prediction log. This is the whole point.

    Without real historical lines, the only honest way to test props is to
    write down what we think BEFORE the game and grade it after. One night is
    59 rows and proves nothing; a season is a real answer.
    """
    if m.empty:
        return
    cols = ["logged_at", "commence", "event", "player", "line", "books",
            "best_over", "proj", "diff", "side"]
    out = m[cols]
    header = not LOG.exists()
    out.to_csv(LOG, mode="a", header=header, index=False)
    print(f"logged {len(out)} predictions -> {LOG.relative_to(BASE)}")


if __name__ == "__main__":
    m = compare()
    if not m.empty:
        log_predictions(m)
        print("\nlargest disagreements:")
        top = m.reindex(m["diff"].abs().sort_values(ascending=False).index).head(10)
        print(top[["player", "line", "proj", "diff", "side", "books"]].to_string(
            index=False, formatters={"proj": "{:.1f}".format, "diff": "{:+.1f}".format}))
