import os
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import ScoreboardV2
import time
from src.utils.config import PREDICTIONS_DIR

def normalize_team(name):
    return name.lower().replace("the ", "").split()[-1].strip()  # Use nickname

def fetch_actual_results(date: str, model_version: str):
    file_path = PREDICTIONS_DIR / f"predictions_{date}_{model_version}.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return

    print(f"📡 Fetching game results for {date}...")
    scoreboard = ScoreboardV2(game_date=date)
    time.sleep(1.5)  # Rate limit buffer

    games_header = scoreboard.game_header.get_data_frame()
    line_scores = scoreboard.line_score.get_data_frame()

    results = {}
    for game_id in games_header["GAME_ID"].unique():
        game_info = games_header[games_header["GAME_ID"] == game_id].iloc[0]
        game_lines = line_scores[line_scores["GAME_ID"] == game_id]

        if len(game_lines) != 2:
            continue

        home_row = game_lines[game_lines["TEAM_ID"] == game_info["HOME_TEAM_ID"]].iloc[0]
        away_row = game_lines[game_lines["TEAM_ID"] == game_info["VISITOR_TEAM_ID"]].iloc[0]

        home_team = normalize_team(home_row["TEAM_NAME"])
        away_team = normalize_team(away_row["TEAM_NAME"])
        home_score = home_row["PTS"]
        away_score = away_row["PTS"]

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        winner = "HOME" if home_score > away_score else "AWAY"
        results[(home_team, away_team)] = winner

    def match_result(row):
        home = normalize_team(row["hometeam"])
        away = normalize_team(row["awayteam"])
        return results.get((home, away), "UNKNOWN")

    df["actual_winner"] = df.apply(match_result, axis=1)
    df.to_csv(file_path, index=False)

    print(f"\n✅ Updated prediction file with actual winners: {file_path}")
    print("\n🎯 Game Results:")
    print(df[["hometeam", "awayteam", "prediction", "actual_winner"]].to_string(index=False))
