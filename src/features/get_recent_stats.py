from nba_api.stats.endpoints import teamgamelog
import pandas as pd

def get_recent_win_pct(team_name_clean, num_games=5):
    """
    Returns win % over last `num_games` for a team.
    """
    try:
        gamelog = teamgamelog.TeamGameLog(team_name=team_name_clean, season="2024")
        df = gamelog.get_data_frames()[0].head(num_games)
        wins = df["WL"].str.upper().value_counts().get("W", 0)
        return wins / num_games
    except:
        return 0.5  # fallback

def get_recent_avg_pts(team_name_clean, num_games=5):
    """
    Returns average points over last `num_games` for a team.
    """
    try:
        gamelog = teamgamelog.TeamGameLog(team_name=team_name_clean, season="2024")
        df = gamelog.get_data_frames()[0].head(num_games)
        return df["PTS"].mean()
    except:
        return 110  # fallback average
