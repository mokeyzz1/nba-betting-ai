import pandas as pd
from nba_api.stats.endpoints import TeamGameLog

def get_recent_stats(team_id, num_games=5, season="2024-25"):
    gamelog = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
    recent_games = gamelog.head(num_games)
    win_pct = (recent_games["WL"] == "W").sum() / num_games
    avg_pts = recent_games["PTS"].mean()
    return round(win_pct, 2), round(avg_pts, 1)

# Example teams
miami_id = 1610612748
hornets_id = 1610612766

miami_recent_wins, miami_recent_points = get_recent_stats(miami_id)
hornets_recent_wins, hornets_recent_points = get_recent_stats(hornets_id)

df = pd.DataFrame({
    "Team": ["Miami Heat", "Charlotte Hornets"],
    "Recent Win %": [miami_recent_wins, hornets_recent_wins],
    "Avg Points (last 5 games)": [miami_recent_points, hornets_recent_points]
})

print(df)
