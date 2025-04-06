import pandas as pd
import time
from nba_api.stats.endpoints import TeamDashboardByGeneralSplits, LeagueDashTeamStats

SEASON = "2024-25"
team_ids = [
    1610612738,  # Boston Celtics
    1610612747,  # Los Angeles Lakers
]

team_features = []

# Get full advanced team stats once
stats_df = LeagueDashTeamStats(season=SEASON, measure_type_detailed="Advanced").get_data_frames()[0]

print(f"Available TEAM_IDs in API response: {stats_df['TEAM_ID'].tolist()}")

for team_id in team_ids:
    try:
        stats_row = stats_df[stats_df["TEAM_ID"] == team_id]
        if stats_row.empty:
            raise ValueError(f"Team ID {team_id} not found in API response.")

        stats_row = stats_row.iloc[0]

        # Extract team-level stats
        season_wins = stats_row["W"]
        pace = stats_row["PACE"]
        off_rating = stats_row["OFF_RATING"]
        def_rating = stats_row["DEF_RATING"]
        efg_pct = stats_row["EFG_PCT"]

        # Recent stats
        recent_df = TeamDashboardByGeneralSplits(team_id=team_id, season=SEASON).get_data_frames()[1]
        last_5 = recent_df[recent_df["GROUP_VALUE"] == "Last 5 Games"]
        if last_5.empty:
            raise ValueError("Recent stats not found.")

        last_5 = last_5.iloc[0]
        recent_wins = last_5["W_PCT"]
        recent_points = last_5["PTS"]

        # Still dummy values for player-level stats
        team_features.append({
            "TEAM_ID": team_id,
            "season_wins": season_wins,
            "recent_wins": recent_wins,
            "recent_points": recent_points,
            "pointsFastBreak": 15.0,
            "pointsInThePaint": 46.0,
            "PACE": pace,
            "OFF_RATING": off_rating,
            "DEF_RATING": def_rating,
            "EFG_PCT": efg_pct
        })

        time.sleep(1.2)
    except Exception as e:
        team_features.append({"TEAM_ID": team_id, "error": str(e)})

df = pd.DataFrame(team_features)
print(df)
