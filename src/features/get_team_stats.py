import pandas as pd

# Load team advanced stats from CSV
team_stats_df = pd.read_csv("data/team_advanced_stats.csv")
team_stats_df["team_clean"] = team_stats_df["TEAM_NAME"].str.lower().str.strip()

# Build lookup
team_stats_dict = {
    row["team_clean"]: row for _, row in team_stats_df.iterrows()
}

def get_team_stats(team_name_clean):
    """
    Return advanced stats (OFF_RATING, DEF_RATING, PACE, EFG_PCT) for a cleaned team name.
    """
    row = team_stats_dict.get(team_name_clean)
    if row is None:
        return None

    return {
        "OFF_RATING": row.get("OFF_RATING"),
        "DEF_RATING": row.get("DEF_RATING"),
        "PACE": row.get("PACE"),
        "EFG_PCT": row.get("EFG_PCT")
    }
