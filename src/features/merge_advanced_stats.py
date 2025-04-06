import pandas as pd
from utils.team_name_mapping import TEAM_NAME_MAPPING

# Load base games file
games = pd.read_csv("data/merged_games_with_odds.csv", low_memory=False)

# Load advanced stats
adv = pd.read_csv("team_advanced_stats.csv")

# 🧠 Map home/away short names to official full names
games["home_team_mapped"] = games["home_clean"].str.lower().map(TEAM_NAME_MAPPING)
games["away_team_mapped"] = games["away_clean"].str.lower().map(TEAM_NAME_MAPPING)

# Rename columns in advanced stats to prepare for merging
home_stats = adv.rename(columns=lambda x: f"home_{x}" if x not in ['TEAM_NAME', 'SEASON'] else x)
away_stats = adv.rename(columns=lambda x: f"away_{x}" if x not in ['TEAM_NAME', 'SEASON'] else x)

# Merge home team stats
games = games.merge(
    home_stats,
    left_on=['home_team_mapped', 'season'],
    right_on=['TEAM_NAME', 'SEASON'],
    how='left'
)

# Merge away team stats
games = games.merge(
    away_stats,
    left_on=['away_team_mapped', 'season'],
    right_on=['TEAM_NAME', 'SEASON'],
    how='left'
)

# Drop merge keys we no longer need
games.drop(columns=["TEAM_NAME_x", "TEAM_NAME_y", "SEASON_x", "SEASON_y"], inplace=True)

# Save to new file
games.to_csv("data/final_games_with_advanced_stats.csv", index=False)
print("✅ Merged advanced stats. Saved to data/final_games_with_advanced_stats.csv")
