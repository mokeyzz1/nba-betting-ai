# verify_odds_merge.py

import pandas as pd

# Load files
games = pd.read_csv("data/your_final_games_data.csv")
odds = pd.read_csv("data/nba_historical_odds.csv")

# Format the dates to match
games["gameDate"] = pd.to_datetime(games["gameDate"]).dt.date
odds["date"] = pd.to_datetime(odds["date"]).dt.date

# Create team and opponent combos for fuzzy matching
games["home_clean"] = games["TEAM_NAME_home"].str.lower().str.replace(" ", "")
games["away_clean"] = games["TEAM_NAME_away"].str.lower().str.replace(" ", "")
odds["team_clean"] = odds["team"].str.lower().str.replace(" ", "")
odds["opp_clean"] = odds["opponent"].str.lower().str.replace(" ", "")

# Match based on date + teams
merged = pd.merge(
    games,
    odds,
    how="inner",
    left_on=["gameDate", "home_clean", "away_clean"],
    right_on=["date", "team_clean", "opp_clean"],
)

print(f"✅ Matched {len(merged)} games with betting odds.")
print("\n🎯 Sample matches:")
print(merged[["gameDate", "TEAM_NAME_home", "TEAM_NAME_away", "moneyLine", "opponentMoneyLine"]].head())

# Save matched file
merged.to_csv("data/verified_matched_odds.csv", index=False)
