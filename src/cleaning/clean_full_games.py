import pandas as pd

# Load full dataset
df = pd.read_csv("data/games.csv")

print(f"📦 Original rows: {len(df)}")

# Filter: Keep only regular season
df = df[df["gameType"].str.lower() == "regular"]
print(f"✅ Regular season rows: {len(df)}")

# Drop games with missing critical info
df = df.dropna(subset=["gameDate", "hometeamName", "awayteamName", "homeScore", "awayScore"])

# Optional: Clean team names for later odds matching
df["home_team"] = df["hometeamName"].str.strip().str.lower()
df["away_team"] = df["awayteamName"].str.strip().str.lower()

# Save cleaned file
df.to_csv("data/games_cleaned_full.csv", index=False)
print("✅ Saved cleaned full dataset to: data/games_cleaned_full.csv")

