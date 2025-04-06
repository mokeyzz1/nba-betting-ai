import pandas as pd

# === Load Data ===
df = pd.read_csv("data/final_games_full_features.csv", low_memory=False)
original_len = len(df)

# === Fill NaNs with smart values ===
fill_median = ["home_recent_avg_pts", "away_recent_avg_pts"]
fill_half = ["home_recent_win_pct", "away_recent_win_pct"]

for col in fill_median:
    df[col] = df[col].fillna(df[col].median())

for col in fill_half:
    df[col] = df[col].fillna(0.5)

# === Drop only if core stat columns are missing ===
required_features = [
    "home_OFF_RATING", "away_OFF_RATING",
    "home_DEF_RATING", "away_DEF_RATING",
    "home_EFG_PCT", "away_EFG_PCT",
    "home_odds", "away_odds"
]

df = df.dropna(subset=required_features)
final_len = len(df)

# === Save cleaned file ===
df.to_csv("data/final_games_full_features_cleaned.csv", index=False)
print(f"✅ Cleaned and saved! Kept {final_len}/{original_len} rows.")
