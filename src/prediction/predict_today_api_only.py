import pandas as pd
import joblib
from datetime import datetime
from utils.team_name_mapping import TEAM_NAME_TO_ODDS_NAME
from get_team_stats import get_team_stats
from get_recent_stats import get_recent_win_pct, get_recent_avg_pts

# === Load model ===
model = joblib.load("nba_model_v4_1.pkl")

# === Load today’s odds CSV ===
today_str = datetime.today().strftime('%Y-%m-%d')
odds_file = f"data/nba_odds_{today_str}.csv"

try:
    odds_df = pd.read_csv(odds_file)
except FileNotFoundError:
    print(f"❌ Odds file not found: {odds_file}")
    exit()

# === Features for prediction ===
features = [
    "off_rating_diff", "def_rating_diff", "recent_win_diff", "pace_diff",
    "home_recent_avg_pts", "away_recent_avg_pts",
    "home_EFG_PCT", "away_EFG_PCT",
    "home_odds", "away_odds", "odds_diff",
    "implied_home_win_pct", "implied_away_win_pct", "implied_win_diff"
]

rows = []
for _, row in odds_df.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    home_odds = row["home_odds"]
    away_odds = row["away_odds"]

    home_clean = home.lower().strip()
    away_clean = away.lower().strip()

    try:
        # Get stats
        home_stats = get_team_stats(home_clean)
        away_stats = get_team_stats(away_clean)
        home_recent_win = get_recent_win_pct(home_clean)
        away_recent_win = get_recent_win_pct(away_clean)
        home_recent_pts = get_recent_avg_pts(home_clean)
        away_recent_pts = get_recent_avg_pts(away_clean)

        # Check for missing data
        if None in (home_stats, away_stats):
            print(f"⚠️ Skipping {home} vs {away} (missing stats)")
            continue

        # Engineered features
        off_rating_diff = home_stats["OFF_RATING"] - away_stats["DEF_RATING"]
        def_rating_diff = home_stats["DEF_RATING"] - away_stats["OFF_RATING"]
        recent_win_diff = home_recent_win - away_recent_win
        pace_diff = home_stats["PACE"] - away_stats["PACE"]
        odds_diff = home_odds - away_odds
        implied_home = 1 / abs(home_odds) if home_odds != 0 else 0.5
        implied_away = 1 / abs(away_odds) if away_odds != 0 else 0.5
        implied_diff = implied_home - implied_away

        row_data = {
            "hometeam": home,
            "awayteam": away,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "off_rating_diff": off_rating_diff,
            "def_rating_diff": def_rating_diff,
            "recent_win_diff": recent_win_diff,
            "pace_diff": pace_diff,
            "home_recent_avg_pts": home_recent_pts,
            "away_recent_avg_pts": away_recent_pts,
            "home_EFG_PCT": home_stats["EFG_PCT"],
            "away_EFG_PCT": away_stats["EFG_PCT"],
            "odds_diff": odds_diff,
            "implied_home_win_pct": implied_home,
            "implied_away_win_pct": implied_away,
            "implied_win_diff": implied_diff
        }

        rows.append(row_data)

    except Exception as e:
        print(f"❌ Error for {home} vs {away}: {e}")
        continue

# === Run predictions ===
if not rows:
    print("❌ No valid games to predict.")
    exit()

df = pd.DataFrame(rows)
X = df[features].apply(pd.to_numeric, errors='coerce').dropna()
df = df.loc[X.index]

df["model_win_prob"] = model.predict_proba(X)[:, 1]
df["value_gap"] = df["model_win_prob"] - df["implied_home_win_pct"]
df["prediction"] = df["model_win_prob"].apply(lambda p: "HOME" if p >= 0.5 else "AWAY")
df["value_flag"] = df["value_gap"].apply(lambda g: "👍 Value Bet" if g > 0.05 else ("⚠️ Trap" if g < -0.05 else "–"))

# === Output ===
print("\n🏀 Predictions for Today’s NBA Games:")
print(df[[
    "hometeam", "awayteam", "home_odds", "away_odds",
    "model_win_prob", "implied_home_win_pct", "value_gap",
    "prediction", "value_flag"
]].sort_values(by="model_win_prob", ascending=False).to_string(index=False))
