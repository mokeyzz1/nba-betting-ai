import pandas as pd
import joblib
from datetime import datetime
import os
from src.utils.team_name_mapping import TEAM_NAME_TO_ODDS_NAME
from src.features.get_team_stats import get_team_stats
from src.features.get_recent_stats import get_recent_win_pct, get_recent_avg_pts
from src.utils.config import DATA_DIR, PREDICTIONS_DIR

# === Load model ===
model = joblib.load("models/nba_model_v4_2.pkl")

def implied_prob(odds):
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return -odds / (-odds + 100)

def run_predictions():
    print("🧠 Using Model: nba_model_v4_2.pkl (tuned + calibrated)")

    # === Load today’s odds CSV ===
    today_str = datetime.today().strftime('%Y-%m-%d')
    odds_file = DATA_DIR / f"nba_odds_{today_str}.csv"
    save_path = PREDICTIONS_DIR / f"predictions_{today_str}_v4_2.csv"

    # === Skip if predictions already exist with value_flag
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path)
            if "value_flag" in existing_df.columns and not existing_df["value_flag"].isnull().all():
                print(f"🛑 Predictions already exist for today — skipping prediction step.")
                return
        except Exception as e:
            print(f"⚠️ Error checking existing predictions: {e}")

    try:
        odds_df = pd.read_csv(odds_file)
    except FileNotFoundError:
        print(f"❌ Odds file not found: {odds_file}")
        return

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
            home_stats = get_team_stats(home_clean)
            away_stats = get_team_stats(away_clean)
            home_recent_win = get_recent_win_pct(home_clean)
            away_recent_win = get_recent_win_pct(away_clean)
            home_recent_pts = get_recent_avg_pts(home_clean)
            away_recent_pts = get_recent_avg_pts(away_clean)

            if None in (home_stats, away_stats):
                print(f"⚠️ Skipping {home} vs {away} (missing stats)")
                continue

            off_rating_diff = home_stats["OFF_RATING"] - away_stats["DEF_RATING"]
            def_rating_diff = home_stats["DEF_RATING"] - away_stats["OFF_RATING"]
            recent_win_diff = home_recent_win - away_recent_win
            pace_diff = home_stats["PACE"] - away_stats["PACE"]
            odds_diff = home_odds - away_odds

            implied_home = implied_prob(home_odds)
            implied_away = implied_prob(away_odds)
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

    if not rows:
        print("❌ No valid games to predict.")
        return

    # === Predict ===
    df = pd.DataFrame(rows)
    X = df[features].apply(pd.to_numeric, errors='coerce').dropna()
    df = df.loc[X.index]

    df["model_win_prob"] = model.predict_proba(X)[:, 1]
    df["prediction"] = df["model_win_prob"].apply(lambda p: "HOME" if p >= 0.5 else "AWAY")
    df["predicted_odds"] = df.apply(lambda x: x["home_odds"] if x["prediction"] == "HOME" else x["away_odds"], axis=1)
    df["implied_prob"] = df["predicted_odds"].apply(implied_prob)
    df["value_gap"] = df["model_win_prob"] - df["implied_prob"]
    df["value_flag"] = df["value_gap"].apply(
        lambda g: "👍 Value Bet" if g > 0.03 else ("⚠️ Caution Bet" if g < -0.03 else "–")
    )

    # === Output ===
    print("\n🏀 Predictions for Today’s NBA Games:")
    print(df[[
        "hometeam", "awayteam", "home_odds", "away_odds",
        "model_win_prob", "implied_prob", "value_gap",
        "prediction", "value_flag"
    ]].sort_values(by="model_win_prob", ascending=False).to_string(index=False))

    # === Save ===
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n✅ Predictions saved to {save_path}")
