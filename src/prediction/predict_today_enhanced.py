import pandas as pd
import joblib
from datetime import datetime
import os
from src.utils.team_name_mapping import TEAM_NAME_TO_ODDS_NAME
from src.features.get_team_stats import get_team_stats
from src.features.get_recent_stats import get_recent_win_pct, get_recent_avg_pts
from src.utils.config import DATA_DIR, PREDICTIONS_DIR

# === Load Hybrid Elite model (10.2% ROI) ===
model = joblib.load("models/hybrid_elite_model_v1.pkl")

# Load feature list
with open("models/hybrid_elite_features_v1.txt", "r") as f:
    model_features = [line.strip() for line in f.readlines()]

def implied_prob(odds):
    if pd.isna(odds) or odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return -odds / (-odds + 100)

def calculate_hybrid_elite_features(home_stats, away_stats, home_recent_win, away_recent_win, 
                                   home_recent_pts, away_recent_pts, home_odds, away_odds):
    """Calculate Hybrid Elite features (10.2% ROI model)"""
    
    # 1. NET_gap (most important feature)
    NET_gap = home_stats["NET_RATING"] - away_stats["NET_RATING"]
    
    # 2. PIE_advantage (elite player impact)
    PIE_advantage = home_stats.get("PIE", 0.5) - away_stats.get("PIE", 0.5)
    
    # 3. ranking_advantage (league context - lower rank is better)
    home_pie_rank = home_stats.get("PIE_RANK", 15)
    away_pie_rank = away_stats.get("PIE_RANK", 15)
    ranking_advantage = away_pie_rank - home_pie_rank
    
    # 4. TS_advantage (elite shooting metric)
    TS_advantage = home_stats.get("TS_PCT", 0.55) - away_stats.get("TS_PCT", 0.55)
    
    # 5. recent_form_advantage
    recent_form_advantage = home_recent_win - away_recent_win
    
    # 6. EFG_advantage (basic shooting)
    EFG_advantage = home_stats["EFG_PCT"] - away_stats["EFG_PCT"]
    
    # 7. TOV_advantage (lower turnover % is better)
    TOV_advantage = away_stats.get("TM_TOV_PCT", 14) - home_stats.get("TM_TOV_PCT", 14)
    
    # 8. efficiency_advantage (AST/TO ratio)
    efficiency_advantage = home_stats.get("AST_TO", 1.5) - away_stats.get("AST_TO", 1.5)
    
    # 9. Raw PIE values for context
    home_PIE = home_stats.get("PIE", 0.5)
    away_PIE = away_stats.get("PIE", 0.5)
    
    return {
        # Hybrid Elite feature set (exact order from model)
        'NET_gap': NET_gap,
        'PIE_advantage': PIE_advantage,
        'ranking_advantage': ranking_advantage,
        'TS_advantage': TS_advantage,
        'recent_form_advantage': recent_form_advantage,
        'EFG_advantage': EFG_advantage,
        'TOV_advantage': TOV_advantage,
        'efficiency_advantage': efficiency_advantage,
        'home_PIE': home_PIE,
        'away_PIE': away_PIE
    }

def run_predictions():
    print("🏆 Using Hybrid Elite Model: hybrid_elite_model_v1.pkl (10 features, 10.2% ROI)")

    # === Load today's odds CSV ===
    today_str = datetime.today().strftime('%Y-%m-%d')
    odds_file = DATA_DIR / f"nba_odds_{today_str}.csv"
    save_path = PREDICTIONS_DIR / f"predictions_{today_str}_hybrid_elite.csv"

    # === Skip if predictions already exist ===
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path)
            if "value_flag" in existing_df.columns and not existing_df["value_flag"].isnull().all():
                print(f"🛑 Hybrid Elite predictions already exist for today — skipping prediction step.")
                return
        except Exception as e:
            print(f"⚠️ Error checking existing predictions: {e}")

    try:
        odds_df = pd.read_csv(odds_file)
    except FileNotFoundError:
        print(f"❌ Odds file not found: {odds_file}")
        return

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

            # Calculate Hybrid Elite features
            features_dict = calculate_hybrid_elite_features(
                home_stats, away_stats, home_recent_win, away_recent_win,
                home_recent_pts, away_recent_pts, home_odds, away_odds
            )

            # Add basic game info
            row_data = {
                "hometeam": home,
                "awayteam": away,
                "home_odds": home_odds,
                "away_odds": away_odds,
            }
            
            # Add all features
            row_data.update(features_dict)
            rows.append(row_data)

        except Exception as e:
            print(f"❌ Error for {home} vs {away}: {e}")
            continue

    if not rows:
        print("❌ No valid games to predict.")
        return

    # === Predict using enhanced model ===
    df = pd.DataFrame(rows)
    
    # Ensure we have all required features
    available_features = [f for f in model_features if f in df.columns]
    missing_features = [f for f in model_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features for prediction: {missing_features}")
    
    print(f"✅ Using {len(available_features)} features for prediction")
    
    X = df[available_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Make predictions
    df["model_win_prob"] = model.predict_proba(X)[:, 1]
    df["prediction"] = df["model_win_prob"].apply(lambda p: "HOME" if p >= 0.5 else "AWAY")
    df["predicted_odds"] = df.apply(lambda x: x["home_odds"] if x["prediction"] == "HOME" else x["away_odds"], axis=1)
    df["implied_prob"] = df["predicted_odds"].apply(implied_prob)
    df["value_gap"] = df["model_win_prob"] - df["implied_prob"]
    
    # Hybrid Elite value flagging (optimized for 10.2% ROI model)
    df["value_flag"] = df["value_gap"].apply(
        lambda g: "🔥 Elite Edge" if g > 0.05 else (
            "👍 Value Bet" if g > 0.03 else (
                "⚠️ Caution Bet" if g < -0.03 else "–"
            )
        )
    )
    
    # Add confidence score based on prediction probability
    df["confidence"] = df["model_win_prob"].apply(
        lambda p: "High" if abs(p - 0.5) > 0.15 else (
            "Medium" if abs(p - 0.5) > 0.08 else "Low"
        )
    )

    # === Output ===
    print("\n🏆 Hybrid Elite Predictions for Today's NBA Games (10.2% ROI Model):")
    display_cols = [
        "hometeam", "awayteam", "home_odds", "away_odds",
        "model_win_prob", "implied_prob", "value_gap",
        "prediction", "confidence", "value_flag"
    ]
    
    print(df[display_cols].sort_values(by="model_win_prob", ascending=False).to_string(index=False))

    # === Save ===
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n✅ Hybrid Elite predictions saved to {save_path}")
    
    # Show betting opportunities
    value_bets = df[df["value_gap"] > 0.03]
    if len(value_bets) > 0:
        print(f"\n💰 Found {len(value_bets)} betting opportunities:")
        for _, bet in value_bets.iterrows():
            print(f"   {bet['hometeam']} vs {bet['awayteam']}: {bet['prediction']} ({bet['model_win_prob']:.1%} vs {bet['implied_prob']:.1%}) - {bet['value_flag']}")
    else:
        print("\n🚫 No significant betting opportunities found today.")

if __name__ == "__main__":
    run_predictions()