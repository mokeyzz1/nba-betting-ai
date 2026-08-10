import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
from src.features.build_enhanced_features import build_enhanced_features
from src.training.train_ensemble_model import NBEnsembleModel
from src.utils.config import DATA_DIR, PREDICTIONS_DIR

def implied_prob(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return -odds / (-odds + 100)

def calculate_kelly_stake(win_prob, odds, bankroll=1000, max_stake=0.05):
    """Calculate optimal Kelly Criterion stake"""
    implied_prob_market = implied_prob(odds)
    edge = win_prob - implied_prob_market
    
    if edge <= 0:
        return 0
    
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1
    
    kelly_fraction = edge / (decimal_odds - 1)
    
    # Cap at maximum stake percentage
    kelly_fraction = min(kelly_fraction, max_stake)
    
    return max(0, kelly_fraction * bankroll)

def run_enhanced_predictions():
    """Run predictions using enhanced features and ensemble model"""
    
    print("🧠 Running Enhanced NBA Predictions...")
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    save_path = PREDICTIONS_DIR / f"predictions_enhanced_{today_str}.csv"
    
    # Check if predictions already exist
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        if "confidence_score" in existing_df.columns:
            print(f"🛑 Enhanced predictions already exist for today")
            return existing_df
    
    # Build enhanced features
    print("🔧 Building enhanced features...")
    features_df = build_enhanced_features()
    
    if features_df is None or len(features_df) == 0:
        print("❌ No games to predict today")
        return None
    
    # Load ensemble model
    try:
        print("📥 Loading ensemble model...")
        ensemble = NBEnsembleModel.load_model("models/nba_ensemble_model.pkl")
    except FileNotFoundError:
        print("⚠️ Ensemble model not found, using XGBoost v4.2...")
        ensemble = joblib.load("models/nba_model_v4_2.pkl")
        ensemble_mode = False
    else:
        ensemble_mode = True
    
    # Define feature set
    prediction_features = [
        "off_rating_diff", "def_rating_diff", "recent_win_diff", "pace_diff",
        "home_recent_avg_pts", "away_recent_avg_pts",
        "home_EFG_PCT", "away_EFG_PCT",
        "home_odds", "away_odds", "odds_diff",
        "implied_home_win_pct", "implied_away_win_pct", "implied_win_diff"
    ]
    
    # Add enhanced features if available
    enhanced_features = [
        "home_court_advantage", "momentum_diff", "wins_last_5_diff",
        "clutch_diff", "sos_diff", "injury_differential"
    ]
    
    available_features = [f for f in prediction_features + enhanced_features 
                         if f in features_df.columns]
    
    print(f"🎯 Using {len(available_features)} features for prediction")
    
    # Prepare prediction data
    X = features_df[available_features].fillna(0)
    
    # Make predictions
    if ensemble_mode:
        win_probabilities = ensemble.predict_proba(X)
        predictions = (win_probabilities >= 0.5).astype(int)
        
        # Calculate confidence based on probability distance from 0.5
        confidence_scores = np.abs(win_probabilities - 0.5) * 2
        
    else:
        win_probabilities = ensemble.predict_proba(X)[:, 1]
        predictions = ensemble.predict(X)
        confidence_scores = np.abs(win_probabilities - 0.5) * 2
    
    # Add predictions to dataframe
    features_df["model_win_prob"] = win_probabilities
    features_df["prediction"] = ["HOME" if p == 1 else "AWAY" for p in predictions]
    features_df["confidence_score"] = confidence_scores
    
    # Calculate betting metrics
    features_df["predicted_odds"] = features_df.apply(
        lambda x: x["home_odds"] if x["prediction"] == "HOME" else x["away_odds"], 
        axis=1
    )
    
    features_df["implied_prob"] = features_df["predicted_odds"].apply(implied_prob)
    features_df["value_gap"] = features_df["model_win_prob"] - features_df["implied_prob"]
    
    # Enhanced value classification
    def classify_bet_value(row):
        gap = row["value_gap"]
        confidence = row["confidence_score"]
        
        if gap > 0.05 and confidence > 0.3:
            return "🔥 Strong Value"
        elif gap > 0.03 and confidence > 0.2:
            return "👍 Good Value"
        elif gap > 0.01:
            return "⚖️ Slight Value"
        elif gap < -0.03:
            return "❌ Avoid"
        else:
            return "➖ No Value"
    
    features_df["value_classification"] = features_df.apply(classify_bet_value, axis=1)
    
    # Calculate Kelly stakes
    features_df["kelly_stake"] = features_df.apply(
        lambda x: calculate_kelly_stake(
            x["model_win_prob"], 
            x["predicted_odds"],
            bankroll=1000,
            max_stake=0.05
        ), axis=1
    )
    
    # Sort by value and confidence
    features_df = features_df.sort_values(
        ["value_gap", "confidence_score"], 
        ascending=[False, False]
    )
    
    # Display results
    print("\n🏀 Enhanced NBA Predictions:")
    display_cols = [
        "hometeam", "awayteam", "prediction", "model_win_prob", 
        "confidence_score", "value_gap", "value_classification", "kelly_stake"
    ]
    
    print(features_df[display_cols].round(3).to_string(index=False))
    
    # Save predictions
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    features_df.to_csv(save_path, index=False)
    print(f"\n✅ Enhanced predictions saved to {save_path}")
    
    # Summary statistics
    print(f"\n📊 Prediction Summary:")
    print(f"Total games: {len(features_df)}")
    print(f"High confidence predictions: {(features_df['confidence_score'] > 0.3).sum()}")
    print(f"Value bets identified: {(features_df['value_gap'] > 0.03).sum()}")
    print(f"Strong value bets: {features_df['value_classification'].str.contains('Strong').sum()}")
    
    return features_df

if __name__ == "__main__":
    run_enhanced_predictions()