#!/usr/bin/env python3
"""
Build Player Rebounds Props Model
Predict if player will go Over/Under their rebounds line
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_player_rebounds_data():
    """Load and prepare player data for rebounds props"""
    
    print("🏀 BUILDING PLAYER REBOUNDS PROPS MODEL")
    print("Predicting Over/Under player rebounds lines")
    print("=" * 60)
    
    print("📊 Loading player data...")
    
    # Load recent data efficiently 
    chunk_size = 50000
    recent_data = []
    target_date = datetime.now() - timedelta(days=400)  # ~13 months
    
    for chunk in pd.read_csv('data/PlayerStatistics.csv', chunksize=chunk_size, low_memory=False):
        chunk['gameDate'] = pd.to_datetime(chunk['gameDate'])
        recent_chunk = chunk[chunk['gameDate'] >= target_date]
        
        if len(recent_chunk) > 0:
            recent_data.append(recent_chunk)
            
        if len(recent_data) > 0 and len(pd.concat(recent_data)) > 150000:
            break
    
    df = pd.concat(recent_data, ignore_index=True)
    print(f"✅ Loaded {len(df)} player performances")
    
    # Filter active players only (played minutes)
    df = df[df['numMinutes'] > 0].copy()
    print(f"📊 Active performances: {len(df)}")
    
    # Sort by date for time-series features
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    return df

def engineer_rebounds_features(df):
    """Create predictive features for player rebounds"""
    
    print(f"\n🔧 ENGINEERING REBOUNDS FEATURES")
    print("Creating predictive features for rebounds props")
    
    # Create player identifier
    df['player_name'] = df['firstName'] + ' ' + df['lastName']
    
    # Basic features
    features_data = []
    
    print("   Computing rolling averages and trends...")
    
    for player_id in df['personId'].unique():
        player_data = df[df['personId'] == player_id].copy()
        
        if len(player_data) < 5:  # Need minimum games
            continue
        
        # Rolling averages for rebounds
        player_data['rebounds_last_5'] = player_data['reboundsTotal'].rolling(5, min_periods=1).mean().shift(1)
        player_data['rebounds_last_10'] = player_data['reboundsTotal'].rolling(10, min_periods=1).mean().shift(1)
        player_data['rebounds_last_15'] = player_data['reboundsTotal'].rolling(15, min_periods=1).mean().shift(1)
        
        # Offensive vs Defensive rebounds trends
        player_data['oreb_last_5'] = player_data['reboundsOffensive'].rolling(5, min_periods=1).mean().shift(1)
        player_data['dreb_last_5'] = player_data['reboundsDefensive'].rolling(5, min_periods=1).mean().shift(1)
        
        # Minutes trends (more minutes = more rebounding opportunities)
        player_data['minutes_last_5'] = player_data['numMinutes'].rolling(5, min_periods=1).mean().shift(1)
        player_data['minutes_last_10'] = player_data['numMinutes'].rolling(10, min_periods=1).mean().shift(1)
        
        # Recent form trend (last 3 vs previous 3-6 games)
        player_data['rebounds_trend_3'] = (
            player_data['reboundsTotal'].rolling(3, min_periods=1).mean().shift(1) - 
            player_data['reboundsTotal'].rolling(6, min_periods=3).mean().shift(4)
        )
        
        # Home/Away splits for rebounds
        home_reb_avg = player_data[player_data['home'] == 1]['reboundsTotal'].mean()
        away_reb_avg = player_data[player_data['home'] == 0]['reboundsTotal'].mean()
        player_data['home_away_reb_diff'] = home_reb_avg - away_reb_avg if not pd.isna(home_reb_avg) and not pd.isna(away_reb_avg) else 0
        
        # Season averages
        player_data['season_rpg'] = player_data['reboundsTotal'].expanding(min_periods=1).mean().shift(1)
        player_data['season_oreb_pct'] = player_data['reboundsOffensive'].expanding(min_periods=1).mean().shift(1)
        player_data['season_dreb_pct'] = player_data['reboundsDefensive'].expanding(min_periods=1).mean().shift(1)
        
        # Consistency metrics
        player_data['rebounds_consistency'] = player_data['reboundsTotal'].rolling(10, min_periods=3).std().shift(1)
        
        # Opponent strength proxy
        player_data['opponent_strength'] = player_data['win'].rolling(10, min_periods=1).mean().shift(1)
        
        # Position-based rebounding (estimate based on rebounds pattern)
        total_rebounds = player_data['reboundsTotal'].mean()
        if total_rebounds >= 8:
            position_type = 'big_man'  # Centers/PFs
        elif total_rebounds >= 4:
            position_type = 'wing'     # SFs/versatile players
        else:
            position_type = 'guard'    # PGs/SGs
            
        player_data['position_big'] = 1 if position_type == 'big_man' else 0
        player_data['position_wing'] = 1 if position_type == 'wing' else 0
        
        features_data.append(player_data)
    
    # Combine all player data
    final_df = pd.concat(features_data, ignore_index=True)
    print(f"✅ Features created for {len(final_df)} performances")
    
    # Remove rows with insufficient history
    final_df = final_df.dropna(subset=['rebounds_last_5', 'minutes_last_5'])
    print(f"📊 Final dataset: {len(final_df)} performances")
    
    return final_df

def create_rebounds_betting_targets(df, common_lines=[4.5, 6.5, 8.5, 10.5, 12.5]):
    """Create betting targets for common rebounds lines"""
    
    print(f"\n🎯 CREATING REBOUNDS BETTING TARGETS")
    print(f"Common lines: {common_lines}")
    
    # For each common line, create Over/Under targets
    for line in common_lines:
        df[f'over_{line}_reb'] = (df['reboundsTotal'] > line).astype(int)
    
    # Also predict exact rebounds for regression
    df['target_rebounds'] = df['reboundsTotal']
    
    # Show distribution
    for line in common_lines:
        over_pct = df[f'over_{line}_reb'].mean()
        print(f"   {line} rebounds: {over_pct:.1%} hit Over")
    
    return df

def train_rebounds_models(df):
    """Train both regression and classification models for rebounds"""
    
    print(f"\n🧠 TRAINING REBOUNDS MODELS")
    print("Both regression (exact rebounds) and classification (Over/Under)")
    
    # Feature columns
    feature_cols = [
        'rebounds_last_5', 'rebounds_last_10', 'rebounds_last_15',
        'oreb_last_5', 'dreb_last_5',
        'minutes_last_5', 'minutes_last_10',
        'rebounds_trend_3', 'home_away_reb_diff', 'opponent_strength',
        'season_rpg', 'season_oreb_pct', 'season_dreb_pct',
        'rebounds_consistency', 'position_big', 'position_wing',
        'home'
    ]
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    
    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
    
    models = {}
    
    # 1. REGRESSION MODEL (predict exact rebounds)
    print(f"\\n🎯 Training Rebounds Regression Model...")
    
    y_rebounds = df['target_rebounds']
    y_train_rebounds, y_test_rebounds = y_rebounds.iloc[:split_idx], y_rebounds.iloc[split_idx:]
    
    rebounds_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        random_state=42
    )
    
    rebounds_regressor.fit(X_train, y_train_rebounds)
    
    # Test regression
    y_pred_rebounds = rebounds_regressor.predict(X_test)
    mae = mean_absolute_error(y_test_rebounds, y_pred_rebounds)
    print(f"   Rebounds MAE: {mae:.2f} rebounds")
    
    models['rebounds_regressor'] = rebounds_regressor
    
    # 2. CLASSIFICATION MODELS (Over/Under each line)
    print(f"\\n🎯 Training Over/Under Classification Models...")
    
    common_lines = [4.5, 6.5, 8.5, 10.5, 12.5]
    
    for line in common_lines:
        print(f"   Training {line} rebounds Over/Under model...")
        
        y_line = df[f'over_{line}_reb']
        y_train_line, y_test_line = y_line.iloc[:split_idx], y_line.iloc[split_idx:]
        
        # Skip if not enough positive examples
        if y_train_line.sum() < 50:
            print(f"     Skipping {line} - insufficient positive examples")
            continue
        
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=15,
            class_weight='balanced',
            random_state=42
        )
        
        classifier.fit(X_train, y_train_line)
        
        # Test classification
        y_pred_line = classifier.predict(X_test)
        accuracy = accuracy_score(y_test_line, y_pred_line)
        
        print(f"     {line} Accuracy: {accuracy:.1%}")
        
        models[f'over_{line}_reb_classifier'] = classifier
    
    # 3. FEATURE IMPORTANCE
    print(f"\\n🔝 FEATURE IMPORTANCE (Rebounds Regression):")
    feature_importance = list(zip(feature_cols, rebounds_regressor.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance):
        print(f"   {i+1:2d}. {feature:<25} {importance:.3f}")
    
    return models, feature_cols

def save_rebounds_models(models, feature_cols):
    """Save the trained rebounds models"""
    
    print(f"\\n💾 SAVING PLAYER REBOUNDS MODELS")
    
    # Save each model
    for model_name, model in models.items():
        model_path = f"models/player_rebounds_{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Saved {model_name}")
    
    # Save feature list
    features_path = "models/player_rebounds_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\\n")
    
    print(f"📋 Saved feature list: {len(feature_cols)} features")

def test_rebounds_example(models, feature_cols):
    """Test rebounds prediction on sample data"""
    
    print(f"\\n🧪 TESTING SAMPLE REBOUNDS PREDICTION")
    
    # Create sample player stats (big man)
    sample_data = {
        'rebounds_last_5': 9.2,
        'rebounds_last_10': 8.8,
        'rebounds_last_15': 9.1,
        'oreb_last_5': 2.4,
        'dreb_last_5': 6.8,
        'minutes_last_5': 32.5,
        'minutes_last_10': 33.1,
        'rebounds_trend_3': 0.8,  # Trending up
        'home_away_reb_diff': 1.2,  # Better at home
        'opponent_strength': 0.45,
        'season_rpg': 9.0,
        'season_oreb_pct': 2.3,
        'season_dreb_pct': 6.7,
        'rebounds_consistency': 2.1,
        'position_big': 1,  # Big man
        'position_wing': 0,
        'home': 1
    }
    
    sample_df = pd.DataFrame([sample_data])
    
    # Predict exact rebounds
    if 'rebounds_regressor' in models:
        predicted_rebounds = models['rebounds_regressor'].predict(sample_df)[0]
        print(f"📊 Predicted Rebounds: {predicted_rebounds:.1f}")
        
        # Check against common lines
        for line in [4.5, 6.5, 8.5, 10.5, 12.5]:
            model_name = f'over_{line}_reb_classifier'
            if model_name in models:
                over_prob = models[model_name].predict_proba(sample_df)[0][1]
                recommendation = "OVER" if over_prob > 0.52 else "UNDER" 
                confidence = max(over_prob, 1-over_prob)
                
                print(f"   {line} rebounds Over/Under: {recommendation} ({confidence:.1%} confidence)")

def main():
    """Build complete player rebounds models"""
    
    # Load data
    df = load_player_rebounds_data()
    
    # Engineer features
    df = engineer_rebounds_features(df)
    
    # Create betting targets
    df = create_rebounds_betting_targets(df)
    
    # Train models
    models, feature_cols = train_rebounds_models(df)
    
    # Save models
    save_rebounds_models(models, feature_cols)
    
    # Test example
    test_rebounds_example(models, feature_cols)
    
    print(f"\\n🎉 PLAYER REBOUNDS PROPS MODELS COMPLETE!")
    return models, feature_cols

if __name__ == "__main__":
    main()