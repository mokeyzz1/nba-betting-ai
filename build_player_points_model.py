#!/usr/bin/env python3
"""
Build Player Points Props Model
Predict if player will go Over/Under their points line
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

def load_and_prepare_player_data():
    """Load and prepare player data for points props"""
    
    print("🏀 BUILDING PLAYER POINTS PROPS MODEL")
    print("Predicting Over/Under player points lines")
    print("=" * 60)
    
    print("📊 Loading player data...")
    
    # Load recent data efficiently 
    chunk_size = 50000
    recent_data = []
    target_date = datetime.now() - timedelta(days=400)  # ~13 months
    
    print(f"   Loading games since {target_date.strftime('%Y-%m-%d')}")
    
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

def engineer_player_features(df):
    """Create predictive features for player points"""
    
    print(f"\n🔧 ENGINEERING PLAYER FEATURES")
    print("Creating predictive features for points props")
    
    # Create player identifier
    df['player_name'] = df['firstName'] + ' ' + df['lastName']
    
    # Basic features
    features_data = []
    
    print("   Computing rolling averages and trends...")
    
    for player_id in df['personId'].unique():
        player_data = df[df['personId'] == player_id].copy()
        
        if len(player_data) < 5:  # Need minimum games
            continue
            
        # Rolling averages (last 5, 10, 15 games)
        player_data['points_last_5'] = player_data['points'].rolling(5, min_periods=1).mean().shift(1)
        player_data['points_last_10'] = player_data['points'].rolling(10, min_periods=1).mean().shift(1)
        player_data['points_last_15'] = player_data['points'].rolling(15, min_periods=1).mean().shift(1)
        
        # Minutes trends (affects scoring opportunity)
        player_data['minutes_last_5'] = player_data['numMinutes'].rolling(5, min_periods=1).mean().shift(1)
        player_data['minutes_last_10'] = player_data['numMinutes'].rolling(10, min_periods=1).mean().shift(1)
        
        # Shooting efficiency trends
        player_data['fg_pct_last_5'] = player_data['fieldGoalsPercentage'].rolling(5, min_periods=1).mean().shift(1)
        player_data['fg_attempts_last_5'] = player_data['fieldGoalsAttempted'].rolling(5, min_periods=1).mean().shift(1)
        
        # Recent form (last 3 games trend)
        player_data['points_trend_3'] = (
            player_data['points'].rolling(3, min_periods=1).mean().shift(1) - 
            player_data['points'].rolling(6, min_periods=3).mean().shift(4)
        )
        
        # Home/Away splits
        home_avg = player_data[player_data['home'] == 1]['points'].mean()
        away_avg = player_data[player_data['home'] == 0]['points'].mean()
        player_data['home_away_diff'] = home_avg - away_avg if not pd.isna(home_avg) and not pd.isna(away_avg) else 0
        
        # Opponent strength (using win column as proxy)
        player_data['opponent_strength'] = player_data['win'].rolling(10, min_periods=1).mean().shift(1)
        
        # Season stats (rolling season averages)
        player_data['season_ppg'] = player_data['points'].expanding(min_periods=1).mean().shift(1)
        player_data['season_mpg'] = player_data['numMinutes'].expanding(min_periods=1).mean().shift(1)
        
        # Consistency metrics (standard deviation)
        player_data['points_consistency'] = player_data['points'].rolling(10, min_periods=3).std().shift(1)
        
        features_data.append(player_data)
    
    # Combine all player data
    final_df = pd.concat(features_data, ignore_index=True)
    print(f"✅ Features created for {len(final_df)} performances")
    
    # Remove rows with insufficient history (first few games)
    final_df = final_df.dropna(subset=['points_last_5', 'minutes_last_5'])
    print(f"📊 Final dataset: {len(final_df)} performances")
    
    return final_df

def create_points_betting_targets(df, common_lines=[15.5, 20.5, 25.5, 30.5]):
    """Create betting targets for common points lines"""
    
    print(f"\n🎯 CREATING BETTING TARGETS")
    print(f"Common lines: {common_lines}")
    
    # For each common line, create Over/Under targets
    for line in common_lines:
        df[f'over_{line}'] = (df['points'] > line).astype(int)
    
    # Also predict exact points for regression
    df['target_points'] = df['points']
    
    # Show distribution
    for line in common_lines:
        over_pct = df[f'over_{line}'].mean()
        print(f"   {line} points: {over_pct:.1%} hit Over")
    
    return df

def train_points_models(df):
    """Train both regression and classification models"""
    
    print(f"\n🧠 TRAINING POINTS MODELS")
    print("Both regression (exact points) and classification (Over/Under)")
    
    # Feature columns
    feature_cols = [
        'points_last_5', 'points_last_10', 'points_last_15',
        'minutes_last_5', 'minutes_last_10',
        'fg_pct_last_5', 'fg_attempts_last_5',
        'points_trend_3', 'home_away_diff', 'opponent_strength',
        'season_ppg', 'season_mpg', 'points_consistency',
        'home'  # Home/Away indicator
    ]
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    
    # Time-based split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
    
    models = {}
    
    # 1. REGRESSION MODEL (predict exact points)
    print(f"\\n🎯 Training Points Regression Model...")
    
    y_points = df['target_points']
    y_train_points, y_test_points = y_points.iloc[:split_idx], y_points.iloc[split_idx:]
    
    points_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        random_state=42
    )
    
    points_regressor.fit(X_train, y_train_points)
    
    # Test regression
    y_pred_points = points_regressor.predict(X_test)
    mae = mean_absolute_error(y_test_points, y_pred_points)
    print(f"   Points MAE: {mae:.2f} points")
    
    models['points_regressor'] = points_regressor
    
    # 2. CLASSIFICATION MODELS (Over/Under each line)
    print(f"\\n🎯 Training Over/Under Classification Models...")
    
    common_lines = [15.5, 20.5, 25.5, 30.5]
    
    for line in common_lines:
        print(f"   Training {line} Over/Under model...")
        
        y_line = df[f'over_{line}']
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
        
        # Get probabilities for betting
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        
        print(f"     {line} Accuracy: {accuracy:.1%}")
        
        models[f'over_{line}_classifier'] = classifier
    
    # 3. FEATURE IMPORTANCE
    print(f"\\n🔝 FEATURE IMPORTANCE (Points Regression):")
    feature_importance = list(zip(feature_cols, points_regressor.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance):
        print(f"   {i+1:2d}. {feature:<20} {importance:.3f}")
    
    return models, feature_cols

def save_points_models(models, feature_cols):
    """Save the trained models"""
    
    print(f"\\n💾 SAVING PLAYER POINTS MODELS")
    
    # Save each model
    for model_name, model in models.items():
        model_path = f"models/player_points_{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Saved {model_name}")
    
    # Save feature list
    features_path = "models/player_points_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    print(f"📋 Saved feature list: {len(feature_cols)} features")
    print(f"\\n🎯 PLAYER POINTS MODELS READY!")
    print(f"   • Regression model: Predicts exact points")
    print(f"   • Classification models: Over/Under 15.5, 20.5, 25.5, 30.5")
    print(f"   • Features: Player averages, trends, matchup data")

def test_prediction_example(models, feature_cols):
    """Test prediction on sample data"""
    
    print(f"\\n🧪 TESTING SAMPLE PREDICTION")
    
    # Create sample player stats
    sample_data = {
        'points_last_5': 24.2,
        'points_last_10': 23.8,
        'points_last_15': 24.1,
        'minutes_last_5': 34.5,
        'minutes_last_10': 35.1,
        'fg_pct_last_5': 0.485,
        'fg_attempts_last_5': 18.4,
        'points_trend_3': 1.2,  # Trending up
        'home_away_diff': 2.1,  # Better at home
        'opponent_strength': 0.45,  # Weak opponent
        'season_ppg': 23.9,
        'season_mpg': 35.0,
        'points_consistency': 4.8,
        'home': 1  # Playing at home
    }
    
    sample_df = pd.DataFrame([sample_data])
    
    # Predict exact points
    if 'points_regressor' in models:
        predicted_points = models['points_regressor'].predict(sample_df)[0]
        print(f"📊 Predicted Points: {predicted_points:.1f}")
        
        # Check against common lines
        for line in [15.5, 20.5, 25.5, 30.5]:
            model_name = f'over_{line}_classifier'
            if model_name in models:
                over_prob = models[model_name].predict_proba(sample_df)[0][1]
                recommendation = "OVER" if over_prob > 0.52 else "UNDER" 
                confidence = max(over_prob, 1-over_prob)
                
                print(f"   {line} Over/Under: {recommendation} ({confidence:.1%} confidence)")

def main():
    """Build complete player points models"""
    
    # Load data
    df = load_and_prepare_player_data()
    
    # Engineer features
    df = engineer_player_features(df)
    
    # Create betting targets
    df = create_points_betting_targets(df)
    
    # Train models
    models, feature_cols = train_points_models(df)
    
    # Save models
    save_points_models(models, feature_cols)
    
    # Test example
    test_prediction_example(models, feature_cols)
    
    print(f"\\n🎉 PLAYER POINTS PROPS MODELS COMPLETE!")
    return models, feature_cols

if __name__ == "__main__":
    main()