#!/usr/bin/env python3
"""
Build Player Assists Props Model  
Predict if player will go Over/Under their assists line
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_player_assists_data():
    """Load and prepare player data for assists props"""
    
    print("🏀 BUILDING PLAYER ASSISTS PROPS MODEL")
    print("Predicting Over/Under player assists lines")
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
    
    # Filter active players only
    df = df[df['numMinutes'] > 0].copy()
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} active player performances")
    return df

def engineer_assists_features(df):
    """Create predictive features for player assists"""
    
    print(f"\n🔧 ENGINEERING ASSISTS FEATURES")
    
    df['player_name'] = df['firstName'] + ' ' + df['lastName']
    features_data = []
    
    for player_id in df['personId'].unique():
        player_data = df[df['personId'] == player_id].copy()
        
        if len(player_data) < 5:
            continue
        
        # Rolling averages for assists
        player_data['assists_last_5'] = player_data['assists'].rolling(5, min_periods=1).mean().shift(1)
        player_data['assists_last_10'] = player_data['assists'].rolling(10, min_periods=1).mean().shift(1)
        player_data['assists_last_15'] = player_data['assists'].rolling(15, min_periods=1).mean().shift(1)
        
        # Minutes trends (more minutes = more assist opportunities)
        player_data['minutes_last_5'] = player_data['numMinutes'].rolling(5, min_periods=1).mean().shift(1)
        
        # Usage-related features
        player_data['fga_last_5'] = player_data['fieldGoalsAttempted'].rolling(5, min_periods=1).mean().shift(1)
        player_data['turnovers_last_5'] = player_data['turnovers'].rolling(5, min_periods=1).mean().shift(1)
        
        # Assist-to-turnover ratio trend
        player_data['ast_to_last_5'] = (player_data['assists'] / (player_data['turnovers'] + 1)).rolling(5, min_periods=1).mean().shift(1)
        
        # Recent form trend
        player_data['assists_trend_3'] = (
            player_data['assists'].rolling(3, min_periods=1).mean().shift(1) - 
            player_data['assists'].rolling(6, min_periods=3).mean().shift(4)
        )
        
        # Home/Away splits
        home_ast_avg = player_data[player_data['home'] == 1]['assists'].mean()
        away_ast_avg = player_data[player_data['home'] == 0]['assists'].mean()
        player_data['home_away_ast_diff'] = home_ast_avg - away_ast_avg if not pd.isna(home_ast_avg) and not pd.isna(away_ast_avg) else 0
        
        # Season averages
        player_data['season_apg'] = player_data['assists'].expanding(min_periods=1).mean().shift(1)
        player_data['season_ast_to'] = (player_data['assists'] / (player_data['turnovers'] + 1)).expanding(min_periods=1).mean().shift(1)
        
        # Consistency
        player_data['assists_consistency'] = player_data['assists'].rolling(10, min_periods=3).std().shift(1)
        
        # Position estimation based on assists pattern
        avg_assists = player_data['assists'].mean()
        if avg_assists >= 6:
            position_type = 'playmaker'  # PGs, primary ball handlers
        elif avg_assists >= 3:
            position_type = 'secondary'  # Secondary playmakers, some wings
        else:
            position_type = 'low_usage'  # Low usage players
            
        player_data['position_playmaker'] = 1 if position_type == 'playmaker' else 0
        player_data['position_secondary'] = 1 if position_type == 'secondary' else 0
        
        # Opponent strength
        player_data['opponent_strength'] = player_data['win'].rolling(10, min_periods=1).mean().shift(1)
        
        features_data.append(player_data)
    
    final_df = pd.concat(features_data, ignore_index=True)
    final_df = final_df.dropna(subset=['assists_last_5', 'minutes_last_5'])
    
    print(f"✅ Features created for {len(final_df)} performances")
    return final_df

def create_assists_betting_targets(df, common_lines=[2.5, 3.5, 4.5, 5.5, 6.5, 7.5]):
    """Create betting targets for assists lines"""
    
    print(f"\n🎯 CREATING ASSISTS BETTING TARGETS")
    print(f"Common lines: {common_lines}")
    
    for line in common_lines:
        df[f'over_{line}_ast'] = (df['assists'] > line).astype(int)
    
    df['target_assists'] = df['assists']
    
    # Show distribution
    for line in common_lines:
        over_pct = df[f'over_{line}_ast'].mean()
        print(f"   {line} assists: {over_pct:.1%} hit Over")
    
    return df

def train_assists_models(df):
    """Train assists prediction models"""
    
    print(f"\n🧠 TRAINING ASSISTS MODELS")
    
    feature_cols = [
        'assists_last_5', 'assists_last_10', 'assists_last_15',
        'minutes_last_5', 'fga_last_5', 'turnovers_last_5',
        'ast_to_last_5', 'assists_trend_3', 'home_away_ast_diff',
        'season_apg', 'season_ast_to', 'assists_consistency',
        'position_playmaker', 'position_secondary', 'opponent_strength',
        'home'
    ]
    
    X = df[feature_cols].fillna(0)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
    
    models = {}
    
    # 1. REGRESSION MODEL
    print(f"\\n🎯 Training Assists Regression Model...")
    
    y_assists = df['target_assists']
    y_train_assists, y_test_assists = y_assists.iloc[:split_idx], y_assists.iloc[split_idx:]
    
    assists_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        random_state=42
    )
    
    assists_regressor.fit(X_train, y_train_assists)
    y_pred_assists = assists_regressor.predict(X_test)
    mae = mean_absolute_error(y_test_assists, y_pred_assists)
    
    print(f"   Assists MAE: {mae:.2f} assists")
    models['assists_regressor'] = assists_regressor
    
    # 2. CLASSIFICATION MODELS
    print(f"\\n🎯 Training Over/Under Classification Models...")
    
    common_lines = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    
    for line in common_lines:
        print(f"   Training {line} assists Over/Under model...")
        
        y_line = df[f'over_{line}_ast']
        y_train_line, y_test_line = y_line.iloc[:split_idx], y_line.iloc[split_idx:]
        
        if y_train_line.sum() < 30:
            print(f"     Skipping {line} - insufficient positive examples")
            continue
        
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42
        )
        
        classifier.fit(X_train, y_train_line)
        y_pred_line = classifier.predict(X_test)
        accuracy = accuracy_score(y_test_line, y_pred_line)
        
        print(f"     {line} Accuracy: {accuracy:.1%}")
        models[f'over_{line}_ast_classifier'] = classifier
    
    # 3. FEATURE IMPORTANCE
    print(f"\\n🔝 FEATURE IMPORTANCE (Assists Regression):")
    feature_importance = list(zip(feature_cols, assists_regressor.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance):
        print(f"   {i+1:2d}. {feature:<25} {importance:.3f}")
    
    return models, feature_cols

def save_assists_models(models, feature_cols):
    """Save assists models"""
    
    print(f"\\n💾 SAVING PLAYER ASSISTS MODELS")
    
    for model_name, model in models.items():
        model_path = f"models/player_assists_{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Saved {model_name}")
    
    features_path = "models/player_assists_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\\n")
    
    print(f"📋 Saved feature list: {len(feature_cols)} features")

def test_assists_example(models, feature_cols):
    """Test assists prediction"""
    
    print(f"\\n🧪 TESTING SAMPLE ASSISTS PREDICTION")
    
    # Sample playmaker stats
    sample_data = {
        'assists_last_5': 7.2,
        'assists_last_10': 6.8,
        'assists_last_15': 7.1,
        'minutes_last_5': 35.2,
        'fga_last_5': 14.2,
        'turnovers_last_5': 3.1,
        'ast_to_last_5': 2.3,
        'assists_trend_3': 0.5,
        'home_away_ast_diff': 0.8,
        'season_apg': 7.0,
        'season_ast_to': 2.2,
        'assists_consistency': 1.8,
        'position_playmaker': 1,
        'position_secondary': 0,
        'opponent_strength': 0.5,
        'home': 1
    }
    
    sample_df = pd.DataFrame([sample_data])
    
    if 'assists_regressor' in models:
        predicted_assists = models['assists_regressor'].predict(sample_df)[0]
        print(f"📊 Predicted Assists: {predicted_assists:.1f}")
        
        for line in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
            model_name = f'over_{line}_ast_classifier'
            if model_name in models:
                over_prob = models[model_name].predict_proba(sample_df)[0][1]
                recommendation = "OVER" if over_prob > 0.52 else "UNDER" 
                confidence = max(over_prob, 1-over_prob)
                
                print(f"   {line} assists Over/Under: {recommendation} ({confidence:.1%} confidence)")

def main():
    """Build complete player assists models"""
    
    df = load_player_assists_data()
    df = engineer_assists_features(df)
    df = create_assists_betting_targets(df)
    models, feature_cols = train_assists_models(df)
    save_assists_models(models, feature_cols)
    test_assists_example(models, feature_cols)
    
    print(f"\\n🎉 PLAYER ASSISTS PROPS MODELS COMPLETE!")
    return models, feature_cols

if __name__ == "__main__":
    main()