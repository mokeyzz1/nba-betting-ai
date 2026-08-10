#!/usr/bin/env python3
"""
Build Combo Props Model (Points + Rebounds + Assists)
Predict combined PRA totals and Over/Under lines
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_player_data():
    """Load player performance data for combo props modeling"""
    
    print("🏀 BUILDING COMBO PROPS MODEL (PRA)")
    print("Predicting Points + Rebounds + Assists combined totals")
    print("=" * 60)
    
    print("📊 Loading player data...")
    
    # Load the player statistics dataset
    df = pd.read_csv('data/PlayerStatistics.csv', low_memory=False)
    df = df.dropna(subset=['points', 'reboundsTotal', 'assists'])
    
    # Convert date if available
    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'])
    
    print(f"✅ Loaded {len(df)} player performances with {len(df.columns)} features")
    
    # Use recent data only for faster training (last 200k records)
    if len(df) > 200000:
        print(f"📊 Using recent {200000} records for faster training...")
        df = df.tail(200000).reset_index(drop=True)
    
    return df

def engineer_combo_features(df):
    """Create features specifically for combo props prediction"""
    
    print(f"\n🔧 ENGINEERING COMBO PROPS FEATURES")
    print("Creating features for PRA (Points + Rebounds + Assists) prediction")
    
    # Calculate PRA total for each game
    df['pra_total'] = df['points'] + df['reboundsTotal'] + df['assists']
    print(f"✅ Created PRA total column")
    
    # Sort by player and date for rolling calculations
    if 'gameDate' in df.columns:
        df = df.sort_values(['firstName', 'lastName', 'gameDate']).reset_index(drop=True)
    else:
        df = df.sort_values(['firstName', 'lastName']).reset_index(drop=True)
    
    print("🔄 Calculating rolling averages...")
    
    # Calculate rolling stats using transform to avoid index issues
    df['points_last_5'] = df.groupby(['firstName', 'lastName'])['points'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    df['points_last_10'] = df.groupby(['firstName', 'lastName'])['points'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1)
    )
    df['rebounds_last_5'] = df.groupby(['firstName', 'lastName'])['reboundsTotal'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    df['rebounds_last_10'] = df.groupby(['firstName', 'lastName'])['reboundsTotal'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1)
    )
    df['assists_last_5'] = df.groupby(['firstName', 'lastName'])['assists'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    df['assists_last_10'] = df.groupby(['firstName', 'lastName'])['assists'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1)
    )
    
    # Rolling averages for PRA total
    df['pra_last_5'] = df.groupby(['firstName', 'lastName'])['pra_total'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    df['pra_last_10'] = df.groupby(['firstName', 'lastName'])['pra_total'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1)
    )
    df['pra_last_15'] = df.groupby(['firstName', 'lastName'])['pra_total'].transform(
        lambda x: x.rolling(15, min_periods=1).mean().shift(1)
    )
    
    print("✅ Rolling averages calculated")
    
    # Recent form trends
    df['pra_trend_3'] = df.groupby(['firstName', 'lastName'])['pra_total'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1) - x.rolling(6, min_periods=3).mean().shift(4)
    )
    
    # Individual stat trends
    df['points_trend_3'] = df.groupby(['firstName', 'lastName'])['points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1) - x.rolling(6, min_periods=3).mean().shift(4)
    )
    df['rebounds_trend_3'] = df.groupby(['firstName', 'lastName'])['reboundsTotal'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1) - x.rolling(6, min_periods=3).mean().shift(4)
    )
    df['assists_trend_3'] = df.groupby(['firstName', 'lastName'])['assists'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1) - x.rolling(6, min_periods=3).mean().shift(4)
    )
    
    print("✅ Trend features calculated")
    
    # Playing time features
    if 'numMinutes' in df.columns:
        df['minutes_last_5'] = df.groupby(['firstName', 'lastName'])['numMinutes'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        df['minutes_last_10'] = df.groupby(['firstName', 'lastName'])['numMinutes'].transform(
            lambda x: x.rolling(10, min_periods=1).mean().shift(1)
        )
        print("✅ Minutes played features")
    
    # Usage features
    if 'fieldGoalsAttempted' in df.columns:
        df['fga_last_5'] = df.groupby(['firstName', 'lastName'])['fieldGoalsAttempted'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        print("✅ Usage features")
    
    # Efficiency features
    if 'fieldGoalsMade' in df.columns and 'fieldGoalsAttempted' in df.columns:
        df['fg_pct'] = df['fieldGoalsMade'] / df['fieldGoalsAttempted'].replace(0, 1)
        df['fg_pct_last_5'] = df.groupby(['firstName', 'lastName'])['fg_pct'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        print("✅ Shooting efficiency features")
    
    # Consistency metrics
    df['pra_std_5'] = df.groupby(['firstName', 'lastName'])['pra_total'].transform(
        lambda x: x.rolling(5, min_periods=2).std().shift(1)
    )
    df['points_std_5'] = df.groupby(['firstName', 'lastName'])['points'].transform(
        lambda x: x.rolling(5, min_periods=2).std().shift(1)
    )
    
    print("✅ Consistency metrics")
    
    # Position-based features (estimated from stats)
    df['position_big'] = ((df['rebounds_last_5'] > 8) & (df['assists_last_5'] < 3)).astype(int)
    df['position_wing'] = ((df['points_last_5'] > 15) & (df['rebounds_last_5'] > 4) & (df['assists_last_5'] < 6)).astype(int)
    df['position_guard'] = ((df['assists_last_5'] > 4) & (df['rebounds_last_5'] < 6)).astype(int)
    df['position_playmaker'] = (df['assists_last_5'] > 6).astype(int)
    
    print("✅ Position-based features")
    
    # Season-level averages (for players with enough games)
    player_season_stats = df.groupby(['firstName', 'lastName']).agg({
        'points': 'mean',
        'reboundsTotal': 'mean', 
        'assists': 'mean',
        'pra_total': 'mean',
        'numMinutes': 'mean' if 'numMinutes' in df.columns else lambda x: 25
    }).add_suffix('_season_avg')
    
    df = df.merge(player_season_stats, left_on=['firstName', 'lastName'], right_index=True, how='left')
    print("✅ Season averages")
    
    # Home/Away differential (if available)
    if 'home' in df.columns:
        # Calculate home vs away performance for each player
        home_stats = df[df['home'] == 1].groupby(['firstName', 'lastName'])['pra_total'].mean()
        away_stats = df[df['home'] == 0].groupby(['firstName', 'lastName'])['pra_total'].mean()
        home_away_diff = (home_stats - away_stats).fillna(0)
        
        df = df.merge(home_away_diff.rename('pra_home_away_diff'), 
                     left_on=['firstName', 'lastName'], right_index=True, how='left')
        df['pra_home_away_diff'] = df['pra_home_away_diff'].fillna(0)
        print("✅ Home/Away differential")
    
    # Composite PRA efficiency score
    df['pra_efficiency'] = (
        df['pra_last_10'] / df.get('minutes_last_10', 30) * 36  # Per 36 minutes
    )
    
    print("✅ PRA efficiency score")
    
    return df

def create_pra_targets(df):
    """Create PRA combo targets from game results"""
    
    print(f"\n🎯 CREATING PRA TARGETS")
    
    # PRA total is already calculated
    print(f"✅ Using actual PRA total: points + rebounds + assists")
    
    # Create Over/Under targets for common PRA lines
    common_pra_lines = [15.5, 20.5, 25.5, 30.5, 35.5, 40.5, 45.5, 50.5, 55.5, 60.5]
    
    print(f"📊 Creating Over/Under targets for {len(common_pra_lines)} PRA lines")
    
    for pra_line in common_pra_lines:
        df[f'pra_over_{pra_line}'] = (df['pra_total'] > pra_line).astype(int)
    
    # Show PRA distribution
    print(f"\n📈 PRA DISTRIBUTION:")
    print(f"   Average PRA: {df['pra_total'].mean():.1f}")
    print(f"   Std deviation: {df['pra_total'].std():.1f}")
    print(f"   High PRA (40+ total): {(df['pra_total'] > 40).mean():.1%}")
    print(f"   Low PRA (20- total): {(df['pra_total'] < 20).mean():.1%}")
    
    # Show breakdown by estimated position
    print(f"\n📊 PRA BY POSITION:")
    for pos_name, pos_col in [('Guards', 'position_guard'), ('Wings', 'position_wing'), 
                              ('Bigs', 'position_big'), ('Playmakers', 'position_playmaker')]:
        if pos_col in df.columns:
            pos_avg = df[df[pos_col] == 1]['pra_total'].mean()
            print(f"   {pos_name}: {pos_avg:.1f} average PRA")
    
    return df

def train_pra_models(df):
    """Train both regression and classification models for PRA"""
    
    print(f"\n🧠 TRAINING PRA COMBO MODELS")
    print("Both regression (exact PRA) and classification (Over/Under lines)")
    
    # Define feature columns
    feature_cols = [
        'points_last_5', 'points_last_10', 'rebounds_last_5', 'rebounds_last_10',
        'assists_last_5', 'assists_last_10', 'pra_last_5', 'pra_last_10', 'pra_last_15',
        'pra_trend_3', 'points_trend_3', 'rebounds_trend_3', 'assists_trend_3',
        'minutes_last_5', 'minutes_last_10', 'fga_last_5', 'fg_pct_last_5',
        'pra_std_5', 'points_std_5', 'position_big', 'position_wing', 'position_guard',
        'position_playmaker', 'points_season_avg', 'reboundsTotal_season_avg',
        'assists_season_avg', 'pra_total_season_avg', 'pra_home_away_diff',
        'pra_efficiency'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"📊 Using {len(available_features)} features for training")
    
    # Remove rows with insufficient data (first few games per player)
    df_clean = df.dropna(subset=['pra_last_5', 'pra_total'])
    
    # Prepare data and clean infinite values
    X = df_clean[available_features].fillna(0)
    
    # Replace infinite values with reasonable bounds
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Cap extreme values (3 standard deviations)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            mean_val = X[col].mean()
            std_val = X[col].std()
            if std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                X[col] = X[col].clip(lower_bound, upper_bound)
    
    # Time-based split (80% train, 20% test)
    split_idx = int(len(df_clean) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
    
    models = {}
    
    # 1. PRA REGRESSION MODEL (predict exact PRA total)
    print(f"\n🎯 Training PRA Regression Model...")
    
    y_pra = df_clean['pra_total']
    y_train_pra, y_test_pra = y_pra.iloc[:split_idx], y_pra.iloc[split_idx:]
    
    # Try both Random Forest and Linear Regression
    rf_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        random_state=42
    )
    
    rf_regressor.fit(X_train, y_train_pra)
    y_pred_pra_rf = rf_regressor.predict(X_test)
    mae_rf = mean_absolute_error(y_test_pra, y_pred_pra_rf)
    
    # Linear regression for comparison
    lr_regressor = LinearRegression()
    lr_regressor.fit(X_train, y_train_pra)
    y_pred_pra_lr = lr_regressor.predict(X_test)
    mae_lr = mean_absolute_error(y_test_pra, y_pred_pra_lr)
    
    print(f"   Random Forest MAE: {mae_rf:.2f} PRA")
    print(f"   Linear Regression MAE: {mae_lr:.2f} PRA")
    
    # Choose better model
    if mae_rf < mae_lr:
        models['pra_regressor'] = rf_regressor
        print(f"   ✅ Using Random Forest (better MAE)")
        pra_mae = mae_rf
    else:
        models['pra_regressor'] = lr_regressor
        print(f"   ✅ Using Linear Regression (better MAE)")
        pra_mae = mae_lr
    
    # 2. PRA CLASSIFICATION MODELS
    print(f"\n🎯 Training PRA Over/Under Classification Models...")
    
    common_pra_lines = [20.5, 25.5, 30.5, 35.5, 40.5, 45.5, 50.5]
    pra_accuracies = {}
    
    for pra_line in common_pra_lines:
        col_name = f'pra_over_{pra_line}'
        if col_name in df_clean.columns:
            print(f"   Training {pra_line} PRA model...")
            
            y_over = df_clean[col_name]
            y_train_over, y_test_over = y_over.iloc[:split_idx], y_over.iloc[split_idx:]
            
            # Skip if not enough examples
            if y_train_over.sum() < 50 or (len(y_train_over) - y_train_over.sum()) < 50:
                print(f"     Skipping {pra_line} - insufficient examples")
                continue
            
            pra_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=15,
                class_weight='balanced',
                random_state=42
            )
            
            pra_classifier.fit(X_train, y_train_over)
            y_pred_over = pra_classifier.predict(X_test)
            accuracy = accuracy_score(y_test_over, y_pred_over)
            
            models[f'pra_over_{pra_line}_classifier'] = pra_classifier
            pra_accuracies[pra_line] = accuracy
            
            print(f"     PRA Over {pra_line} accuracy: {accuracy:.1%}")
    
    # 3. FEATURE IMPORTANCE
    print(f"\n🔝 FEATURE IMPORTANCE (PRA Prediction):")
    if hasattr(models['pra_regressor'], 'feature_importances_'):
        feature_importance = list(zip(available_features, models['pra_regressor'].feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:15]):  # Top 15
            print(f"   {i+1:2d}. {feature:<25} {importance:.3f}")
    
    return models, available_features, pra_mae, pra_accuracies

def save_pra_models(models, feature_cols):
    """Save the PRA combo models"""
    
    print(f"\n💾 SAVING PRA COMBO MODELS")
    
    # Save each model
    for model_name, model in models.items():
        model_path = f"models/combo_pra_{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Saved {model_name}")
    
    # Save feature list
    features_path = "models/combo_pra_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    print(f"📋 Saved feature list: {len(feature_cols)} features")

def test_pra_predictions(models, feature_cols):
    """Test PRA prediction on sample players"""
    
    print(f"\n🧪 TESTING PRA PREDICTIONS")
    print("Sample player scenarios")
    
    # High PRA player (star)
    star_scenario = {
        'points_last_5': 28.5,           # High scorer
        'points_last_10': 27.8,
        'rebounds_last_5': 8.2,          # Good rebounds
        'rebounds_last_10': 8.0,
        'assists_last_5': 7.5,           # Good playmaking
        'assists_last_10': 7.2,
        'pra_last_5': 44.2,              # High PRA
        'pra_last_10': 43.0,
        'pra_last_15': 42.5,
        'pra_trend_3': 2.1,              # Recent uptick
        'points_trend_3': 1.5,
        'rebounds_trend_3': 0.3,
        'assists_trend_3': 0.3,
        'minutes_last_5': 36.2,          # High minutes
        'minutes_last_10': 35.8,
        'fga_last_5': 22.5,              # High usage
        'fg_pct_last_5': 0.485,
        'pra_std_5': 6.2,                # Some variance
        'points_std_5': 4.1,
        'position_big': 0,
        'position_wing': 1,
        'position_guard': 0,
        'position_playmaker': 1,
        'points_season_avg': 27.2,
        'reboundsTotal_season_avg': 7.8,
        'assists_season_avg': 7.0,
        'pra_total_season_avg': 42.0,
        'pra_home_away_diff': 2.5,       # Better at home
        'pra_efficiency': 42.8
    }
    
    # Role player scenario
    role_scenario = {
        'points_last_5': 14.2,           # Moderate scorer
        'points_last_10': 13.8,
        'rebounds_last_5': 5.1,          # Decent rebounds
        'rebounds_last_10': 5.0,
        'assists_last_5': 3.2,           # Some assists
        'assists_last_10': 3.1,
        'pra_last_5': 22.5,              # Moderate PRA
        'pra_last_10': 21.9,
        'pra_last_15': 21.5,
        'pra_trend_3': -0.8,             # Recent dip
        'points_trend_3': -0.5,
        'rebounds_trend_3': -0.2,
        'assists_trend_3': -0.1,
        'minutes_last_5': 24.8,          # Moderate minutes
        'minutes_last_10': 25.2,
        'fga_last_5': 12.1,              # Lower usage
        'fg_pct_last_5': 0.445,
        'pra_std_5': 4.8,                # Consistent
        'points_std_5': 3.2,
        'position_big': 0,
        'position_wing': 1,
        'position_guard': 0,
        'position_playmaker': 0,
        'points_season_avg': 13.5,
        'reboundsTotal_season_avg': 4.9,
        'assists_season_avg': 3.0,
        'pra_total_season_avg': 21.4,
        'pra_home_away_diff': 1.2,
        'pra_efficiency': 31.2
    }
    
    scenarios = [
        ("Star Player", star_scenario),
        ("Role Player", role_scenario)
    ]
    
    for scenario_name, scenario_data in scenarios:
        print(f"\n📊 {scenario_name}:")
        
        # Create DataFrame
        sample_df = pd.DataFrame([{f: scenario_data.get(f, 0) for f in feature_cols}])
        
        # Predict PRA total
        if 'pra_regressor' in models:
            predicted_pra = models['pra_regressor'].predict(sample_df)[0]
            print(f"   Predicted PRA: {predicted_pra:.1f}")
            
            # Test common PRA lines
            for pra_line in [25.5, 30.5, 35.5, 40.5, 45.5]:
                model_key = f'pra_over_{pra_line}_classifier'
                if model_key in models:
                    over_prob = models[model_key].predict_proba(sample_df)[0][1]
                    recommendation = "OVER" if over_prob > 0.52 else "UNDER"
                    confidence = max(over_prob, 1-over_prob)
                    
                    print(f"   {pra_line:4.1f} PRA: {recommendation} ({confidence:.1%} confidence)")

def main():
    """Build complete PRA combo models"""
    
    # Load data
    df = load_player_data()
    
    # Engineer features
    df = engineer_combo_features(df)
    
    # Create targets
    df = create_pra_targets(df)
    
    # Train models
    models, feature_cols, pra_mae, pra_accuracies = train_pra_models(df)
    
    # Save models
    save_pra_models(models, feature_cols)
    
    # Test predictions
    test_pra_predictions(models, feature_cols)
    
    # Final summary
    print(f"\n🎉 PRA COMBO MODELS COMPLETE!")
    print(f"📊 PRA prediction MAE: {pra_mae:.2f}")
    print(f"🎯 PRA Over/Under accuracies: {len(pra_accuracies)} lines trained")
    for pra_line, acc in pra_accuracies.items():
        print(f"   Over {pra_line:4.1f}: {acc:.1%} accuracy")
    
    print(f"\n🏀 ENHANCED PLAYER PROPS NOW AVAILABLE:")
    print(f"   ✅ Points Props")
    print(f"   ✅ Rebounds Props") 
    print(f"   ✅ Assists Props")
    print(f"   ✅ PRA Combo Props ({pra_mae:.1f} MAE)")
    
    return models, feature_cols

if __name__ == "__main__":
    main()