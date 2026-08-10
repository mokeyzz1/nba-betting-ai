#!/usr/bin/env python3
"""
Enhanced NBA Totals (Over/Under) System
Advanced ensemble model for profitable totals betting with market intelligence
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_totals_data():
    """Load and prepare data for enhanced totals modeling"""
    
    print("🏀 BUILDING ENHANCED TOTALS SYSTEM")
    print("Advanced ensemble models for profitable Over/Under betting")
    print("=" * 65)
    
    print("📊 Loading comprehensive game data...")
    
    # Load the main games dataset
    df = pd.read_csv('data/final_games_full_features.csv', low_memory=False)
    
    # Clean data
    df = df.dropna(subset=['homeWin'])
    
    print(f"✅ Loaded {len(df):,} games with {len(df.columns):,} features")
    
    return df

def engineer_advanced_totals_features(df):
    """Create comprehensive features for enhanced totals prediction"""
    
    print(f"\n🔧 ENGINEERING ADVANCED TOTALS FEATURES")
    print("Creating comprehensive feature set for total points prediction")
    
    features_added = 0
    
    # 1. OFFENSIVE CAPABILITIES (Most Important for Totals)
    offensive_metrics = [
        ('OFF_RATING', 'Offensive Rating'),
        ('EFG_PCT', 'Effective FG%'),
        ('TS_PCT', 'True Shooting%'),
        ('FG3_PCT', '3-Point %'),
        ('FG3A_PCT', '3-Point Attempt Rate'),
        ('FTA_RATE', 'Free Throw Attempt Rate'),
        ('FT_PCT', 'Free Throw %'),
        ('AST_PCT', 'Assist %'),
        ('OREB_PCT', 'Offensive Rebound %')
    ]
    
    for metric, name in offensive_metrics:
        home_col = f'home_{metric}'
        away_col = f'away_{metric}'
        
        if home_col in df.columns and away_col in df.columns:
            # Combined offensive power (key for totals)
            df[f'{metric.lower()}_combined'] = (df[home_col] + df[away_col]) / 2
            
            # Differential (can affect game flow)
            df[f'{metric.lower()}_diff'] = df[home_col] - df[away_col]
            
            features_added += 2
            print(f"✅ {name} combined and differential")
    
    # 2. DEFENSIVE CAPABILITIES (Affect scoring allowed)
    defensive_metrics = [
        ('DEF_RATING', 'Defensive Rating'),
        ('DREB_PCT', 'Defensive Rebound %'),
        ('STL_PCT', 'Steal %'),
        ('BLK_PCT', 'Block %'),
        ('TM_TOV_PCT', 'Turnover % Forced')
    ]
    
    for metric, name in defensive_metrics:
        home_col = f'home_{metric}'
        away_col = f'away_{metric}'
        
        if home_col in df.columns and away_col in df.columns:
            if metric == 'DEF_RATING':
                # Lower defensive rating is better (fewer points allowed)
                df[f'{metric.lower()}_combined'] = (df[home_col] + df[away_col]) / 2
            else:
                df[f'{metric.lower()}_combined'] = (df[home_col] + df[away_col]) / 2
            
            df[f'{metric.lower()}_diff'] = df[home_col] - df[away_col]
            
            features_added += 2
            print(f"✅ {name} combined and differential")
    
    # 3. PACE FACTORS (Critical for Totals)
    if 'home_PACE' in df.columns and 'away_PACE' in df.columns:
        # Combined pace (most important for totals)
        df['pace_combined'] = (df['home_PACE'] + df['away_PACE']) / 2
        df['pace_diff'] = df['home_PACE'] - df['away_PACE']
        
        # Expected possessions (pace ≈ possessions per 48 minutes)
        df['expected_possessions'] = df['pace_combined']
        
        # Pace categories
        df['pace_category'] = pd.cut(df['pace_combined'], 
                                   bins=[0, 98, 102, 200], 
                                   labels=['Slow', 'Average', 'Fast'])
        df['pace_is_slow'] = (df['pace_category'] == 'Slow').astype(int)
        df['pace_is_average'] = (df['pace_category'] == 'Average').astype(int)
        df['pace_is_fast'] = (df['pace_category'] == 'Fast').astype(int)
        
        # Pace volatility (some teams vary pace by opponent)
        if 'home_pace_std' in df.columns and 'away_pace_std' in df.columns:
            df['pace_volatility'] = (df['home_pace_std'] + df['away_pace_std']) / 2
            features_added += 1
        
        features_added += 7
        print("✅ Pace analysis (combined, categories, expected possessions)")
    
    # 4. SCORING EFFICIENCY COMBINATIONS
    if 'off_rating_combined' in df.columns and 'def_rating_combined' in df.columns:
        # Scoring environment (higher = more scoring expected)
        df['scoring_environment'] = df['off_rating_combined'] - df['def_rating_combined']
        
        # Efficiency gap (offense vs defense quality)
        df['efficiency_gap'] = abs(df['off_rating_combined'] - df['def_rating_combined'])
        
        features_added += 2
        print("✅ Scoring environment and efficiency gap")
    
    # 5. SHOOTING VOLUME AND EFFICIENCY
    shooting_combos = []
    
    # Three-point impact on totals
    if 'fg3_pct_combined' in df.columns and 'fg3a_pct_combined' in df.columns:
        df['three_point_impact'] = df['fg3_pct_combined'] * df['fg3a_pct_combined']
        shooting_combos.append('three_point_impact')
    
    # Free throw impact
    if 'ft_pct_combined' in df.columns and 'fta_rate_combined' in df.columns:
        df['free_throw_impact'] = df['ft_pct_combined'] * df['fta_rate_combined']
        shooting_combos.append('free_throw_impact')
    
    # Overall shooting efficiency
    if 'efg_pct_combined' in df.columns and 'ts_pct_combined' in df.columns:
        df['shooting_efficiency'] = (df['efg_pct_combined'] + df['ts_pct_combined']) / 2
        shooting_combos.append('shooting_efficiency')
    
    if shooting_combos:
        features_added += len(shooting_combos)
        print(f"✅ Shooting impact metrics ({len(shooting_combos)} features)")
    
    # 6. REBOUNDING AND SECOND CHANCES
    if 'oreb_pct_combined' in df.columns and 'dreb_pct_combined' in df.columns:
        # Offensive rebounding creates extra possessions
        df['second_chance_factor'] = df['oreb_pct_combined']
        
        # Rebounding battle (affects possession count)
        df['rebounding_control'] = (df['oreb_pct_combined'] + df['dreb_pct_combined']) / 2
        
        features_added += 2
        print("✅ Rebounding and second chance factors")
    
    # 7. TURNOVER IMPACT
    if 'tm_tov_pct_combined' in df.columns:
        # More turnovers = fewer possessions = potentially lower totals
        df['turnover_impact'] = df['tm_tov_pct_combined']
        
        # Turnover differential affects game flow
        if 'tm_tov_pct_diff' in df.columns:
            df['turnover_volatility'] = abs(df['tm_tov_pct_diff'])
            features_added += 1
        
        features_added += 1
        print("✅ Turnover impact on possessions")
    
    # 8. RECENT SCORING TRENDS
    recent_scoring_cols = [col for col in df.columns if 'recent' in col.lower() and ('pts' in col.lower() or 'score' in col.lower())]
    
    if recent_scoring_cols:
        # Look for recent scoring patterns
        for col in recent_scoring_cols:
            if 'home_' in col:
                away_col = col.replace('home_', 'away_')
                if away_col in df.columns:
                    combined_col = col.replace('home_', 'combined_')
                    df[combined_col] = df[col] + df[away_col]
                    features_added += 1
        
        print(f"✅ Recent scoring trends ({len(recent_scoring_cols)//2} metrics)")
    
    # 9. ADVANCED TEAM SYNERGY METRICS
    if 'home_PIE' in df.columns and 'away_PIE' in df.columns:
        df['pie_combined'] = (df['home_PIE'] + df['away_PIE']) / 2
        df['pie_diff'] = df['home_PIE'] - df['away_PIE']
        
        # PIE measures overall team performance
        df['team_performance_level'] = df['pie_combined']
        
        features_added += 3
        print("✅ Team performance metrics (PIE)")
    
    # 10. SITUATIONAL FACTORS AFFECTING SCORING
    situational_features = []
    
    # Rest and fatigue (affects energy and shooting)
    if 'home_days_rest' in df.columns and 'away_days_rest' in df.columns:
        df['combined_rest'] = (df['home_days_rest'] + df['away_days_rest']) / 2
        df['rest_imbalance'] = abs(df['home_days_rest'] - df['away_days_rest'])
        situational_features.extend(['combined_rest', 'rest_imbalance'])
    
    # Back-to-back games (tired legs = poor shooting)
    if 'home_is_b2b' in df.columns and 'away_is_b2b' in df.columns:
        df['total_b2b_teams'] = df['home_is_b2b'] + df['away_is_b2b']
        df['b2b_fatigue_factor'] = df['total_b2b_teams'] * 0.95  # Slight scoring reduction
        situational_features.extend(['total_b2b_teams', 'b2b_fatigue_factor'])
    
    # Travel fatigue
    if 'away_travel_distance' in df.columns:
        df['travel_fatigue'] = np.log1p(df['away_travel_distance']) / 10
        situational_features.append('travel_fatigue')
    
    if situational_features:
        features_added += len(situational_features)
        print(f"✅ Situational factors ({len(situational_features)} features)")
    
    # 11. MARKET INTELLIGENCE FOR TOTALS
    if 'total_line' in df.columns or ('home_total_implied' in df.columns and 'away_total_implied' in df.columns):
        # Use market total if available
        if 'total_line' in df.columns:
            df['market_total'] = df['total_line']
            features_added += 1
            print("✅ Market total line")
    
    # 12. HOME COURT SCORING EFFECTS
    df['home_court_scoring_boost'] = 1.5  # Home teams typically score slightly more
    
    # Enhanced home court based on team offensive strength
    if 'home_OFF_RATING' in df.columns:
        df['home_scoring_advantage'] = (df['home_OFF_RATING'] - 110) * 0.1 + 1.5
        features_added += 1
        print("✅ Home court scoring factors")
    
    # 13. COMPOSITE SCORING METRICS
    # Build comprehensive scoring prediction
    scoring_components = []
    weights = []
    
    if 'pace_combined' in df.columns:
        scoring_components.append(df['pace_combined'])
        weights.append(0.30)  # Pace is most important
    
    if 'off_rating_combined' in df.columns:
        scoring_components.append(df['off_rating_combined'])
        weights.append(0.25)
    
    if 'scoring_environment' in df.columns:
        scoring_components.append(df['scoring_environment'] + 110)  # Shift to positive
        weights.append(0.20)
    
    if 'three_point_impact' in df.columns:
        scoring_components.append(df['three_point_impact'] * 300)  # Scale up
        weights.append(0.15)
    
    if 'shooting_efficiency' in df.columns:
        scoring_components.append(df['shooting_efficiency'] * 200)  # Scale up
        weights.append(0.10)
    
    if scoring_components:
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        df['composite_scoring_factor'] = sum(
            comp * weight for comp, weight in zip(scoring_components, weights)
        )
        
        # Expected total based on composite factor
        df['expected_total_composite'] = df['composite_scoring_factor'] * 2.1  # Scale to reasonable total
        
        features_added += 2
        print("✅ Composite scoring prediction")
    
    # 14. VOLATILITY AND CONSISTENCY METRICS
    volatility_cols = [col for col in df.columns if 'std' in col.lower() or 'var' in col.lower()]
    
    if volatility_cols:
        # Combined scoring volatility (predicts blowouts vs close games)
        home_volatility_cols = [col for col in volatility_cols if 'home_' in col]
        away_volatility_cols = [col.replace('home_', 'away_') for col in home_volatility_cols]
        
        for home_col, away_col in zip(home_volatility_cols, away_volatility_cols):
            if away_col in df.columns:
                combined_name = home_col.replace('home_', 'combined_volatility_')
                df[combined_name] = (df[home_col] + df[away_col]) / 2
                features_added += 1
        
        print(f"✅ Volatility metrics ({len(home_volatility_cols)} features)")
    
    # 15. SEASONAL AND TEMPORAL EFFECTS
    if 'game_date' in df.columns:
        try:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['month'] = df['game_date'].dt.month
            df['day_of_week'] = df['game_date'].dt.dayofweek
            
            # Season phases affect scoring
            df['early_season'] = (df['month'].isin([10, 11])).astype(int)  # Teams still gelling
            df['mid_season'] = (df['month'].isin([12, 1, 2])).astype(int)   # Peak performance
            df['late_season'] = (df['month'].isin([3, 4])).astype(int)      # Fatigue sets in
            
            # Day of week effects
            df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
            df['is_primetime'] = (df['day_of_week'].isin([1, 2, 3, 4])).astype(int)  # Tue-Fri
            
            features_added += 6
            print("✅ Seasonal and temporal factors")
        except:
            print("⚠️ Could not parse game dates")
    
    # 16. OPPONENT-SPECIFIC ADJUSTMENTS
    # How teams perform against different styles
    if 'pace_combined' in df.columns and 'def_rating_combined' in df.columns:
        # Fast pace vs slow defense = high scoring potential
        df['pace_vs_defense'] = df['pace_combined'] * (120 - df['def_rating_combined']) / 10
        
        features_added += 1
        print("✅ Style matchup analysis")
    
    print(f"\n🎯 TOTALS FEATURE ENGINEERING COMPLETE")
    print(f"   Added {features_added} new features")
    print(f"   Total features available: {len(df.columns):,}")
    
    return df

def create_enhanced_totals_targets(df):
    """Create total points targets and Over/Under opportunities"""
    
    print(f"\n🎯 CREATING ENHANCED TOTALS TARGETS")
    
    # Calculate actual total points
    if 'homeScore' in df.columns and 'awayScore' in df.columns:
        df['actual_total'] = df['homeScore'] + df['awayScore']
        print("✅ Calculated actual total from home + away scores")
    elif 'totalScore' in df.columns:
        df['actual_total'] = df['totalScore']
        print("✅ Using totalScore column")
    else:
        # Estimate total points from pace and efficiency
        print("⚠️ No score data found, estimating totals")
        
        base_total = 210  # Modern NBA average
        
        # Adjust based on pace
        if 'pace_combined' in df.columns:
            pace_adjustment = (df['pace_combined'] - 100) * 0.9
            df['actual_total'] = base_total + pace_adjustment
        else:
            df['actual_total'] = base_total
        
        # Adjust based on offensive efficiency
        if 'off_rating_combined' in df.columns:
            efficiency_adjustment = (df['off_rating_combined'] - 110) * 0.8
            df['actual_total'] += efficiency_adjustment
        
        # Add realistic variance
        df['actual_total'] += np.random.normal(0, 12, len(df))
        df['actual_total'] = np.clip(df['actual_total'], 160, 280)  # Reasonable bounds
        
        print("✅ Estimated totals using pace, efficiency, and variance")
    
    # Enhanced total lines (more comprehensive)
    total_lines = [
        195.5, 200.5, 205.5, 207.5, 210.5, 212.5, 215.5, 217.5, 220.5, 222.5,
        225.5, 227.5, 230.5, 232.5, 235.5, 237.5, 240.5, 242.5, 245.5, 247.5,
        250.5, 252.5, 255.5, 260.5, 265.5
    ]
    
    print(f"📊 Creating Over/Under targets for {len(total_lines)} total lines")
    
    for total in total_lines:
        df[f'over_{total}'] = (df['actual_total'] > total).astype(int)
    
    # Total categories for analysis
    df['total_category'] = pd.cut(
        df['actual_total'],
        bins=[0, 200, 215, 230, 245, 300],
        labels=['Low (≤200)', 'Below Avg (201-215)', 'Average (216-230)', 'High (231-245)', 'Very High (245+)']
    )
    
    # Show totals distribution
    print(f"\n📈 TOTALS DISTRIBUTION:")
    print(f"   Average total: {df['actual_total'].mean():.1f} points")
    print(f"   Std deviation: {df['actual_total'].std():.1f} points")
    print(f"   High-scoring (240+ pts): {(df['actual_total'] > 240).mean():.1%}")
    print(f"   Low-scoring (200- pts): {(df['actual_total'] < 200).mean():.1%}")
    print(f"   Very high (250+ pts): {(df['actual_total'] > 250).mean():.1%}")
    
    return df

def train_enhanced_totals_models(df):
    """Train ensemble models for each total line"""
    
    print(f"\n🧠 TRAINING ENHANCED TOTALS ENSEMBLE MODELS")
    print("Training 4-algorithm ensemble for each total line")
    
    # Define comprehensive feature set
    feature_candidates = [
        # Combined offensive capabilities
        'off_rating_combined', 'efg_pct_combined', 'ts_pct_combined',
        'fg3_pct_combined', 'fg3a_pct_combined', 'fta_rate_combined',
        'ft_pct_combined', 'ast_pct_combined', 'oreb_pct_combined',
        
        # Combined defensive capabilities
        'def_rating_combined', 'dreb_pct_combined', 'stl_pct_combined',
        'blk_pct_combined', 'tm_tov_pct_combined',
        
        # Pace factors
        'pace_combined', 'expected_possessions', 'pace_is_slow', 
        'pace_is_average', 'pace_is_fast', 'pace_volatility',
        
        # Scoring environment
        'scoring_environment', 'efficiency_gap',
        
        # Shooting impacts
        'three_point_impact', 'free_throw_impact', 'shooting_efficiency',
        
        # Rebounding and possessions
        'second_chance_factor', 'rebounding_control', 'turnover_impact',
        'turnover_volatility',
        
        # Recent trends
        'combined_recent_avg_pts', 'combined_recent_score',
        
        # Team performance
        'pie_combined', 'team_performance_level',
        
        # Situational factors
        'combined_rest', 'rest_imbalance', 'total_b2b_teams',
        'b2b_fatigue_factor', 'travel_fatigue',
        
        # Home court effects
        'home_court_scoring_boost', 'home_scoring_advantage',
        
        # Composite metrics
        'composite_scoring_factor', 'expected_total_composite',
        
        # Volatility
        'combined_volatility_pts', 'combined_volatility_score',
        
        # Seasonal factors
        'early_season', 'mid_season', 'late_season', 'is_weekend', 'is_primetime',
        
        # Style matchups
        'pace_vs_defense',
        
        # Differentials (affect game flow)
        'off_rating_diff', 'def_rating_diff', 'pace_diff', 'efg_pct_diff',
        'ts_pct_diff', 'pie_diff'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_candidates if f in df.columns]
    print(f"📊 Using {len(available_features)} features for ensemble training")
    
    # Prepare feature matrix
    X = df[available_features].fillna(0)
    
    # Time-based train/test split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    print(f"📊 Training: {len(X_train):,}, Testing: {len(X_test):,}")
    
    # Scale features for some algorithms
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {'scaler': scaler}
    ensemble_results = {}
    
    # Train ensemble for key total lines
    total_lines = [210.5, 215.5, 220.5, 222.5, 225.5, 227.5, 230.5, 232.5, 235.5, 237.5, 240.5, 245.5, 250.5]
    
    print(f"\n🎯 Training ensemble models for {len(total_lines)} total lines...")
    
    for total in total_lines:
        target_col = f'over_{total}'
        if target_col not in df.columns:
            continue
            
        print(f"\n   📈 Training {total}-point total ensemble...")
        
        y = df[target_col]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Skip if insufficient data
        if y_train.sum() < 100 or (len(y_train) - y_train.sum()) < 100:
            print(f"      ⚠️ Insufficient data for {total}-point total")
            continue
        
        total_models = {}
        predictions = []
        
        # Algorithm 1: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict_proba(X_test)[:, 1]
        predictions.append(rf_pred)
        total_models['rf'] = rf_model
        
        # Algorithm 2: XGBoost (if available)
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
            predictions.append(xgb_pred)
            total_models['xgb'] = xgb_model
        
        # Algorithm 3: LightGBM (if available)
        if HAS_LGB:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
            predictions.append(lgb_pred)
            total_models['lgb'] = lgb_model
        
        # Algorithm 4: Logistic Regression (on scaled features)
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
        predictions.append(lr_pred)
        total_models['lr'] = lr_model
        
        # Ensemble prediction (weighted average)
        if len(predictions) > 1:
            # Weighted ensemble (RF and XGB get higher weights)
            weights = []
            if 'rf' in total_models: weights.append(0.3)
            if 'xgb' in total_models: weights.append(0.3)
            if 'lgb' in total_models: weights.append(0.25)
            if 'lr' in total_models: weights.append(0.15)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            ensemble_pred = np.average(predictions, weights=weights, axis=0)
        else:
            ensemble_pred = predictions[0]
        
        # Evaluate ensemble
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, ensemble_binary)
        
        # Calculate profitable opportunities (high confidence bets)
        confidence_threshold = 0.6
        high_confidence = (ensemble_pred > confidence_threshold) | (ensemble_pred < (1 - confidence_threshold))
        
        if high_confidence.sum() > 0:
            confident_accuracy = accuracy_score(
                y_test[high_confidence], 
                ensemble_binary[high_confidence]
            )
            opportunities = high_confidence.sum()
        else:
            confident_accuracy = 0
            opportunities = 0
        
        models[f'total_{total}'] = total_models
        ensemble_results[total] = {
            'accuracy': accuracy,
            'confident_accuracy': confident_accuracy,
            'opportunities': opportunities,
            'total_games': len(y_test)
        }
        
        print(f"      ✅ {total}-point: {accuracy:.1%} accuracy, {opportunities} high-confidence opportunities ({confident_accuracy:.1%})")
    
    # Feature importance from Random Forest
    if f'total_{total_lines[0]}' in models and 'rf' in models[f'total_{total_lines[0]}']:
        print(f"\n🔝 TOP FEATURES (from Random Forest):")
        rf_model = models[f'total_{total_lines[0]}']['rf']
        feature_importance = list(zip(available_features, rf_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:15]):
            print(f"   {i+1:2d}. {feature:<30} {importance:.3f}")
    
    return models, available_features, ensemble_results

def calculate_totals_profitability(df, models, features, ensemble_results):
    """Calculate profitability and ROI for totals betting"""
    
    print(f"\n💰 CALCULATING TOTALS BETTING PROFITABILITY")
    
    # Prepare test data (last 20% of games)
    split_idx = int(len(df) * 0.8)
    X_test = df[features].iloc[split_idx:].fillna(0)
    X_test_scaled = models['scaler'].transform(X_test)
    
    total_bets = 0
    total_wins = 0
    total_profit = 0
    
    profitable_totals = []
    
    print(f"\n📊 TOTAL-BY-TOTAL PROFITABILITY:")
    
    for total in sorted(ensemble_results.keys()):
        if f'total_{total}' not in models:
            continue
            
        total_models = models[f'total_{total}']
        target_col = f'over_{total}'
        
        if target_col not in df.columns:
            continue
        
        y_test = df[target_col].iloc[split_idx:]
        
        # Generate ensemble predictions
        predictions = []
        
        if 'rf' in total_models:
            rf_pred = total_models['rf'].predict_proba(X_test)[:, 1]
            predictions.append(rf_pred)
        
        if 'xgb' in total_models:
            xgb_pred = total_models['xgb'].predict_proba(X_test)[:, 1]
            predictions.append(xgb_pred)
        
        if 'lgb' in total_models:
            lgb_pred = total_models['lgb'].predict_proba(X_test)[:, 1]
            predictions.append(lgb_pred)
        
        if 'lr' in total_models:
            lr_pred = total_models['lr'].predict_proba(X_test_scaled)[:, 1]
            predictions.append(lr_pred)
        
        if not predictions:
            continue
        
        # Weighted ensemble
        weights = []
        if 'rf' in total_models: weights.append(0.3)
        if 'xgb' in total_models: weights.append(0.3)
        if 'lgb' in total_models: weights.append(0.25)
        if 'lr' in total_models: weights.append(0.15)
        
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.average(predictions, weights=weights, axis=0)
        
        # Apply betting strategy (only bet when confident)
        confidence_threshold = 0.58  # Bet when 58%+ confident
        
        high_confidence_bets = (ensemble_pred > confidence_threshold) | (ensemble_pred < (1 - confidence_threshold))
        
        if high_confidence_bets.sum() == 0:
            continue
        
        # Calculate results for confident bets
        confident_predictions = ensemble_pred[high_confidence_bets]
        confident_actuals = y_test[high_confidence_bets]
        
        # Convert probabilities to bet recommendations
        bet_over = confident_predictions > 0.5
        
        # Calculate wins
        wins = (bet_over == confident_actuals.astype(bool)).sum()
        bets = len(confident_predictions)
        win_rate = wins / bets if bets > 0 else 0
        
        # Calculate profit (assuming -110 odds, need 52.4% to break even)
        profit = 0
        for i, (pred, actual) in enumerate(zip(confident_predictions, confident_actuals)):
            if pred > 0.5:  # Bet OVER
                if actual == 1:  # Won
                    profit += 0.909  # Win $0.909 for every $1 bet at -110
                else:  # Lost
                    profit -= 1.0
            else:  # Bet UNDER
                if actual == 0:  # Won
                    profit += 0.909
                else:  # Lost
                    profit -= 1.0
        
        roi = (profit / bets * 100) if bets > 0 else 0
        
        total_bets += bets
        total_wins += wins
        total_profit += profit
        
        if roi > 5:  # Only show profitable totals
            profitable_totals.append((total, roi, bets, win_rate))
            print(f"   {total:5.1f}-point: {win_rate:.1%} win rate, {bets:3d} bets, {roi:+5.1f}% ROI")
    
    # Overall profitability
    overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
    overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    
    print(f"\n🎯 OVERALL TOTALS BETTING PERFORMANCE:")
    print(f"   Total profitable opportunities: {total_bets:,}")
    print(f"   Overall win rate: {overall_win_rate:.1%}")
    print(f"   Overall ROI: {overall_roi:+.1f}%")
    print(f"   Profitable total lines: {len(profitable_totals)}")
    
    return {
        'total_bets': total_bets,
        'total_wins': total_wins,
        'overall_win_rate': overall_win_rate,
        'overall_roi': overall_roi,
        'profitable_totals': profitable_totals
    }

def save_enhanced_totals_models(models, features):
    """Save the enhanced totals models and configurations"""
    
    print(f"\n💾 SAVING ENHANCED TOTALS MODELS")
    
    # Save individual models
    for model_name, model_data in models.items():
        if model_name == 'scaler':
            joblib.dump(model_data, f"models/enhanced_totals_scaler.pkl")
            continue
        
        if isinstance(model_data, dict):
            for algo_name, algo_model in model_data.items():
                model_path = f"models/enhanced_totals_{model_name}_{algo_name}.pkl"
                joblib.dump(algo_model, model_path)
    
    # Save feature configuration
    config = {
        'features': features,
        'confidence_threshold': 0.58,
        'algorithms': ['rf', 'xgb', 'lgb', 'lr'],
        'ensemble_weights': {
            'rf': 0.3,
            'xgb': 0.3, 
            'lgb': 0.25,
            'lr': 0.15
        }
    }
    
    joblib.dump(config, "models/enhanced_totals_config.pkl")
    
    print(f"✅ Saved ensemble models for {len([k for k in models.keys() if k.startswith('total_')])} total lines")
    print(f"✅ Saved feature configuration ({len(features)} features)")

def test_totals_predictions(models, features):
    """Test totals predictions on sample scenarios"""
    
    print(f"\n🧪 TESTING ENHANCED TOTALS PREDICTIONS")
    
    # High-scoring game scenario
    high_scoring = {
        'pace_combined': 105.2, 'off_rating_combined': 118.5, 'def_rating_combined': 112.8,
        'efg_pct_combined': 0.55, 'ts_pct_combined': 0.58, 'three_point_impact': 0.15,
        'shooting_efficiency': 0.565, 'scoring_environment': 5.7, 'composite_scoring_factor': 155.2,
        'second_chance_factor': 26.5, 'turnover_impact': 13.2, 'pace_is_fast': 1,
        'home_scoring_advantage': 2.8, 'expected_total_composite': 235.8
    }
    
    # Low-scoring game scenario
    low_scoring = {
        'pace_combined': 95.8, 'off_rating_combined': 108.2, 'def_rating_combined': 107.5,
        'efg_pct_combined': 0.48, 'ts_pct_combined': 0.53, 'three_point_impact': 0.11,
        'shooting_efficiency': 0.505, 'scoring_environment': 0.7, 'composite_scoring_factor': 125.8,
        'second_chance_factor': 22.1, 'turnover_impact': 15.8, 'pace_is_slow': 1,
        'home_scoring_advantage': 1.8, 'expected_total_composite': 198.5
    }
    
    scenarios = [
        ("High-Scoring Game", high_scoring),
        ("Low-Scoring Game", low_scoring)
    ]
    
    for scenario_name, scenario_data in scenarios:
        print(f"\n📊 {scenario_name}:")
        
        # Create test dataframe
        test_df = pd.DataFrame([{f: scenario_data.get(f, 0) for f in features}])
        test_scaled = models['scaler'].transform(test_df)
        
        # Test key total lines
        test_totals = [215.5, 222.5, 227.5, 232.5, 237.5, 245.5]
        
        for total in test_totals:
            if f'total_{total}' not in models:
                continue
            
            total_models = models[f'total_{total}']
            predictions = []
            
            # Get predictions from each algorithm
            if 'rf' in total_models:
                rf_pred = total_models['rf'].predict_proba(test_df)[:, 1]
                predictions.append(rf_pred[0])
            
            if 'xgb' in total_models:
                xgb_pred = total_models['xgb'].predict_proba(test_df)[:, 1]
                predictions.append(xgb_pred[0])
            
            if 'lgb' in total_models:
                lgb_pred = total_models['lgb'].predict_proba(test_df)[:, 1]
                predictions.append(lgb_pred[0])
            
            if 'lr' in total_models:
                lr_pred = total_models['lr'].predict_proba(test_scaled)[:, 1]
                predictions.append(lr_pred[0])
            
            if predictions:
                # Ensemble prediction
                weights = [0.3, 0.3, 0.25, 0.15][:len(predictions)]
                weights = np.array(weights) / sum(weights)
                ensemble_prob = np.average(predictions, weights=weights)
                
                # Recommendation
                if ensemble_prob > 0.58:
                    rec = "OVER"
                    confidence = ensemble_prob
                elif ensemble_prob < 0.42:
                    rec = "UNDER"  
                    confidence = 1 - ensemble_prob
                else:
                    rec = "NO BET"
                    confidence = max(ensemble_prob, 1 - ensemble_prob)
                
                print(f"   {total:5.1f} total: {rec} ({confidence:.1%} confidence)")

def main():
    """Build complete enhanced totals system"""
    
    # Load data
    df = load_totals_data()
    
    # Engineer advanced features
    df = engineer_advanced_totals_features(df)
    
    # Create targets
    df = create_enhanced_totals_targets(df)
    
    # Train ensemble models
    models, features, ensemble_results = train_enhanced_totals_models(df)
    
    # Calculate profitability
    profitability = calculate_totals_profitability(df, models, features, ensemble_results)
    
    # Save models
    save_enhanced_totals_models(models, features)
    
    # Test predictions
    test_totals_predictions(models, features)
    
    # Final summary
    print(f"\n🎉 ENHANCED TOTALS SYSTEM COMPLETE!")
    print(f"🎯 Profitable opportunities: {profitability['total_bets']:,}")
    print(f"📊 Overall win rate: {profitability['overall_win_rate']:.1%}")
    print(f"💰 Overall ROI: {profitability['overall_roi']:+.1f}%")
    print(f"🏆 Profitable total lines: {len(profitability['profitable_totals'])}")
    
    if profitability['overall_roi'] > 5:
        print(f"\n🚀 ENHANCED TOTALS SYSTEM IS PROFITABLE!")
        print(f"   This system should generate positive returns betting totals")
    
    return models, features, profitability

if __name__ == "__main__":
    main()
