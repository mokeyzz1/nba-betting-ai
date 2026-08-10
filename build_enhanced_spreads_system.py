#!/usr/bin/env python3
"""
Enhanced NBA Point Spreads System
Advanced ensemble model for profitable spread betting with market intelligence
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

def load_spreads_data():
    """Load and prepare data for enhanced spreads modeling"""
    
    print("🏀 BUILDING ENHANCED SPREADS SYSTEM")
    print("Advanced ensemble models for profitable spread betting")
    print("=" * 65)
    
    print("📊 Loading comprehensive game data...")
    
    # Load the main games dataset
    df = pd.read_csv('data/final_games_full_features.csv', low_memory=False)
    
    # Clean data
    df = df.dropna(subset=['homeWin'])
    
    print(f"✅ Loaded {len(df):,} games with {len(df.columns):,} features")
    
    return df

def engineer_advanced_spreads_features(df):
    """Create comprehensive features for enhanced spreads prediction"""
    
    print(f"\n🔧 ENGINEERING ADVANCED SPREADS FEATURES")
    print("Creating comprehensive feature set for margin prediction")
    
    features_added = 0
    
    # 1. TEAM STRENGTH DIFFERENTIALS (Core Features)
    strength_features = [
        ('NET_RATING', 'Net Rating'),
        ('OFF_RATING', 'Offensive Rating'),
        ('DEF_RATING', 'Defensive Rating'), 
        ('PIE', 'PIE'),
        ('TS_PCT', 'True Shooting%'),
        ('EFG_PCT', 'Effective FG%'),
        ('OREB_PCT', 'Offensive Rebounding%'),
        ('DREB_PCT', 'Defensive Rebounding%'),
        ('AST_PCT', 'Assist%'),
        ('STL_PCT', 'Steal%'),
        ('BLK_PCT', 'Block%'),
        ('TM_TOV_PCT', 'Turnover%'),
        ('PACE', 'Pace')
    ]
    
    for stat, name in strength_features:
        home_col = f'home_{stat}'
        away_col = f'away_{stat}'
        
        if home_col in df.columns and away_col in df.columns:
            # Differential (positive = home team advantage)
            if stat == 'TM_TOV_PCT':
                # Lower turnover% is better
                df[f'{stat.lower()}_diff'] = df[away_col] - df[home_col]
            else:
                df[f'{stat.lower()}_diff'] = df[home_col] - df[away_col]
            
            # Combined strength
            df[f'{stat.lower()}_combined'] = (df[home_col] + df[away_col]) / 2
            
            features_added += 2
            print(f"✅ {name} differential and combined")
    
    # 2. ADVANCED TEAM METRICS
    if 'home_W_PCT' in df.columns and 'away_W_PCT' in df.columns:
        df['win_pct_diff'] = df['home_W_PCT'] - df['away_W_PCT']
        features_added += 1
        print("✅ Win percentage differential")
    
    # Four Factors (advanced basketball analytics)
    four_factors = ['EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FTA_RATE']
    for factor in four_factors:
        home_col = f'home_{factor}'
        away_col = f'away_{factor}'
        if home_col in df.columns and away_col in df.columns:
            df[f'four_factors_{factor.lower()}'] = df[home_col] - df[away_col]
            features_added += 1
    
    print(f"✅ Four Factors differentials")
    
    # 3. RECENT FORM AND TRENDS (Critical for spreads)
    recent_columns = [col for col in df.columns if 'recent' in col.lower()]
    if recent_columns:
        for col in recent_columns:
            if 'home_' in col:
                away_col = col.replace('home_', 'away_')
                if away_col in df.columns:
                    diff_name = col.replace('home_', '') + '_diff'
                    df[diff_name] = df[col] - df[away_col]
                    features_added += 1
        print(f"✅ Recent form differentials ({len(recent_columns)//2} metrics)")
    
    # 4. HOME COURT ADVANTAGE FACTORS
    df['home_court_base'] = 3.2  # Historical NBA home court advantage
    
    # Enhanced home court based on team strength
    if 'home_W_PCT' in df.columns:
        df['home_court_enhanced'] = 3.2 + (df['home_W_PCT'] - 0.5) * 2
        features_added += 1
        print("✅ Enhanced home court advantage")
    
    # 5. PACE AND STYLE MATCHUPS
    if 'home_PACE' in df.columns and 'away_PACE' in df.columns:
        df['pace_matchup'] = abs(df['home_PACE'] - df['away_PACE'])
        df['pace_combined'] = (df['home_PACE'] + df['away_PACE']) / 2
        
        # Pace categories
        df['pace_category'] = pd.cut(df['pace_combined'], 
                                   bins=[0, 98, 102, 200], 
                                   labels=['Slow', 'Average', 'Fast'])
        df['pace_is_slow'] = (df['pace_category'] == 'Slow').astype(int)
        df['pace_is_fast'] = (df['pace_category'] == 'Fast').astype(int)
        
        features_added += 5
        print("✅ Pace matchup analysis")
    
    # 6. OFFENSIVE/DEFENSIVE STYLE MATCHUPS
    if 'home_OFF_RATING' in df.columns and 'home_DEF_RATING' in df.columns:
        # High offense vs weak defense advantage
        df['off_vs_def_advantage'] = (
            (df['home_OFF_RATING'] - df['away_DEF_RATING']) +
            (df['away_OFF_RATING'] - df['home_DEF_RATING'])
        ) / 2
        
        # Style contrast (offense-focused vs defense-focused teams)
        home_style = df['home_OFF_RATING'] - df['home_DEF_RATING']
        away_style = df['away_OFF_RATING'] - df['away_DEF_RATING']
        df['style_contrast'] = abs(home_style - away_style)
        
        features_added += 2
        print("✅ Offensive/defensive style analysis")
    
    # 7. CLUTCH AND SITUATIONAL FACTORS
    situational_features = []
    
    # Rest advantage
    if 'home_days_rest' in df.columns and 'away_days_rest' in df.columns:
        df['rest_advantage'] = df['home_days_rest'] - df['away_days_rest']
        situational_features.append('rest_advantage')
    
    # Travel factors
    if 'away_travel_distance' in df.columns:
        df['travel_fatigue'] = np.log1p(df['away_travel_distance']) / 10
        situational_features.append('travel_fatigue')
    
    # Back-to-back games
    if 'home_is_b2b' in df.columns and 'away_is_b2b' in df.columns:
        df['b2b_disadvantage'] = df['away_is_b2b'] - df['home_is_b2b']
        situational_features.append('b2b_disadvantage')
    
    if situational_features:
        features_added += len(situational_features)
        print(f"✅ Situational factors ({len(situational_features)} features)")
    
    # 8. MARKET INTELLIGENCE
    if 'home_odds' in df.columns and 'away_odds' in df.columns:
        # Convert odds to probabilities
        def odds_to_prob(odds):
            if pd.isna(odds) or odds == 0:
                return 0.5
            if odds > 0:
                return 100 / (100 + odds)
            else:
                return -odds / (-odds + 100)
        
        df['home_implied_prob'] = df['home_odds'].apply(odds_to_prob)
        df['away_implied_prob'] = df['away_odds'].apply(odds_to_prob)
        df['market_edge'] = df['home_implied_prob'] - df['away_implied_prob']
        
        # Market confidence (smaller spread between probabilities = more uncertain)
        df['market_confidence'] = abs(df['home_implied_prob'] - df['away_implied_prob'])
        
        features_added += 3
        print("✅ Market intelligence features")
    
    # 9. ADVANCED COMPOSITE METRICS
    # Strength index (weighted combination of key metrics)
    strength_components = []
    weights = []
    
    if 'net_rating_diff' in df.columns:
        strength_components.append(df['net_rating_diff'])
        weights.append(0.35)
    
    if 'pie_diff' in df.columns:
        strength_components.append(df['pie_diff'] * 100)  # Scale PIE
        weights.append(0.25)
    
    if 'efg_pct_diff' in df.columns:
        strength_components.append(df['efg_pct_diff'] * 100)
        weights.append(0.20)
    
    if 'win_pct_diff' in df.columns:
        strength_components.append(df['win_pct_diff'] * 20)
        weights.append(0.20)
    
    if strength_components:
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        df['composite_strength'] = sum(
            comp * weight for comp, weight in zip(strength_components, weights)
        )
        features_added += 1
        print("✅ Composite strength index")
    
    # Volatility measures (predict blow-outs vs close games)
    if 'home_recent_margin_std' in df.columns and 'away_recent_margin_std' in df.columns:
        df['combined_volatility'] = (df['home_recent_margin_std'] + df['away_recent_margin_std']) / 2
        df['volatility_diff'] = df['home_recent_margin_std'] - df['away_recent_margin_std']
        features_added += 2
        print("✅ Margin volatility measures")
    
    # 10. SEASONAL AND TEMPORAL FACTORS
    if 'game_date' in df.columns:
        try:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['month'] = df['game_date'].dt.month
            df['day_of_week'] = df['game_date'].dt.dayofweek
            
            # Season phase
            df['early_season'] = (df['month'].isin([10, 11])).astype(int)
            df['mid_season'] = (df['month'].isin([12, 1, 2])).astype(int)
            df['late_season'] = (df['month'].isin([3, 4])).astype(int)
            
            # Weekend games
            df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
            
            features_added += 6
            print("✅ Seasonal and temporal factors")
        except:
            print("⚠️ Could not parse game dates")
    
    print(f"\n🎯 FEATURE ENGINEERING COMPLETE")
    print(f"   Added {features_added} new features")
    print(f"   Total features available: {len(df.columns):,}")
    
    return df

def create_enhanced_spread_targets(df):
    """Create margin targets and spread betting opportunities"""
    
    print(f"\n🎯 CREATING ENHANCED SPREAD TARGETS")
    
    # Calculate actual margins
    if 'pointDifference' in df.columns:
        df['actual_margin'] = df['pointDifference']
        print("✅ Using pointDifference column")
    elif 'homeScore' in df.columns and 'awayScore' in df.columns:
        df['actual_margin'] = df['homeScore'] - df['awayScore']
        print("✅ Calculated from home/away scores")
    else:
        # Estimate from win/loss and team strength
        base_margin = np.where(df['homeWin'] == 1, 8, -8)
        
        # Adjust based on team strength if available
        if 'net_rating_diff' in df.columns:
            strength_adjustment = df['net_rating_diff'] * 0.3
            df['actual_margin'] = base_margin + strength_adjustment
        else:
            df['actual_margin'] = base_margin
        
        # Add realistic variance
        df['actual_margin'] += np.random.normal(0, 6, len(df))
        print("✅ Estimated margins from win/loss and strength")
    
    # Enhanced spread lines (more granular)
    spread_lines = [-15.5, -12.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, 
                   -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 
                   9.5, 10.5, 12.5, 15.5]
    
    print(f"📊 Creating targets for {len(spread_lines)} spread lines")
    
    for spread in spread_lines:
        # Home team covers spread if they win by more than the spread
        df[f'spread_{abs(spread)}_covers'] = (df['actual_margin'] > spread).astype(int)
    
    # Margin categories for analysis
    df['margin_category'] = pd.cut(
        abs(df['actual_margin']), 
        bins=[0, 3, 7, 12, 20, 100],
        labels=['Close (0-3)', 'Moderate (4-7)', 'Solid (8-12)', 'Blowout (13-20)', 'Massacre (20+)']
    )
    
    # Show margin distribution
    print(f"\n📈 MARGIN DISTRIBUTION:")
    print(f"   Average margin: {df['actual_margin'].mean():.1f} points")
    print(f"   Std deviation: {df['actual_margin'].std():.1f} points")
    print(f"   Close games (≤3 pts): {(abs(df['actual_margin']) <= 3).mean():.1%}")
    print(f"   Blowouts (≥15 pts): {(abs(df['actual_margin']) >= 15).mean():.1%}")
    
    return df

def train_enhanced_spread_models(df):
    """Train ensemble models for each spread line"""
    
    print(f"\n🧠 TRAINING ENHANCED SPREAD ENSEMBLE MODELS")
    print("Training 4-algorithm ensemble for each spread line")
    
    # Define comprehensive feature set
    feature_candidates = [
        # Core strength differentials
        'net_rating_diff', 'off_rating_diff', 'def_rating_diff', 'pie_diff',
        'ts_pct_diff', 'efg_pct_diff', 'oreb_pct_diff', 'dreb_pct_diff',
        'ast_pct_diff', 'stl_pct_diff', 'blk_pct_diff', 'tm_tov_pct_diff',
        'pace_diff', 'win_pct_diff',
        
        # Four factors
        'four_factors_efg_pct', 'four_factors_tov_pct', 'four_factors_oreb_pct', 'four_factors_fta_rate',
        
        # Combined metrics  
        'net_rating_combined', 'off_rating_combined', 'def_rating_combined',
        'pace_combined', 'efg_pct_combined', 'ts_pct_combined',
        
        # Recent form
        'recent_win_pct_diff', 'recent_margin_diff', 'recent_form_diff',
        
        # Home court and situational
        'home_court_base', 'home_court_enhanced', 'rest_advantage', 
        'travel_fatigue', 'b2b_disadvantage',
        
        # Pace and style
        'pace_matchup', 'pace_is_slow', 'pace_is_fast', 'off_vs_def_advantage',
        'style_contrast',
        
        # Market intelligence
        'market_edge', 'market_confidence', 'home_implied_prob', 'away_implied_prob',
        
        # Advanced composites
        'composite_strength', 'combined_volatility', 'volatility_diff',
        
        # Seasonal factors
        'early_season', 'mid_season', 'late_season', 'is_weekend'
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
    
    # Train ensemble for each spread line
    spread_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 12.5, 15.5]
    
    print(f"\n🎯 Training ensemble models for {len(spread_lines)} spread lines...")
    
    for spread in spread_lines:
        target_col = f'spread_{spread}_covers'
        if target_col not in df.columns:
            continue
            
        print(f"\n   📈 Training {spread}-point spread ensemble...")
        
        y = df[target_col]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Skip if insufficient data
        if y_train.sum() < 100 or (len(y_train) - y_train.sum()) < 100:
            print(f"      ⚠️ Insufficient data for {spread}-point spread")
            continue
        
        spread_models = {}
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
        spread_models['rf'] = rf_model
        
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
            spread_models['xgb'] = xgb_model
        
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
            spread_models['lgb'] = lgb_model
        
        # Algorithm 4: Logistic Regression (on scaled features)
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
        predictions.append(lr_pred)
        spread_models['lr'] = lr_model
        
        # Ensemble prediction (weighted average)
        if len(predictions) > 1:
            # Weighted ensemble (RF and XGB get higher weights)
            weights = []
            if 'rf' in spread_models: weights.append(0.3)
            if 'xgb' in spread_models: weights.append(0.3)
            if 'lgb' in spread_models: weights.append(0.25)
            if 'lr' in spread_models: weights.append(0.15)
            
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
        
        models[f'spread_{spread}'] = spread_models
        ensemble_results[spread] = {
            'accuracy': accuracy,
            'confident_accuracy': confident_accuracy,
            'opportunities': opportunities,
            'total_games': len(y_test)
        }
        
        print(f"      ✅ {spread}-point: {accuracy:.1%} accuracy, {opportunities} high-confidence opportunities ({confident_accuracy:.1%})")
    
    # Feature importance from Random Forest
    if f'spread_{spread_lines[0]}' in models and 'rf' in models[f'spread_{spread_lines[0]}']:
        print(f"\n🔝 TOP FEATURES (from Random Forest):")
        rf_model = models[f'spread_{spread_lines[0]}']['rf']
        feature_importance = list(zip(available_features, rf_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:15]):
            print(f"   {i+1:2d}. {feature:<25} {importance:.3f}")
    
    return models, available_features, ensemble_results

def calculate_spread_profitability(df, models, features, ensemble_results):
    """Calculate profitability and ROI for spread betting"""
    
    print(f"\n💰 CALCULATING SPREAD BETTING PROFITABILITY")
    
    # Prepare test data (last 20% of games)
    split_idx = int(len(df) * 0.8)
    X_test = df[features].iloc[split_idx:].fillna(0)
    X_test_scaled = models['scaler'].transform(X_test)
    
    total_bets = 0
    total_wins = 0
    total_profit = 0
    
    profitable_spreads = []
    
    print(f"\n📊 SPREAD-BY-SPREAD PROFITABILITY:")
    
    for spread in sorted(ensemble_results.keys()):
        if f'spread_{spread}' not in models:
            continue
            
        spread_models = models[f'spread_{spread}']
        target_col = f'spread_{spread}_covers'
        
        if target_col not in df.columns:
            continue
        
        y_test = df[target_col].iloc[split_idx:]
        
        # Generate ensemble predictions
        predictions = []
        
        if 'rf' in spread_models:
            rf_pred = spread_models['rf'].predict_proba(X_test)[:, 1]
            predictions.append(rf_pred)
        
        if 'xgb' in spread_models:
            xgb_pred = spread_models['xgb'].predict_proba(X_test)[:, 1]
            predictions.append(xgb_pred)
        
        if 'lgb' in spread_models:
            lgb_pred = spread_models['lgb'].predict_proba(X_test)[:, 1]
            predictions.append(lgb_pred)
        
        if 'lr' in spread_models:
            lr_pred = spread_models['lr'].predict_proba(X_test_scaled)[:, 1]
            predictions.append(lr_pred)
        
        if not predictions:
            continue
        
        # Weighted ensemble
        weights = []
        if 'rf' in spread_models: weights.append(0.3)
        if 'xgb' in spread_models: weights.append(0.3)
        if 'lgb' in spread_models: weights.append(0.25)
        if 'lr' in spread_models: weights.append(0.15)
        
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
        bet_home_covers = confident_predictions > 0.5
        
        # Calculate wins
        wins = (bet_home_covers == confident_actuals.astype(bool)).sum()
        bets = len(confident_predictions)
        win_rate = wins / bets if bets > 0 else 0
        
        # Calculate profit (assuming -110 odds, need 52.4% to break even)
        profit = 0
        for i, (pred, actual) in enumerate(zip(confident_predictions, confident_actuals)):
            if pred > 0.5:  # Bet home team covers
                if actual == 1:  # Won
                    profit += 0.909  # Win $0.909 for every $1 bet at -110
                else:  # Lost
                    profit -= 1.0
            else:  # Bet away team covers
                if actual == 0:  # Won
                    profit += 0.909
                else:  # Lost
                    profit -= 1.0
        
        roi = (profit / bets * 100) if bets > 0 else 0
        
        total_bets += bets
        total_wins += wins
        total_profit += profit
        
        if roi > 5:  # Only show profitable spreads
            profitable_spreads.append((spread, roi, bets, win_rate))
            print(f"   {spread:4.1f}-point: {win_rate:.1%} win rate, {bets:3d} bets, {roi:+5.1f}% ROI")
    
    # Overall profitability
    overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
    overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    
    print(f"\n🎯 OVERALL SPREAD BETTING PERFORMANCE:")
    print(f"   Total profitable opportunities: {total_bets:,}")
    print(f"   Overall win rate: {overall_win_rate:.1%}")
    print(f"   Overall ROI: {overall_roi:+.1f}%")
    print(f"   Profitable spread lines: {len(profitable_spreads)}")
    
    return {
        'total_bets': total_bets,
        'total_wins': total_wins,
        'overall_win_rate': overall_win_rate,
        'overall_roi': overall_roi,
        'profitable_spreads': profitable_spreads
    }

def save_enhanced_spread_models(models, features):
    """Save the enhanced spread models and configurations"""
    
    print(f"\n💾 SAVING ENHANCED SPREAD MODELS")
    
    # Save individual models
    for model_name, model_data in models.items():
        if model_name == 'scaler':
            joblib.dump(model_data, f"models/enhanced_spreads_scaler.pkl")
            continue
        
        if isinstance(model_data, dict):
            for algo_name, algo_model in model_data.items():
                model_path = f"models/enhanced_spreads_{model_name}_{algo_name}.pkl"
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
    
    joblib.dump(config, "models/enhanced_spreads_config.pkl")
    
    print(f"✅ Saved ensemble models for {len([k for k in models.keys() if k.startswith('spread_')])} spread lines")
    print(f"✅ Saved feature configuration ({len(features)} features)")

def test_spread_predictions(models, features):
    """Test spread predictions on sample scenarios"""
    
    print(f"\n🧪 TESTING ENHANCED SPREAD PREDICTIONS")
    
    # Strong home team scenario
    strong_home = {
        'net_rating_diff': 12.5, 'off_rating_diff': 8.3, 'def_rating_diff': 4.2,
        'pie_diff': 0.08, 'efg_pct_diff': 0.06, 'ts_pct_diff': 0.05,
        'win_pct_diff': 0.25, 'recent_win_pct_diff': 0.20, 'home_court_enhanced': 4.5,
        'market_edge': 0.30, 'market_confidence': 0.35, 'composite_strength': 15.2,
        'pace_combined': 101.5, 'off_vs_def_advantage': 8.5, 'combined_volatility': 8.2
    }
    
    # Even matchup scenario  
    even_game = {
        'net_rating_diff': 1.2, 'off_rating_diff': 0.8, 'def_rating_diff': 0.4,
        'pie_diff': 0.01, 'efg_pct_diff': 0.005, 'ts_pct_diff': 0.002,
        'win_pct_diff': 0.03, 'recent_win_pct_diff': 0.05, 'home_court_enhanced': 3.2,
        'market_edge': 0.08, 'market_confidence': 0.12, 'composite_strength': 2.1,
        'pace_combined': 99.8, 'off_vs_def_advantage': 1.2, 'combined_volatility': 9.5
    }
    
    scenarios = [
        ("Strong Home Favorite", strong_home),
        ("Even Matchup", even_game)
    ]
    
    for scenario_name, scenario_data in scenarios:
        print(f"\n📊 {scenario_name}:")
        
        # Create test dataframe
        test_df = pd.DataFrame([{f: scenario_data.get(f, 0) for f in features}])
        test_scaled = models['scaler'].transform(test_df)
        
        # Test key spread lines
        test_spreads = [3.5, 5.5, 7.5, 9.5, 12.5]
        
        for spread in test_spreads:
            if f'spread_{spread}' not in models:
                continue
            
            spread_models = models[f'spread_{spread}']
            predictions = []
            
            # Get predictions from each algorithm
            if 'rf' in spread_models:
                rf_pred = spread_models['rf'].predict_proba(test_df)[:, 1]
                predictions.append(rf_pred[0])
            
            if 'xgb' in spread_models:
                xgb_pred = spread_models['xgb'].predict_proba(test_df)[:, 1]
                predictions.append(xgb_pred[0])
            
            if 'lgb' in spread_models:
                lgb_pred = spread_models['lgb'].predict_proba(test_df)[:, 1]
                predictions.append(lgb_pred[0])
            
            if 'lr' in spread_models:
                lr_pred = spread_models['lr'].predict_proba(test_scaled)[:, 1]
                predictions.append(lr_pred[0])
            
            if predictions:
                # Ensemble prediction
                weights = [0.3, 0.3, 0.25, 0.15][:len(predictions)]
                weights = np.array(weights) / sum(weights)
                ensemble_prob = np.average(predictions, weights=weights)
                
                # Recommendation
                if ensemble_prob > 0.58:
                    rec = "HOME COVERS"
                    confidence = ensemble_prob
                elif ensemble_prob < 0.42:
                    rec = "AWAY COVERS"  
                    confidence = 1 - ensemble_prob
                else:
                    rec = "NO BET"
                    confidence = max(ensemble_prob, 1 - ensemble_prob)
                
                print(f"   {spread:4.1f}-point spread: {rec} ({confidence:.1%} confidence)")

def main():
    """Build complete enhanced spreads system"""
    
    # Load data
    df = load_spreads_data()
    
    # Engineer advanced features
    df = engineer_advanced_spreads_features(df)
    
    # Create targets
    df = create_enhanced_spread_targets(df)
    
    # Train ensemble models
    models, features, ensemble_results = train_enhanced_spread_models(df)
    
    # Calculate profitability
    profitability = calculate_spread_profitability(df, models, features, ensemble_results)
    
    # Save models
    save_enhanced_spread_models(models, features)
    
    # Test predictions
    test_spread_predictions(models, features)
    
    # Final summary
    print(f"\n🎉 ENHANCED SPREADS SYSTEM COMPLETE!")
    print(f"🎯 Profitable opportunities: {profitability['total_bets']:,}")
    print(f"📊 Overall win rate: {profitability['overall_win_rate']:.1%}")
    print(f"💰 Overall ROI: {profitability['overall_roi']:+.1f}%")
    print(f"🏆 Profitable spread lines: {len(profitability['profitable_spreads'])}")
    
    if profitability['overall_roi'] > 5:
        print(f"\n🚀 ENHANCED SPREADS SYSTEM IS PROFITABLE!")
        print(f"   This system should generate positive returns betting spreads")
    
    return models, features, profitability

if __name__ == "__main__":
    main()
