#!/usr/bin/env python3
"""
Train Enhanced 36-Feature NBA Betting Model
Uses proven feature set from backtest results
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Load and prepare training data"""
    
    print("📊 Loading training data...")
    
    try:
        df = pd.read_csv('data/final_games_full_features.csv')
        
        # Clean data
        df = df.dropna(subset=['homeWin', 'home_odds', 'away_odds'])
        df = df[df['home_odds'] != 0]
        df = df[df['away_odds'] != 0]
        
        print(f"✅ Loaded {len(df)} games for training")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def engineer_enhanced_features(df):
    """Create the proven 36 enhanced features"""
    
    print("🔧 Engineering enhanced features...")
    
    # Basic differentials
    if 'off_rating_diff' not in df.columns:
        df['off_rating_diff'] = df['home_OFF_RATING'] - df['away_DEF_RATING']
    if 'def_rating_diff' not in df.columns:
        df['def_rating_diff'] = df['home_DEF_RATING'] - df['away_OFF_RATING']
    if 'net_rating_diff' not in df.columns:
        df['net_rating_diff'] = df['home_NET_RATING'] - df['away_NET_RATING']
    if 'recent_win_diff' not in df.columns:
        df['recent_win_diff'] = df['home_recent_win_pct'] - df['away_recent_win_pct']
    if 'pace_diff' not in df.columns:
        df['pace_diff'] = df['home_PACE'] - df['away_PACE']
    
    # Four factors advantages
    if 'efg_advantage' not in df.columns:
        df['efg_advantage'] = df['home_EFG_PCT'] - df['away_EFG_PCT']
    if 'tov_advantage' not in df.columns:
        df['tov_advantage'] = df['away_TM_TOV_PCT'] - df['home_TM_TOV_PCT']
    if 'oreb_advantage' not in df.columns:
        df['oreb_advantage'] = df['home_OREB_PCT'] - df['away_DREB_PCT']
    if 'ft_advantage' not in df.columns:
        # Create FT rate if missing
        if 'home_FT_RATE' not in df.columns:
            df['home_FT_RATE'] = 0.25  # Average FT rate
            df['away_FT_RATE'] = 0.25
        df['ft_advantage'] = df['home_FT_RATE'] - df['away_FT_RATE']
    
    # Home court advantage
    if 'home_court_advantage' not in df.columns:
        df['home_court_advantage'] = 0.04
    
    # Composite features
    if 'four_factors_composite' not in df.columns:
        df['four_factors_composite'] = (
            df['efg_advantage'] * 0.4 + 
            df['tov_advantage'] * 0.25 + 
            df['oreb_advantage'] * 0.2 + 
            df.get('ft_advantage', 0) * 0.15
        )
    
    if 'overall_advantage' not in df.columns:
        df['overall_advantage'] = (
            df['net_rating_diff'] * 0.3 + 
            df['efg_advantage'] * 100 * 0.3 + 
            df['four_factors_composite'] * 0.4
        )
    
    # Market features
    def american_to_prob(odds):
        if pd.isna(odds) or odds == 0:
            return 0.5
        if odds > 0:
            return 100 / (100 + odds)
        else:
            return -odds / (-odds + 100)
    
    if 'implied_home_prob' not in df.columns:
        df['implied_home_prob'] = df['home_odds'].apply(american_to_prob)
    if 'implied_away_prob' not in df.columns:
        df['implied_away_prob'] = df['away_odds'].apply(american_to_prob)
    if 'odds_diff' not in df.columns:
        df['odds_diff'] = df['home_odds'] - df['away_odds']
    
    print("✅ Enhanced features created")
    return df

def define_enhanced_features():
    """Define the proven 36-feature set"""
    
    return [
        # Original core features (14)
        'home_OFF_RATING', 'away_OFF_RATING', 'home_DEF_RATING', 'away_DEF_RATING',
        'home_EFG_PCT', 'away_EFG_PCT', 'home_PACE', 'away_PACE',
        'home_recent_win_pct', 'away_recent_win_pct',
        'off_rating_diff', 'def_rating_diff', 'recent_win_diff', 'pace_diff',
        
        # Enhanced features (22 additional)
        'home_NET_RATING', 'away_NET_RATING', 'net_rating_diff',
        'home_OREB_PCT', 'away_OREB_PCT', 'home_DREB_PCT', 'away_DREB_PCT',
        'home_TM_TOV_PCT', 'away_TM_TOV_PCT', 'home_AST_PCT', 'away_AST_PCT',
        'efg_advantage', 'tov_advantage', 'oreb_advantage', 'home_court_advantage',
        'four_factors_composite', 'overall_advantage',
        'home_odds', 'away_odds', 'implied_home_prob', 'implied_away_prob', 'odds_diff'
    ]

def train_enhanced_model(df, features):
    """Train enhanced model with available sklearn algorithms"""
    
    print(f"\n🤖 Training Enhanced Model...")
    print(f"📊 Features: {len(features)}")
    
    # Prepare data
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    print(f"✅ Using {len(available_features)} available features")
    
    X = df[available_features].fillna(df[available_features].median())
    y = df['homeWin']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Test multiple models and pick the best
    models = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("🔄 Testing multiple algorithms...")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"   {name}: {test_score:.3f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    print(f"🏆 Best model: {best_name}")
    model = best_model
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_logloss = log_loss(y_train, train_proba)
    test_logloss = log_loss(y_test, test_proba)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"   Train Accuracy: {train_acc:.3f}")
    print(f"   Test Accuracy: {test_acc:.3f}")
    print(f"   Train Log Loss: {train_logloss:.3f}")
    print(f"   Test Log Loss: {test_logloss:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
    else:
        # For logistic regression, use coefficient magnitude
        importance = np.abs(model.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return model, available_features, feature_importance

def save_model(model, features, feature_importance):
    """Save the trained model and metadata"""
    
    print(f"\n💾 Saving enhanced model...")
    
    # Save model
    model_path = 'models/nba_model_v4_3_enhanced.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save feature list
    features_path = 'models/features_v4_3_enhanced.txt'
    with open(features_path, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"✅ Features saved: {features_path}")
    
    # Save feature importance
    importance_path = 'models/feature_importance_v4_3_enhanced.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"✅ Feature importance saved: {importance_path}")
    
    print(f"\n🚀 ENHANCED MODEL READY FOR DEPLOYMENT!")
    print(f"📈 Upgrade from v4.2 (14 features) to v4.3 (36 features)")

def main():
    """Train and save the enhanced model"""
    
    print("🏀 TRAINING ENHANCED NBA MODEL v4.3")
    print("Upgrading from 14 to 36 features")
    print("=" * 50)
    
    # Load data
    df = load_training_data()
    if df is None:
        return
    
    # Engineer features
    df = engineer_enhanced_features(df)
    
    # Define feature set
    features = define_enhanced_features()
    
    # Train model
    model, used_features, feature_importance = train_enhanced_model(df, features)
    
    # Save everything
    save_model(model, used_features, feature_importance)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"🎯 Ready to deploy enhanced NBA betting system")

if __name__ == "__main__":
    main()