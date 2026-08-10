#!/usr/bin/env python3
"""
BUILD PROFITABLE MONEYLINE MODEL
Train on VALUE detection, not winner prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def implied_probability(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def kelly_bet_size(win_prob, odds):
    """Calculate optimal Kelly Criterion bet size"""
    if odds > 0:
        b = odds / 100  # Net odds received
    else:
        b = 100 / abs(odds)
    
    kelly = (win_prob * (b + 1) - 1) / b
    return max(0, kelly)  # Never bet negative

def build_value_detection_model():
    """Build model that detects VALUE, not just winners"""
    
    print("🎯 BUILDING PROFITABLE MONEYLINE MODEL")
    print("Focus: VALUE DETECTION not winner prediction")
    print("=" * 50)
    
    # Load historical data with odds
    df = pd.read_csv('/Users/mk/nba-betting-ai/data/final_games_full_features.csv', low_memory=False)
    
    # Filter to games with odds
    df = df.dropna(subset=['home_odds', 'away_odds', 'homeWin'])
    df = df[(df['home_odds'] != 0) & (df['away_odds'] != 0)]
    
    print(f"📊 Loaded {len(df)} games with odds")
    
    # Calculate market probabilities
    df['home_market_prob'] = df['home_odds'].apply(implied_probability)
    df['away_market_prob'] = df['away_odds'].apply(implied_probability)
    
    # Build features for VALUE detection
    print("\n🔧 BUILDING VALUE-FOCUSED FEATURES")
    
    # Core basketball features
    features = []
    
    if 'home_NET_RATING' in df.columns:
        df['NET_gap'] = df['home_NET_RATING'] - df['away_NET_RATING']
        features.append('NET_gap')
    
    if 'home_PIE' in df.columns:
        df['PIE_advantage'] = df['home_PIE'] - df['away_PIE']
        features.append('PIE_advantage')
        features.extend(['home_PIE', 'away_PIE'])
    
    if 'home_TS_PCT' in df.columns:
        df['TS_advantage'] = df['home_TS_PCT'] - df['away_TS_PCT']
        features.append('TS_advantage')
    
    if 'home_EFG_PCT' in df.columns:
        df['EFG_advantage'] = df['home_EFG_PCT'] - df['away_EFG_PCT']
        features.append('EFG_advantage')
    
    if 'home_TOV_PCT' in df.columns:
        df['TOV_advantage'] = df['away_TOV_PCT'] - df['home_TOV_PCT']  # Lower is better
        features.append('TOV_advantage')
    
    # VALUE-SPECIFIC FEATURES
    print("   💰 Adding value-detection features...")
    
    # Market inefficiency indicators
    df['market_total_prob'] = df['home_market_prob'] + df['away_market_prob']  # Should be ~1.05-1.10
    df['market_juice'] = df['market_total_prob'] - 1.0
    features.extend(['market_total_prob', 'market_juice'])
    
    # Contrarian indicators
    df['home_is_heavy_favorite'] = (df['home_market_prob'] > 0.75).astype(int)
    df['away_is_heavy_favorite'] = (df['away_market_prob'] > 0.75).astype(int)
    df['is_coinflip_game'] = ((df['home_market_prob'] > 0.45) & (df['home_market_prob'] < 0.55)).astype(int)
    features.extend(['home_is_heavy_favorite', 'away_is_heavy_favorite', 'is_coinflip_game'])
    
    # Statistical vs Market comparison
    if 'NET_gap' in df.columns:
        # Estimate "true" probability from NET rating
        df['stat_based_home_prob'] = 1 / (1 + np.exp(-df['NET_gap'] * 0.05))  # Sigmoid conversion
        df['market_vs_stats_diff'] = df['home_market_prob'] - df['stat_based_home_prob']
        features.extend(['stat_based_home_prob', 'market_vs_stats_diff'])
    
    print(f"   ✅ Created {len(features)} value-focused features")
    
    # Remove rows with missing features
    df_clean = df.dropna(subset=features + ['homeWin'])
    print(f"   📊 Clean dataset: {len(df_clean)} games")
    
    # CREATE VALUE-BASED TARGETS
    print("\n🎯 CREATING VALUE-BASED TARGETS")
    
    # TARGET 1: Profitable home bets (home wins AND odds justify it)
    df_clean['home_value_bet'] = (
        (df_clean['homeWin'] == 1) & 
        (df_clean['home_market_prob'] < 0.65)  # Don't bet heavy favorites
    ).astype(int)
    
    # TARGET 2: Profitable away bets
    df_clean['away_value_bet'] = (
        (df_clean['homeWin'] == 0) & 
        (df_clean['away_market_prob'] < 0.65)  # Don't bet heavy favorites
    ).astype(int)
    
    # TARGET 3: Any value bet (home or away)
    df_clean['any_value_bet'] = (
        df_clean['home_value_bet'] | df_clean['away_value_bet']
    ).astype(int)
    
    home_value_rate = df_clean['home_value_bet'].mean()
    away_value_rate = df_clean['away_value_bet'].mean()
    any_value_rate = df_clean['any_value_bet'].mean()
    
    print(f"   📊 Home value bet rate: {home_value_rate:.3f}")
    print(f"   📊 Away value bet rate: {away_value_rate:.3f}")
    print(f"   📊 Any value bet rate: {any_value_rate:.3f}")
    
    # TRAIN MODELS
    print("\n🤖 TRAINING VALUE-DETECTION MODELS")
    
    X = df_clean[features]
    
    # Sort by date for proper train/test split
    if 'gameDate' in df_clean.columns:
        df_clean['gameDate'] = pd.to_datetime(df_clean['gameDate'])
        df_clean = df_clean.sort_values('gameDate')
        
        # Use first 80% for training, last 20% for testing
        split_idx = int(len(df_clean) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        
        y_home_train = df_clean['home_value_bet'].iloc[:split_idx]
        y_home_test = df_clean['home_value_bet'].iloc[split_idx:]
        
        y_away_train = df_clean['away_value_bet'].iloc[:split_idx]
        y_away_test = df_clean['away_value_bet'].iloc[split_idx:]
        
        print(f"   📊 Train: {len(X_train)} games, Test: {len(X_test)} games")
    else:
        X_train, X_test, y_home_train, y_home_test = train_test_split(
            X, df_clean['home_value_bet'], test_size=0.2, random_state=42
        )
        _, _, y_away_train, y_away_test = train_test_split(
            X, df_clean['away_value_bet'], test_size=0.2, random_state=42
        )
    
    # Train models
    print("   🏠 Training home value model...")
    home_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    home_model.fit(X_train, y_home_train)
    
    print("   ✈️ Training away value model...")
    away_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    away_model.fit(X_train, y_away_train)
    
    # Evaluate models
    print("\n📊 MODEL EVALUATION")
    
    home_pred = home_model.predict_proba(X_test)[:, 1]
    away_pred = away_model.predict_proba(X_test)[:, 1]
    
    home_auc = roc_auc_score(y_home_test, home_pred)
    away_auc = roc_auc_score(y_away_test, away_pred)
    
    print(f"   🏠 Home value AUC: {home_auc:.3f}")
    print(f"   ✈️ Away value AUC: {away_auc:.3f}")
    
    # Feature importance
    print("\n🔍 TOP VALUE-DETECTION FEATURES")
    
    home_importance = pd.DataFrame({
        'feature': features,
        'importance': home_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("   🏠 Home value features:")
    for _, row in home_importance.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.3f}")
    
    # Save models
    print("\n💾 SAVING VALUE-DETECTION MODELS")
    
    joblib.dump(home_model, 'models/home_value_model.pkl')
    joblib.dump(away_model, 'models/away_value_model.pkl')
    
    # Save feature list
    with open('models/value_features.txt', 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    
    print("   ✅ Models saved!")
    
    return home_model, away_model, features, df_clean

def test_value_betting_strategy(df, home_model, away_model, features):
    """Test the value betting strategy"""
    
    print("\n💰 TESTING VALUE BETTING STRATEGY")
    print("=" * 40)
    
    # Get test data (last 20%)
    df_sorted = df.sort_values('gameDate') if 'gameDate' in df.columns else df
    split_idx = int(len(df_sorted) * 0.8)
    test_df = df_sorted.iloc[split_idx:].copy()
    
    X_test = test_df[features]
    
    # Get value predictions
    home_value_prob = home_model.predict_proba(X_test)[:, 1]
    away_value_prob = away_model.predict_proba(X_test)[:, 1]
    
    test_df = test_df.copy()
    test_df['home_value_prob'] = home_value_prob
    test_df['away_value_prob'] = away_value_prob
    
    # VALUE BETTING STRATEGY
    print("🎯 IMPLEMENTING VALUE STRATEGY")
    
    total_profit = 0
    total_bets = 0
    winning_bets = 0
    
    bet_threshold = 0.15  # Only bet when model is 15%+ confident of value
    
    for idx, row in test_df.iterrows():
        home_bet = False
        away_bet = False
        
        # Home value bet
        if row['home_value_prob'] > bet_threshold:
            if row['homeWin'] == 1:
                # Win home bet
                if row['home_odds'] > 0:
                    profit = row['home_odds'] / 100
                else:
                    profit = 100 / abs(row['home_odds'])
                total_profit += profit
                winning_bets += 1
            else:
                # Lose home bet
                total_profit -= 1
            total_bets += 1
            home_bet = True
        
        # Away value bet (if we didn't bet home)
        elif row['away_value_prob'] > bet_threshold:
            if row['homeWin'] == 0:
                # Win away bet
                if row['away_odds'] > 0:
                    profit = row['away_odds'] / 100
                else:
                    profit = 100 / abs(row['away_odds'])
                total_profit += profit
                winning_bets += 1
            else:
                # Lose away bet
                total_profit -= 1
            total_bets += 1
            away_bet = True
    
    if total_bets > 0:
        win_rate = winning_bets / total_bets
        roi = (total_profit / total_bets) * 100
        
        print(f"   📊 Total bets: {total_bets}")
        print(f"   📊 Win rate: {win_rate:.3f}")
        print(f"   📊 ROI: {roi:.1f}%")
        print(f"   📊 Total profit: ${total_profit:.2f}")
        
        if roi > 0:
            print("   🎉 PROFITABLE STRATEGY!")
        else:
            print("   ⚠️ Still not profitable - need higher thresholds")
            
        # Test different thresholds
        print("\n🔧 TESTING DIFFERENT THRESHOLDS")
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
            bets, wins, profit = 0, 0, 0
            
            for idx, row in test_df.iterrows():
                if row['home_value_prob'] > threshold:
                    bets += 1
                    if row['homeWin'] == 1:
                        wins += 1
                        profit += row['home_odds']/100 if row['home_odds'] > 0 else 100/abs(row['home_odds'])
                    else:
                        profit -= 1
                elif row['away_value_prob'] > threshold:
                    bets += 1
                    if row['homeWin'] == 0:
                        wins += 1
                        profit += row['away_odds']/100 if row['away_odds'] > 0 else 100/abs(row['away_odds'])
                    else:
                        profit -= 1
            
            if bets > 0:
                wr = wins/bets
                roi = (profit/bets)*100
                print(f"   {threshold:.2f} threshold: {bets} bets, {wr:.3f} win rate, {roi:.1f}% ROI")

if __name__ == "__main__":
    home_model, away_model, features, df = build_value_detection_model()
    test_value_betting_strategy(df, home_model, away_model, features)
    
    print("\n🎯 VALUE-DETECTION MODEL COMPLETE!")
    print("Key changes from winner-prediction model:")
    print("1. Trains on VALUE detection, not winner prediction")
    print("2. Focuses on contrarian opportunities") 
    print("3. Avoids heavy favorites")
    print("4. Uses market inefficiency indicators")
    print("5. Optimizes for ROI, not accuracy")