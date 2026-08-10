#!/usr/bin/env python3
"""
REALISTIC PROFITABLE MODEL - Offseason Version
Use only data we can actually get consistently
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

def build_realistic_profitable_model():
    """Build profitable model using ONLY realistic data sources"""
    
    print("🏀 REALISTIC PROFITABLE NBA MODEL")
    print("Using only data we can consistently access")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('/Users/mk/nba-betting-ai/data/final_games_full_features.csv', low_memory=False)
    df = df.dropna(subset=['home_odds', 'away_odds', 'homeWin'])
    df = df[(df['home_odds'] != 0) & (df['away_odds'] != 0)]
    
    print(f"📊 Total games with odds: {len(df)}")
    
    # Market probabilities
    def implied_probability(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    df['home_market_prob'] = df['home_odds'].apply(implied_probability)
    df['away_market_prob'] = df['away_odds'].apply(implied_probability)
    
    # BUILD FEATURES WE CAN ACTUALLY GET
    print("\n🔧 BUILDING REALISTIC FEATURES")
    
    features = []
    
    # 1. CORE TEAM STATS (always available from official NBA stats)
    if 'home_NET_RATING' in df.columns:
        df['NET_gap'] = df['home_NET_RATING'] - df['away_NET_RATING']
        features.append('NET_gap')
        print("   ✅ NET Rating gap")
    
    if 'home_PIE' in df.columns:
        df['PIE_advantage'] = df['home_PIE'] - df['away_PIE']
        features.extend(['PIE_advantage', 'home_PIE', 'away_PIE'])
        print("   ✅ PIE stats")
    
    # 2. SHOOTING EFFICIENCY (always available)
    if 'home_TS_PCT' in df.columns:
        df['TS_advantage'] = df['home_TS_PCT'] - df['away_TS_PCT']
        features.append('TS_advantage')
        print("   ✅ True Shooting advantage")
    
    if 'home_EFG_PCT' in df.columns:
        df['EFG_advantage'] = df['home_EFG_PCT'] - df['away_EFG_PCT'] 
        features.append('EFG_advantage')
        print("   ✅ Effective FG advantage")
    
    # 3. MARKET INEFFICIENCY INDICATORS
    df['market_juice'] = (df['home_market_prob'] + df['away_market_prob']) - 1.0
    features.append('market_juice')
    print("   ✅ Market juice indicator")
    
    # 4. CONTRARIAN BETTING SPOTS (most profitable angles)
    
    # Heavy favorite fade opportunities
    df['home_heavy_favorite'] = (df['home_market_prob'] > 0.70).astype(int)
    df['away_heavy_favorite'] = (df['away_market_prob'] > 0.70).astype(int)
    features.extend(['home_heavy_favorite', 'away_heavy_favorite'])
    print("   ✅ Heavy favorite indicators")
    
    # Sweet spot underdogs (30-45% implied probability)
    df['home_sweet_underdog'] = ((df['home_market_prob'] > 0.30) & (df['home_market_prob'] < 0.45)).astype(int)
    df['away_sweet_underdog'] = ((df['away_market_prob'] > 0.30) & (df['away_market_prob'] < 0.45)).astype(int)
    features.extend(['home_sweet_underdog', 'away_sweet_underdog'])
    print("   ✅ Sweet spot underdog indicators")
    
    # 5. STATISTICAL VS MARKET COMPARISON
    if 'NET_gap' in df.columns:
        # Convert NET rating to win probability
        df['stat_home_prob'] = 1 / (1 + np.exp(-df['NET_gap'] * 0.04))
        df['market_vs_stats'] = df['home_market_prob'] - df['stat_home_prob'] 
        features.extend(['stat_home_prob', 'market_vs_stats'])
        print("   ✅ Market vs stats comparison")
    
    # 6. SITUATIONAL FACTORS (if available in data)
    
    # Rest advantage
    if 'home_days_rest' in df.columns and 'away_days_rest' in df.columns:
        df['rest_advantage'] = df['home_days_rest'] - df['away_days_rest']
        features.append('rest_advantage')
        print("   ✅ Rest advantage")
    
    # Back-to-back games
    if 'home_b2b' in df.columns and 'away_b2b' in df.columns:
        df['b2b_advantage'] = df['away_b2b'].astype(int) - df['home_b2b'].astype(int)
        features.append('b2b_advantage')
        print("   ✅ Back-to-back advantage")
    
    print(f"\n📊 Final feature count: {len(features)}")
    
    # Clean data
    df_clean = df.dropna(subset=features + ['homeWin'])
    print(f"📊 Clean games: {len(df_clean)}")
    
    # CREATE PROFIT-FOCUSED TARGETS
    print("\n🎯 CREATING PROFIT TARGETS")
    
    # HOME VALUE BETS: Home wins + reasonable odds
    df_clean['home_profit_bet'] = (
        (df_clean['homeWin'] == 1) & 
        (df_clean['home_market_prob'] > 0.25) &  # Not super longshots
        (df_clean['home_market_prob'] < 0.65)    # Not heavy favorites
    ).astype(int)
    
    # AWAY VALUE BETS: Away wins + reasonable odds  
    df_clean['away_profit_bet'] = (
        (df_clean['homeWin'] == 0) &
        (df_clean['away_market_prob'] > 0.25) &  # Not super longshots
        (df_clean['away_market_prob'] < 0.65)    # Not heavy favorites
    ).astype(int)
    
    home_rate = df_clean['home_profit_bet'].mean()
    away_rate = df_clean['away_profit_bet'].mean()
    
    print(f"   🏠 Home profitable rate: {home_rate:.3f}")
    print(f"   ✈️ Away profitable rate: {away_rate:.3f}")
    
    # TRAIN MODELS
    print("\n🤖 TRAINING PROFIT-FOCUSED MODELS")
    
    X = df_clean[features]
    
    # Chronological split (more realistic)
    if 'gameDate' in df_clean.columns:
        df_clean['gameDate'] = pd.to_datetime(df_clean['gameDate'])
        df_clean = df_clean.sort_values('gameDate')
        
        split_idx = int(len(df_clean) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_home_train = df_clean['home_profit_bet'].iloc[:split_idx]
        y_home_test = df_clean['home_profit_bet'].iloc[split_idx:]
        y_away_train = df_clean['away_profit_bet'].iloc[:split_idx] 
        y_away_test = df_clean['away_profit_bet'].iloc[split_idx:]
        test_df = df_clean.iloc[split_idx:].copy()
        
        print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
    else:
        X_train, X_test, y_home_train, y_home_test = train_test_split(X, df_clean['home_profit_bet'], test_size=0.2, random_state=42)
        _, _, y_away_train, y_away_test = train_test_split(X, df_clean['away_profit_bet'], test_size=0.2, random_state=42)
        test_df = df_clean.iloc[-len(X_test):].copy()
    
    # Train models optimized for profit
    home_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,  # Prevent overfitting
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced targets
    )
    
    away_model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    
    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)
    
    # Evaluate
    home_pred = home_model.predict_proba(X_test)[:, 1]
    away_pred = away_model.predict_proba(X_test)[:, 1]
    
    home_auc = roc_auc_score(y_home_test, home_pred) if len(np.unique(y_home_test)) > 1 else 0
    away_auc = roc_auc_score(y_away_test, away_pred) if len(np.unique(y_away_test)) > 1 else 0
    
    print(f"   🏠 Home AUC: {home_auc:.3f}")
    print(f"   ✈️ Away AUC: {away_auc:.3f}")
    
    # REALISTIC BETTING SIMULATION
    print("\n💰 REALISTIC BETTING SIMULATION")
    
    test_df['home_profit_prob'] = home_pred
    test_df['away_profit_prob'] = away_pred
    
    # Test multiple conservative thresholds
    best_roi = -100
    best_strategy = {}
    
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        total_profit = 0
        total_bets = 0
        wins = 0
        
        for _, row in test_df.iterrows():
            bet_made = False
            
            # Home bet
            if row['home_profit_prob'] > threshold and not bet_made:
                total_bets += 1
                if row['homeWin'] == 1:
                    # Win
                    payout = 1 * (100 / abs(row['home_odds'])) if row['home_odds'] < 0 else 1 * (row['home_odds'] / 100)
                    total_profit += payout
                    wins += 1
                else:
                    # Loss
                    total_profit -= 1
                bet_made = True
            
            # Away bet (if no home bet)
            elif row['away_profit_prob'] > threshold:
                total_bets += 1
                if row['homeWin'] == 0:
                    # Win
                    payout = 1 * (100 / abs(row['away_odds'])) if row['away_odds'] < 0 else 1 * (row['away_odds'] / 100)
                    total_profit += payout
                    wins += 1
                else:
                    # Loss
                    total_profit -= 1
        
        if total_bets > 0:
            roi = (total_profit / total_bets) * 100
            win_rate = wins / total_bets
            
            print(f"   {threshold:.2f}: {total_bets} bets, {win_rate:.3f} WR, {roi:.1f}% ROI")
            
            if roi > best_roi:
                best_roi = roi
                best_strategy = {
                    'threshold': threshold,
                    'bets': total_bets,
                    'win_rate': win_rate,
                    'roi': roi,
                    'profit': total_profit
                }
    
    print(f"\n🏆 BEST REALISTIC STRATEGY:")
    if best_strategy:
        print(f"   Threshold: {best_strategy['threshold']}")
        print(f"   Bets: {best_strategy['bets']}")
        print(f"   Win Rate: {best_strategy['win_rate']:.3f}")
        print(f"   ROI: {best_strategy['roi']:.1f}%")
        print(f"   Profit: ${best_strategy['profit']:.2f}")
        
        if best_strategy['roi'] > 2:
            print("   🎉 PROFITABLE WITH REALISTIC DATA!")
        elif best_strategy['roi'] > 0:
            print("   ✅ Slightly profitable - consider live implementation")
        else:
            print("   ⚠️ Still needs improvement")
    
    # Save models
    joblib.dump(home_model, 'models/realistic_home_profit_model.pkl')
    joblib.dump(away_model, 'models/realistic_away_profit_model.pkl')
    
    with open('models/realistic_profit_features.txt', 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    
    print("\n✅ REALISTIC PROFITABLE MODEL SAVED!")
    
    return home_model, away_model, features, best_strategy

if __name__ == "__main__":
    build_realistic_profitable_model()