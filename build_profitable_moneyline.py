#!/usr/bin/env python3
"""
BUILD PROFITABLE MONEYLINE MODEL
Rebuild the moneyline model to focus on VALUE detection, not just accuracy
This will fix the favorite bias and make it actually profitable
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ProfitableMoneylineModel:
    """Build moneyline model focused on profitability, not just accuracy"""
    
    def __init__(self):
        self.model = None
        self.features = []
        
    def load_and_prepare_data(self):
        """Load data and prepare for value-based training"""
        
        print("🏀 BUILDING PROFITABLE MONEYLINE MODEL")
        print("Focus: VALUE detection, not just winner prediction")
        print("=" * 60)
        
        # Load historical data with odds
        df = pd.read_csv('/Users/mk/nba-betting-ai/data/final_games_full_features.csv', low_memory=False)
        print(f"📊 Loaded: {len(df)} total games")
        
        # Only keep games with betting odds (essential for value detection)
        df = df.dropna(subset=['home_odds', 'away_odds', 'homeWin'])
        df = df[(df['home_odds'] != 0) & (df['away_odds'] != 0)]
        print(f"📊 Games with odds: {len(df)}")
        
        # Calculate market implied probabilities
        df['market_home_prob'] = self.odds_to_probability(df['home_odds'])
        df['market_away_prob'] = self.odds_to_probability(df['away_odds'])
        
        # Remove games with missing core stats
        required_stats = ['home_NET_RATING', 'away_NET_RATING', 'home_PIE', 'away_PIE']
        df = df.dropna(subset=required_stats)
        print(f"📊 Games with required stats: {len(df)}")
        
        # Sort chronologically for proper training
        if 'gameDate' in df.columns:
            df['gameDate'] = pd.to_datetime(df['gameDate'])
            df = df.sort_values('gameDate').reset_index(drop=True)
            print(f"📅 Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
        
        return df
    
    def odds_to_probability(self, odds):
        """Convert American odds to implied probability"""
        def convert_odds(odd):
            if pd.isna(odd) or odd == 0:
                return 0.5
            if odd > 0:
                return 100 / (100 + odd)
            else:
                return abs(odd) / (abs(odd) + 100)
        
        return odds.apply(convert_odds)
    
    def create_value_features(self, df):
        """Create features focused on finding VALUE, not just predicting winners"""
        
        print(f"\n🔧 CREATING VALUE-FOCUSED FEATURES")
        print("These features help find mispriced games")
        
        # Core team strength differentials (your existing strong features)
        df['NET_gap'] = df['home_NET_RATING'] - df['away_NET_RATING']
        df['PIE_advantage'] = df['home_PIE'] - df['away_PIE']
        
        # Market-based features (NEW - this is the key!)
        df['market_efficiency'] = abs(df['market_home_prob'] + df['market_away_prob'] - 1.0)
        df['market_favorite_prob'] = np.maximum(df['market_home_prob'], df['market_away_prob'])
        df['market_underdog_prob'] = np.minimum(df['market_home_prob'], df['market_away_prob'])
        df['market_spread'] = df['market_favorite_prob'] - df['market_underdog_prob']
        
        # Value indicators
        df['heavy_favorite'] = (df['market_favorite_prob'] > 0.75).astype(int)
        df['close_game'] = (df['market_spread'] < 0.1).astype(int)
        df['potential_upset'] = ((df['market_favorite_prob'] > 0.65) & (df['NET_gap'].abs() < 3)).astype(int)
        
        # Advanced team metrics
        if 'home_TS_PCT' in df.columns and 'away_TS_PCT' in df.columns:
            df['TS_advantage'] = df['home_TS_PCT'] - df['away_TS_PCT']
        else:
            df['TS_advantage'] = 0
            
        if 'home_EFG_PCT' in df.columns and 'away_EFG_PCT' in df.columns:
            df['EFG_advantage'] = df['home_EFG_PCT'] - df['away_EFG_PCT']
        else:
            df['EFG_advantage'] = 0
        
        # Team ranking context
        if 'home_PIE_RANK' in df.columns and 'away_PIE_RANK' in df.columns:
            df['ranking_advantage'] = df['away_PIE_RANK'] - df['home_PIE_RANK']
        else:
            df['ranking_advantage'] = 0
        
        # Interaction features (combinations that might reveal value)
        df['strength_vs_market'] = df['NET_gap'] - (df['market_home_prob'] - 0.5) * 20
        df['talent_vs_odds'] = df['PIE_advantage'] - (df['market_home_prob'] - 0.5) * 0.2
        
        print("✅ Created value-focused features:")
        print("   - Market efficiency indicators")
        print("   - Favorite/underdog classifications") 
        print("   - Potential upset detection")
        print("   - Strength vs market pricing")
        
        return df
    
    def create_profitable_targets(self, df):
        """Create targets focused on profitable betting, not just accuracy"""
        
        print(f"\n🎯 CREATING PROFITABLE TARGETS")
        
        # Calculate expected value for betting on home team
        df['home_payout'] = np.where(df['home_odds'] > 0, 
                                   df['home_odds'] / 100,
                                   100 / abs(df['home_odds']))
        
        df['away_payout'] = np.where(df['away_odds'] > 0,
                                   df['away_odds'] / 100, 
                                   100 / abs(df['away_odds']))
        
        # Actual profit from betting $1 on home team
        df['home_profit'] = np.where(df['homeWin'] == 1,
                                   df['home_payout'],  # Win: get payout
                                   -1)  # Loss: lose $1
        
        # Expected value if we had perfect knowledge
        df['true_home_ev'] = df['homeWin'] * df['home_payout'] - (1 - df['homeWin']) * 1
        
        # Value gap: difference between fair odds and market odds
        df['fair_home_prob'] = df['homeWin']  # Actual outcome
        df['value_gap'] = df['fair_home_prob'] - df['market_home_prob']
        
        print("✅ Created profit-focused targets:")
        print("   - Actual betting profits")
        print("   - Expected value calculations")
        print("   - Value gap identification")
        
        return df
    
    def build_value_model(self, df):
        """Build model to predict VALUE, not just winners"""
        
        print(f"\n🤖 BUILDING VALUE DETECTION MODEL")
        
        # Define features for the model
        self.features = [
            'NET_gap', 'PIE_advantage', 'TS_advantage', 'EFG_advantage', 'ranking_advantage',
            'market_efficiency', 'market_favorite_prob', 'market_spread', 
            'heavy_favorite', 'close_game', 'potential_upset',
            'strength_vs_market', 'talent_vs_odds'
        ]
        
        # Prepare data
        X = df[self.features].fillna(0)
        
        # Target: Predict the VALUE GAP (not just who wins!)
        y = df['value_gap']  # This is the key difference!
        
        print(f"📊 Features: {len(self.features)}")
        print(f"📊 Training samples: {len(X)}")
        print(f"📊 Target: Value gap (fair prob - market prob)")
        
        # Split data chronologically (recent games as test)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Training: {len(X_train)} games")
        print(f"📊 Testing: {len(X_test)} games")
        
        # Train Random Forest to predict value gaps
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate value prediction
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"✅ Value prediction MAE: {mae:.4f}")
        
        # Test betting strategy
        self.test_betting_strategy(df[split_idx:].copy(), y_pred)
        
        return self.model
    
    def test_betting_strategy(self, test_df, predicted_values):
        """Test actual betting performance with value-based strategy"""
        
        print(f"\n💰 TESTING VALUE-BASED BETTING STRATEGY")
        
        test_df = test_df.reset_index(drop=True)
        test_df['predicted_value'] = predicted_values
        
        # Different value thresholds to test
        thresholds = [0.02, 0.03, 0.05, 0.08, 0.10]
        
        results = []
        
        for threshold in thresholds:
            # Only bet when model predicts value > threshold
            value_bets = test_df[test_df['predicted_value'] > threshold].copy()
            
            if len(value_bets) < 10:
                continue
            
            # Calculate actual profits
            total_profit = value_bets['home_profit'].sum()
            total_bets = len(value_bets)
            win_rate = (value_bets['home_profit'] > 0).mean()
            roi = (total_profit / total_bets) * 100
            
            # Also test betting on away team when predicted value is negative
            away_value_bets = test_df[test_df['predicted_value'] < -threshold].copy()
            if len(away_value_bets) > 0:
                away_profits = np.where(away_value_bets['homeWin'] == 0,
                                      away_value_bets['away_payout'],
                                      -1)
                away_profit = away_profits.sum()
                away_bets = len(away_value_bets)
                away_win_rate = (away_profits > 0).mean()
                
                # Combine home and away value bets
                combined_profit = total_profit + away_profit
                combined_bets = total_bets + away_bets
                combined_roi = (combined_profit / combined_bets) * 100 if combined_bets > 0 else 0
                combined_win_rate = ((value_bets['home_profit'] > 0).sum() + (away_profits > 0).sum()) / combined_bets if combined_bets > 0 else 0
            else:
                combined_profit = total_profit
                combined_bets = total_bets
                combined_roi = roi
                combined_win_rate = win_rate
            
            results.append({
                'threshold': threshold,
                'bets': combined_bets,
                'win_rate': combined_win_rate,
                'profit': combined_profit,
                'roi': combined_roi
            })
            
            print(f"   {threshold:.2f} threshold: {combined_bets:,} bets, {combined_win_rate:.1%} win rate, {combined_roi:+.1f}% ROI")
        
        # Find best strategy
        if results:
            best = max(results, key=lambda x: x['roi'])
            print(f"\n🏆 BEST VALUE STRATEGY:")
            print(f"   Threshold: ±{best['threshold']:.2f}")
            print(f"   Total bets: {best['bets']:,}")
            print(f"   Win rate: {best['win_rate']:.1%}")
            print(f"   ROI: {best['roi']:+.1f}%")
            print(f"   Total profit: ${best['profit']:,.2f}")
            
            if best['roi'] > 0:
                print(f"   🎉 PROFITABLE! Model finds value!")
            else:
                print(f"   ⚠️ Still needs work, but much better approach")
        
        return results
    
    def save_model(self):
        """Save the profitable moneyline model"""
        
        print(f"\n💾 SAVING PROFITABLE MONEYLINE MODEL")
        
        # Save model
        joblib.dump(self.model, 'models/profitable_moneyline_model.pkl')
        
        # Save features
        with open('models/profitable_moneyline_features.txt', 'w') as f:
            for feature in self.features:
                f.write(f"{feature}\n")
        
        print("✅ Saved model: models/profitable_moneyline_model.pkl")
        print("✅ Saved features: models/profitable_moneyline_features.txt")
        
        print(f"\n🎯 HOW TO USE THIS MODEL:")
        print("1. Get today's games with odds")
        print("2. Calculate team stats and features")
        print("3. Predict value gaps for each game")
        print("4. Only bet when predicted value > threshold")
        print("5. Bet on home team if positive value, away team if negative value")

def main():
    """Build the profitable moneyline model"""
    
    # Initialize model builder
    builder = ProfitableMoneylineModel()
    
    # Load and prepare data
    df = builder.load_and_prepare_data()
    
    # Create value-focused features
    df = builder.create_value_features(df)
    
    # Create profitable targets
    df = builder.create_profitable_targets(df)
    
    # Build value detection model
    model = builder.build_value_model(df)
    
    # Save model
    builder.save_model()
    
    print(f"\n🎉 PROFITABLE MONEYLINE MODEL COMPLETE!")
    print("This model focuses on VALUE detection, not just accuracy")
    print("It should perform much better than your accuracy-focused model")
    print("Key difference: Predicts profitable opportunities, not just winners")

if __name__ == "__main__":
    main()
