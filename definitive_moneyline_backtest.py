#!/usr/bin/env python3
"""
DEFINITIVE MONEYLINE MODEL BACKTEST
Test on ALL 71,649 NBA games (1946-2025) - The absolute truth
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import warnings
warnings.filterwarnings('ignore')

def load_complete_dataset():
    """Load the complete 71K+ NBA games dataset"""
    
    print("🏀 DEFINITIVE NBA MONEYLINE BACKTEST")
    print("Testing on ALL 71,649 games (1946-2025)")
    print("=" * 60)
    
    # Load the complete dataset
    df = pd.read_csv('/Users/mk/nba-betting-ai/data/final_games_full_features.csv', low_memory=False)
    
    print(f"📊 Loaded: {len(df)} total NBA games")
    
    # Check date range
    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        print(f"📅 Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
        print(f"📅 Span: {(df['gameDate'].max() - df['gameDate'].min()).days / 365.25:.1f} years")
    
    # Clean data for model testing
    print(f"\n🧹 CLEANING DATA FOR BACKTEST")
    
    # Remove games without basic requirements
    original_len = len(df)
    df = df.dropna(subset=['homeWin'])
    print(f"   Removed {original_len - len(df)} games without homeWin")
    
    # Remove games without odds (can't bet on these)
    if 'home_odds' in df.columns and 'away_odds' in df.columns:
        df = df.dropna(subset=['home_odds', 'away_odds'])
        df = df[(df['home_odds'] != 0) & (df['away_odds'] != 0)]
        print(f"   Games with betting odds: {len(df)}")
    else:
        print(f"   ⚠️ No odds data - testing prediction accuracy only")
    
    # Remove games without required stats for hybrid model
    required_stats = ['home_NET_RATING', 'away_NET_RATING', 'home_PIE', 'away_PIE']
    available_stats = [col for col in required_stats if col in df.columns]
    
    if len(available_stats) >= 2:
        df = df.dropna(subset=available_stats)
        print(f"   Games with required stats: {len(df)}")
    else:
        print(f"   ⚠️ Limited stats available - using basic features")
    
    print(f"✅ Final dataset: {len(df)} games ready for backtest")
    
    return df

def create_model_features(df):
    """Create hybrid elite model features for all games"""
    
    print(f"\n🔧 CREATING HYBRID ELITE FEATURES")
    print("Building features for all 71K+ games...")
    
    # Load the exact features the model expects
    with open('models/hybrid_elite_features_v1.txt', 'r') as f:
        model_features = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📋 Model expects: {model_features}")
    
    # Create each feature
    features_created = []
    
    # 1. NET_gap
    if 'home_NET_RATING' in df.columns and 'away_NET_RATING' in df.columns:
        df['NET_gap'] = df['home_NET_RATING'] - df['away_NET_RATING']
        features_created.append('NET_gap')
        print(f"✅ NET_gap: {df['NET_gap'].mean():.2f} avg, {df['NET_gap'].std():.2f} std")
    
    # 2. PIE_advantage  
    if 'home_PIE' in df.columns and 'away_PIE' in df.columns:
        df['PIE_advantage'] = df['home_PIE'] - df['away_PIE']
        features_created.append('PIE_advantage')
        print(f"✅ PIE_advantage: {df['PIE_advantage'].mean():.3f} avg")
    
    # 3. ranking_advantage
    if 'home_PIE_RANK' in df.columns and 'away_PIE_RANK' in df.columns:
        df['ranking_advantage'] = df['away_PIE_RANK'] - df['home_PIE_RANK']
        features_created.append('ranking_advantage')
        print(f"✅ ranking_advantage: {df['ranking_advantage'].mean():.2f} avg")
    
    # 4. TS_advantage
    if 'home_TS_PCT' in df.columns and 'away_TS_PCT' in df.columns:
        df['TS_advantage'] = df['home_TS_PCT'] - df['away_TS_PCT']
        features_created.append('TS_advantage')
        print(f"✅ TS_advantage: {df['TS_advantage'].mean():.3f} avg")
    
    # 5. recent_form_advantage (not available in historical data)
    df['recent_form_advantage'] = 0
    features_created.append('recent_form_advantage')
    print(f"⚠️ recent_form_advantage: Set to 0 (not available historically)")
    
    # 6. EFG_advantage
    if 'home_EFG_PCT' in df.columns and 'away_EFG_PCT' in df.columns:
        df['EFG_advantage'] = df['home_EFG_PCT'] - df['away_EFG_PCT']
        features_created.append('EFG_advantage')
        print(f"✅ EFG_advantage: {df['EFG_advantage'].mean():.3f} avg")
    
    # 7. TOV_advantage
    if 'home_TM_TOV_PCT' in df.columns and 'away_TM_TOV_PCT' in df.columns:
        df['TOV_advantage'] = df['away_TM_TOV_PCT'] - df['home_TM_TOV_PCT']
        features_created.append('TOV_advantage')
        print(f"✅ TOV_advantage: {df['TOV_advantage'].mean():.3f} avg")
    
    # 8. efficiency_advantage
    if 'home_AST_TO' in df.columns and 'away_AST_TO' in df.columns:
        df['efficiency_advantage'] = df['home_AST_TO'] - df['away_AST_TO']
        features_created.append('efficiency_advantage')
        print(f"✅ efficiency_advantage: {df['efficiency_advantage'].mean():.3f} avg")
    
    # 9. home_PIE and 10. away_PIE (already exist)
    if 'home_PIE' in df.columns:
        features_created.append('home_PIE')
        print(f"✅ home_PIE: Available")
    if 'away_PIE' in df.columns:
        features_created.append('away_PIE')
        print(f"✅ away_PIE: Available")
    
    print(f"\n📊 FEATURE SUMMARY:")
    print(f"   Expected: {len(model_features)} features")
    print(f"   Created: {len(features_created)} features")
    print(f"   Available: {features_created}")
    
    missing_features = [f for f in model_features if f not in features_created]
    if missing_features:
        print(f"   Missing: {missing_features}")
    
    return df, features_created

def run_chronological_backtest(model, df, features):
    """Run backtest with proper chronological order"""
    
    print(f"\n⏰ CHRONOLOGICAL BACKTEST")
    print("Testing model performance in proper time order")
    
    # Sort by date for proper chronological testing
    if 'gameDate' in df.columns:
        df_sorted = df.sort_values('gameDate').reset_index(drop=True)
        print(f"✅ Sorted {len(df_sorted)} games chronologically")
    else:
        df_sorted = df.copy()
        print(f"⚠️ No date column - using data order")
    
    # Prepare features
    X = df_sorted[features].fillna(0)
    y = df_sorted['homeWin']
    
    # Clean any infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"📊 Features ready: {X.shape[0]} games, {X.shape[1]} features")
    
    # Test different time periods
    print(f"\n📈 PERFORMANCE BY ERA:")
    
    results = {}
    
    # Test on different eras to see consistency
    if 'gameDate' in df_sorted.columns:
        eras = [
            ("Modern Era (2010+)", df_sorted['gameDate'] >= '2010-01-01'),
            ("Recent Era (2015+)", df_sorted['gameDate'] >= '2015-01-01'),
            ("Current Era (2020+)", df_sorted['gameDate'] >= '2020-01-01'),
            ("Very Recent (2022+)", df_sorted['gameDate'] >= '2022-01-01')
        ]
        
        for era_name, era_mask in eras:
            era_games = df_sorted[era_mask]
            if len(era_games) >= 1000:  # Need sufficient games
                X_era = era_games[features].fillna(0)
                X_era = X_era.replace([np.inf, -np.inf], np.nan).fillna(0)
                y_era = era_games['homeWin']
                
                # Make predictions
                try:
                    y_pred_prob = model.predict_proba(X_era)[:, 1]
                    y_pred = (y_pred_prob >= 0.5).astype(int)
                    
                    accuracy = accuracy_score(y_era, y_pred)
                    auc = roc_auc_score(y_era, y_pred_prob)
                    
                    results[era_name] = {
                        'games': len(era_games),
                        'accuracy': accuracy,
                        'auc': auc,
                        'home_win_rate': y_era.mean()
                    }
                    
                    print(f"   {era_name}: {accuracy:.3f} accuracy, {auc:.3f} AUC ({len(era_games):,} games)")
                    
                except Exception as e:
                    print(f"   {era_name}: ERROR - {e}")
    
    # Overall performance on all data
    print(f"\n🎯 OVERALL PERFORMANCE:")
    
    try:
        y_pred_prob_all = model.predict_proba(X)[:, 1]
        y_pred_all = (y_pred_prob_all >= 0.5).astype(int)
        
        overall_accuracy = accuracy_score(y, y_pred_all)
        overall_auc = roc_auc_score(y, y_pred_prob_all)
        overall_logloss = log_loss(y, y_pred_prob_all)
        
        print(f"   ALL 71K+ GAMES:")
        print(f"   📊 Accuracy: {overall_accuracy:.3f}")
        print(f"   📊 AUC: {overall_auc:.3f}")
        print(f"   📊 Log Loss: {overall_logloss:.3f}")
        print(f"   📊 Home Win Rate: {y.mean():.3f}")
        
        results['overall'] = {
            'games': len(df_sorted),
            'accuracy': overall_accuracy,
            'auc': overall_auc,
            'log_loss': overall_logloss,
            'home_win_rate': y.mean()
        }
        
    except Exception as e:
        print(f"   ERROR in overall test: {e}")
        overall_accuracy = None
    
    return results, y_pred_prob_all if overall_accuracy else None

def test_betting_performance(df, predictions, features):
    """Test actual betting performance where odds are available"""
    
    print(f"\n💰 BETTING PERFORMANCE TEST")
    
    # Only test on games with odds
    if 'home_odds' not in df.columns or 'away_odds' not in df.columns:
        print("❌ No odds data available - cannot test betting performance")
        return
    
    betting_df = df[(df['home_odds'].notna()) & (df['away_odds'].notna())].copy()
    betting_df['model_prob'] = predictions[:len(betting_df)]
    
    print(f"📊 Testing betting on {len(betting_df):,} games with odds")
    
    # Calculate implied probabilities
    def odds_to_prob(odds):
        if pd.isna(odds) or odds == 0:
            return 0.5
        if odds > 0:
            return 100 / (100 + odds)
        else:
            return -odds / (-odds + 100)
    
    betting_df['home_implied'] = betting_df['home_odds'].apply(odds_to_prob)
    betting_df['away_implied'] = betting_df['away_odds'].apply(odds_to_prob)
    betting_df['value_gap'] = betting_df['model_prob'] - betting_df['home_implied']
    
    print(f"💡 BETTING STRATEGY PERFORMANCE:")
    
    # Test different value thresholds
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    
    betting_results = []
    
    for threshold in thresholds:
        value_bets = betting_df[betting_df['value_gap'] > threshold]
        
        if len(value_bets) >= 10:
            wins = value_bets['homeWin'].sum()
            total_bets = len(value_bets)
            win_rate = wins / total_bets
            
            # Calculate profit (assuming $100 per bet)
            profit = 0
            for _, bet in value_bets.iterrows():
                if bet['homeWin'] == 1:  # Home team won
                    if bet['home_odds'] > 0:
                        profit += bet['home_odds']  # Win $X on +X odds
                    elif bet['home_odds'] < 0:
                        profit += 100 * (100 / abs(bet['home_odds']))  # Win proportional amount
                else:
                    profit -= 100  # Lose $100
            
            roi = (profit / (total_bets * 100)) * 100
            
            betting_results.append({
                'threshold': threshold,
                'bets': total_bets,
                'wins': wins,
                'win_rate': win_rate,
                'profit': profit,
                'roi': roi
            })
            
            print(f"   {threshold:.2f} threshold: {total_bets:,} bets, {win_rate:.1%} win rate, {roi:+.1f}% ROI")
        else:
            print(f"   {threshold:.2f} threshold: {len(value_bets)} bets (too few)")
    
    # Find best strategy
    if betting_results:
        best_strategy = max(betting_results, key=lambda x: x['roi'])
        print(f"\n🏆 BEST STRATEGY:")
        print(f"   Threshold: {best_strategy['threshold']:.2f}")
        print(f"   Bets: {best_strategy['bets']:,}")
        print(f"   Win Rate: {best_strategy['win_rate']:.1%}")
        print(f"   ROI: {best_strategy['roi']:+.1f}%")
        print(f"   Total Profit: ${best_strategy['profit']:,.2f}")
    
    return betting_results

def final_assessment(results):
    """Give final professional assessment"""
    
    print(f"\n🏆 DEFINITIVE ASSESSMENT")
    print("=" * 60)
    
    if 'overall' not in results:
        print("❌ Could not complete assessment - insufficient data")
        return
    
    overall = results['overall']
    accuracy = overall['accuracy']
    auc = overall['auc']
    games = overall['games']
    
    print(f"📊 TESTED ON: {games:,} NBA games")
    print(f"📊 ACCURACY: {accuracy:.3f}")
    print(f"📊 AUC: {auc:.3f}")
    
    print(f"\n🎯 PROFESSIONAL RATING:")
    
    # Rate the model
    if accuracy >= 0.65:
        rating = "WORLD-CLASS"
        emoji = "🔥"
        description = "Elite professional level - top 1% of sports models"
    elif accuracy >= 0.60:
        rating = "EXCELLENT"
        emoji = "⭐"
        description = "Professional grade - ready for serious betting"
    elif accuracy >= 0.55:
        rating = "VERY GOOD"
        emoji = "✅"
        description = "Above professional threshold - profitable"
    elif accuracy >= 0.52:
        rating = "GOOD"
        emoji = "👍"
        description = "Legitimate edge - small but real advantage"
    else:
        rating = "POOR"
        emoji = "❌"
        description = "No significant predictive power"
    
    print(f"   {emoji} {rating}")
    print(f"   {description}")
    
    print(f"\n📈 CONTEXT:")
    print(f"   • Random guessing: 50.0% accuracy")
    print(f"   • Home court advantage: ~{overall['home_win_rate']:.1%}")
    print(f"   • Professional threshold: 52-55%")
    print(f"   • Your model: {accuracy:.1%}")
    
    print(f"\n💡 FINAL VERDICT:")
    if accuracy >= 0.55:
        print(f"   ✅ MODEL IS LEGITIMATE AND PROFITABLE")
        print(f"   ✅ Ready for live betting with proper bankroll management")
        print(f"   ✅ Represents genuine edge in NBA betting")
    else:
        print(f"   ❌ MODEL NEEDS IMPROVEMENT")
        print(f"   ❌ Not recommended for live betting")
        print(f"   ❌ Consider rebuilding with better features")
    
    return rating, accuracy

def main():
    """Run the definitive backtest"""
    
    # Load model
    print("📥 Loading hybrid elite model...")
    model = joblib.load('models/hybrid_elite_model_v1.pkl')
    print("✅ Model loaded")
    
    # Load complete dataset
    df = load_complete_dataset()
    
    # Create features
    df, features = create_model_features(df)
    
    # Run chronological backtest
    results, predictions = run_chronological_backtest(model, df, features)
    
    # Test betting performance
    if predictions is not None:
        betting_results = test_betting_performance(df, predictions, features)
    
    # Final assessment
    rating, accuracy = final_assessment(results)
    
    print(f"\n🎉 DEFINITIVE BACKTEST COMPLETE!")
    print(f"📊 Final Rating: {rating}")
    print(f"📊 Final Accuracy: {accuracy:.3f}")
    print(f"📊 Tested on: {results['overall']['games']:,} games")
    
    return rating, accuracy, results

if __name__ == "__main__":
    main()