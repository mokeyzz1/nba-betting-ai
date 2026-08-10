#!/usr/bin/env python3
"""
COMPREHENSIVE PLAYER PROPS ACCURACY TEST
Tests Player Props models against historical data to validate real-world performance
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
sys.path.append('/Users/mk/nba-betting-ai')
from src.prediction.predict_player_props import PlayerPropsPredictor

class PlayerPropsAccuracyTest:
    """Test Player Props models against historical performance"""
    
    def __init__(self):
        """Initialize test with player data"""
        print("🧪 PLAYER PROPS ACCURACY TEST")
        print("=" * 60)
        
        # Load historical player data
        self.load_test_data()
        
        # Initialize predictor
        self.predictor = PlayerPropsPredictor()
        
    def load_test_data(self):
        """Load historical player performance data for testing"""
        
        try:
            # Try to load your player statistics data
            data_files = [
                '/Users/mk/nba-betting-ai/data/PlayerStatistics.csv',
                '/Users/mk/nba-betting-ai/data/data/Players.csv',
                '/Users/mk/nba-betting-ai/data/player_stats.csv'
            ]
            
            self.player_data = None
            for file_path in data_files:
                if os.path.exists(file_path):
                    print(f"📂 Loading player data: {file_path}")
                    self.player_data = pd.read_csv(file_path)
                    break
            
            if self.player_data is None:
                print("⚠️ No player data file found, using synthetic test data")
                self.create_synthetic_test_data()
            else:
                print(f"✅ Loaded {len(self.player_data)} player performances")
                self.prepare_real_data()
                
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            self.create_synthetic_test_data()
    
    def prepare_real_data(self):
        """Prepare real player data for testing"""
        
        # Clean column names
        self.player_data.columns = [col.lower().strip() for col in self.player_data.columns]
        
        # Create test cases from last 30 days of data
        if 'gamedate' in self.player_data.columns:
            self.player_data['gamedate'] = pd.to_datetime(self.player_data['gamedate'])
            recent_data = self.player_data[
                self.player_data['gamedate'] >= (datetime.now() - timedelta(days=30))
            ]
        else:
            # Use last 1000 games if no date column
            recent_data = self.player_data.tail(1000)
        
        # If no recent data, fall back to synthetic
        if len(recent_data) < 100:
            print("⚠️ Insufficient recent data, using synthetic test data")
            self.create_synthetic_test_data()
            return
        
        print(f"📊 Using {len(recent_data)} recent performances for testing")
        self.test_cases = recent_data
    
    def create_synthetic_test_data(self):
        """Create realistic synthetic player performance data for testing"""
        
        print("🎲 Creating synthetic test data...")
        
        # Create realistic player performance scenarios
        test_scenarios = []
        
        # Star Players (high production)
        for i in range(50):
            scenario = {
                'player_name': f'Star Player {i+1}',
                'actual_points': np.random.normal(27.5, 6.0),
                'actual_rebounds': np.random.normal(8.5, 3.0),
                'actual_assists': np.random.normal(7.2, 2.5),
                'season_ppg': np.random.normal(26.8, 4.0),
                'season_rpg': np.random.normal(8.1, 2.5),
                'season_apg': np.random.normal(6.9, 2.0),
                'recent_form': np.random.uniform(0.8, 1.2),
                'home_game': np.random.choice([0, 1]),
                'player_type': 'star'
            }
            test_scenarios.append(scenario)
        
        # Role Players (lower production)
        for i in range(100):
            scenario = {
                'player_name': f'Role Player {i+1}',
                'actual_points': np.random.normal(12.3, 4.5),
                'actual_rebounds': np.random.normal(4.8, 2.2),
                'actual_assists': np.random.normal(2.1, 1.5),
                'season_ppg': np.random.normal(11.9, 3.5),
                'season_rpg': np.random.normal(4.5, 2.0),
                'season_apg': np.random.normal(2.0, 1.2),
                'recent_form': np.random.uniform(0.7, 1.3),
                'home_game': np.random.choice([0, 1]),
                'player_type': 'role'
            }
            test_scenarios.append(scenario)
        
        # Bench Players (very low production)
        for i in range(25):
            scenario = {
                'player_name': f'Bench Player {i+1}',
                'actual_points': np.random.normal(6.8, 3.2),
                'actual_rebounds': np.random.normal(2.3, 1.5),
                'actual_assists': np.random.normal(0.9, 0.8),
                'season_ppg': np.random.normal(6.5, 2.8),
                'season_rpg': np.random.normal(2.1, 1.2),
                'season_apg': np.random.normal(0.8, 0.6),
                'recent_form': np.random.uniform(0.6, 1.4),
                'home_game': np.random.choice([0, 1]),
                'player_type': 'bench'
            }
            test_scenarios.append(scenario)
        
        self.test_cases = pd.DataFrame(test_scenarios)
        print(f"✅ Created {len(self.test_cases)} synthetic test cases")
    
    def create_features_from_test_case(self, test_case):
        """Convert test case to model features"""
        
        if isinstance(test_case, pd.Series):
            case = test_case
        else:
            case = test_case.iloc[0] if hasattr(test_case, 'iloc') else test_case
        
        # Create features based on available data
        features = {}
        
        # Points features
        base_ppg = case.get('season_ppg', case.get('seasonppg', 15.0))
        form_factor = case.get('recent_form', 1.0)
        
        features.update({
            'points_last_5': base_ppg * form_factor,
            'points_last_10': base_ppg * (form_factor * 0.95 + 0.05),
            'points_last_15': base_ppg,
            'minutes_last_5': base_ppg * 1.4,  # Rough minutes correlation
            'minutes_last_10': base_ppg * 1.35,
            'fg_pct_last_5': 0.45 + (base_ppg - 15) * 0.01,
            'fg_attempts_last_5': base_ppg * 0.8,
            'points_trend_3': (form_factor - 1.0) * 5,
            'home_away_diff': 2.1 if case.get('home_game', 1) else -1.8,
            'season_ppg': base_ppg,
            'season_mpg': base_ppg * 1.3,
            'points_consistency': abs(np.random.normal(0, 3)),
            'opponent_strength': np.random.uniform(0.3, 0.7),
            'home': case.get('home_game', 1)
        })
        
        # Rebounds features
        base_rpg = case.get('season_rpg', case.get('seasonrpg', 4.5))
        features.update({
            'rebounds_last_5': base_rpg * form_factor,
            'rebounds_last_10': base_rpg * (form_factor * 0.95 + 0.05),
            'rebounds_last_15': base_rpg,
            'oreb_last_5': base_rpg * 0.25,
            'dreb_last_5': base_rpg * 0.75,
            'rebounds_trend_3': (form_factor - 1.0) * 2,
            'home_away_reb_diff': 1.2 if case.get('home_game', 1) else -0.8,
            'season_rpg': base_rpg,
            'season_oreb_pct': base_rpg * 0.2,
            'season_dreb_pct': base_rpg * 0.8,
            'rebounds_consistency': abs(np.random.normal(0, 1.5)),
            'position_big': 1 if base_rpg > 8 else 0,
            'position_wing': 1 if 4 < base_rpg <= 8 else 0
        })
        
        # Assists features
        base_apg = case.get('season_apg', case.get('seasonapg', 2.5))
        features.update({
            'assists_last_5': base_apg * form_factor,
            'assists_last_10': base_apg * (form_factor * 0.95 + 0.05),
            'assists_last_15': base_apg,
            'fga_last_5': base_ppg * 0.8,
            'turnovers_last_5': base_apg * 0.5,
            'ast_to_last_5': base_apg / max(base_apg * 0.5, 1),
            'assists_trend_3': (form_factor - 1.0) * 1.5,
            'home_away_ast_diff': 0.8 if case.get('home_game', 1) else -0.5,
            'season_apg': base_apg,
            'season_ast_to': 2.0,
            'assists_consistency': abs(np.random.normal(0, 1)),
            'position_playmaker': 1 if base_apg > 5 else 0,
            'position_secondary': 1 if 2 < base_apg <= 5 else 0
        })
        
        return features
    
    def test_prediction_accuracy(self):
        """Test model predictions against actual outcomes"""
        
        print(f"\n🎯 TESTING PREDICTION ACCURACY")
        print("=" * 60)
        
        results = {
            'points': {'correct': 0, 'total': 0, 'mae': []},
            'rebounds': {'correct': 0, 'total': 0, 'mae': []},
            'assists': {'correct': 0, 'total': 0, 'mae': []}
        }
        
        line_accuracy = {
            'points': {15.5: {'correct': 0, 'total': 0}, 20.5: {'correct': 0, 'total': 0}, 
                      25.5: {'correct': 0, 'total': 0}, 30.5: {'correct': 0, 'total': 0}},
            'rebounds': {4.5: {'correct': 0, 'total': 0}, 6.5: {'correct': 0, 'total': 0}, 
                        8.5: {'correct': 0, 'total': 0}, 10.5: {'correct': 0, 'total': 0}, 12.5: {'correct': 0, 'total': 0}},
            'assists': {2.5: {'correct': 0, 'total': 0}, 3.5: {'correct': 0, 'total': 0}, 
                       4.5: {'correct': 0, 'total': 0}, 5.5: {'correct': 0, 'total': 0}, 
                       6.5: {'correct': 0, 'total': 0}, 7.5: {'correct': 0, 'total': 0}}
        }
        
        # Test each case
        for i, test_case in self.test_cases.iterrows():
            if i % 25 == 0:
                print(f"📊 Testing case {i+1}/{len(self.test_cases)}")
            
            try:
                # Get actual performance
                actual_points = test_case.get('actual_points', test_case.get('points', 0))
                actual_rebounds = test_case.get('actual_rebounds', test_case.get('rebounds', 0))
                actual_assists = test_case.get('actual_assists', test_case.get('assists', 0))
                
                # Create features
                features = self.create_features_from_test_case(test_case)
                
                # Make prediction
                player_name = test_case.get('player_name', f'Player {i}')
                prediction = self.predictor.predict_player_props(features, player_name)
                
                # Test Points
                if 'predicted' in prediction['points']:
                    pred_points = prediction['points']['predicted']
                    results['points']['mae'].append(abs(pred_points - actual_points))
                    
                    # Test each line
                    for line in [15.5, 20.5, 25.5, 30.5]:
                        if str(line) in prediction['points']:
                            pred_over = prediction['points'][str(line)]['recommendation'] == 'OVER'
                            actual_over = actual_points > line
                            line_accuracy['points'][line]['total'] += 1
                            if pred_over == actual_over:
                                line_accuracy['points'][line]['correct'] += 1
                
                # Test Rebounds
                if 'predicted' in prediction['rebounds']:
                    pred_rebounds = prediction['rebounds']['predicted']
                    results['rebounds']['mae'].append(abs(pred_rebounds - actual_rebounds))
                    
                    for line in [4.5, 6.5, 8.5, 10.5, 12.5]:
                        if str(line) in prediction['rebounds']:
                            pred_over = prediction['rebounds'][str(line)]['recommendation'] == 'OVER'
                            actual_over = actual_rebounds > line
                            line_accuracy['rebounds'][line]['total'] += 1
                            if pred_over == actual_over:
                                line_accuracy['rebounds'][line]['correct'] += 1
                
                # Test Assists
                if 'predicted' in prediction['assists']:
                    pred_assists = prediction['assists']['predicted']
                    results['assists']['mae'].append(abs(pred_assists - actual_assists))
                    
                    for line in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
                        if str(line) in prediction['assists']:
                            pred_over = prediction['assists'][str(line)]['recommendation'] == 'OVER'
                            actual_over = actual_assists > line
                            line_accuracy['assists'][line]['total'] += 1
                            if pred_over == actual_over:
                                line_accuracy['assists'][line]['correct'] += 1
                                
            except Exception as e:
                print(f"⚠️ Error testing case {i}: {e}")
                continue
        
        return results, line_accuracy
    
    def display_results(self, results, line_accuracy):
        """Display comprehensive test results"""
        
        print(f"\n📊 PLAYER PROPS ACCURACY TEST RESULTS")
        print("=" * 80)
        
        # Overall MAE results
        for stat_type in ['points', 'rebounds', 'assists']:
            if results[stat_type]['mae']:
                mae = np.mean(results[stat_type]['mae'])
                print(f"🎯 {stat_type.upper()} MAE: {mae:.2f}")
        
        print(f"\n📈 LINE-BY-LINE ACCURACY:")
        print("-" * 50)
        
        # Points lines
        print("🏀 POINTS LINES:")
        for line in [15.5, 20.5, 25.5, 30.5]:
            if line_accuracy['points'][line]['total'] > 0:
                accuracy = line_accuracy['points'][line]['correct'] / line_accuracy['points'][line]['total']
                print(f"   {line:>4} line: {accuracy:.1%} ({line_accuracy['points'][line]['correct']}/{line_accuracy['points'][line]['total']})")
        
        # Rebounds lines
        print("\n🏀 REBOUNDS LINES:")
        for line in [4.5, 6.5, 8.5, 10.5, 12.5]:
            if line_accuracy['rebounds'][line]['total'] > 0:
                accuracy = line_accuracy['rebounds'][line]['correct'] / line_accuracy['rebounds'][line]['total']
                print(f"   {line:>4} line: {accuracy:.1%} ({line_accuracy['rebounds'][line]['correct']}/{line_accuracy['rebounds'][line]['total']})")
        
        # Assists lines
        print("\n🎯 ASSISTS LINES:")
        for line in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
            if line_accuracy['assists'][line]['total'] > 0:
                accuracy = line_accuracy['assists'][line]['correct'] / line_accuracy['assists'][line]['total']
                print(f"   {line:>4} line: {accuracy:.1%} ({line_accuracy['assists'][line]['correct']}/{line_accuracy['assists'][line]['total']})")
        
        # Calculate overall accuracy
        total_correct = sum(
            sum(line_data['correct'] for line_data in prop_lines.values()) 
            for prop_lines in line_accuracy.values()
        )
        total_predictions = sum(
            sum(line_data['total'] for line_data in prop_lines.values()) 
            for prop_lines in line_accuracy.values()
        )
        
        if total_predictions > 0:
            overall_accuracy = total_correct / total_predictions
            print(f"\n🎯 OVERALL ACCURACY: {overall_accuracy:.1%} ({total_correct}/{total_predictions})")
        
        # Profitability estimation
        print(f"\n💰 ESTIMATED PROFITABILITY:")
        profitable_lines = 0
        total_lines = 0
        
        for prop_type in line_accuracy:
            for line, data in line_accuracy[prop_type].items():
                if data['total'] > 10:  # Only count lines with sufficient data
                    accuracy = data['correct'] / data['total']
                    total_lines += 1
                    if accuracy > 0.52:  # Profitable threshold (accounting for juice)
                        profitable_lines += 1
        
        if total_lines > 0:
            profit_rate = profitable_lines / total_lines
            print(f"   {profitable_lines}/{total_lines} lines profitable ({profit_rate:.1%})")
            
            if profit_rate > 0.6:
                print("   🔥 EXCELLENT - High profit potential!")
            elif profit_rate > 0.4:
                print("   ✅ GOOD - Solid profit potential")
            else:
                print("   ⚠️ NEEDS IMPROVEMENT - Limited profit potential")

def main():
    """Run comprehensive Player Props accuracy test"""
    
    # Initialize test
    tester = PlayerPropsAccuracyTest()
    
    # Run accuracy test
    results, line_accuracy = tester.test_prediction_accuracy()
    
    # Display results
    tester.display_results(results, line_accuracy)
    
    print(f"\n✅ PLAYER PROPS ACCURACY TEST COMPLETE!")
    print("🎯 Models tested against realistic game scenarios")
    print("💡 Use these results to validate betting confidence")

if __name__ == "__main__":
    main()
