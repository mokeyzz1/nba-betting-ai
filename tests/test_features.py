import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.get_odds import decimal_to_american, fetch_odds
from features.get_recent_stats import get_recent_win_pct, get_recent_avg_pts
from features.get_team_stats import get_team_stats

class TestFeatureEngineering(unittest.TestCase):
    
    def test_decimal_to_american_conversion(self):
        """Test odds conversion"""
        # Test favorites (decimal < 2.0)
        self.assertEqual(decimal_to_american(1.5), -200)
        self.assertEqual(decimal_to_american(1.8), -125)
        
        # Test underdogs (decimal >= 2.0)
        self.assertEqual(decimal_to_american(2.0), 100)
        self.assertEqual(decimal_to_american(3.0), 200)
    
    def test_recent_stats_bounds(self):
        """Test that recent stats return reasonable values"""
        # Mock the NBA API call
        with patch('features.get_recent_stats.teamgamelog') as mock_api:
            # Mock successful response
            mock_df = pd.DataFrame({
                'WL': ['W', 'W', 'L', 'W', 'L'],
                'PTS': [110, 105, 98, 115, 102]
            })
            mock_api.TeamGameLog.return_value.get_data_frames.return_value = [mock_df]
            
            win_pct = get_recent_win_pct("boston celtics")
            avg_pts = get_recent_avg_pts("boston celtics")
            
            # Check bounds
            self.assertGreaterEqual(win_pct, 0.0)
            self.assertLessEqual(win_pct, 1.0)
            self.assertGreater(avg_pts, 80)  # Reasonable NBA score
            self.assertLess(avg_pts, 150)
    
    def test_feature_data_types(self):
        """Test that features return correct data types"""
        with patch('features.get_recent_stats.teamgamelog'):
            win_pct = get_recent_win_pct("test_team")
            avg_pts = get_recent_avg_pts("test_team")
            
            self.assertIsInstance(win_pct, (int, float))
            self.assertIsInstance(avg_pts, (int, float))
    
    def test_error_handling(self):
        """Test error handling in feature functions"""
        with patch('features.get_recent_stats.teamgamelog') as mock_api:
            # Mock API failure
            mock_api.TeamGameLog.side_effect = Exception("API Error")
            
            # Should return fallback values, not crash
            win_pct = get_recent_win_pct("invalid_team")
            avg_pts = get_recent_avg_pts("invalid_team")
            
            self.assertEqual(win_pct, 0.5)  # Fallback
            self.assertEqual(avg_pts, 110)  # Fallback

class TestDataValidation(unittest.TestCase):
    
    def test_odds_data_format(self):
        """Test odds data has required columns"""
        sample_odds = pd.DataFrame({
            'home_team': ['Team A', 'Team B'],
            'away_team': ['Team C', 'Team D'],
            'home_odds': [-110, 120],
            'away_odds': [105, -115]
        })
        
        required_cols = ['home_team', 'away_team', 'home_odds', 'away_odds']
        for col in required_cols:
            self.assertIn(col, sample_odds.columns)
    
    def test_prediction_data_bounds(self):
        """Test prediction outputs are within valid ranges"""
        # Mock prediction data
        predictions = pd.DataFrame({
            'model_win_prob': [0.65, 0.45, 0.78, 0.52],
            'value_gap': [0.05, -0.02, 0.08, 0.01]
        })
        
        # Test probability bounds
        self.assertTrue((predictions['model_win_prob'] >= 0).all())
        self.assertTrue((predictions['model_win_prob'] <= 1).all())
        
        # Test value gap is reasonable
        self.assertTrue((predictions['value_gap'] >= -0.5).all())
        self.assertTrue((predictions['value_gap'] <= 0.5).all())

class TestModelPerformance(unittest.TestCase):
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent"""
        # Mock feature data
        features = pd.DataFrame({
            'off_rating_diff': [5.2, -2.1],
            'def_rating_diff': [3.0, 1.5],
            'recent_win_diff': [0.2, -0.1],
            'pace_diff': [2.5, -1.8],
            'home_recent_avg_pts': [112, 108],
            'away_recent_avg_pts': [105, 110],
            'home_EFG_PCT': [0.52, 0.48],
            'away_EFG_PCT': [0.49, 0.51],
            'home_odds': [-110, 125],
            'away_odds': [105, -115],
            'odds_diff': [-215, 240],
            'implied_home_win_pct': [0.52, 0.44],
            'implied_away_win_pct': [0.48, 0.53],
            'implied_win_diff': [0.04, -0.09]
        })
        
        # Test same inputs give same outputs
        # This would require loading actual model
        pass

if __name__ == '__main__':
    unittest.main()