#!/usr/bin/env python3
"""
ENHANCED PLAYER PROPS SYSTEM
Building on your existing player props models with profitability focus
Combines your models with market timing and value detection
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import sys
sys.path.append('/Users/mk/nba-betting-ai')
from src.prediction.predict_player_props import PlayerPropsPredictor

class EnhancedPlayerPropsSystem:
    """Enhanced profitable player props system"""
    
    def __init__(self):
        """Initialize enhanced system with market intelligence"""
        
        print("🚀 ENHANCED PLAYER PROPS SYSTEM")
        print("Market-aware profitable player props betting")
        print("=" * 60)
        
        # Load your existing models
        self.base_predictor = PlayerPropsPredictor()
        
        # Enhanced market features
        self.market_thresholds = {
            'high_confidence': 0.65,   # Only bet when very confident
            'medium_confidence': 0.58, # Secondary opportunities
            'min_edge': 0.03          # Minimum edge required
        }
        
        print("✅ Enhanced player props system initialized")
    
    def add_market_intelligence(self, base_prediction, market_data):
        """Add market timing and value analysis to predictions"""
        
        enhanced_prediction = base_prediction.copy()
        
        # Add market factors to each prop
        for prop_type in ['points', 'rebounds', 'assists']:
            for line_key, line_pred in enhanced_prediction[prop_type].items():
                if isinstance(line_pred, dict) and 'confidence' in line_pred:
                    
                    # Get market data for this prop
                    prop_market = market_data.get(prop_type, {}).get(line_key, {})
                    
                    # Calculate enhanced metrics
                    market_prob = prop_market.get('implied_prob', 0.5)
                    public_percentage = prop_market.get('public_pct', 50)
                    line_movement = prop_market.get('line_move', 0)
                    
                    # Calculate true edge
                    model_prob = line_pred['over_probability']
                    if line_pred['recommendation'] == 'UNDER':
                        model_prob = 1 - model_prob
                        market_prob = 1 - market_prob
                    
                    edge = model_prob - market_prob
                    
                    # Market timing factors
                    sharp_money_indicator = abs(line_movement) > 0.5 and public_percentage < 40
                    public_trap = public_percentage > 70
                    
                    # Enhanced confidence calculation
                    base_confidence = line_pred['confidence']
                    
                    # Boost confidence for favorable market conditions
                    if sharp_money_indicator:
                        base_confidence *= 1.15  # Sharp money agrees
                    if edge > 0.05:
                        base_confidence *= 1.10  # Strong model edge
                    if public_trap and model_prob < 0.45:
                        base_confidence *= 1.12  # Fade the public
                    
                    # Reduce confidence for unfavorable conditions
                    if edge < 0:
                        base_confidence *= 0.85  # Negative edge
                    if public_percentage > 60 and model_prob > 0.55:
                        base_confidence *= 0.90  # Following the public
                    
                    # Update prediction with enhanced data
                    enhanced_prediction[prop_type][line_key].update({
                        'enhanced_confidence': min(base_confidence, 0.95),
                        'market_edge': edge,
                        'value_score': edge * base_confidence,
                        'market_factors': {
                            'public_pct': public_percentage,
                            'line_movement': line_movement,
                            'sharp_money': sharp_money_indicator,
                            'public_trap': public_trap
                        }
                    })
        
        return enhanced_prediction
    
    def get_mock_market_data(self):
        """Generate realistic market data for testing"""
        
        market_data = {
            'points': {
                '15.5': {'implied_prob': 0.52, 'public_pct': 45, 'line_move': 0.0},
                '20.5': {'implied_prob': 0.48, 'public_pct': 72, 'line_move': -0.5},
                '25.5': {'implied_prob': 0.45, 'public_pct': 38, 'line_move': 1.0},
                '30.5': {'implied_prob': 0.35, 'public_pct': 55, 'line_move': 0.0}
            },
            'rebounds': {
                '4.5': {'implied_prob': 0.55, 'public_pct': 60, 'line_move': 0.0},
                '6.5': {'implied_prob': 0.50, 'public_pct': 45, 'line_move': 0.5},
                '8.5': {'implied_prob': 0.42, 'public_pct': 35, 'line_move': -1.0},
                '10.5': {'implied_prob': 0.38, 'public_pct': 65, 'line_move': 0.0},
                '12.5': {'implied_prob': 0.25, 'public_pct': 75, 'line_move': 0.0}
            },
            'assists': {
                '2.5': {'implied_prob': 0.58, 'public_pct': 50, 'line_move': 0.0},
                '3.5': {'implied_prob': 0.52, 'public_pct': 42, 'line_move': 0.5},
                '4.5': {'implied_prob': 0.47, 'public_pct': 68, 'line_move': -0.5},
                '5.5': {'implied_prob': 0.41, 'public_pct': 35, 'line_move': 1.0},
                '6.5': {'implied_prob': 0.35, 'public_pct': 58, 'line_move': 0.0},
                '7.5': {'implied_prob': 0.28, 'public_pct': 72, 'line_move': 0.0}
            }
        }
        
        return market_data
    
    def identify_profitable_props(self, enhanced_predictions):
        """Identify most profitable player prop opportunities"""
        
        print(f"\n💰 PROFITABLE PLAYER PROPS ANALYSIS")
        print("=" * 80)
        
        profitable_bets = []
        
        for prediction in enhanced_predictions:
            player = prediction['player']
            
            for prop_type in ['points', 'rebounds', 'assists']:
                for line_key, line_pred in prediction[prop_type].items():
                    if isinstance(line_pred, dict) and 'enhanced_confidence' in line_pred:
                        
                        confidence = line_pred['enhanced_confidence']
                        edge = line_pred['market_edge']
                        value_score = line_pred['value_score']
                        
                        # Profitable criteria (similar to your enhanced moneyline system)
                        if (confidence > self.market_thresholds['high_confidence'] and 
                            edge > self.market_thresholds['min_edge']):
                            
                            profitable_bets.append({
                                'player': player,
                                'prop_type': prop_type,
                                'line': line_key,
                                'recommendation': line_pred['recommendation'],
                                'confidence': confidence,
                                'edge': edge,
                                'value_score': value_score,
                                'market_factors': line_pred['market_factors'],
                                'base_confidence': line_pred['confidence']
                            })
        
        # Sort by value score (edge * confidence)
        profitable_bets.sort(key=lambda x: x['value_score'], reverse=True)
        
        return profitable_bets
    
    def display_profitable_bets(self, profitable_bets):
        """Display profitable betting opportunities"""
        
        if not profitable_bets:
            print("📊 No profitable player props opportunities found today")
            print("💡 System is being selective - waiting for better spots")
            return
        
        print(f"🔥 HIGH-VALUE PLAYER PROPS ({len(profitable_bets)} opportunities)")
        print("-" * 80)
        print(f"{'Player':<20} {'Prop':<15} {'Bet':<6} {'Conf':<8} {'Edge':<8} {'Value':<8} {'Factors'}")
        print("-" * 80)
        
        total_value = 0
        
        for bet in profitable_bets[:10]:  # Show top 10
            prop_display = f"{bet['prop_type'].title()} {bet['line']}"
            confidence_display = f"{bet['confidence']:.1%}"
            edge_display = f"{bet['edge']:+.1%}"
            value_display = f"{bet['value_score']:.3f}"
            
            # Market factors summary
            factors = bet['market_factors']
            factor_summary = ""
            if factors['sharp_money']:
                factor_summary += "💎"  # Sharp money
            if factors['public_trap']:
                factor_summary += "🎣"  # Public trap
            if abs(factors['line_movement']) > 0.5:
                factor_summary += "📈"  # Line movement
            
            print(f"{bet['player']:<20} {prop_display:<15} {bet['recommendation']:<6} "
                  f"{confidence_display:<8} {edge_display:<8} {value_display:<8} {factor_summary}")
            
            total_value += bet['value_score']
        
        print("-" * 80)
        print(f"💰 TOTAL PORTFOLIO VALUE SCORE: {total_value:.3f}")
        
        # Show expected performance
        if len(profitable_bets) > 0:
            avg_confidence = np.mean([bet['confidence'] for bet in profitable_bets[:10]])
            avg_edge = np.mean([bet['edge'] for bet in profitable_bets[:10]])
            
            print(f"\n📊 EXPECTED PERFORMANCE:")
            print(f"   Average Confidence: {avg_confidence:.1%}")
            print(f"   Average Edge: {avg_edge:+.1%}")
            print(f"   Expected Win Rate: {avg_confidence:.1%}")
            print(f"   Expected ROI: {avg_edge*100:+.1f}%")
        
        return profitable_bets
    
    def generate_betting_card(self, profitable_bets):
        """Generate professional betting card"""
        
        print(f"\n🎯 TODAY'S PLAYER PROPS BETTING CARD")
        print("=" * 80)
        
        if len(profitable_bets) == 0:
            print("💎 NO BETS - System being selective")
            print("🎯 Waiting for higher-value opportunities")
            return
        
        print("🔥 PREMIUM PICKS:")
        print()
        
        for i, bet in enumerate(profitable_bets[:5], 1):
            print(f"#{i} 🌟 {bet['player']} - {bet['prop_type'].title()} {bet['line']}")
            print(f"    💰 BET: {bet['recommendation']}")
            print(f"    📊 Confidence: {bet['confidence']:.1%} (Enhanced from {bet['base_confidence']:.1%})")
            print(f"    💎 Edge: {bet['edge']:+.1%}")
            print(f"    📈 Value Score: {bet['value_score']:.3f}")
            
            # Explain the edge
            factors = bet['market_factors']
            print(f"    🔍 Edge Sources:")
            if factors['sharp_money']:
                print(f"       💎 Sharp money movement detected")
            if factors['public_trap']:
                print(f"       🎣 Public heavily on other side")
            if bet['edge'] > 0.05:
                print(f"       🧠 Strong model disagreement with market")
            if abs(factors['line_movement']) > 0.5:
                print(f"       📈 Significant line movement")
            print()
        
        # Summary
        total_expected_profit = sum(bet['value_score'] * 100 for bet in profitable_bets[:5])
        print(f"💰 EXPECTED PROFIT (per $100 unit): ${total_expected_profit:.2f}")
        print(f"🎯 Recommended unit size: $50-100 per bet")
        print(f"📊 Based on enhanced model consensus + market timing")
    
    def run_enhanced_analysis(self, sample_players=None):
        """Run complete enhanced player props analysis"""
        
        print("🚀 RUNNING ENHANCED PLAYER PROPS ANALYSIS")
        print("=" * 60)
        
        # Use sample data or provided players
        if sample_players is None:
            sample_players = self.create_sample_players()
        
        # Get market data (in real system, this would be live)
        market_data = self.get_mock_market_data()
        
        all_enhanced_predictions = []
        
        # Process each player
        for player_name, features in sample_players:
            print(f"📊 Analyzing {player_name}...")
            
            # Get base prediction from your existing system
            base_prediction = self.base_predictor.predict_player_props(features, player_name)
            
            # Enhance with market intelligence
            enhanced_prediction = self.add_market_intelligence(base_prediction, market_data)
            
            all_enhanced_predictions.append(enhanced_prediction)
        
        # Find profitable opportunities
        profitable_bets = self.identify_profitable_props(all_enhanced_predictions)
        
        # Display results
        self.display_profitable_bets(profitable_bets)
        
        # Generate betting card
        self.generate_betting_card(profitable_bets)
        
        return profitable_bets
    
    def create_sample_players(self):
        """Create sample players for testing"""
        
        return [
            ('LeBron James', {
                'points_last_5': 27.2, 'points_last_10': 26.8, 'points_last_15': 27.1,
                'minutes_last_5': 36.5, 'minutes_last_10': 37.1, 'fg_pct_last_5': 0.485,
                'fg_attempts_last_5': 19.4, 'points_trend_3': 1.2, 'home_away_diff': 2.1,
                'season_ppg': 26.9, 'season_mpg': 37.0, 'points_consistency': 5.2,
                'opponent_strength': 0.45, 'home': 1,
                'rebounds_last_5': 8.2, 'rebounds_last_10': 7.8, 'rebounds_last_15': 8.1,
                'oreb_last_5': 1.4, 'dreb_last_5': 6.8, 'rebounds_trend_3': 0.8,
                'home_away_reb_diff': 1.2, 'season_rpg': 8.0, 'season_oreb_pct': 1.3,
                'season_dreb_pct': 6.7, 'rebounds_consistency': 2.1, 'position_big': 0,
                'position_wing': 1,
                'assists_last_5': 7.2, 'assists_last_10': 6.8, 'assists_last_15': 7.1,
                'fga_last_5': 19.2, 'turnovers_last_5': 3.8, 'ast_to_last_5': 1.9,
                'assists_trend_3': 0.5, 'home_away_ast_diff': 0.8, 'season_apg': 7.0,
                'season_ast_to': 1.8, 'assists_consistency': 1.8, 'position_playmaker': 0,
                'position_secondary': 1
            }),
            ('Nikola Jokic', {
                'points_last_5': 24.8, 'points_last_10': 25.2, 'points_last_15': 24.9,
                'minutes_last_5': 34.2, 'minutes_last_10': 34.8, 'fg_pct_last_5': 0.615,
                'fg_attempts_last_5': 16.8, 'points_trend_3': -0.3, 'home_away_diff': 1.8,
                'season_ppg': 25.1, 'season_mpg': 34.5, 'points_consistency': 4.8,
                'opponent_strength': 0.52, 'home': 1,
                'rebounds_last_5': 12.8, 'rebounds_last_10': 13.2, 'rebounds_last_15': 12.9,
                'oreb_last_5': 2.1, 'dreb_last_5': 10.7, 'rebounds_trend_3': 0.4,
                'home_away_reb_diff': 1.5, 'season_rpg': 13.0, 'season_oreb_pct': 2.0,
                'season_dreb_pct': 11.0, 'rebounds_consistency': 2.8, 'position_big': 1,
                'position_wing': 0,
                'assists_last_5': 9.8, 'assists_last_10': 9.4, 'assists_last_15': 9.7,
                'fga_last_5': 16.5, 'turnovers_last_5': 3.2, 'ast_to_last_5': 3.1,
                'assists_trend_3': 0.8, 'home_away_ast_diff': 1.1, 'season_apg': 9.6,
                'season_ast_to': 3.0, 'assists_consistency': 2.2, 'position_playmaker': 1,
                'position_secondary': 0
            }),
            ('Role Player X', {
                'points_last_5': 12.2, 'points_last_10': 11.8, 'points_last_15': 12.1,
                'minutes_last_5': 22.5, 'minutes_last_10': 23.1, 'fg_pct_last_5': 0.445,
                'fg_attempts_last_5': 8.4, 'points_trend_3': -0.2, 'home_away_diff': 1.1,
                'season_ppg': 11.9, 'season_mpg': 22.0, 'points_consistency': 3.8,
                'opponent_strength': 0.55, 'home': 0,
                'rebounds_last_5': 4.2, 'rebounds_last_10': 3.8, 'rebounds_last_15': 4.1,
                'oreb_last_5': 0.8, 'dreb_last_5': 3.4, 'rebounds_trend_3': 0.1,
                'home_away_reb_diff': 0.5, 'season_rpg': 4.0, 'season_oreb_pct': 0.7,
                'season_dreb_pct': 3.3, 'rebounds_consistency': 1.5, 'position_big': 0,
                'position_wing': 0,
                'assists_last_5': 2.2, 'assists_last_10': 2.0, 'assists_last_15': 2.1,
                'fga_last_5': 8.2, 'turnovers_last_5': 1.1, 'ast_to_last_5': 2.0,
                'assists_trend_3': 0.1, 'home_away_ast_diff': 0.2, 'season_apg': 2.1,
                'season_ast_to': 1.9, 'assists_consistency': 0.8, 'position_playmaker': 0,
                'position_secondary': 0
            })
        ]

def main():
    """Run the enhanced player props system"""
    
    # Initialize enhanced system
    enhanced_system = EnhancedPlayerPropsSystem()
    
    # Run complete analysis
    profitable_bets = enhanced_system.run_enhanced_analysis()
    
    print(f"\n✅ ENHANCED PLAYER PROPS ANALYSIS COMPLETE!")
    print(f"💰 Found {len(profitable_bets)} profitable opportunities")
    print(f"🎯 System combines your existing models with market intelligence")
    print(f"📊 Similar approach to your profitable enhanced moneyline system")

if __name__ == "__main__":
    main()
