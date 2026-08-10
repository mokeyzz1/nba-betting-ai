#!/usr/bin/env python3
"""
NBA Master Controller System
Unified interface for all profitable NBA betting systems
Ranks all betting opportunities by expected value and confidence
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all our enhanced prediction systems
try:
    from src.prediction.predict_player_props import EnhancedPlayerPropsPredictor
except ImportError:
    print("⚠️ Player props predictor not available")
    EnhancedPlayerPropsPredictor = None

class NBAMasterController:
    """Unified NBA betting system controller"""
    
    def __init__(self):
        """Initialize all betting systems"""
        
        print("🎯 NBA MASTER CONTROLLER SYSTEM")
        print("Unified interface for maximum profitability")
        print("=" * 60)
        
        self.systems = {}
        self.all_opportunities = []
        
        # Load all enhanced systems
        self._load_moneyline_system()
        self._load_spreads_system() 
        self._load_totals_system()
        self._load_player_props_system()
        
        print(f"✅ Master Controller initialized with {len(self.systems)} systems")
    
    def _load_moneyline_system(self):
        """Load enhanced moneyline system"""
        try:
            models = {}
            
            # Load moneyline ensemble models
            for algo in ['rf', 'xgb', 'lgb', 'lr']:
                model_path = f"models/enhanced_profitable_moneyline_{algo}.pkl"
                if os.path.exists(model_path):
                    models[algo] = joblib.load(model_path)
            
            if models:
                # Load configuration
                config_path = "models/enhanced_profitable_config.pkl"
                if os.path.exists(config_path):
                    config = joblib.load(config_path)
                    
                    self.systems['moneyline'] = {
                        'models': models,
                        'features': config['features'],
                        'scaler': joblib.load("models/enhanced_profitable_scaler.pkl"),
                        'type': 'team_game',
                        'roi': 5.3,
                        'confidence_threshold': 0.35
                    }
                    print("✅ Enhanced Moneyline system loaded")
                else:
                    print("⚠️ Moneyline config not found")
            else:
                print("⚠️ Enhanced Moneyline models not found")
                
        except Exception as e:
            print(f"⚠️ Error loading moneyline system: {e}")
    
    def _load_spreads_system(self):
        """Load enhanced spreads system"""
        try:
            # Load spreads configuration
            config_path = "models/enhanced_spreads_config.pkl"
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                
                # Load spread line models
                spread_models = {}
                spread_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 12.5, 15.5]
                
                for line in spread_lines:
                    line_models = {}
                    for algo in ['rf', 'xgb', 'lgb', 'lr']:
                        model_path = f"models/enhanced_spreads_spread_{line}_{algo}.pkl"
                        if os.path.exists(model_path):
                            line_models[algo] = joblib.load(model_path)
                    
                    if line_models:
                        spread_models[f'spread_{line}'] = line_models
                
                if spread_models:
                    self.systems['spreads'] = {
                        'models': spread_models,
                        'features': config['features'],
                        'scaler': joblib.load("models/enhanced_spreads_scaler.pkl"),
                        'type': 'team_spread',
                        'roi': 10.9,
                        'confidence_threshold': 0.58
                    }
                    print(f"✅ Enhanced Spreads system loaded ({len(spread_models)} lines)")
                else:
                    print("⚠️ No spread models found")
            else:
                print("⚠️ Spreads config not found")
                
        except Exception as e:
            print(f"⚠️ Error loading spreads system: {e}")
    
    def _load_totals_system(self):
        """Load enhanced totals system"""
        try:
            # Load totals configuration
            config_path = "models/enhanced_totals_config.pkl"
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                
                # Load total line models
                total_models = {}
                total_lines = [210.5, 215.5, 220.5, 222.5, 225.5, 227.5, 230.5, 232.5, 235.5, 237.5, 240.5, 245.5, 250.5]
                
                for line in total_lines:
                    line_models = {}
                    for algo in ['rf', 'xgb', 'lgb', 'lr']:
                        model_path = f"models/enhanced_totals_total_{line}_{algo}.pkl"
                        if os.path.exists(model_path):
                            line_models[algo] = joblib.load(model_path)
                    
                    if line_models:
                        total_models[f'total_{line}'] = line_models
                
                if total_models:
                    self.systems['totals'] = {
                        'models': total_models,
                        'features': config['features'],
                        'scaler': joblib.load("models/enhanced_totals_scaler.pkl"),
                        'type': 'team_total',
                        'roi': 17.6,
                        'confidence_threshold': 0.58
                    }
                    print(f"✅ Enhanced Totals system loaded ({len(total_models)} lines)")
                else:
                    print("⚠️ No total models found")
            else:
                print("⚠️ Totals config not found")
                
        except Exception as e:
            print(f"⚠️ Error loading totals system: {e}")
    
    def _load_player_props_system(self):
        """Load enhanced player props system"""
        try:
            if EnhancedPlayerPropsPredictor:
                self.systems['player_props'] = {
                    'predictor': EnhancedPlayerPropsPredictor(),
                    'type': 'player_props',
                    'roi': 53.6,  # Average of points (61.4%), assists (59.7%), rebounds (39.6%)
                    'confidence_threshold': 0.62
                }
                print("✅ Enhanced Player Props system loaded")
            else:
                print("⚠️ Player Props system not available")
                
        except Exception as e:
            print(f"⚠️ Error loading player props system: {e}")
    
    def analyze_today_games(self, games_data=None, players_data=None):
        """Analyze all betting opportunities for today's games"""
        
        print(f"\n🔍 ANALYZING TODAY'S NBA GAMES")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        self.all_opportunities = []
        
        # Sample data for demonstration (in production, this would come from live feeds)
        if games_data is None:
            games_data = self._create_sample_games_data()
        
        if players_data is None:
            players_data = self._create_sample_players_data()
        
        print(f"📊 Processing {len(games_data)} games with {len(players_data)} players")
        
        # Analyze team-level betting opportunities
        for game in games_data:
            self._analyze_game_opportunities(game)
        
        # Analyze player props opportunities
        for player in players_data:
            self._analyze_player_opportunities(player)
        
        # Rank all opportunities by value
        self._rank_opportunities()
        
        return self.all_opportunities
    
    def _analyze_game_opportunities(self, game_data):
        """Analyze moneyline, spreads, and totals for a single game"""
        
        game_id = game_data['game_id']
        home_team = game_data['home_team']
        away_team = game_data['away_team']
        
        # MONEYLINE ANALYSIS
        if 'moneyline' in self.systems:
            try:
                ml_system = self.systems['moneyline']
                
                # Prepare features
                features_df = pd.DataFrame([{f: game_data.get(f, 0) for f in ml_system['features']}])
                features_scaled = ml_system['scaler'].transform(features_df)
                
                # Get ensemble prediction
                predictions = []
                for algo, model in ml_system['models'].items():
                    if algo == 'lr':
                        pred = model.predict_proba(features_scaled)[0][1]
                    else:
                        pred = model.predict_proba(features_df)[0][1]
                    predictions.append(pred)
                
                if predictions:
                    ensemble_prob = np.mean(predictions)
                    
                    if ensemble_prob > ml_system['confidence_threshold']:
                        self.all_opportunities.append({
                            'type': 'moneyline',
                            'game_id': game_id,
                            'description': f"{home_team} vs {away_team} - Home Win",
                            'recommendation': 'HOME',
                            'probability': ensemble_prob,
                            'confidence': ensemble_prob,
                            'expected_roi': ml_system['roi'],
                            'value_score': (ensemble_prob - 0.5) * 2 * ml_system['roi'] / 100,
                            'system': 'Enhanced Moneyline'
                        })
                    elif ensemble_prob < (1 - ml_system['confidence_threshold']):
                        self.all_opportunities.append({
                            'type': 'moneyline',
                            'game_id': game_id,
                            'description': f"{home_team} vs {away_team} - Away Win",
                            'recommendation': 'AWAY',
                            'probability': 1 - ensemble_prob,
                            'confidence': 1 - ensemble_prob,
                            'expected_roi': ml_system['roi'],
                            'value_score': (0.5 - ensemble_prob) * 2 * ml_system['roi'] / 100,
                            'system': 'Enhanced Moneyline'
                        })
                        
            except Exception as e:
                print(f"⚠️ Error analyzing moneyline for {game_id}: {e}")
        
        # SPREADS ANALYSIS
        if 'spreads' in self.systems:
            try:
                spreads_system = self.systems['spreads']
                
                # Prepare features
                features_df = pd.DataFrame([{f: game_data.get(f, 0) for f in spreads_system['features']}])
                features_scaled = spreads_system['scaler'].transform(features_df)
                
                # Test each spread line
                for model_key, line_models in spreads_system['models'].items():
                    line = float(model_key.replace('spread_', ''))
                    
                    # Get ensemble prediction
                    predictions = []
                    for algo, model in line_models.items():
                        if algo == 'lr':
                            pred = model.predict_proba(features_scaled)[0][1]
                        else:
                            pred = model.predict_proba(features_df)[0][1]
                        predictions.append(pred)
                    
                    if predictions:
                        weights = [0.3, 0.3, 0.25, 0.15][:len(predictions)]
                        weights = np.array(weights) / sum(weights)
                        ensemble_prob = np.average(predictions, weights=weights)
                        
                        if ensemble_prob > spreads_system['confidence_threshold']:
                            self.all_opportunities.append({
                                'type': 'spread',
                                'game_id': game_id,
                                'description': f"{home_team} vs {away_team} - Home -{line}",
                                'recommendation': 'HOME COVERS',
                                'probability': ensemble_prob,
                                'confidence': ensemble_prob,
                                'expected_roi': spreads_system['roi'],
                                'value_score': (ensemble_prob - 0.5) * 2 * spreads_system['roi'] / 100,
                                'system': 'Enhanced Spreads',
                                'line': line
                            })
                        elif ensemble_prob < (1 - spreads_system['confidence_threshold']):
                            self.all_opportunities.append({
                                'type': 'spread',
                                'game_id': game_id,
                                'description': f"{home_team} vs {away_team} - Away +{line}",
                                'recommendation': 'AWAY COVERS',
                                'probability': 1 - ensemble_prob,
                                'confidence': 1 - ensemble_prob,
                                'expected_roi': spreads_system['roi'],
                                'value_score': (0.5 - ensemble_prob) * 2 * spreads_system['roi'] / 100,
                                'system': 'Enhanced Spreads',
                                'line': line
                            })
                            
            except Exception as e:
                print(f"⚠️ Error analyzing spreads for {game_id}: {e}")
        
        # TOTALS ANALYSIS
        if 'totals' in self.systems:
            try:
                totals_system = self.systems['totals']
                
                # Prepare features
                features_df = pd.DataFrame([{f: game_data.get(f, 0) for f in totals_system['features']}])
                features_scaled = totals_system['scaler'].transform(features_df)
                
                # Test each total line
                for model_key, line_models in totals_system['models'].items():
                    line = float(model_key.replace('total_', ''))
                    
                    # Get ensemble prediction
                    predictions = []
                    for algo, model in line_models.items():
                        if algo == 'lr':
                            pred = model.predict_proba(features_scaled)[0][1]
                        else:
                            pred = model.predict_proba(features_df)[0][1]
                        predictions.append(pred)
                    
                    if predictions:
                        weights = [0.3, 0.3, 0.25, 0.15][:len(predictions)]
                        weights = np.array(weights) / sum(weights)
                        ensemble_prob = np.average(predictions, weights=weights)
                        
                        if ensemble_prob > totals_system['confidence_threshold']:
                            self.all_opportunities.append({
                                'type': 'total',
                                'game_id': game_id,
                                'description': f"{home_team} vs {away_team} - Over {line}",
                                'recommendation': 'OVER',
                                'probability': ensemble_prob,
                                'confidence': ensemble_prob,
                                'expected_roi': totals_system['roi'],
                                'value_score': (ensemble_prob - 0.5) * 2 * totals_system['roi'] / 100,
                                'system': 'Enhanced Totals',
                                'line': line
                            })
                        elif ensemble_prob < (1 - totals_system['confidence_threshold']):
                            self.all_opportunities.append({
                                'type': 'total',
                                'game_id': game_id,
                                'description': f"{home_team} vs {away_team} - Under {line}",
                                'recommendation': 'UNDER',
                                'probability': 1 - ensemble_prob,
                                'confidence': 1 - ensemble_prob,
                                'expected_roi': totals_system['roi'],
                                'value_score': (0.5 - ensemble_prob) * 2 * totals_system['roi'] / 100,
                                'system': 'Enhanced Totals',
                                'line': line
                            })
                            
            except Exception as e:
                print(f"⚠️ Error analyzing totals for {game_id}: {e}")
    
    def _analyze_player_opportunities(self, player_data):
        """Analyze player props opportunities"""
        
        if 'player_props' not in self.systems:
            return
        
        try:
            props_system = self.systems['player_props']
            predictor = props_system['predictor']
            
            player_name = player_data['name']
            player_features = player_data['features']
            
            # Get player props predictions
            predictions = predictor.predict_player_props(player_features, player_name)
            
            # Convert to opportunities
            for prop_type in ['points', 'rebounds', 'assists']:
                for line, pred_data in predictions[prop_type].items():
                    if isinstance(pred_data, dict) and pred_data.get('recommendation') in ['OVER', 'UNDER']:
                        if pred_data['confidence'] > props_system['confidence_threshold']:
                            self.all_opportunities.append({
                                'type': 'player_prop',
                                'game_id': player_data.get('game_id', 'unknown'),
                                'description': f"{player_name} - {prop_type.title()} {line}",
                                'recommendation': pred_data['recommendation'],
                                'probability': pred_data['over_probability'] if pred_data['recommendation'] == 'OVER' else 1 - pred_data['over_probability'],
                                'confidence': pred_data['confidence'],
                                'expected_roi': props_system['roi'],
                                'value_score': pred_data.get('value_score', 0) * props_system['roi'] / 100,
                                'system': 'Enhanced Player Props',
                                'player': player_name,
                                'prop_type': prop_type,
                                'line': line
                            })
                            
        except Exception as e:
            print(f"⚠️ Error analyzing player props for {player_data.get('name', 'unknown')}: {e}")
    
    def _rank_opportunities(self):
        """Rank all opportunities by value score"""
        
        # Sort by value score (combination of confidence, edge, and expected ROI)
        self.all_opportunities.sort(key=lambda x: x['value_score'], reverse=True)
        
        # Add rankings
        for i, opp in enumerate(self.all_opportunities):
            opp['rank'] = i + 1
    
    def display_best_opportunities(self, max_opportunities=20, min_value_score=0.02):
        """Display the best betting opportunities"""
        
        print(f"\n🏆 TOP BETTING OPPORTUNITIES")
        print(f"📊 Ranked by Expected Value")
        print("=" * 100)
        
        # Filter by minimum value score
        top_opportunities = [opp for opp in self.all_opportunities if opp['value_score'] >= min_value_score][:max_opportunities]
        
        if not top_opportunities:
            print("No high-value opportunities found today.")
            return
        
        print(f"{'Rank':<4} {'System':<20} {'Description':<35} {'Bet':<12} {'Confidence':<11} {'Value':<8}")
        print("-" * 100)
        
        for opp in top_opportunities:
            rank = opp['rank']
            system = opp['system'][:19]
            description = opp['description'][:34]
            recommendation = opp['recommendation'][:11]
            confidence = f"{opp['confidence']:.1%}"
            value_stars = "★" * min(5, int(opp['value_score'] * 25))
            
            print(f"{rank:<4} {system:<20} {description:<35} {recommendation:<12} {confidence:<11} {value_stars:<8}")
        
        # Summary statistics
        print(f"\n📈 PORTFOLIO SUMMARY:")
        print(f"   Total opportunities: {len(self.all_opportunities)}")
        print(f"   High-value bets: {len(top_opportunities)}")
        print(f"   Average confidence: {np.mean([opp['confidence'] for opp in top_opportunities]):.1%}")
        print(f"   Average expected ROI: {np.mean([opp['expected_roi'] for opp in top_opportunities]):.1f}%")
        
        # By system breakdown
        system_counts = {}
        for opp in top_opportunities:
            system = opp['system']
            system_counts[system] = system_counts.get(system, 0) + 1
        
        print(f"\n🎯 BY BETTING SYSTEM:")
        for system, count in sorted(system_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {system}: {count} opportunities")
    
    def _create_sample_games_data(self):
        """Create sample games data for demonstration"""
        
        return [
            {
                'game_id': 'LAL_vs_GSW',
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                # Sample enhanced features (would come from live data feeds)
                'net_rating_diff': 5.2, 'off_rating_diff': 3.1, 'def_rating_diff': 2.1,
                'recent_win_pct_diff': 0.15, 'win_pct_diff': 0.08, 'market_edge': 0.12,
                'composite_strength': 8.5, 'pace_combined': 101.5, 'scoring_environment': 4.2,
                'home_court_enhanced': 4.1
            },
            {
                'game_id': 'BOS_vs_MIA',
                'home_team': 'Boston Celtics',
                'away_team': 'Miami Heat',
                'net_rating_diff': 2.8, 'off_rating_diff': 1.5, 'def_rating_diff': 1.3,
                'recent_win_pct_diff': 0.05, 'win_pct_diff': 0.12, 'market_edge': 0.08,
                'composite_strength': 4.2, 'pace_combined': 98.5, 'scoring_environment': 2.1,
                'home_court_enhanced': 3.5
            }
        ]
    
    def _create_sample_players_data(self):
        """Create sample players data for demonstration"""
        
        return [
            {
                'name': 'LeBron James',
                'game_id': 'LAL_vs_GSW',
                'features': {
                    # Enhanced player features (would come from live data feeds)
                    'points_last_5': 27.2, 'rebounds_last_5': 8.2, 'assists_last_5': 7.2,
                    'minutes_last_5': 36.5, 'usg_rate_last_5': 28.5, 'ts_pct_last_5': 0.575,
                    'is_star': 1, 'is_primary_scorer': 1, 'opponent_def_rating': 108.5,
                    'home_advantage': 1, 'form_factor': 1.15
                }
            },
            {
                'name': 'Stephen Curry',
                'game_id': 'LAL_vs_GSW',
                'features': {
                    'points_last_5': 29.8, 'rebounds_last_5': 4.5, 'assists_last_5': 6.1,
                    'minutes_last_5': 35.2, 'usg_rate_last_5': 32.1, 'ts_pct_last_5': 0.635,
                    'is_star': 1, 'is_primary_scorer': 1, 'opponent_def_rating': 110.2,
                    'home_advantage': 0, 'form_factor': 1.08
                }
            }
        ]

def main():
    """Run the Master Controller System"""
    
    # Initialize Master Controller
    controller = NBAMasterController()
    
    # Analyze today's games
    opportunities = controller.analyze_today_games()
    
    # Display best opportunities
    controller.display_best_opportunities()
    
    print(f"\n🎉 NBA MASTER CONTROLLER ANALYSIS COMPLETE!")
    print(f"🎯 Ready for profitable NBA betting across all markets")
    
    return controller

if __name__ == "__main__":
    main()
