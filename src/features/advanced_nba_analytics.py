"""
Advanced NBA Analytics - Four Factors + Elite Metrics
Basketball's Four Factors: EFG%, TOV%, OREB%, FT Rate
Plus advanced situational analytics
"""
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import teamdashboardbyfour, teamestimatedmetrics, hustlestatsboxscore
import time
from datetime import datetime, timedelta
from src.utils.config import DATA_DIR

class AdvancedNBAAnalytics:
    
    def __init__(self):
        self.rate_limit = 1.5
        self.current_season = "2024-25"
        
    def get_four_factors(self, team_name):
        """
        Get Basketball's Four Factors - the core stats that determine wins
        1. Effective Field Goal % (EFG%)
        2. Turnover Rate (TOV%)  
        3. Offensive Rebound % (OREB%)
        4. Free Throw Rate (FT/FGA)
        """
        
        try:
            print(f"📊 Getting Four Factors for {team_name}")
            
            # Get four factors data
            four_factors = teamdashboardbyfour.TeamDashboardByFourFactors(
                team_id=self._get_team_id(team_name),
                season=self.current_season
            )
            time.sleep(self.rate_limit)
            
            team_stats = four_factors.get_data_frames()[0].iloc[0]
            
            return {
                # Offensive Four Factors
                'off_efg_pct': team_stats.get('EFG_PCT', 0.50),
                'off_tov_pct': team_stats.get('TOV_PCT', 15.0),
                'off_oreb_pct': team_stats.get('OREB_PCT', 25.0),
                'off_ft_rate': team_stats.get('FT_FGA', 0.20),
                
                # Defensive Four Factors (opponent stats)
                'def_efg_pct': team_stats.get('OPP_EFG_PCT', 0.50),
                'def_tov_pct': team_stats.get('OPP_TOV_PCT', 15.0),
                'def_oreb_pct': team_stats.get('OPP_OREB_PCT', 25.0),
                'def_ft_rate': team_stats.get('OPP_FT_FGA', 0.20),
                
                # Derived metrics
                'efg_differential': team_stats.get('EFG_PCT', 0.50) - team_stats.get('OPP_EFG_PCT', 0.50),
                'tov_differential': team_stats.get('OPP_TOV_PCT', 15.0) - team_stats.get('TOV_PCT', 15.0),
                'oreb_differential': team_stats.get('OREB_PCT', 25.0) - team_stats.get('OPP_OREB_PCT', 25.0),
                'ft_differential': team_stats.get('FT_FGA', 0.20) - team_stats.get('OPP_FT_FGA', 0.20)
            }
            
        except Exception as e:
            print(f"⚠️ Error getting four factors for {team_name}: {e}")
            return self._get_mock_four_factors()
    
    def get_pace_and_efficiency(self, team_name):
        """Get pace and efficiency metrics"""
        
        try:
            # Get basic team stats for pace calculation
            team_stats = self._get_basic_team_stats(team_name)
            
            return {
                'pace': team_stats.get('PACE', 100.0),
                'off_rating': team_stats.get('OFF_RATING', 110.0),
                'def_rating': team_stats.get('DEF_RATING', 110.0),
                'net_rating': team_stats.get('NET_RATING', 0.0),
                'poss_per_game': team_stats.get('POSS', 100.0)
            }
            
        except Exception as e:
            print(f"⚠️ Error getting pace data: {e}")
            return self._get_mock_pace_data()
    
    def get_clutch_performance(self, team_name):
        """Get clutch time performance (last 5 minutes, score within 5)"""
        
        try:
            # This would use clutch stats endpoint
            # For now, generate realistic mock data
            return {
                'clutch_record': f"{np.random.randint(5, 15)}-{np.random.randint(3, 12)}",
                'clutch_off_rating': np.random.uniform(105, 125),
                'clutch_def_rating': np.random.uniform(105, 125),
                'clutch_net_rating': np.random.uniform(-10, 10),
                'close_game_experience': np.random.randint(15, 35)  # Games within 5 points
            }
            
        except Exception as e:
            return self._get_mock_clutch_data()
    
    def get_rest_and_travel_impact(self, team_name, game_date):
        """Calculate rest days and travel impact"""
        
        try:
            # This would analyze schedule data
            # Mock implementation for now
            
            rest_days = np.random.randint(0, 4)
            travel_miles = np.random.randint(0, 3000)
            
            # Calculate fatigue factors
            rest_advantage = self._calculate_rest_advantage(rest_days)
            travel_fatigue = self._calculate_travel_fatigue(travel_miles)
            
            return {
                'rest_days': rest_days,
                'travel_miles': travel_miles,
                'rest_advantage': rest_advantage,
                'travel_fatigue_factor': travel_fatigue,
                'back_to_back': rest_days == 0,
                'well_rested': rest_days >= 2
            }
            
        except Exception as e:
            return self._get_mock_rest_data()
    
    def get_situational_trends(self, team_name):
        """Get situational performance trends"""
        
        try:
            # Home/Away splits, vs strong/weak teams, etc.
            return {
                'home_record': f"{np.random.randint(15, 25)}-{np.random.randint(5, 15)}",
                'away_record': f"{np.random.randint(10, 20)}-{np.random.randint(10, 20)}",
                'vs_above_500': f"{np.random.randint(8, 18)}-{np.random.randint(12, 22)}",
                'vs_below_500': f"{np.random.randint(12, 22)}-{np.random.randint(3, 13)}",
                'in_division': f"{np.random.randint(6, 12)}-{np.random.randint(4, 10)}",
                'favored_record': f"{np.random.randint(20, 30)}-{np.random.randint(8, 18)}",
                'underdog_record': f"{np.random.randint(5, 15)}-{np.random.randint(15, 25)}"
            }
            
        except Exception as e:
            return self._get_mock_situational_data()
    
    def calculate_game_factors_differential(self, home_team, away_team, game_date=None):
        """Calculate comprehensive game analysis"""
        
        print(f"🏀 Advanced Analysis: {away_team} @ {home_team}")
        
        # Get four factors for both teams
        home_factors = self.get_four_factors(home_team)
        away_factors = self.get_four_factors(away_team)
        
        # Get pace and efficiency
        home_pace = self.get_pace_and_efficiency(home_team)
        away_pace = self.get_pace_and_efficiency(away_team)
        
        # Get clutch performance
        home_clutch = self.get_clutch_performance(home_team)
        away_clutch = self.get_clutch_performance(away_team)
        
        # Get rest/travel
        home_rest = self.get_rest_and_travel_impact(home_team, game_date)
        away_rest = self.get_rest_and_travel_impact(away_team, game_date)
        
        # Get situational trends
        home_situations = self.get_situational_trends(home_team)
        away_situations = self.get_situational_trends(away_team)
        
        # Calculate key differentials
        differentials = {
            # Four Factors Advantages
            'efg_advantage': home_factors['efg_differential'] - away_factors['efg_differential'],
            'turnover_advantage': home_factors['tov_differential'] - away_factors['tov_differential'],
            'rebounding_advantage': home_factors['oreb_differential'] - away_factors['oreb_differential'],
            'free_throw_advantage': home_factors['ft_differential'] - away_factors['ft_differential'],
            
            # Pace and Efficiency
            'pace_differential': home_pace['pace'] - away_pace['pace'],
            'offensive_rating_diff': home_pace['off_rating'] - away_pace['def_rating'],
            'defensive_rating_diff': home_pace['def_rating'] - away_pace['off_rating'],
            'net_rating_diff': home_pace['net_rating'] - away_pace['net_rating'],
            
            # Clutch Performance
            'clutch_net_rating_diff': home_clutch['clutch_net_rating'] - away_clutch['clutch_net_rating'],
            
            # Rest and Travel
            'rest_advantage': home_rest['rest_advantage'] - away_rest['rest_advantage'],
            'travel_advantage': away_rest['travel_fatigue_factor'] - home_rest['travel_fatigue_factor'],
            
            # Composite Scores
            'four_factors_composite': (
                home_factors['efg_differential'] - away_factors['efg_differential'] +
                home_factors['tov_differential'] - away_factors['tov_differential'] +
                home_factors['oreb_differential'] - away_factors['oreb_differential'] +
                home_factors['ft_differential'] - away_factors['ft_differential']
            ) / 4,
            
            'situational_advantage': self._calculate_situational_advantage(
                home_situations, away_situations, home_team
            )
        }
        
        return {
            'home_team_data': {
                'four_factors': home_factors,
                'pace_efficiency': home_pace,
                'clutch': home_clutch,
                'rest_travel': home_rest,
                'situational': home_situations
            },
            'away_team_data': {
                'four_factors': away_factors,
                'pace_efficiency': away_pace,
                'clutch': away_clutch,
                'rest_travel': away_rest,
                'situational': away_situations
            },
            'differentials': differentials
        }
    
    def _calculate_rest_advantage(self, rest_days):
        """Calculate advantage from rest days"""
        if rest_days == 0:
            return -0.15  # Back-to-back penalty
        elif rest_days == 1:
            return 0.0    # Normal rest
        elif rest_days == 2:
            return 0.05   # Good rest
        else:
            return 0.10   # Extra rest
    
    def _calculate_travel_fatigue(self, miles):
        """Calculate travel fatigue factor"""
        if miles < 500:
            return 0.0
        elif miles < 1500:
            return 0.02
        elif miles < 2500:
            return 0.05
        else:
            return 0.08
    
    def _calculate_situational_advantage(self, home_sit, away_sit, home_team):
        """Calculate situational advantages"""
        # This would be more sophisticated in practice
        return np.random.uniform(-0.05, 0.10)  # Home court advantage range
    
    def _get_team_id(self, team_name):
        """Get NBA API team ID"""
        # Simplified mapping
        team_ids = {
            'boston celtics': 1610612738,
            'brooklyn nets': 1610612751,
            # Add all teams...
        }
        return team_ids.get(team_name.lower(), 1610612738)
    
    def _get_basic_team_stats(self, team_name):
        """Get basic team statistics"""
        # Mock data for now
        return {
            'PACE': np.random.uniform(95, 105),
            'OFF_RATING': np.random.uniform(105, 120),
            'DEF_RATING': np.random.uniform(105, 120),
            'NET_RATING': np.random.uniform(-10, 10),
            'POSS': np.random.uniform(95, 105)
        }
    
    def _get_mock_four_factors(self):
        """Generate mock four factors data"""
        return {
            'off_efg_pct': np.random.uniform(0.48, 0.58),
            'off_tov_pct': np.random.uniform(12, 18),
            'off_oreb_pct': np.random.uniform(20, 30),
            'off_ft_rate': np.random.uniform(0.15, 0.25),
            'def_efg_pct': np.random.uniform(0.48, 0.58),
            'def_tov_pct': np.random.uniform(12, 18),
            'def_oreb_pct': np.random.uniform(20, 30),
            'def_ft_rate': np.random.uniform(0.15, 0.25),
            'efg_differential': np.random.uniform(-0.05, 0.05),
            'tov_differential': np.random.uniform(-3, 3),
            'oreb_differential': np.random.uniform(-5, 5),
            'ft_differential': np.random.uniform(-0.03, 0.03)
        }
    
    def _get_mock_pace_data(self):
        """Generate mock pace data"""
        return {
            'pace': np.random.uniform(95, 105),
            'off_rating': np.random.uniform(105, 120),
            'def_rating': np.random.uniform(105, 120),
            'net_rating': np.random.uniform(-10, 10),
            'poss_per_game': np.random.uniform(95, 105)
        }
    
    def _get_mock_clutch_data(self):
        """Generate mock clutch data"""
        return {
            'clutch_record': f"{np.random.randint(5, 15)}-{np.random.randint(3, 12)}",
            'clutch_off_rating': np.random.uniform(105, 125),
            'clutch_def_rating': np.random.uniform(105, 125),
            'clutch_net_rating': np.random.uniform(-10, 10),
            'close_game_experience': np.random.randint(15, 35)
        }
    
    def _get_mock_rest_data(self):
        """Generate mock rest/travel data"""
        rest_days = np.random.randint(0, 4)
        return {
            'rest_days': rest_days,
            'travel_miles': np.random.randint(0, 3000),
            'rest_advantage': self._calculate_rest_advantage(rest_days),
            'travel_fatigue_factor': np.random.uniform(0, 0.08),
            'back_to_back': rest_days == 0,
            'well_rested': rest_days >= 2
        }
    
    def _get_mock_situational_data(self):
        """Generate mock situational data"""
        return {
            'home_record': f"{np.random.randint(15, 25)}-{np.random.randint(5, 15)}",
            'away_record': f"{np.random.randint(10, 20)}-{np.random.randint(10, 20)}",
            'vs_above_500': f"{np.random.randint(8, 18)}-{np.random.randint(12, 22)}",
            'vs_below_500': f"{np.random.randint(12, 22)}-{np.random.randint(3, 13)}",
            'in_division': f"{np.random.randint(6, 12)}-{np.random.randint(4, 10)}",
            'favored_record': f"{np.random.randint(20, 30)}-{np.random.randint(8, 18)}",
            'underdog_record': f"{np.random.randint(5, 15)}-{np.random.randint(15, 25)}"
        }

# Global instance
nba_analytics = AdvancedNBAAnalytics()

def get_advanced_game_analysis(home_team, away_team, game_date=None):
    """Get comprehensive advanced analytics for a game"""
    return nba_analytics.calculate_game_factors_differential(home_team, away_team, game_date)

if __name__ == "__main__":
    # Test advanced analytics
    analysis = get_advanced_game_analysis("Boston Celtics", "Los Angeles Lakers")
    print("Advanced game analysis completed")
    print("Four factors composite:", analysis['differentials']['four_factors_composite'])