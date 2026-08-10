"""
Player-Level NBA Data Integration
Tracks key players, injuries, minutes, performance trends
"""
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    playergamelog, playerinfo, teamplayerdashboard, 
    leaguedashplayerstats, teamroster, injuryreport
)
import time
import requests
from datetime import datetime, timedelta
from src.utils.config import DATA_DIR

class PlayerDataManager:
    
    def __init__(self):
        self.rate_limit_delay = 1.5  # Seconds between API calls
        self.player_cache = {}
        self.team_rosters = {}
        
    def get_team_key_players(self, team_name, season="2024-25"):
        """Get key players for a team based on minutes/impact"""
        
        try:
            print(f"👨‍💼 Getting key players for {team_name}...")
            
            # Get team roster
            roster = teamroster.TeamRoster(team_id=self._get_team_id(team_name), season=season)
            time.sleep(self.rate_limit_delay)
            
            roster_df = roster.get_data_frames()[0]
            
            # Get player stats for the team
            team_players = teamplayerdashboard.TeamPlayerDashboard(
                team_id=self._get_team_id(team_name),
                season=season
            )
            time.sleep(self.rate_limit_delay)
            
            players_df = team_players.get_data_frames()[1]  # Player stats
            
            # Identify key players (top 8 by minutes)
            key_players = players_df.nlargest(8, 'MIN')
            
            player_data = []
            for _, player in key_players.iterrows():
                player_data.append({
                    'player_id': player['PLAYER_ID'],
                    'player_name': player['PLAYER_NAME'],
                    'position': roster_df[roster_df['PLAYER_ID'] == player['PLAYER_ID']]['POSITION'].iloc[0] if len(roster_df[roster_df['PLAYER_ID'] == player['PLAYER_ID']]) > 0 else 'G',
                    'minutes_per_game': player['MIN'],
                    'points_per_game': player['PTS'],
                    'assists_per_game': player['AST'],
                    'rebounds_per_game': player['REB'],
                    'usage_rate': player.get('USG_PCT', 20.0),
                    'true_shooting': player.get('TS_PCT', 0.55)
                })
            
            return player_data
            
        except Exception as e:
            print(f"⚠️ Error getting players for {team_name}: {e}")
            return self._get_mock_key_players(team_name)
    
    def get_player_recent_performance(self, player_id, games=10):
        """Get recent performance trends for a player"""
        
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season="2024-25"
            )
            time.sleep(self.rate_limit_delay)
            
            recent_games = gamelog.get_data_frames()[0].head(games)
            
            if len(recent_games) == 0:
                return self._get_mock_player_performance()
            
            return {
                'games_played': len(recent_games),
                'avg_points': recent_games['PTS'].mean(),
                'avg_minutes': recent_games['MIN'].mean(),
                'avg_fg_pct': recent_games['FG_PCT'].mean(),
                'avg_plus_minus': recent_games['PLUS_MINUS'].mean(),
                'trending_up': recent_games['PTS'].tail(3).mean() > recent_games['PTS'].head(3).mean()
            }
            
        except Exception as e:
            print(f"⚠️ Error getting recent performance: {e}")
            return self._get_mock_player_performance()
    
    def check_injury_status(self, team_name):
        """Check for injuries that could impact the game"""
        
        try:
            # This would use NBA injury report API
            # For now, return mock data structure
            return {
                'injured_players': [],
                'questionable_players': [],
                'injury_impact_score': 0.0  # 0-1 scale
            }
            
        except Exception as e:
            print(f"⚠️ Error checking injuries: {e}")
            return {'injured_players': [], 'questionable_players': [], 'injury_impact_score': 0.0}
    
    def calculate_team_player_impact(self, team_name):
        """Calculate overall team impact from player performance"""
        
        print(f"🎯 Calculating player impact for {team_name}")
        
        key_players = self.get_team_key_players(team_name)
        injury_status = self.check_injury_status(team_name)
        
        total_impact = 0
        available_impact = 0
        
        for player in key_players:
            # Weight impact by minutes and usage
            minutes_weight = min(player['minutes_per_game'] / 36, 1.0)  # Cap at 36 minutes
            usage_weight = player['usage_rate'] / 30.0  # Normalize usage rate
            
            player_impact = minutes_weight * usage_weight * player['true_shooting']
            total_impact += player_impact
            
            # Check if player is available
            if player['player_name'] not in injury_status['injured_players']:
                available_impact += player_impact
        
        # Calculate availability percentage
        availability_pct = available_impact / total_impact if total_impact > 0 else 1.0
        
        # Get recent performance trends
        trending_players = 0
        for player in key_players[:5]:  # Top 5 players
            recent_perf = self.get_player_recent_performance(player['player_id'])
            if recent_perf['trending_up']:
                trending_players += 1
        
        momentum_factor = trending_players / 5.0
        
        return {
            'key_players_count': len(key_players),
            'availability_percentage': availability_pct,
            'injury_impact_score': injury_status['injury_impact_score'],
            'momentum_factor': momentum_factor,
            'total_player_impact': total_impact,
            'available_player_impact': available_impact
        }
    
    def get_player_matchup_advantages(self, home_team, away_team):
        """Analyze player matchup advantages"""
        
        print(f"⚔️ Analyzing player matchups: {away_team} @ {home_team}")
        
        home_players = self.get_team_key_players(home_team)
        away_players = self.get_team_key_players(away_team)
        
        # Simplified matchup analysis
        home_star_power = sum(p['usage_rate'] * p['true_shooting'] for p in home_players[:3])
        away_star_power = sum(p['usage_rate'] * p['true_shooting'] for p in away_players[:3])
        
        # Depth comparison (players 4-8)
        home_depth = sum(p['minutes_per_game'] for p in home_players[3:6]) / 3
        away_depth = sum(p['minutes_per_game'] for p in away_players[3:6]) / 3
        
        return {
            'home_star_power': home_star_power,
            'away_star_power': away_star_power,
            'star_power_advantage': home_star_power - away_star_power,
            'home_depth_minutes': home_depth,
            'away_depth_minutes': away_depth,
            'depth_advantage': home_depth - away_depth
        }
    
    def _get_team_id(self, team_name):
        """Convert team name to NBA API team ID"""
        
        # Simplified mapping - would expand this
        team_ids = {
            'boston celtics': 1610612738,
            'brooklyn nets': 1610612751,
            'new york knicks': 1610612752,
            'philadelphia 76ers': 1610612755,
            'toronto raptors': 1610612761,
            # Add all 30 teams...
        }
        
        return team_ids.get(team_name.lower(), 1610612738)  # Default to Celtics
    
    def _get_mock_key_players(self, team_name):
        """Generate mock player data for testing"""
        
        import random
        
        mock_players = []
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        
        for i in range(8):
            mock_players.append({
                'player_id': f"mock_{i}",
                'player_name': f"{team_name} Player {i+1}",
                'position': random.choice(positions),
                'minutes_per_game': random.uniform(15, 36),
                'points_per_game': random.uniform(8, 25),
                'assists_per_game': random.uniform(1, 8),
                'rebounds_per_game': random.uniform(2, 10),
                'usage_rate': random.uniform(15, 30),
                'true_shooting': random.uniform(0.45, 0.65)
            })
        
        return mock_players
    
    def _get_mock_player_performance(self):
        """Mock recent player performance"""
        
        import random
        
        return {
            'games_played': 10,
            'avg_points': random.uniform(10, 25),
            'avg_minutes': random.uniform(20, 35),
            'avg_fg_pct': random.uniform(0.40, 0.55),
            'avg_plus_minus': random.uniform(-10, 10),
            'trending_up': random.choice([True, False])
        }

# Global instance
player_data_manager = PlayerDataManager()

def get_team_player_impact(team_name):
    """Get player impact analysis for a team"""
    return player_data_manager.calculate_team_player_impact(team_name)

def get_player_matchups(home_team, away_team):
    """Get player matchup analysis"""
    return player_data_manager.get_player_matchup_advantages(home_team, away_team)

if __name__ == "__main__":
    # Test player data
    impact = get_team_player_impact("Boston Celtics")
    print("Player impact analysis:", impact)