import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import teamgamelog, leaguegamefinder
import time

def get_team_strength_of_schedule(team_name, num_games=10):
    """
    Calculate strength of schedule for recent games
    """
    try:
        # Get recent games
        gamelog = teamgamelog.TeamGameLog(team_name=team_name, season="2024-25")
        time.sleep(1)  # Rate limit
        
        recent_games = gamelog.get_data_frames()[0].head(num_games)
        
        # Calculate opponent win percentages (simplified)
        opponent_strength = 0.5  # Placeholder - would calculate actual opponent records
        
        return opponent_strength
        
    except Exception as e:
        print(f"⚠️ Error getting SOS for {team_name}: {e}")
        return 0.5

def get_home_away_splits(team_name):
    """
    Get team's home vs away performance splits
    """
    try:
        gamelog = teamgamelog.TeamGameLog(team_name=team_name, season="2024-25")
        time.sleep(1)
        
        games_df = gamelog.get_data_frames()[0]
        
        # Split home/away games
        home_games = games_df[games_df['MATCHUP'].str.contains('vs.')]
        away_games = games_df[games_df['MATCHUP'].str.contains('@')]
        
        home_win_pct = (home_games['WL'] == 'W').mean() if len(home_games) > 0 else 0.5
        away_win_pct = (away_games['WL'] == 'W').mean() if len(away_games) > 0 else 0.5
        
        home_avg_pts = home_games['PTS'].mean() if len(home_games) > 0 else 110
        away_avg_pts = away_games['PTS'].mean() if len(away_games) > 0 else 110
        
        return {
            'home_win_pct': home_win_pct,
            'away_win_pct': away_win_pct,
            'home_avg_pts': home_avg_pts,
            'away_avg_pts': away_avg_pts
        }
        
    except Exception as e:
        print(f"⚠️ Error getting splits for {team_name}: {e}")
        return {
            'home_win_pct': 0.5,
            'away_win_pct': 0.5,
            'home_avg_pts': 110,
            'away_avg_pts': 110
        }

def get_momentum_features(team_name, games=5):
    """
    Calculate team momentum indicators
    """
    try:
        gamelog = teamgamelog.TeamGameLog(team_name=team_name, season="2024-25")
        time.sleep(1)
        
        recent_games = gamelog.get_data_frames()[0].head(games)
        
        # Calculate momentum metrics
        wins_last_5 = (recent_games['WL'] == 'W').sum()
        avg_margin = (recent_games['PTS'] - recent_games['OPP_PTS']).mean()
        
        # Trend analysis (are they improving or declining?)
        first_half = recent_games.tail(3)
        second_half = recent_games.head(2)
        
        first_avg_margin = (first_half['PTS'] - first_half['OPP_PTS']).mean()
        second_avg_margin = (second_half['PTS'] - second_half['OPP_PTS']).mean()
        
        momentum_trend = second_avg_margin - first_avg_margin
        
        return {
            'wins_last_5': wins_last_5,
            'avg_point_margin': avg_margin,
            'momentum_trend': momentum_trend
        }
        
    except Exception as e:
        print(f"⚠️ Error getting momentum for {team_name}: {e}")
        return {
            'wins_last_5': 2.5,
            'avg_point_margin': 0,
            'momentum_trend': 0
        }

def get_clutch_performance(team_name):
    """
    Get team performance in close games (simplified)
    """
    try:
        # This would require more complex queries to get clutch stats
        # For now, return placeholder
        return {
            'clutch_win_pct': 0.5,
            'close_game_margin': 0
        }
        
    except Exception as e:
        return {
            'clutch_win_pct': 0.5,
            'close_game_margin': 0
        }