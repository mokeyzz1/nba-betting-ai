import pandas as pd
import numpy as np
from src.utils.offseason_mode import is_offseason

def get_recent_win_pct(team_name_clean, num_games=5):
    """
    Returns win % over last `num_games` for a team.
    Enhanced with fallback for offseason testing.
    """
    
    # During offseason or if NBA API fails, use mock data
    if is_offseason():
        return _get_mock_recent_win_pct(team_name_clean, num_games)
    
    try:
        from nba_api.stats.endpoints import teamgamelog
        gamelog = teamgamelog.TeamGameLog(team_name=team_name_clean, season="2024-25")
        df = gamelog.get_data_frames()[0].head(num_games)
        wins = df["WL"].str.upper().value_counts().get("W", 0)
        return wins / num_games
    except Exception as e:
        print(f"⚠️ NBA API failed for {team_name_clean}, using mock data: {e}")
        return _get_mock_recent_win_pct(team_name_clean, num_games)

def get_recent_avg_pts(team_name_clean, num_games=5):
    """
    Returns average points over last `num_games` for a team.
    Enhanced with fallback for offseason testing.
    """
    
    # During offseason or if NBA API fails, use mock data
    if is_offseason():
        return _get_mock_recent_avg_pts(team_name_clean, num_games)
    
    try:
        from nba_api.stats.endpoints import teamgamelog
        gamelog = teamgamelog.TeamGameLog(team_name=team_name_clean, season="2024-25")
        df = gamelog.get_data_frames()[0].head(num_games)
        return df["PTS"].mean()
    except Exception as e:
        print(f"⚠️ NBA API failed for {team_name_clean}, using mock data: {e}")
        return _get_mock_recent_avg_pts(team_name_clean, num_games)

def get_recent_performance_trends(team_name_clean, num_games=10):
    """
    Get comprehensive recent performance analysis
    """
    
    if is_offseason():
        return _get_mock_performance_trends(team_name_clean, num_games)
    
    try:
        from nba_api.stats.endpoints import teamgamelog
        gamelog = teamgamelog.TeamGameLog(team_name=team_name_clean, season="2024-25")
        recent_games = gamelog.get_data_frames()[0].head(num_games)
        
        if len(recent_games) == 0:
            return _get_mock_performance_trends(team_name_clean, num_games)
        
        # Calculate trends
        wins = (recent_games["WL"] == "W").sum()
        avg_pts = recent_games["PTS"].mean()
        avg_opp_pts = recent_games["OPP_PTS"].mean()
        avg_margin = avg_pts - avg_opp_pts
        
        # Momentum analysis
        first_half = recent_games.tail(num_games//2)
        second_half = recent_games.head(num_games//2)
        
        recent_trend = second_half["PTS"].mean() - first_half["PTS"].mean()
        
        return {
            "games_played": len(recent_games),
            "win_pct": wins / len(recent_games),
            "avg_points": avg_pts,
            "avg_opp_points": avg_opp_pts,
            "avg_margin": avg_margin,
            "momentum_trend": recent_trend,
            "trending_up": recent_trend > 0,
            "hot_streak": wins >= (num_games * 0.7),  # 70%+ win rate
            "cold_streak": wins <= (num_games * 0.3)   # 30%- win rate
        }
        
    except Exception as e:
        print(f"⚠️ Performance trends failed for {team_name_clean}: {e}")
        return _get_mock_performance_trends(team_name_clean, num_games)

def _get_mock_recent_win_pct(team_name_clean, num_games):
    """Generate realistic mock recent win percentage"""
    # Seed based on team name for consistency
    np.random.seed(hash(team_name_clean) % 2**32)
    
    # Generate realistic win percentage (30-80% range)
    base_win_pct = np.random.uniform(0.3, 0.8)
    
    # Add some recent performance variation
    recent_variation = np.random.uniform(-0.2, 0.2)
    recent_win_pct = np.clip(base_win_pct + recent_variation, 0.0, 1.0)
    
    return recent_win_pct

def _get_mock_recent_avg_pts(team_name_clean, num_games):
    """Generate realistic mock recent scoring average"""
    # Seed based on team name for consistency
    np.random.seed(hash(team_name_clean) % 2**32)
    
    # Generate realistic NBA scoring (100-125 range)
    base_scoring = np.random.uniform(105, 120)
    
    # Add recent form variation
    recent_variation = np.random.uniform(-8, 8)
    recent_avg_pts = max(95, base_scoring + recent_variation)
    
    return recent_avg_pts

def _get_mock_performance_trends(team_name_clean, num_games):
    """Generate mock comprehensive performance trends"""
    # Seed based on team name for consistency
    np.random.seed(hash(team_name_clean) % 2**32)
    
    win_pct = _get_mock_recent_win_pct(team_name_clean, num_games)
    avg_points = _get_mock_recent_avg_pts(team_name_clean, num_games)
    avg_opp_points = np.random.uniform(105, 118)
    
    return {
        "games_played": num_games,
        "win_pct": win_pct,
        "avg_points": avg_points,
        "avg_opp_points": avg_opp_points,
        "avg_margin": avg_points - avg_opp_points,
        "momentum_trend": np.random.uniform(-5, 5),
        "trending_up": np.random.choice([True, False]),
        "hot_streak": win_pct > 0.7,
        "cold_streak": win_pct < 0.3
    }
