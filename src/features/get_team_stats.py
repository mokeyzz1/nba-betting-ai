import pandas as pd
import numpy as np
from src.utils.offseason_mode import is_offseason

# Load team advanced stats from CSV (fallback to mock data if not available)
try:
    team_stats_df = pd.read_csv("data/team_advanced_stats.csv")
    team_stats_df["team_clean"] = team_stats_df["TEAM_NAME"].str.lower().str.strip()
    team_stats_dict = {row["team_clean"]: row for _, row in team_stats_df.iterrows()}
    print("✅ Loaded team stats from CSV")
except FileNotFoundError:
    print("⚠️ team_advanced_stats.csv not found - using mock data")
    team_stats_dict = {}

def get_team_stats(team_name_clean):
    """
    Return advanced stats with enhanced four factors and situational metrics
    """
    row = team_stats_dict.get(team_name_clean)
    
    if row is None or is_offseason():
        # Generate realistic mock stats for testing
        return _generate_mock_team_stats(team_name_clean)

    # Enhanced stats including four factors
    return {
        # Basic ratings
        "OFF_RATING": row.get("OFF_RATING", 110.0),
        "DEF_RATING": row.get("DEF_RATING", 110.0),
        "PACE": row.get("PACE", 100.0),
        "EFG_PCT": row.get("EFG_PCT", 0.52),
        
        # Four factors (if available in data)
        "TOV_PCT": row.get("TOV_PCT", 15.0),
        "OREB_PCT": row.get("OREB_PCT", 25.0),
        "FT_RATE": row.get("FT_RATE", 0.20),
        
        # Defensive four factors
        "OPP_EFG_PCT": row.get("OPP_EFG_PCT", 0.52),
        "OPP_TOV_PCT": row.get("OPP_TOV_PCT", 15.0),
        "DREB_PCT": row.get("DREB_PCT", 75.0),
        
        # Advanced metrics
        "NET_RATING": row.get("NET_RATING", 0.0),
        "TS_PCT": row.get("TS_PCT", 0.56),
        "WIN_PCT": row.get("W_PCT", 0.50)
    }

def get_team_four_factors_differential(team_name_clean):
    """
    Calculate four factors advantage vs league average
    """
    stats = get_team_stats(team_name_clean)
    
    # League averages (approximate)
    league_avg = {
        "EFG_PCT": 0.52,
        "TOV_PCT": 15.0,
        "OREB_PCT": 25.0,
        "FT_RATE": 0.20
    }
    
    return {
        "efg_advantage": stats["EFG_PCT"] - league_avg["EFG_PCT"],
        "tov_advantage": league_avg["TOV_PCT"] - stats["TOV_PCT"],  # Lower is better
        "oreb_advantage": stats["OREB_PCT"] - league_avg["OREB_PCT"],
        "ft_advantage": stats["FT_RATE"] - league_avg["FT_RATE"]
    }

def get_team_situational_stats(team_name_clean):
    """
    Get situational performance (home/away, vs strong teams, etc.)
    """
    base_stats = get_team_stats(team_name_clean)
    
    # Mock situational data - would be replaced with real data
    return {
        "home_off_rating": base_stats["OFF_RATING"] + np.random.uniform(2, 6),
        "away_off_rating": base_stats["OFF_RATING"] - np.random.uniform(0, 4),
        "vs_strong_teams": base_stats["NET_RATING"] - np.random.uniform(2, 8),
        "vs_weak_teams": base_stats["NET_RATING"] + np.random.uniform(3, 10),
        "back_to_back_rating": base_stats["NET_RATING"] - np.random.uniform(3, 8),
        "well_rested_rating": base_stats["NET_RATING"] + np.random.uniform(1, 5)
    }

def _generate_mock_team_stats(team_name_clean):
    """
    Generate realistic mock team stats for testing
    """
    # Seed with team name for consistency
    np.random.seed(hash(team_name_clean) % 2**32)
    
    # Generate realistic NBA team stats
    off_rating = np.random.uniform(105, 120)
    def_rating = np.random.uniform(105, 120)
    
    return {
        "OFF_RATING": off_rating,
        "DEF_RATING": def_rating,
        "PACE": np.random.uniform(95, 105),
        "EFG_PCT": np.random.uniform(0.48, 0.58),
        "TOV_PCT": np.random.uniform(12, 18),
        "OREB_PCT": np.random.uniform(20, 30),
        "FT_RATE": np.random.uniform(0.15, 0.25),
        "OPP_EFG_PCT": np.random.uniform(0.48, 0.58),
        "OPP_TOV_PCT": np.random.uniform(12, 18),
        "DREB_PCT": np.random.uniform(70, 80),
        "NET_RATING": off_rating - def_rating,
        "TS_PCT": np.random.uniform(0.52, 0.62),
        "WIN_PCT": np.random.uniform(0.25, 0.75)
    }
