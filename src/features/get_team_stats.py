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
    Situational splits, approximated deterministically from league averages.

    This used to return unseeded np.random values on every call, in season as
    well as out. Two live features built on it -- home_court_advantage and
    overall_advantage -- were therefore pure noise that changed between runs,
    so the same game could be scored differently twice in a row.

    These are now fixed league-average adjustments: crude, but deterministic
    and honest about being an approximation. Real per-team splits require
    home/away game logs; until those are wired in, a constant offset is
    strictly better than a random one, because it at least cannot masquerade
    as signal.
    """
    base_stats = get_team_stats(team_name_clean)
    net = base_stats["NET_RATING"]
    off = base_stats["OFF_RATING"]

    # League-average magnitudes, not team-specific measurements.
    HOME_OFF_BOOST = 2.0      # teams score ~2 pts/100 better at home
    AWAY_OFF_PENALTY = 2.0
    STRONG_OPP_DRAG = 5.0
    WEAK_OPP_BOOST = 5.0
    B2B_DRAG = 3.0            # back-to-backs cost roughly 3 pts/100
    RESTED_BOOST = 1.5

    return {
        "home_off_rating": off + HOME_OFF_BOOST,
        "away_off_rating": off - AWAY_OFF_PENALTY,
        "vs_strong_teams": net - STRONG_OPP_DRAG,
        "vs_weak_teams": net + WEAK_OPP_BOOST,
        "back_to_back_rating": net - B2B_DRAG,
        "well_rested_rating": net + RESTED_BOOST,
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
