import requests
import pandas as pd
from datetime import datetime
import time
from src.utils.config import DATA_DIR
from src.utils.team_name_mapping import TEAM_NAME_TO_ODDS_NAME

def get_injury_report():
    """
    Fetch injury data for NBA teams. 
    This is a placeholder - you'll need to integrate with a real injury API
    like ESPN, RotoWire, or NBA.com
    """
    print("📋 Fetching NBA injury reports...")
    
    try:
        # For now, return empty dict - implement with real API later
        # Real implementation would use ESPN API or similar
        injury_data = {}
        
        # Example structure:
        # injury_data = {
        #     "Boston Celtics": {
        #         "injured_players": ["Player Name"],
        #         "injury_impact_score": 0.15  # 0-1 scale of team impact
        #     }
        # }
        
        return injury_data
        
    except Exception as e:
        print(f"⚠️ Error fetching injury data: {e}")
        return {}

def calculate_injury_impact(home_team, away_team, injury_data):
    """
    Calculate injury impact differential between teams
    """
    home_impact = injury_data.get(home_team, {}).get('injury_impact_score', 0)
    away_impact = injury_data.get(away_team, {}).get('injury_impact_score', 0)
    
    # Positive means home team less impacted by injuries
    return away_impact - home_impact

def get_rest_days(team_name):
    """
    Calculate days of rest for a team (placeholder)
    Real implementation would track last game date
    """
    # Placeholder - implement with real game log data
    return 1  # Default 1 day rest

def is_back_to_back(team_name):
    """
    Check if team is playing back-to-back games
    """
    rest_days = get_rest_days(team_name)
    return rest_days == 0