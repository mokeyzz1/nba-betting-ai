"""
Offseason Mode - Generate mock data for testing during NBA offseason
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from src.utils.config import DATA_DIR

# NBA teams for mock data
NBA_TEAMS = [
    "Boston Celtics", "Brooklyn Nets", "New York Knicks", "Philadelphia 76ers", "Toronto Raptors",
    "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Milwaukee Bucks",
    "Atlanta Hawks", "Charlotte Hornets", "Miami Heat", "Orlando Magic", "Washington Wizards",
    "Denver Nuggets", "Minnesota Timberwolves", "Oklahoma City Thunder", "Portland Trail Blazers", "Utah Jazz",
    "Golden State Warriors", "Los Angeles Clippers", "Los Angeles Lakers", "Phoenix Suns", "Sacramento Kings",
    "Dallas Mavericks", "Houston Rockets", "Memphis Grizzlies", "New Orleans Pelicans", "San Antonio Spurs"
]

class OffseasonMode:
    def __init__(self):
        self.is_offseason = True
        
    def generate_mock_odds(self, date_str=None, num_games=6):
        """Generate realistic mock NBA odds data"""
        
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🎭 OFFSEASON MODE: Generating mock odds for {date_str}")
        
        # Randomly select teams for games
        teams_available = NBA_TEAMS.copy()
        random.shuffle(teams_available)
        
        mock_games = []
        
        for i in range(num_games):
            if len(teams_available) < 2:
                break
                
            away_team = teams_available.pop()
            home_team = teams_available.pop()
            
            # Generate realistic odds
            # Home team slight advantage (usually -110 to +110 range)
            home_advantage = random.uniform(-0.05, 0.15)  # 5% to 15% edge
            
            if home_advantage > 0:
                # Home team favored
                home_prob = 0.5 + home_advantage
                home_odds = self._prob_to_american_odds(home_prob)
                away_odds = self._prob_to_american_odds(1 - home_prob)
            else:
                # Away team favored  
                away_prob = 0.5 - home_advantage
                home_odds = self._prob_to_american_odds(1 - away_prob)
                away_odds = self._prob_to_american_odds(away_prob)
            
            # Add some variation to make it realistic
            home_odds += random.randint(-15, 15)
            away_odds += random.randint(-15, 15)
            
            mock_games.append({
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "commence_time": f"{date_str}T19:00:00Z"
            })
        
        # Save mock odds
        df = pd.DataFrame(mock_games)
        filename = DATA_DIR / f"nba_odds_{date_str}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ Generated {len(mock_games)} mock games saved to {filename}")
        return df
    
    def _prob_to_american_odds(self, prob):
        """Convert probability to American odds"""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)
    
    def generate_mock_team_stats(self):
        """Generate realistic team stats for testing"""
        
        mock_stats = {}
        
        for team in NBA_TEAMS:
            team_key = team.lower().replace(" ", "_")
            
            # Generate realistic NBA team stats
            mock_stats[team_key] = {
                "OFF_RATING": random.uniform(105, 120),  # Offensive rating
                "DEF_RATING": random.uniform(105, 120),  # Defensive rating  
                "EFG_PCT": random.uniform(0.48, 0.58),   # Effective FG%
                "PACE": random.uniform(95, 105),         # Pace
                "WIN_PCT": random.uniform(0.25, 0.75)    # Win percentage
            }
        
        return mock_stats
    
    def generate_mock_recent_stats(self, team_name, games=5):
        """Generate mock recent performance stats"""
        
        wins = random.randint(0, games)
        win_pct = wins / games
        avg_pts = random.uniform(100, 125)
        
        return {
            "recent_win_pct": win_pct,
            "recent_avg_pts": avg_pts,
            "games_played": games
        }
    
    def create_historical_test_data(self, start_date="2024-01-01", end_date="2024-04-15"):
        """Create historical data for backtesting"""
        
        print(f"🏀 Creating historical test data from {start_date} to {end_date}")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start
        total_games = 0
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip some days (not every day has games)
            if random.random() < 0.3:
                current_date += timedelta(days=1)
                continue
            
            # Generate 4-12 games per day
            num_games = random.randint(4, 12)
            df = self.generate_mock_odds(date_str, num_games)
            
            # Generate results for completed games
            self._add_mock_results(df, date_str)
            
            total_games += len(df)
            current_date += timedelta(days=1)
        
        print(f"✅ Created {total_games} games of historical test data")
    
    def _add_mock_results(self, df, date_str):
        """Add mock game results to odds data"""
        
        results = []
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            home_odds = row['home_odds']
            
            # Simulate realistic results based on odds
            home_prob = self._american_odds_to_prob(home_odds)
            
            # Add some randomness but generally favor the favorite
            actual_result = "HOME" if random.random() < home_prob else "AWAY"
            
            results.append({
                "game_id": f"{date_str}_{home_team}_{away_team}",
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "actual_winner": actual_result,
                "home_score": random.randint(95, 130),
                "away_score": random.randint(95, 130)
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_file = DATA_DIR / f"mock_results_{date_str}.csv"
        results_df.to_csv(results_file, index=False)
    
    def _american_odds_to_prob(self, odds):
        """Convert American odds to probability"""
        if odds > 0:
            return 100 / (100 + odds)
        else:
            return -odds / (-odds + 100)

# Global instance
offseason = OffseasonMode()

def is_offseason():
    """Whether to substitute mock data. OPT-IN, and deliberately so.

    This used to return True automatically for June-September, which meant
    every feature getter silently swapped in randomly generated team ratings
    without failing or warning. That is how predictions/predictions_2025-08-17
    came to pair real sportsbook odds with fabricated stats -- output that
    looks exactly like a real prediction file and is worth nothing.

    Mock data is fine for exercising the plumbing. It is not fine as a silent
    fallback in a path that writes files someone might later trust. So it now
    requires an explicit opt-in:

        NBA_MOCK_DATA=1 python run_pipeline.py

    Out of season without the flag, the pipeline fails instead of inventing
    numbers -- which is the correct outcome, because there are no games.
    """
    import os
    return os.getenv("NBA_MOCK_DATA", "").strip().lower() in ("1", "true", "yes")

def get_mock_data_if_needed():
    """Get mock data if in offseason"""
    if is_offseason():
        print("🎭 Offseason detected - using mock data mode")
        today = datetime.now().strftime('%Y-%m-%d')
        return offseason.generate_mock_odds(today)
    return None