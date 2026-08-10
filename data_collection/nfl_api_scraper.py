#!/usr/bin/env python3
"""
NFL API Data Scraper
Uses multiple sources to get NFL data
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

class NFLAPIScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_nfl_api_data(self, url):
        """Generic function to get data from NFL API"""
        try:
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get data from {url}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def get_fantasy_data(self, year=2023, week='all'):
        """Get fantasy football data which includes key stats"""
        print(f"Getting fantasy data for {year}...")
        
        # Try ESPN fantasy API
        url = f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/0"
        
        # Alternative: try a more public API
        # FantasyData.com has free tier
        fantasy_url = f"https://api.fantasydata.net/v3/nfl/stats/json/PlayerSeasonStats/{year}"
        
        # For now, let's create some sample data structure
        sample_data = self.create_sample_nfl_data(year)
        return sample_data
    
    def create_sample_nfl_data(self, year):
        """Create sample NFL data structure for testing"""
        print(f"Creating sample NFL data for {year}...")
        
        # QB sample data
        qb_data = [
            {'player': 'Josh Allen', 'team': 'BUF', 'position': 'QB', 'passing_yards': 4306, 'passing_tds': 29, 'games': 17},
            {'player': 'Tua Tagovailoa', 'team': 'MIA', 'position': 'QB', 'passing_yards': 3548, 'passing_tds': 25, 'games': 17},
            {'player': 'Patrick Mahomes', 'team': 'KC', 'position': 'QB', 'passing_yards': 4183, 'passing_tds': 27, 'games': 16},
            {'player': 'Dak Prescott', 'team': 'DAL', 'position': 'QB', 'passing_yards': 2860, 'passing_tds': 23, 'games': 12},
            {'player': 'Lamar Jackson', 'team': 'BAL', 'position': 'QB', 'passing_yards': 3678, 'passing_tds': 24, 'games': 17},
        ]
        
        # RB sample data
        rb_data = [
            {'player': 'Christian McCaffrey', 'team': 'SF', 'position': 'RB', 'rushing_yards': 1459, 'rushing_tds': 14, 'games': 16},
            {'player': 'Josh Jacobs', 'team': 'LV', 'position': 'RB', 'rushing_yards': 1653, 'rushing_tds': 12, 'games': 17},
            {'player': 'Nick Chubb', 'team': 'CLE', 'position': 'RB', 'rushing_yards': 1525, 'rushing_tds': 12, 'games': 17},
            {'player': 'Derrick Henry', 'team': 'TEN', 'position': 'RB', 'rushing_yards': 1538, 'rushing_tds': 13, 'games': 16},
            {'player': 'Saquon Barkley', 'team': 'NYG', 'position': 'RB', 'rushing_yards': 1312, 'rushing_tds': 10, 'games': 16},
        ]
        
        # WR sample data
        wr_data = [
            {'player': 'Tyreek Hill', 'team': 'MIA', 'position': 'WR', 'receiving_yards': 1799, 'receiving_tds': 7, 'receptions': 119, 'games': 17},
            {'player': 'Stefon Diggs', 'team': 'BUF', 'position': 'WR', 'receiving_yards': 1429, 'receiving_tds': 11, 'receptions': 108, 'games': 17},
            {'player': 'Davante Adams', 'team': 'LV', 'position': 'WR', 'receiving_yards': 1516, 'receiving_tds': 14, 'receptions': 100, 'games': 17},
            {'player': 'Travis Kelce', 'team': 'KC', 'position': 'TE', 'receiving_yards': 1338, 'receiving_tds': 12, 'receptions': 110, 'games': 16},
            {'player': 'CeeDee Lamb', 'team': 'DAL', 'position': 'WR', 'receiving_yards': 1359, 'receiving_tds': 9, 'receptions': 107, 'games': 17},
        ]
        
        return {
            'quarterbacks': pd.DataFrame(qb_data),
            'running_backs': pd.DataFrame(rb_data),
            'wide_receivers': pd.DataFrame(wr_data)
        }
    
    def generate_game_logs(self, season_data):
        """Generate realistic game logs from season data"""
        print("Generating game logs from season data...")
        
        all_game_logs = []
        
        for position, df in season_data.items():
            for _, player in df.iterrows():
                games = player.get('games', 16)
                
                if position == 'quarterbacks':
                    # Generate QB game logs
                    avg_yards = player['passing_yards'] / games
                    avg_tds = player['passing_tds'] / games
                    
                    for game in range(1, games + 1):
                        # Add some variance to make it realistic
                        variance = 0.3  # 30% variance
                        game_yards = int(avg_yards * (1 + (0.5 - np.random.random()) * variance))
                        game_tds = max(0, int(avg_tds * (1 + (0.5 - np.random.random()) * variance)))
                        
                        game_log = {
                            'player': player['player'],
                            'team': player['team'],
                            'position': player['position'],
                            'game_number': game,
                            'week': game,
                            'stat_type': 'passing_yards',
                            'stat_value': game_yards,
                            'season_total': player['passing_yards'],
                            'year': 2023
                        }
                        all_game_logs.append(game_log)
                
                elif position == 'running_backs':
                    # Generate RB game logs
                    avg_yards = player['rushing_yards'] / games
                    avg_tds = player['rushing_tds'] / games
                    
                    for game in range(1, games + 1):
                        variance = 0.4  # 40% variance for RBs
                        game_yards = int(avg_yards * (1 + (0.5 - np.random.random()) * variance))
                        game_tds = max(0, int(avg_tds * (1 + (0.5 - np.random.random()) * variance)))
                        
                        game_log = {
                            'player': player['player'],
                            'team': player['team'],
                            'position': player['position'],
                            'game_number': game,
                            'week': game,
                            'stat_type': 'rushing_yards',
                            'stat_value': game_yards,
                            'season_total': player['rushing_yards'],
                            'year': 2023
                        }
                        all_game_logs.append(game_log)
                
                elif position == 'wide_receivers':
                    # Generate WR game logs
                    avg_yards = player['receiving_yards'] / games
                    avg_tds = player['receiving_tds'] / games
                    
                    for game in range(1, games + 1):
                        variance = 0.5  # 50% variance for WRs
                        game_yards = int(avg_yards * (1 + (0.5 - np.random.random()) * variance))
                        game_tds = max(0, int(avg_tds * (1 + (0.5 - np.random.random()) * variance)))
                        
                        game_log = {
                            'player': player['player'],
                            'team': player['team'],
                            'position': player['position'],
                            'game_number': game,
                            'week': game,
                            'stat_type': 'receiving_yards',
                            'stat_value': game_yards,
                            'season_total': player['receiving_yards'],
                            'year': 2023
                        }
                        all_game_logs.append(game_log)
        
        return pd.DataFrame(all_game_logs)
    
    def create_prop_betting_lines(self, game_logs):
        """Create prop betting lines based on historical performance"""
        print("Creating prop betting lines...")
        
        prop_lines = []
        
        # Group by player and stat type
        for (player, stat_type), group in game_logs.groupby(['player', 'stat_type']):
            stat_values = group['stat_value'].values
            
            # Calculate percentiles for common prop lines
            if stat_type == 'passing_yards':
                lines = [225.5, 250.5, 275.5, 300.5, 325.5]
            elif stat_type == 'rushing_yards':
                lines = [50.5, 65.5, 80.5, 95.5, 110.5]
            elif stat_type == 'receiving_yards':
                lines = [45.5, 60.5, 75.5, 90.5, 105.5]
            else:
                continue
            
            for line in lines:
                over_count = sum(1 for val in stat_values if val > line)
                over_rate = over_count / len(stat_values)
                
                prop_lines.append({
                    'player': player,
                    'stat_type': stat_type,
                    'line': line,
                    'over_rate': over_rate,
                    'over_count': over_count,
                    'total_games': len(stat_values),
                    'avg_value': stat_values.mean(),
                    'year': 2023
                })
        
        return pd.DataFrame(prop_lines)

def main():
    """Main function to test NFL data scraping"""
    scraper = NFLAPIScraper()
    
    # Get sample data
    season_data = scraper.get_fantasy_data(2023)
    
    print("Season Data Summary:")
    for position, df in season_data.items():
        print(f"{position}: {len(df)} players")
        print(df.head())
        print()
    
    # Generate game logs
    game_logs = scraper.generate_game_logs(season_data)
    print(f"Generated {len(game_logs)} game log entries")
    
    # Create prop lines
    prop_lines = scraper.create_prop_betting_lines(game_logs)
    print(f"Created {len(prop_lines)} prop betting lines")
    
    # Save data
    import os
    os.makedirs('data/nfl', exist_ok=True)
    
    # Save season data
    for position, df in season_data.items():
        filename = f'data/nfl/nfl_{position}_2023.csv'
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
    
    # Save game logs
    game_logs.to_csv('data/nfl/nfl_game_logs_2023.csv', index=False)
    print("Saved: data/nfl/nfl_game_logs_2023.csv")
    
    # Save prop lines
    prop_lines.to_csv('data/nfl/nfl_prop_lines_2023.csv', index=False)
    print("Saved: data/nfl/nfl_prop_lines_2023.csv")
    
    print("\n--- Sample Prop Lines ---")
    sample_props = prop_lines[prop_lines['player'].isin(['Josh Allen', 'Christian McCaffrey', 'Tyreek Hill'])]
    print(sample_props.to_string(index=False))

if __name__ == "__main__":
    main()