#!/usr/bin/env python3
"""
ESPN NFL Data Scraper
Uses ESPN's API endpoints for NFL data
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime

class ESPNNFLScraper:
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.session = requests.Session()
        
    def get_teams(self):
        """Get all NFL teams"""
        url = f"{self.base_url}/teams"
        response = self.session.get(url)
        
        if response.status_code != 200:
            print(f"Failed to get teams: {response.status_code}")
            return []
            
        data = response.json()
        teams = []
        
        for team in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
            team_info = team.get('team', {})
            teams.append({
                'id': team_info.get('id'),
                'name': team_info.get('displayName'),
                'abbreviation': team_info.get('abbreviation'),
                'location': team_info.get('location')
            })
            
        return teams
    
    def get_season_stats(self, year=2023, stat_type='passing'):
        """Get season statistics for all players"""
        print(f"Getting {stat_type} stats for {year}...")
        
        # ESPN stats categories
        stat_categories = {
            'passing': 'passing',
            'rushing': 'rushing', 
            'receiving': 'receiving'
        }
        
        category = stat_categories.get(stat_type, 'passing')
        url = f"{self.base_url}/seasons/{year}/types/2/leaders"
        
        response = self.session.get(url)
        
        if response.status_code != 200:
            print(f"Failed to get {stat_type} stats: {response.status_code}")
            return pd.DataFrame()
            
        data = response.json()
        
        # Parse leaders data
        all_stats = []
        leaders = data.get('leaders', [])
        
        for leader_category in leaders:
            category_name = leader_category.get('displayName', '').lower()
            
            # Find the right category
            if stat_type.lower() in category_name or category.lower() in category_name:
                leaders_list = leader_category.get('leaders', [])
                
                for leader in leaders_list:
                    athlete = leader.get('athlete', {})
                    team = athlete.get('team', {})
                    
                    stat_data = {
                        'player_id': athlete.get('id'),
                        'player_name': athlete.get('displayName'),
                        'team_id': team.get('id'),
                        'team_name': team.get('displayName'),
                        'team_abbr': team.get('abbreviation'),
                        'position': athlete.get('position', {}).get('abbreviation'),
                        'stat_value': leader.get('value'),
                        'stat_display': leader.get('displayValue'),
                        'year': year,
                        'category': leader_category.get('name')
                    }
                    
                    all_stats.append(stat_data)
        
        return pd.DataFrame(all_stats)
    
    def get_player_stats(self, player_id, year=2023):
        """Get detailed stats for a specific player"""
        url = f"{self.base_url}/seasons/{year}/athletes/{player_id}/statistics"
        response = self.session.get(url)
        
        if response.status_code != 200:
            return {}
            
        data = response.json()
        return data.get('statistics', {})
    
    def get_scoreboard(self, year=2023, week=1):
        """Get games for a specific week"""
        url = f"{self.base_url}/seasons/{year}/types/2/weeks/{week}/events"
        response = self.session.get(url)
        
        if response.status_code != 200:
            return []
            
        data = response.json()
        return data.get('events', [])
    
    def scrape_comprehensive_data(self, years=[2021, 2022, 2023]):
        """Scrape comprehensive NFL data"""
        all_data = {}
        
        for year in years:
            print(f"\n--- Scraping {year} Season ---")
            all_data[year] = {}
            
            # Get passing stats
            passing_df = self.get_season_stats(year, 'passing')
            if not passing_df.empty:
                all_data[year]['passing'] = passing_df
                print(f"Found {len(passing_df)} passing records")
            
            time.sleep(1)
            
            # Get rushing stats  
            rushing_df = self.get_season_stats(year, 'rushing')
            if not rushing_df.empty:
                all_data[year]['rushing'] = rushing_df
                print(f"Found {len(rushing_df)} rushing records")
            
            time.sleep(1)
            
            # Get receiving stats
            receiving_df = self.get_season_stats(year, 'receiving')
            if not receiving_df.empty:
                all_data[year]['receiving'] = receiving_df
                print(f"Found {len(receiving_df)} receiving records")
            
            time.sleep(1)
        
        return all_data
    
    def save_data(self, data, output_dir='data/nfl'):
        """Save scraped data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for year, year_data in data.items():
            for stat_type, df in year_data.items():
                if not df.empty:
                    filename = f"{output_dir}/espn_{stat_type}_{year}_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                    print(f"Saved: {filename}")

def test_espn_scraper():
    """Test the ESPN scraper"""
    scraper = ESPNNFLScraper()
    
    # Test teams
    print("Getting NFL teams...")
    teams = scraper.get_teams()
    print(f"Found {len(teams)} teams")
    if teams:
        print("Sample teams:", teams[:3])
    
    # Test 2023 passing stats
    print("\nGetting 2023 passing stats...")
    passing_stats = scraper.get_season_stats(2023, 'passing')
    print(f"Found {len(passing_stats)} passing records")
    
    if not passing_stats.empty:
        print("Columns:", list(passing_stats.columns))
        print("Top 5 passers:")
        print(passing_stats.head().to_string(index=False))
        
        # Save test data
        passing_stats.to_csv('data/nfl/test_espn_passing_2023.csv', index=False)
        print("Saved test data to data/nfl/test_espn_passing_2023.csv")

if __name__ == "__main__":
    test_espn_scraper()