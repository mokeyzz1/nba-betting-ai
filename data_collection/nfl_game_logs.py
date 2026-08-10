#!/usr/bin/env python3
"""
NFL Game Logs Scraper
Gets detailed game-by-game statistics for building prop betting models
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime
import json
import re

class NFLGameLogScraper:
    def __init__(self):
        self.base_url = "https://www.pro-football-reference.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def get_top_players_by_position(self, year, position_type='QB', min_games=8):
        """Get list of top players to scrape game logs for"""
        
        if position_type == 'QB':
            url = f"{self.base_url}/years/{year}/passing.htm"
            stat_filter = 'pass_yds'
        elif position_type == 'RB':
            url = f"{self.base_url}/years/{year}/rushing.htm"
            stat_filter = 'rush_yds'
        elif position_type == 'WR':
            url = f"{self.base_url}/years/{year}/receiving.htm"
            stat_filter = 'rec_yds'
        else:
            return []
            
        response = self.session.get(url)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main stats table
        table = soup.find('table')
        if not table:
            return []
            
        players = []
        tbody = table.find('tbody')
        
        for row in tbody.find_all('tr'):
            if 'thead' in row.get('class', []):
                continue
                
            # Get player link and basic stats
            player_cell = row.find('td', {'data-stat': 'player'})
            if not player_cell:
                continue
                
            player_link = player_cell.find('a')
            if not player_link:
                continue
                
            player_name = player_link.text.strip()
            player_url = player_link.get('href')
            
            # Get games played
            games_cell = row.find('td', {'data-stat': 'g'})
            games = int(games_cell.text.strip()) if games_cell and games_cell.text.strip().isdigit() else 0
            
            # Get primary stat (yards)
            stat_cell = row.find('td', {'data-stat': stat_filter})
            stat_value = int(stat_cell.text.strip()) if stat_cell and stat_cell.text.strip().replace(',', '').isdigit() else 0
            
            if games >= min_games and stat_value > 0:
                players.append({
                    'name': player_name,
                    'url': player_url,
                    'games': games,
                    'total_yards': stat_value,
                    'position': position_type
                })
        
        # Sort by total yards and return top players
        players.sort(key=lambda x: x['total_yards'], reverse=True)
        return players[:50]  # Top 50 players
    
    def scrape_player_game_log(self, player_url, player_name, year, position):
        """Scrape detailed game log for a specific player"""
        print(f"Scraping {player_name} ({position}) game log for {year}...")
        
        url = f"{self.base_url}{player_url}/gamelog/{year}/"
        response = self.session.get(url)
        
        if response.status_code != 200:
            print(f"Failed to get game log for {player_name}")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find stats table
        table = soup.find('table', {'id': 'stats'})
        if not table:
            print(f"No stats table found for {player_name}")
            return pd.DataFrame()
        
        # Get headers
        headers = []
        header_row = table.find('thead').find('tr')
        for th in header_row.find_all('th'):
            stat = th.get('data-stat', th.text.strip())
            headers.append(stat)
        
        # Parse game data
        games = []
        tbody = table.find('tbody')
        
        for row in tbody.find_all('tr'):
            if 'thead' in row.get('class', []):
                continue
                
            game_data = {'player_name': player_name, 'position': position, 'year': year}
            
            for i, cell in enumerate(row.find_all(['th', 'td'])):
                if i < len(headers):
                    stat_name = headers[i]
                    value = cell.text.strip()
                    
                    # Clean up the value
                    if value == '':
                        value = None
                    elif value.replace('.', '').replace('-', '').isdigit():
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    
                    game_data[stat_name] = value
            
            # Only include regular season games (skip playoffs)
            if game_data.get('game_date') and game_data.get('opp'):
                games.append(game_data)
        
        return pd.DataFrame(games)
    
    def scrape_position_game_logs(self, year, position='QB'):
        """Scrape game logs for all top players at a position"""
        print(f"\n=== Scraping {position} Game Logs for {year} ===")
        
        # Get list of top players
        top_players = self.get_top_players_by_position(year, position)
        print(f"Found {len(top_players)} top {position}s to scrape")
        
        all_game_logs = []
        
        for i, player in enumerate(top_players):
            try:
                game_log = self.scrape_player_game_log(
                    player['url'], 
                    player['name'], 
                    year, 
                    position
                )
                
                if not game_log.empty:
                    all_game_logs.append(game_log)
                
                # Rate limiting
                time.sleep(1)
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(top_players)} players")
                    
            except Exception as e:
                print(f"Error scraping {player['name']}: {e}")
                continue
        
        # Combine all game logs
        if all_game_logs:
            combined_df = pd.concat(all_game_logs, ignore_index=True)
            print(f"Scraped {len(combined_df)} total games for {position}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def scrape_all_positions_all_years(self, years=[2021, 2022, 2023, 2024]):
        """Scrape game logs for all positions and years"""
        positions = ['QB', 'RB', 'WR']
        all_data = {}
        
        for year in years:
            all_data[year] = {}
            
            for position in positions:
                try:
                    position_data = self.scrape_position_game_logs(year, position)
                    all_data[year][position] = position_data
                    
                    # Save intermediate results
                    if not position_data.empty:
                        filename = f"data/nfl/game_logs_{position}_{year}.csv"
                        position_data.to_csv(filename, index=False)
                        print(f"Saved: {filename}")
                    
                    time.sleep(3)  # Longer pause between positions
                    
                except Exception as e:
                    print(f"Error scraping {position} for {year}: {e}")
                    all_data[year][position] = pd.DataFrame()
        
        return all_data
    
    def create_prop_betting_dataset(self, game_logs_dir='data/nfl'):
        """Process game logs into prop betting format"""
        print("\n=== Creating Prop Betting Dataset ===")
        
        # Load all game log files
        qb_files = [f for f in os.listdir(game_logs_dir) if 'QB' in f and f.endswith('.csv')]
        rb_files = [f for f in os.listdir(game_logs_dir) if 'RB' in f and f.endswith('.csv')]
        wr_files = [f for f in os.listdir(game_logs_dir) if 'WR' in f and f.endswith('.csv')]
        
        # Process QB data for passing yards props
        qb_props = []
        for file in qb_files:
            df = pd.read_csv(os.path.join(game_logs_dir, file))
            if 'pass_yds' in df.columns:
                for _, row in df.iterrows():
                    if pd.notnull(row['pass_yds']):
                        qb_props.append({
                            'player': row['player_name'],
                            'date': row['game_date'],
                            'opponent': row['opp'],
                            'stat_type': 'passing_yards',
                            'actual_value': row['pass_yds'],
                            'year': row['year']
                        })
        
        # Process RB data for rushing yards props
        rb_props = []
        for file in rb_files:
            df = pd.read_csv(os.path.join(game_logs_dir, file))
            if 'rush_yds' in df.columns:
                for _, row in df.iterrows():
                    if pd.notnull(row['rush_yds']):
                        rb_props.append({
                            'player': row['player_name'],
                            'date': row['game_date'],
                            'opponent': row['opp'],
                            'stat_type': 'rushing_yards',
                            'actual_value': row['rush_yds'],
                            'year': row['year']
                        })
        
        # Process WR data for receiving yards props
        wr_props = []
        for file in wr_files:
            df = pd.read_csv(os.path.join(game_logs_dir, file))
            if 'rec_yds' in df.columns:
                for _, row in df.iterrows():
                    if pd.notnull(row['rec_yds']):
                        wr_props.append({
                            'player': row['player_name'],
                            'date': row['game_date'],
                            'opponent': row['opp'],
                            'stat_type': 'receiving_yards',
                            'actual_value': row['rec_yds'],
                            'year': row['year']
                        })
        
        # Combine all props
        all_props = qb_props + rb_props + wr_props
        props_df = pd.DataFrame(all_props)
        
        if not props_df.empty:
            output_file = f"{game_logs_dir}/nfl_prop_betting_data.csv"
            props_df.to_csv(output_file, index=False)
            print(f"Created prop betting dataset: {output_file}")
            print(f"Total records: {len(props_df)}")
        
        return props_df

def main():
    """Main execution function"""
    scraper = NFLGameLogScraper()
    
    # Scrape game logs for recent years
    years = [2021, 2022, 2023, 2024]
    print(f"Starting NFL game log scraping for: {years}")
    
    # Scrape all data
    all_data = scraper.scrape_all_positions_all_years(years)
    
    # Create prop betting dataset
    props_data = scraper.create_prop_betting_dataset()
    
    print("\n=== Scraping Complete ===")
    for year in years:
        for position in ['QB', 'RB', 'WR']:
            if year in all_data and position in all_data[year]:
                count = len(all_data[year][position])
                print(f"{year} {position}: {count} games")

if __name__ == "__main__":
    main()