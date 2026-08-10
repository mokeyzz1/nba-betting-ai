#!/usr/bin/env python3
"""
NFL COMPLETE DATA SCRAPER
Gets REAL NFL data for 2021-2024 seasons
Comprehensive player statistics and game logs

Run this yourself - it will take time but get you quality data!
"""

import requests
import pandas as pd
import json
import time
import re
import os
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class NFLCompleteScraper:
    def __init__(self):
        self.session = requests.Session()
        self.setup_session()
        self.driver = None
        
    def setup_session(self):
        """Setup requests session with proper headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        self.session.headers.update(headers)
        print("✅ Requests session configured")
    
    def setup_selenium(self):
        """Setup Selenium for sites that block requests"""
        try:
            print("🔧 Setting up Selenium WebDriver...")
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✅ Selenium WebDriver ready")
            return True
        except Exception as e:
            print(f"❌ Selenium setup failed: {e}")
            print("   Install Chrome browser if not available")
            return False
    
    def scrape_pro_football_reference(self, year):
        """
        Scrape Pro Football Reference - the gold standard for NFL stats
        Gets comprehensive player statistics
        """
        print(f"\n📊 SCRAPING PRO FOOTBALL REFERENCE - {year}")
        print("-" * 50)
        
        base_url = "https://www.pro-football-reference.com"
        urls = {
            'passing': f"{base_url}/years/{year}/passing.htm",
            'rushing': f"{base_url}/years/{year}/rushing.htm", 
            'receiving': f"{base_url}/years/{year}/receiving.htm"
        }
        
        all_data = {}
        
        for stat_type, url in urls.items():
            print(f"\n🏈 Scraping {stat_type.upper()} stats for {year}...")
            print(f"URL: {url}")
            
            # Try multiple methods
            data = None
            
            # Method 1: Direct requests
            print("  Method 1: Direct HTTP request...")
            try:
                response = self.session.get(url, timeout=15)
                print(f"    Status: {response.status_code}")
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    table = soup.find('table', {'id': stat_type})
                    
                    if table:
                        data = self.parse_pfr_table(table, stat_type, year)
                        if not data.empty:
                            print(f"    ✅ SUCCESS: Got {len(data)} players")
                        else:
                            print(f"    ⚠️ Table found but no data parsed")
                    else:
                        print(f"    ❌ No table with id '{stat_type}' found")
                else:
                    print(f"    ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"    ❌ Request failed: {e}")
            
            # Method 2: Selenium if direct failed
            if data is None or data.empty:
                print("  Method 2: Selenium WebDriver...")
                
                if self.driver is None:
                    if not self.setup_selenium():
                        print("    ❌ Selenium not available")
                        continue
                
                try:
                    self.driver.get(url)
                    print("    Waiting for page to load...")
                    
                    # Wait for the table to load
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.ID, stat_type))
                    )
                    
                    html = self.driver.page_source
                    soup = BeautifulSoup(html, 'html.parser')
                    table = soup.find('table', {'id': stat_type})
                    
                    if table:
                        data = self.parse_pfr_table(table, stat_type, year)
                        if not data.empty:
                            print(f"    ✅ SELENIUM SUCCESS: Got {len(data)} players")
                        else:
                            print(f"    ⚠️ Selenium: Table found but no data")
                    else:
                        print(f"    ❌ Selenium: No table found")
                        
                except Exception as e:
                    print(f"    ❌ Selenium failed: {e}")
            
            # Save data if we got it
            if data is not None and not data.empty:
                all_data[stat_type] = data
                print(f"    💾 Saved {stat_type} data: {len(data)} players")
            else:
                print(f"    ❌ No {stat_type} data retrieved")
            
            # Be respectful - wait between requests
            print("    ⏱️ Waiting 5 seconds...")
            time.sleep(5)
        
        return all_data
    
    def parse_pfr_table(self, table, stat_type, year):
        """
        Parse Pro Football Reference table
        Extracts all statistics properly
        """
        try:
            print(f"      📋 Parsing {stat_type} table...")
            
            # Get headers
            headers = []
            thead = table.find('thead')
            if thead:
                # PFR sometimes has multiple header rows
                header_rows = thead.find_all('tr')
                for row in header_rows:
                    row_headers = []
                    for th in row.find_all('th'):
                        data_stat = th.get('data-stat')
                        text = th.get_text(strip=True)
                        
                        # Use data-stat if available, otherwise use text
                        header = data_stat if data_stat else text
                        if header and header not in ['', ' ']:
                            row_headers.append(header)
                    
                    if len(row_headers) > len(headers):
                        headers = row_headers
            
            print(f"      Found {len(headers)} columns: {headers[:8]}...")
            
            # Get data rows
            data = []
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                print(f"      Processing {len(rows)} data rows...")
                
                for i, row in enumerate(rows):
                    # Skip sub-header rows
                    if 'thead' in row.get('class', []):
                        continue
                    
                    row_data = {
                        'year': year,
                        'stat_type': stat_type,
                        'rank': i + 1
                    }
                    
                    cells = row.find_all(['th', 'td'])
                    
                    for j, cell in enumerate(cells):
                        if j < len(headers):
                            header = headers[j]
                            value = cell.get_text(strip=True)
                            
                            # Convert numeric values
                            if value and value not in ['', '--', 'N/A']:
                                # Remove commas and try to convert to number
                                clean_value = value.replace(',', '')
                                
                                try:
                                    if '.' in clean_value:
                                        row_data[header] = float(clean_value)
                                    elif clean_value.lstrip('-').isdigit():
                                        row_data[header] = int(clean_value)
                                    else:
                                        row_data[header] = value
                                except:
                                    row_data[header] = value
                            else:
                                row_data[header] = None
                    
                    # Only include rows with a player name
                    if row_data.get('player') or row_data.get('Player'):
                        data.append(row_data)
            
            df = pd.DataFrame(data)
            print(f"      ✅ Parsed {len(df)} player records")
            
            if not df.empty:
                print(f"      Columns: {list(df.columns)}")
                
                # Show sample of key stats
                key_stats = []
                if stat_type == 'passing':
                    key_stats = ['player', 'team', 'pass_yds', 'pass_td', 'pass_att']
                elif stat_type == 'rushing':
                    key_stats = ['player', 'team', 'rush_yds', 'rush_td', 'rush_att']
                elif stat_type == 'receiving':
                    key_stats = ['player', 'team', 'rec_yds', 'rec_td', 'rec']
                
                available_stats = [col for col in key_stats if col in df.columns]
                if available_stats:
                    print(f"      Sample data ({available_stats}):")
                    print(df[available_stats].head(3).to_string(index=False))
            
            return df
            
        except Exception as e:
            print(f"      ❌ Parse error: {e}")
            return pd.DataFrame()
    
    def scrape_player_game_logs(self, year, player_urls, max_players=50):
        """
        Scrape individual player game logs
        This gives us game-by-game performance data
        """
        print(f"\n📋 SCRAPING PLAYER GAME LOGS - {year}")
        print("-" * 50)
        
        all_game_logs = []
        
        for i, (player_name, player_url) in enumerate(player_urls[:max_players]):
            print(f"\n🏃 Player {i+1}/{min(len(player_urls), max_players)}: {player_name}")
            
            # Construct game log URL
            game_log_url = f"https://www.pro-football-reference.com{player_url}/gamelog/{year}/"
            print(f"  URL: {game_log_url}")
            
            try:
                # Try direct request first
                response = self.session.get(game_log_url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for stats table
                    stats_table = soup.find('table', {'id': 'stats'})
                    
                    if stats_table:
                        game_data = self.parse_game_log_table(stats_table, player_name, year)
                        if not game_data.empty:
                            all_game_logs.append(game_data)
                            print(f"    ✅ Got {len(game_data)} games")
                        else:
                            print(f"    ⚠️ No game data parsed")
                    else:
                        print(f"    ❌ No stats table found")
                else:
                    print(f"    ❌ HTTP {response.status_code}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
            
            # Rate limiting
            if i % 10 == 9:
                print(f"    ⏱️ Completed 10 players, waiting 10 seconds...")
                time.sleep(10)
            else:
                time.sleep(2)
        
        # Combine all game logs
        if all_game_logs:
            combined_logs = pd.concat(all_game_logs, ignore_index=True)
            print(f"\n✅ GAME LOGS COMPLETE: {len(combined_logs)} total games")
            return combined_logs
        else:
            print(f"\n❌ NO GAME LOGS RETRIEVED")
            return pd.DataFrame()
    
    def parse_game_log_table(self, table, player_name, year):
        """Parse individual player game log table"""
        try:
            # Get headers
            headers = []
            thead = table.find('thead')
            if thead:
                for th in thead.find_all('th'):
                    data_stat = th.get('data-stat')
                    if data_stat:
                        headers.append(data_stat)
            
            # Get game data
            game_data = []
            tbody = table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    if 'thead' in row.get('class', []):
                        continue
                    
                    row_data = {
                        'player': player_name,
                        'year': year
                    }
                    
                    cells = row.find_all(['th', 'td'])
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            header = headers[i]
                            value = cell.get_text(strip=True)
                            
                            # Convert numeric values
                            if value and value not in ['', '--']:
                                try:
                                    if '.' in value:
                                        row_data[header] = float(value)
                                    elif value.lstrip('-').isdigit():
                                        row_data[header] = int(value)
                                    else:
                                        row_data[header] = value
                                except:
                                    row_data[header] = value
                            else:
                                row_data[header] = None
                    
                    # Only include regular season games
                    if row_data.get('game_date'):
                        game_data.append(row_data)
            
            return pd.DataFrame(game_data)
            
        except Exception as e:
            print(f"    Parse error: {e}")
            return pd.DataFrame()
    
    def scrape_complete_season(self, year):
        """
        Scrape complete season data:
        1. Season statistics for all players
        2. Game logs for top players
        """
        print(f"\n🏈 COMPLETE SEASON SCRAPE: {year}")
        print("=" * 60)
        
        # Step 1: Get season statistics
        season_stats = self.scrape_pro_football_reference(year)
        
        if not season_stats:
            print(f"❌ No season stats for {year} - skipping game logs")
            return {}
        
        # Step 2: Extract top players for game logs
        print(f"\n📋 Extracting top players for game log scraping...")
        
        all_player_urls = []
        
        for stat_type, df in season_stats.items():
            if not df.empty and 'player' in df.columns:
                # Get top 30 players of each type
                top_players = df.head(30)
                
                for _, player in top_players.iterrows():
                    player_name = player.get('player', '')
                    
                    # Try to find player URL (this is tricky with PFR)
                    # For now, we'll use a simplified approach
                    if player_name:
                        # Construct likely URL pattern
                        name_parts = player_name.split()
                        if len(name_parts) >= 2:
                            first = name_parts[0][:2].lower()
                            last = name_parts[-1][:6].lower()
                            player_id = f"{last}{first}00"  # PFR pattern
                            player_url = f"/players/{last[0].upper()}/{player_id}.htm"
                            all_player_urls.append((player_name, player_url))
        
        print(f"Found {len(all_player_urls)} players for game log scraping")
        
        # Step 3: Get game logs (limit to prevent overwhelming)
        game_logs = self.scrape_player_game_logs(year, all_player_urls, max_players=20)
        
        return {
            'season_stats': season_stats,
            'game_logs': game_logs
        }
    
    def scrape_all_years(self, years=[2021, 2022, 2023, 2024]):
        """
        Scrape all requested years
        This is the main function you'll run
        """
        print(f"\n🚀 NFL COMPLETE DATA SCRAPER STARTING")
        print(f"Years to scrape: {years}")
        print(f"Started at: {datetime.now()}")
        print("=" * 80)
        
        all_data = {}
        
        for i, year in enumerate(years):
            print(f"\n\n🏈 YEAR {i+1}/{len(years)}: {year}")
            print("=" * 80)
            
            year_data = self.scrape_complete_season(year)
            
            if year_data:
                all_data[year] = year_data
                
                # Save intermediate results
                self.save_year_data(year, year_data)
                
                print(f"\n✅ {year} COMPLETE!")
                
                # Progress update
                stats_count = sum(len(df) for df in year_data.get('season_stats', {}).values())
                games_count = len(year_data.get('game_logs', pd.DataFrame()))
                print(f"   Season stats: {stats_count} player records")
                print(f"   Game logs: {games_count} games")
                
            else:
                print(f"\n❌ {year} FAILED!")
                all_data[year] = {}
            
            # Wait between years
            if i < len(years) - 1:
                print(f"\n⏱️ Waiting 30 seconds before next year...")
                time.sleep(30)
        
        print(f"\n\n🏆 ALL YEARS COMPLETE!")
        print(f"Finished at: {datetime.now()}")
        
        # Save combined data
        self.save_combined_data(all_data)
        
        return all_data
    
    def save_year_data(self, year, year_data):
        """Save data for a single year"""
        os.makedirs('data/nfl', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save season stats
        if 'season_stats' in year_data:
            for stat_type, df in year_data['season_stats'].items():
                if not df.empty:
                    filename = f"data/nfl/nfl_{stat_type}_{year}_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                    print(f"💾 Saved: {filename}")
        
        # Save game logs
        if 'game_logs' in year_data and not year_data['game_logs'].empty:
            filename = f"data/nfl/nfl_game_logs_{year}_{timestamp}.csv"
            year_data['game_logs'].to_csv(filename, index=False)
            print(f"💾 Saved: {filename}")
    
    def save_combined_data(self, all_data):
        """Save combined multi-year datasets"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Combine season stats by type
        for stat_type in ['passing', 'rushing', 'receiving']:
            combined_stats = []
            
            for year_data in all_data.values():
                if 'season_stats' in year_data and stat_type in year_data['season_stats']:
                    combined_stats.append(year_data['season_stats'][stat_type])
            
            if combined_stats:
                combined_df = pd.concat(combined_stats, ignore_index=True)
                filename = f"data/nfl/nfl_{stat_type}_all_years_{timestamp}.csv"
                combined_df.to_csv(filename, index=False)
                print(f"💾 Combined: {filename} ({len(combined_df)} records)")
        
        # Combine all game logs
        combined_game_logs = []
        for year_data in all_data.values():
            if 'game_logs' in year_data and not year_data['game_logs'].empty:
                combined_game_logs.append(year_data['game_logs'])
        
        if combined_game_logs:
            combined_df = pd.concat(combined_game_logs, ignore_index=True)
            filename = f"data/nfl/nfl_game_logs_all_years_{timestamp}.csv"
            combined_df.to_csv(filename, index=False)
            print(f"💾 Combined: {filename} ({len(combined_df)} games)")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
            print("🧹 Selenium driver closed")

def main():
    """
    MAIN FUNCTION - RUN THIS TO GET ALL NFL DATA
    
    This will take 30-60 minutes but give you comprehensive data
    """
    print("🏈 NFL COMPLETE DATA SCRAPER")
    print("This will scrape REAL NFL data for 2021-2024")
    print("It will take time but give you quality data!")
    print("=" * 60)
    
    scraper = NFLCompleteScraper()
    
    try:
        # Scrape all years: 2021, 2022, 2023, 2024
        all_data = scraper.scrape_all_years([2021, 2022, 2023, 2024])
        
        # Print final summary
        print("\n" + "=" * 80)
        print("🏆 SCRAPING COMPLETE!")
        print("=" * 80)
        
        total_players = 0
        total_games = 0
        
        for year, year_data in all_data.items():
            print(f"\n{year}:")
            
            if 'season_stats' in year_data:
                year_players = sum(len(df) for df in year_data['season_stats'].values())
                total_players += year_players
                print(f"  Season stats: {year_players} player records")
            
            if 'game_logs' in year_data:
                year_games = len(year_data['game_logs'])
                total_games += year_games
                print(f"  Game logs: {year_games} games")
        
        print(f"\n📊 TOTAL SUMMARY:")
        print(f"   Total player-seasons: {total_players}")
        print(f"   Total games: {total_games}")
        print(f"   Data files saved in: data/nfl/")
        
        print(f"\n🎯 Ready to build NFL prop betting models!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Scraping interrupted by user")
        print("Partial data may have been saved to data/nfl/")
        
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()