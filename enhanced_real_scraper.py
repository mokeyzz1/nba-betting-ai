#!/usr/bin/env python3
"""
Enhanced Real NFL Scraper
Fixed to get actual statistics, not just player names
"""

import requests
import pandas as pd
import json
import time
import re
from datetime import datetime
from bs4 import BeautifulSoup

class EnhancedNFLScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/',
        })
    
    def scrape_espn_stats_detailed(self, year=2023):
        """Scrape ESPN with detailed stat parsing"""
        print(f"🏈 Scraping ESPN detailed stats for {year}...")
        
        base_url = "https://www.espn.com/nfl/stats/player"
        
        # ESPN URLs for each stat type
        urls = {
            'passing': f"{base_url}/_/stat/passing/season/{year}/seasontype/2",
            'rushing': f"{base_url}/_/stat/rushing/season/{year}/seasontype/2",
            'receiving': f"{base_url}/_/stat/receiving/season/{year}/seasontype/2"
        }
        
        all_data = {}
        
        for stat_type, url in urls.items():
            print(f"  Getting {stat_type} stats...")
            
            try:
                response = self.session.get(url, timeout=15)
                print(f"    Status: {response.status_code}")
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for the stats table
                    table = soup.find('table', class_='Table')
                    
                    if table:
                        df = self.parse_espn_detailed_table(table, stat_type, year)
                        if not df.empty:
                            all_data[stat_type] = df
                            print(f"    ✅ Got {len(df)} {stat_type} records with {len(df.columns)} columns")
                        else:
                            print(f"    ⚠️ Table found but no data parsed")
                    else:
                        print(f"    ❌ No table found")
                        # Debug: save the HTML to see what we got
                        with open(f'debug_{stat_type}_{year}.html', 'w') as f:
                            f.write(response.text)
                        print(f"    🔍 Saved debug HTML: debug_{stat_type}_{year}.html")
                
                time.sleep(2)  # Be respectful
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        return all_data
    
    def parse_espn_detailed_table(self, table, stat_type, year):
        """Parse ESPN table with all statistics"""
        try:
            data = []
            
            # Get headers from thead
            thead = table.find('thead')
            headers = []
            
            if thead:
                # ESPN sometimes has multiple header rows
                header_rows = thead.find_all('tr')
                for row in header_rows:
                    row_headers = []
                    for th in row.find_all('th'):
                        text = th.text.strip()
                        if text and text not in ['', ' ']:
                            row_headers.append(text)
                    if row_headers:
                        headers = row_headers
                        break
            
            print(f"    Headers found: {headers[:10]}...")  # Show first 10 headers
            
            # Get data from tbody
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                print(f"    Data rows found: {len(rows)}")
                
                for i, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    
                    if len(cells) >= 2:  # Must have at least player name + 1 stat
                        row_data = {
                            'year': year,
                            'stat_type': stat_type,
                            'rank': i + 1
                        }
                        
                        for j, cell in enumerate(cells):
                            # Get cell text
                            cell_text = cell.text.strip()
                            
                            # First cell is usually player name
                            if j == 0:
                                # Extract player name and team
                                # Format is usually like "Player NameTEAM"
                                match = re.match(r'^(.+?)([A-Z]{2,3})$', cell_text)
                                if match:
                                    player_name = match.group(1).strip()
                                    team = match.group(2)
                                    row_data['player'] = player_name
                                    row_data['team'] = team
                                else:
                                    row_data['player'] = cell_text
                                    row_data['team'] = 'UNK'
                            
                            # Other cells are stats
                            elif j < len(headers):
                                header = headers[j] if j < len(headers) else f'col_{j}'
                                
                                # Clean header name
                                header = header.lower().replace(' ', '_').replace('/', '_')
                                
                                # Convert numeric values
                                if cell_text and cell_text.replace(',', '').replace('.', '').replace('-', '').isdigit():
                                    cell_text = cell_text.replace(',', '')
                                    try:
                                        if '.' in cell_text:
                                            row_data[header] = float(cell_text)
                                        else:
                                            row_data[header] = int(cell_text)
                                    except:
                                        row_data[header] = cell_text
                                else:
                                    row_data[header] = cell_text
                        
                        # Only add if we have a player name
                        if row_data.get('player'):
                            data.append(row_data)
            
            df = pd.DataFrame(data)
            print(f"    Parsed {len(df)} rows with columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            print(f"    ❌ Parse error: {e}")
            return pd.DataFrame()
    
    def scrape_footballdb_stats(self, year=2023):
        """Try FootballDB as alternative source"""
        print(f"🏈 Trying FootballDB for {year}...")
        
        # FootballDB URLs
        urls = {
            'passing': f"https://www.footballdb.com/statistics/nfl/player-stats/passing/{year}",
            'rushing': f"https://www.footballdb.com/statistics/nfl/player-stats/rushing/{year}",
            'receiving': f"https://www.footballdb.com/statistics/nfl/player-stats/receiving/{year}"
        }
        
        all_data = {}
        
        for stat_type, url in urls.items():
            try:
                print(f"  Getting {stat_type}...")
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    table = soup.find('table', class_='statistics')
                    
                    if table:
                        df = self.parse_footballdb_table(table, stat_type, year)
                        if not df.empty:
                            all_data[stat_type] = df
                            print(f"    ✅ Got {len(df)} records")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        return all_data
    
    def parse_footballdb_table(self, table, stat_type, year):
        """Parse FootballDB table"""
        try:
            data = []
            rows = table.find_all('tr')
            
            if len(rows) < 2:
                return pd.DataFrame()
            
            # Get headers
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.text.strip().lower().replace(' ', '_'))
            
            # Get data
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    row_data = {'year': year, 'stat_type': stat_type}
                    
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            value = cell.text.strip()
                            
                            # Convert numbers
                            if value and value.replace(',', '').replace('.', '').replace('-', '').isdigit():
                                value = value.replace(',', '')
                                try:
                                    if '.' in value:
                                        value = float(value)
                                    else:
                                        value = int(value)
                                except:
                                    pass
                            
                            row_data[headers[i]] = value
                    
                    data.append(row_data)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Parse error: {e}")
            return pd.DataFrame()
    
    def scrape_year_comprehensive(self, year):
        """Scrape using multiple sources"""
        print(f"\n🏈 COMPREHENSIVE SCRAPING FOR {year}")
        print("=" * 50)
        
        all_data = {}
        
        # Try ESPN first
        espn_data = self.scrape_espn_stats_detailed(year)
        if espn_data:
            all_data.update(espn_data)
        
        # If ESPN didn't work or incomplete, try FootballDB
        if len(all_data) < 3:
            print(f"\n🔄 Trying FootballDB as backup...")
            footballdb_data = self.scrape_footballdb_stats(year)
            
            for stat_type, df in footballdb_data.items():
                if stat_type not in all_data or all_data[stat_type].empty:
                    all_data[stat_type] = df
        
        return all_data

def main():
    """Test the enhanced scraper"""
    scraper = EnhancedNFLScraper()
    
    # Test with 2023
    data = scraper.scrape_year_comprehensive(2023)
    
    if data:
        print("\n✅ SUCCESS! Real NFL data retrieved:")
        
        for stat_type, df in data.items():
            print(f"\n{stat_type.upper()}: {len(df)} players")
            print(f"Columns: {list(df.columns)}")
            
            if not df.empty:
                print("Sample data:")
                print(df.head(3).to_string(index=False))
                
                # Save the data
                filename = f"data/nfl/enhanced_{stat_type}_2023.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Saved: {filename}")
        
        print(f"\n🎯 READY FOR NFL MODELING!")
    
    else:
        print("❌ All sources failed")

if __name__ == "__main__":
    main()