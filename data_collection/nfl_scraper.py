#!/usr/bin/env python3
"""
NFL Historical Data Scraper
Scrapes player statistics from Pro Football Reference for 2021-2024
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime
import re

class NFLDataScraper:
    def __init__(self):
        self.base_url = "https://www.pro-football-reference.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def scrape_quarterback_stats(self, year):
        """Scrape QB passing statistics for a given year"""
        print(f"Scraping QB stats for {year}...")
        
        url = f"{self.base_url}/years/{year}/passing.htm"
        response = self.session.get(url)
        
        if response.status_code != 200:
            print(f"Failed to fetch QB data for {year}")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'passing'})
        
        if not table:
            print(f"No QB table found for {year}")
            return pd.DataFrame()
            
        # Parse table headers
        headers = []
        header_row = table.find('thead').find('tr')
        for th in header_row.find_all('th'):
            headers.append(th.get('data-stat', th.text.strip()))
        
        # Parse data rows
        data = []
        tbody = table.find('tbody')
        for row in tbody.find_all('tr'):
            if 'thead' in row.get('class', []):
                continue
                
            row_data = {}
            for i, td in enumerate(row.find_all(['th', 'td'])):
                if i < len(headers):
                    row_data[headers[i]] = td.text.strip()
            
            if row_data and row_data.get('player'):
                row_data['year'] = year
                row_data['position'] = 'QB'
                data.append(row_data)
        
        return pd.DataFrame(data)
    
    def scrape_rushing_stats(self, year):
        """Scrape RB rushing statistics for a given year"""
        print(f"Scraping RB stats for {year}...")
        
        url = f"{self.base_url}/years/{year}/rushing.htm"
        response = self.session.get(url)
        
        if response.status_code != 200:
            print(f"Failed to fetch RB data for {year}")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'rushing'})
        
        if not table:
            print(f"No RB table found for {year}")
            return pd.DataFrame()
            
        headers = []
        header_row = table.find('thead').find('tr')
        for th in header_row.find_all('th'):
            headers.append(th.get('data-stat', th.text.strip()))
        
        data = []
        tbody = table.find('tbody')
        for row in tbody.find_all('tr'):
            if 'thead' in row.get('class', []):
                continue
                
            row_data = {}
            for i, td in enumerate(row.find_all(['th', 'td'])):
                if i < len(headers):
                    row_data[headers[i]] = td.text.strip()
            
            if row_data and row_data.get('player'):
                row_data['year'] = year
                row_data['position'] = 'RB'
                data.append(row_data)
        
        return pd.DataFrame(data)
    
    def scrape_receiving_stats(self, year):
        """Scrape WR/TE receiving statistics for a given year"""
        print(f"Scraping WR stats for {year}...")
        
        url = f"{self.base_url}/years/{year}/receiving.htm"
        response = self.session.get(url)
        
        if response.status_code != 200:
            print(f"Failed to fetch WR data for {year}")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'receiving'})
        
        if not table:
            print(f"No WR table found for {year}")
            return pd.DataFrame()
            
        headers = []
        header_row = table.find('thead').find('tr')
        for th in header_row.find_all('th'):
            headers.append(th.get('data-stat', th.text.strip()))
        
        data = []
        tbody = table.find('tbody')
        for row in tbody.find_all('tr'):
            if 'thead' in row.get('class', []):
                continue
                
            row_data = {}
            for i, td in enumerate(row.find_all(['th', 'td'])):
                if i < len(headers):
                    row_data[headers[i]] = td.text.strip()
            
            if row_data and row_data.get('player'):
                row_data['year'] = year
                data.append(row_data)
        
        return pd.DataFrame(data)
    
    def scrape_player_game_logs(self, player_url, year, position):
        """Scrape individual player game logs"""
        url = f"{self.base_url}{player_url}/gamelog/{year}/"
        response = self.session.get(url)
        
        if response.status_code != 200:
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the correct table based on position
        table_id = 'stats' if position == 'QB' else 'stats'
        table = soup.find('table', {'id': table_id})
        
        if not table:
            return pd.DataFrame()
        
        # Parse game log data
        data = []
        tbody = table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                if 'thead' in row.get('class', []):
                    continue
                    
                row_data = {}
                for td in row.find_all(['th', 'td']):
                    stat = td.get('data-stat')
                    if stat:
                        row_data[stat] = td.text.strip()
                
                if row_data and row_data.get('game_date'):
                    row_data['year'] = year
                    row_data['position'] = position
                    data.append(row_data)
        
        return pd.DataFrame(data)
    
    def scrape_all_years(self, years=[2021, 2022, 2023, 2024]):
        """Scrape all data for specified years"""
        all_qb_data = []
        all_rb_data = []
        all_wr_data = []
        
        for year in years:
            print(f"\n--- Scraping {year} Season ---")
            
            # QB data
            qb_df = self.scrape_quarterback_stats(year)
            if not qb_df.empty:
                all_qb_data.append(qb_df)
            
            time.sleep(2)  # Be respectful to the server
            
            # RB data
            rb_df = self.scrape_rushing_stats(year)
            if not rb_df.empty:
                all_rb_data.append(rb_df)
            
            time.sleep(2)
            
            # WR data
            wr_df = self.scrape_receiving_stats(year)
            if not wr_df.empty:
                all_wr_data.append(wr_df)
            
            time.sleep(2)
        
        # Combine all years
        combined_qb = pd.concat(all_qb_data, ignore_index=True) if all_qb_data else pd.DataFrame()
        combined_rb = pd.concat(all_rb_data, ignore_index=True) if all_rb_data else pd.DataFrame()
        combined_wr = pd.concat(all_wr_data, ignore_index=True) if all_wr_data else pd.DataFrame()
        
        return combined_qb, combined_rb, combined_wr
    
    def save_data(self, qb_df, rb_df, wr_df, data_dir='data/nfl'):
        """Save scraped data to CSV files"""
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not qb_df.empty:
            qb_file = f"{data_dir}/nfl_qb_stats_{timestamp}.csv"
            qb_df.to_csv(qb_file, index=False)
            print(f"Saved QB data: {qb_file}")
        
        if not rb_df.empty:
            rb_file = f"{data_dir}/nfl_rb_stats_{timestamp}.csv"
            rb_df.to_csv(rb_file, index=False)
            print(f"Saved RB data: {rb_file}")
        
        if not wr_df.empty:
            wr_file = f"{data_dir}/nfl_wr_stats_{timestamp}.csv"
            wr_df.to_csv(wr_file, index=False)
            print(f"Saved WR data: {wr_file}")

def main():
    """Main function to run the scraper"""
    scraper = NFLDataScraper()
    
    # Scrape 2021-2024 seasons
    years = [2021, 2022, 2023, 2024]
    print(f"Starting NFL data scrape for years: {years}")
    
    qb_data, rb_data, wr_data = scraper.scrape_all_years(years)
    
    # Save the data
    scraper.save_data(qb_data, rb_data, wr_data)
    
    print(f"\n--- Scraping Complete ---")
    print(f"QB records: {len(qb_data)}")
    print(f"RB records: {len(rb_data)}")
    print(f"WR records: {len(wr_data)}")

if __name__ == "__main__":
    main()