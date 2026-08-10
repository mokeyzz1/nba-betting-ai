#!/usr/bin/env python3
"""
REAL NFL Data Scraper
Gets actual NFL player statistics from multiple sources
Uses various techniques to bypass blocks and get real data
"""

import requests
import pandas as pd
import json
import time
import re
from datetime import datetime
from bs4 import BeautifulSoup
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class RealNFLScraper:
    def __init__(self, use_selenium=False):
        self.use_selenium = use_selenium
        self.session = requests.Session()
        
        # Realistic headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        if use_selenium:
            self.setup_selenium()
    
    def setup_selenium(self):
        """Setup Selenium WebDriver for sites that block requests"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✅ Selenium WebDriver initialized")
        except Exception as e:
            print(f"❌ Selenium setup failed: {e}")
            print("Install Chrome browser if not available")
            self.use_selenium = False
    
    def scrape_with_selenium(self, url, wait_selector=None):
        """Scrape using Selenium"""
        if not self.use_selenium:
            return None
            
        try:
            self.driver.get(url)
            
            if wait_selector:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
            else:
                time.sleep(3)
            
            return self.driver.page_source
        except Exception as e:
            print(f"❌ Selenium scraping failed for {url}: {e}")
            return None
    
    def try_pro_football_reference(self, year=2023):
        """Try to get data from Pro Football Reference with different approaches"""
        print(f"🏈 Attempting Pro Football Reference for {year}...")
        
        urls = {
            'passing': f'https://www.pro-football-reference.com/years/{year}/passing.htm',
            'rushing': f'https://www.pro-football-reference.com/years/{year}/rushing.htm', 
            'receiving': f'https://www.pro-football-reference.com/years/{year}/receiving.htm'
        }
        
        all_data = {}
        
        for stat_type, url in urls.items():
            print(f"  Trying {stat_type}...")
            
            # Method 1: Regular requests
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    table = soup.find('table', {'id': stat_type})
                    
                    if table:
                        df = self.parse_pfr_table(table, stat_type, year)
                        if not df.empty:
                            all_data[stat_type] = df
                            print(f"    ✅ Got {len(df)} {stat_type} records")
                            time.sleep(2)  # Be respectful
                            continue
                
                print(f"    ⚠️ Regular request failed: {response.status_code}")
            except Exception as e:
                print(f"    ❌ Regular request error: {e}")
            
            # Method 2: Try with Selenium if available
            if self.use_selenium:
                print(f"    🔄 Trying with Selenium...")
                html = self.scrape_with_selenium(url, 'table')
                
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    table = soup.find('table', {'id': stat_type})
                    
                    if table:
                        df = self.parse_pfr_table(table, stat_type, year)
                        if not df.empty:
                            all_data[stat_type] = df
                            print(f"    ✅ Selenium got {len(df)} {stat_type} records")
                            time.sleep(3)
                            continue
                
                print(f"    ❌ Selenium also failed")
            
            time.sleep(5)  # Wait between attempts
        
        return all_data
    
    def parse_pfr_table(self, table, stat_type, year):
        """Parse Pro Football Reference table"""
        try:
            headers = []
            header_row = table.find('thead')
            if header_row:
                for th in header_row.find_all('th'):
                    stat = th.get('data-stat', th.text.strip())
                    headers.append(stat)
            
            data = []
            tbody = table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    if 'thead' in row.get('class', []):
                        continue
                    
                    row_data = {'year': year, 'stat_type': stat_type}
                    cells = row.find_all(['th', 'td'])
                    
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            value = cell.text.strip()
                            # Clean up numeric values
                            if value and value.replace(',', '').replace('.', '').replace('-', '').isdigit():
                                value = value.replace(',', '')
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            row_data[headers[i]] = value
                    
                    if row_data.get('player'):  # Only add if we have a player name
                        data.append(row_data)
            
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"    ❌ Parse error: {e}")
            return pd.DataFrame()
    
    def try_nfl_api_direct(self, year=2023):
        """Try NFL's own API endpoints"""
        print(f"🏈 Trying NFL.com API for {year}...")
        
        # NFL.com API endpoints (these might work)
        base_url = "https://www.nfl.com/stats"
        
        stat_urls = {
            'passing': f"{base_url}/player-stats/category/passing/season/{year}/REG",
            'rushing': f"{base_url}/player-stats/category/rushing/season/{year}/REG",
            'receiving': f"{base_url}/player-stats/category/receiving/season/{year}/REG"
        }
        
        all_data = {}
        
        for stat_type, url in stat_urls.items():
            try:
                print(f"  Trying {stat_type}...")
                
                if self.use_selenium:
                    html = self.scrape_with_selenium(url, '.nfl-o-table')
                    if html:
                        soup = BeautifulSoup(html, 'html.parser')
                        table = soup.find('table', class_='nfl-o-table')
                        
                        if table:
                            df = self.parse_nfl_table(table, stat_type, year)
                            if not df.empty:
                                all_data[stat_type] = df
                                print(f"    ✅ Got {len(df)} {stat_type} records")
                
                time.sleep(3)
                
            except Exception as e:
                print(f"    ❌ NFL API error: {e}")
        
        return all_data
    
    def parse_nfl_table(self, table, stat_type, year):
        """Parse NFL.com table"""
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
            
            # Get data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= len(headers):
                    row_data = {'year': year, 'stat_type': stat_type}
                    
                    for i, cell in enumerate(cells[:len(headers)]):
                        value = cell.text.strip()
                        # Clean numeric values
                        if value and value.replace(',', '').replace('.', '').replace('-', '').isdigit():
                            value = value.replace(',', '')
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        row_data[headers[i]] = value
                    
                    data.append(row_data)
            
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"    ❌ NFL parse error: {e}")
            return pd.DataFrame()
    
    def try_espn_stats(self, year=2023):
        """Try ESPN stats pages"""
        print(f"🏈 Trying ESPN stats for {year}...")
        
        base_url = "https://www.espn.com/nfl/stats"
        
        stat_urls = {
            'passing': f"{base_url}/player/_/stat/passing/season/{year}/seasontype/2",
            'rushing': f"{base_url}/player/_/stat/rushing/season/{year}/seasontype/2", 
            'receiving': f"{base_url}/player/_/stat/receiving/season/{year}/seasontype/2"
        }
        
        all_data = {}
        
        for stat_type, url in stat_urls.items():
            try:
                print(f"  Trying {stat_type}...")
                
                # Try direct request first
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    table = soup.find('table', class_='Table')
                    
                    if table:
                        df = self.parse_espn_table(table, stat_type, year)
                        if not df.empty:
                            all_data[stat_type] = df
                            print(f"    ✅ Got {len(df)} {stat_type} records")
                            time.sleep(2)
                            continue
                
                # Try with Selenium
                if self.use_selenium:
                    html = self.scrape_with_selenium(url, '.Table')
                    if html:
                        soup = BeautifulSoup(html, 'html.parser')
                        table = soup.find('table', class_='Table')
                        
                        if table:
                            df = self.parse_espn_table(table, stat_type, year)
                            if not df.empty:
                                all_data[stat_type] = df
                                print(f"    ✅ Selenium got {len(df)} {stat_type} records")
                
                time.sleep(3)
                
            except Exception as e:
                print(f"    ❌ ESPN error: {e}")
        
        return all_data
    
    def parse_espn_table(self, table, stat_type, year):
        """Parse ESPN table"""
        try:
            data = []
            
            # Find headers
            thead = table.find('thead')
            headers = []
            if thead:
                for th in thead.find_all('th'):
                    headers.append(th.text.strip().lower().replace(' ', '_'))
            
            # Find data
            tbody = table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= len(headers):
                        row_data = {'year': year, 'stat_type': stat_type}
                        
                        for i, cell in enumerate(cells[:len(headers)]):
                            value = cell.text.strip()
                            # Clean numeric values
                            if value and value.replace(',', '').replace('.', '').replace('-', '').isdigit():
                                value = value.replace(',', '')
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            row_data[headers[i]] = value
                        
                        data.append(row_data)
            
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"    ❌ ESPN parse error: {e}")
            return pd.DataFrame()
    
    def scrape_year(self, year):
        """Scrape all data for a given year using all available methods"""
        print(f"\n🏈 SCRAPING {year} NFL SEASON DATA 🏈")
        print("=" * 50)
        
        all_year_data = {}
        
        # Try each source in order of preference
        sources = [
            ('Pro Football Reference', self.try_pro_football_reference),
            ('ESPN', self.try_espn_stats),
            ('NFL.com', self.try_nfl_api_direct)
        ]
        
        for source_name, scrape_func in sources:
            print(f"\n📊 Trying {source_name}...")
            
            try:
                data = scrape_func(year)
                
                if data:
                    for stat_type, df in data.items():
                        if not df.empty:
                            if stat_type not in all_year_data:
                                all_year_data[stat_type] = df
                                print(f"    ✅ {source_name}: Got {len(df)} {stat_type} records")
                            else:
                                print(f"    ℹ️ {source_name}: Already have {stat_type} data")
                
                # Check if we have all data types
                if len(all_year_data) >= 3:  # passing, rushing, receiving
                    print(f"    🎯 Complete data for {year}!")
                    break
                    
            except Exception as e:
                print(f"    ❌ {source_name} failed: {e}")
            
            time.sleep(5)  # Wait between sources
        
        return all_year_data
    
    def scrape_multiple_years(self, years=[2021, 2022, 2023]):
        """Scrape data for multiple years"""
        print(f"🚀 STARTING COMPREHENSIVE NFL DATA SCRAPE")
        print(f"Years: {years}")
        print(f"Selenium: {'ENABLED' if self.use_selenium else 'DISABLED'}")
        print("=" * 60)
        
        all_data = {}
        
        for year in years:
            year_data = self.scrape_year(year)
            if year_data:
                all_data[year] = year_data
                print(f"\n✅ {year} COMPLETE: {list(year_data.keys())}")
            else:
                print(f"\n❌ {year} FAILED: No data retrieved")
            
            print(f"\n⏱️ Waiting 10 seconds before next year...")
            time.sleep(10)
        
        return all_data
    
    def save_data(self, all_data, output_dir='data/nfl'):
        """Save all scraped data"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save by year and stat type
        for year, year_data in all_data.items():
            for stat_type, df in year_data.items():
                filename = f"{output_dir}/real_{stat_type}_{year}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Saved: {filename} ({len(df)} records)")
        
        # Save combined datasets
        for stat_type in ['passing', 'rushing', 'receiving']:
            combined_dfs = []
            for year_data in all_data.values():
                if stat_type in year_data:
                    combined_dfs.append(year_data[stat_type])
            
            if combined_dfs:
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                filename = f"{output_dir}/real_{stat_type}_all_years_{timestamp}.csv"
                combined_df.to_csv(filename, index=False)
                print(f"💾 Combined: {filename} ({len(combined_df)} records)")
        
        return timestamp
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():
    """Main scraping function"""
    # Try with Selenium first (more likely to work)
    scraper = RealNFLScraper(use_selenium=True)
    
    try:
        # Scrape recent years
        years_to_scrape = [2021, 2022, 2023]
        all_data = scraper.scrape_multiple_years(years_to_scrape)
        
        if all_data:
            # Save all data
            timestamp = scraper.save_data(all_data)
            
            # Print summary
            print("\n" + "=" * 60)
            print("🏆 REAL NFL DATA SCRAPING COMPLETE!")
            print("=" * 60)
            
            total_records = 0
            for year, year_data in all_data.items():
                print(f"\n{year}:")
                for stat_type, df in year_data.items():
                    print(f"  {stat_type}: {len(df)} players")
                    total_records += len(df)
            
            print(f"\n📊 TOTAL RECORDS: {total_records}")
            print(f"📁 Data saved with timestamp: {timestamp}")
            print(f"🎯 Ready for NFL prop betting models!")
        
        else:
            print("\n❌ NO DATA RETRIEVED - All sources failed")
            print("This might be due to:")
            print("1. All sites blocking requests")
            print("2. Selenium not installed properly") 
            print("3. Network issues")
            print("4. Site structure changes")
    
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()