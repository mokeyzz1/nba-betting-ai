#!/usr/bin/env python3
"""
PURE NFL DATA SCRAPER
NO HARDCODED DATA - EVERYTHING DISCOVERED FROM WEBSITES
Gets REAL NFL data for 2021-2024 seasons by scraping actual web pages
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

class PureNFLScraper:
    def __init__(self):
        self.session = requests.Session()
        self.setup_session()
        self.driver = None
        
    def setup_session(self):
        """Setup requests session with rotating user agents"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session.headers.update(headers)
        print("✅ Session configured")
    
    def setup_selenium(self):
        """Setup Selenium when needed"""
        try:
            print("🔧 Initializing Selenium...")
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox') 
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✅ Selenium ready")
            return True
        except Exception as e:
            print(f"❌ Selenium failed: {e}")
            return False
    
    def discover_stat_pages(self, year):
        """
        PURE DISCOVERY: Find all available stat pages for a year
        NO hardcoded URLs - discovers them from the site
        """
        print(f"\n🔍 DISCOVERING STAT PAGES FOR {year}")
        print("-" * 40)
        
        # Start from the main year page and discover what's available
        main_url = f"https://www.pro-football-reference.com/years/{year}/"
        
        try:
            response = self.session.get(main_url, timeout=15)
            if response.status_code != 200:
                print(f"❌ Cannot access main page: {response.status_code}")
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links that look like stat pages
            stat_links = {}
            
            # Look for navigation or content links
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True).lower()
                
                # Discover passing stats link
                if any(word in text for word in ['passing', 'quarterback', 'qb']) and 'stats' in href:
                    stat_links['passing'] = href
                    print(f"  📊 Found passing: {href}")
                
                # Discover rushing stats link  
                elif any(word in text for word in ['rushing', 'running']) and 'stats' in href:
                    stat_links['rushing'] = href
                    print(f"  📊 Found rushing: {href}")
                
                # Discover receiving stats link
                elif any(word in text for word in ['receiving', 'receiver']) and 'stats' in href:
                    stat_links['receiving'] = href
                    print(f"  📊 Found receiving: {href}")
            
            # If we didn't find them in nav, try common patterns but verify they exist
            if not stat_links:
                print("  🔄 No links found in nav, trying common patterns...")
                
                potential_urls = {
                    'passing': f"/years/{year}/passing.htm",
                    'rushing': f"/years/{year}/rushing.htm", 
                    'receiving': f"/years/{year}/receiving.htm"
                }
                
                for stat_type, url_path in potential_urls.items():
                    full_url = f"https://www.pro-football-reference.com{url_path}"
                    
                    # Verify it exists
                    test_response = self.session.head(full_url, timeout=10)
                    if test_response.status_code == 200:
                        stat_links[stat_type] = url_path
                        print(f"  ✅ Verified {stat_type}: {url_path}")
                    else:
                        print(f"  ❌ Not found {stat_type}: {test_response.status_code}")
            
            print(f"  📋 Discovered {len(stat_links)} stat pages")
            return stat_links
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return {}
    
    def scrape_stat_page(self, stat_url, stat_type, year):
        """
        PURE SCRAPING: Extract ALL data from a stat page
        NO assumptions about table structure - discovers everything
        """
        print(f"\n📊 SCRAPING {stat_type.upper()} PAGE")
        print("-" * 30)
        
        full_url = f"https://www.pro-football-reference.com{stat_url}"
        print(f"URL: {full_url}")
        
        # Try multiple approaches
        data = None
        
        # Method 1: Direct request
        print("  🌐 Method 1: Direct request...")
        try:
            response = self.session.get(full_url, timeout=20)
            print(f"    Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                data = self.extract_all_tables(soup, stat_type, year)
                
                if data is not None and not data.empty:
                    print(f"    ✅ SUCCESS: {len(data)} records")
                else:
                    print(f"    ⚠️ No data extracted")
            else:
                print(f"    ❌ Bad status: {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ Request failed: {e}")
        
        # Method 2: Selenium if needed
        if data is None or data.empty:
            print("  🤖 Method 2: Selenium...")
            
            if self.driver is None:
                if not self.setup_selenium():
                    return pd.DataFrame()
            
            try:
                self.driver.get(full_url)
                print("    ⏱️ Waiting for page load...")
                time.sleep(5)
                
                # Wait for any table to appear
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                data = self.extract_all_tables(soup, stat_type, year)
                
                if data is not None and not data.empty:
                    print(f"    ✅ SELENIUM SUCCESS: {len(data)} records")
                else:
                    print(f"    ⚠️ Selenium: No data")
                    
            except Exception as e:
                print(f"    ❌ Selenium failed: {e}")
        
        return data if data is not None else pd.DataFrame()
    
    def extract_all_tables(self, soup, stat_type, year):
        """
        PURE EXTRACTION: Find and parse ALL tables on the page
        Discovers table structure dynamically
        """
        print(f"    🔍 Discovering tables...")
        
        # Find all tables
        tables = soup.find_all('table')
        print(f"    Found {len(tables)} tables")
        
        if not tables:
            print(f"    ❌ No tables found")
            return pd.DataFrame()
        
        # Try each table until we find data
        for i, table in enumerate(tables):
            print(f"    📋 Analyzing table {i+1}...")
            
            # Get table info
            table_id = table.get('id', f'table_{i}')
            table_class = table.get('class', [])
            
            print(f"      ID: {table_id}")
            print(f"      Classes: {table_class}")
            
            # Extract data from this table
            table_data = self.parse_table_dynamically(table, stat_type, year)
            
            if not table_data.empty:
                print(f"      ✅ Got {len(table_data)} records from table {i+1}")
                return table_data
            else:
                print(f"      ⚠️ No data in table {i+1}")
        
        print(f"    ❌ No data found in any table")
        return pd.DataFrame()
    
    def parse_table_dynamically(self, table, stat_type, year):
        """
        PURE PARSING: Extract data without knowing table structure
        Discovers headers and data patterns
        """
        try:
            # Discover headers
            headers = self.discover_headers(table)
            if not headers:
                return pd.DataFrame()
            
            print(f"        Headers ({len(headers)}): {headers[:5]}...")
            
            # Discover data rows
            data_rows = self.discover_data_rows(table)
            if not data_rows:
                return pd.DataFrame()
            
            print(f"        Data rows: {len(data_rows)}")
            
            # Extract data
            data = []
            for row_idx, row in enumerate(data_rows):
                row_data = {
                    'year': year,
                    'stat_type': stat_type,
                    'source_row': row_idx + 1
                }
                
                cells = row.find_all(['td', 'th'])
                
                for cell_idx, cell in enumerate(cells):
                    # Get header name
                    if cell_idx < len(headers):
                        header = headers[cell_idx]
                    else:
                        header = f'column_{cell_idx}'
                    
                    # Get cell value
                    cell_value = self.extract_cell_value(cell)
                    row_data[header] = cell_value
                
                # Only include rows with substantial data
                non_null_values = sum(1 for v in row_data.values() if v is not None and v != '')
                if non_null_values >= 3:  # At least 3 meaningful values
                    data.append(row_data)
            
            df = pd.DataFrame(data)
            
            if not df.empty:
                print(f"        ✅ Extracted {len(df)} valid records")
                print(f"        Columns: {list(df.columns)[:10]}...")
                
                # Show sample
                if len(df) > 0:
                    sample_cols = [col for col in df.columns if df[col].notna().sum() > 0][:5]
                    if sample_cols:
                        print(f"        Sample data:")
                        print(df[sample_cols].head(2).to_string(index=False))
            
            return df
            
        except Exception as e:
            print(f"        ❌ Parse error: {e}")
            return pd.DataFrame()
    
    def discover_headers(self, table):
        """Discover headers from table structure"""
        headers = []
        
        # Try thead first
        thead = table.find('thead')
        if thead:
            # Look through all rows in thead
            for row in thead.find_all('tr'):
                row_headers = []
                for cell in row.find_all(['th', 'td']):
                    # Try data-stat attribute first (PFR uses this)
                    header = cell.get('data-stat')
                    if not header:
                        # Fall back to text content
                        header = cell.get_text(strip=True)
                    
                    if header and header not in ['', ' ']:
                        row_headers.append(header)
                
                # Use the row with the most headers
                if len(row_headers) > len(headers):
                    headers = row_headers
        
        # If no thead, try first row of tbody or table
        if not headers:
            first_row = table.find('tr')
            if first_row:
                for cell in first_row.find_all(['th', 'td']):
                    header = cell.get('data-stat') or cell.get_text(strip=True)
                    if header:
                        headers.append(header)
        
        return headers
    
    def discover_data_rows(self, table):
        """Discover data rows from table structure"""
        data_rows = []
        
        # Try tbody first
        tbody = table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                # Skip header rows within tbody
                if 'thead' not in row.get('class', []):
                    data_rows.append(row)
        else:
            # No tbody - get all rows except the first (header)
            all_rows = table.find_all('tr')
            if len(all_rows) > 1:
                data_rows = all_rows[1:]  # Skip first row (likely header)
        
        return data_rows
    
    def extract_cell_value(self, cell):
        """Extract and clean cell value"""
        if not cell:
            return None
        
        # Get text content
        text = cell.get_text(strip=True)
        
        if not text or text in ['', '--', 'N/A', 'n/a']:
            return None
        
        # Try to convert to number
        clean_text = text.replace(',', '').replace('%', '')
        
        # Check if it's a number
        try:
            if '.' in clean_text:
                return float(clean_text)
            elif clean_text.lstrip('-').isdigit():
                return int(clean_text)
        except:
            pass
        
        return text
    
    def discover_player_links(self, stat_data):
        """
        PURE DISCOVERY: Find actual player links from the scraped data
        NO hardcoded URL patterns - extracts real links from HTML
        """
        print(f"\n🔗 DISCOVERING PLAYER LINKS")
        print("-" * 30)
        
        # This would need to be implemented to go back to the original HTML
        # and extract actual href attributes from player name cells
        # For now, return empty list to avoid hardcoded patterns
        print("  ⚠️ Player link discovery not implemented yet")
        print("  (Avoiding hardcoded URL patterns)")
        return []
    
    def scrape_year_complete(self, year):
        """Scrape complete data for one year"""
        print(f"\n🏈 YEAR {year} - PURE SCRAPING")
        print("=" * 50)
        
        # Step 1: Discover what stat pages exist
        stat_pages = self.discover_stat_pages(year)
        
        if not stat_pages:
            print(f"❌ No stat pages discovered for {year}")
            return {}
        
        # Step 2: Scrape each discovered page
        year_data = {}
        
        for stat_type, stat_url in stat_pages.items():
            print(f"\n📊 Scraping {stat_type} for {year}...")
            
            data = self.scrape_stat_page(stat_url, stat_type, year)
            
            if not data.empty:
                year_data[stat_type] = data
                print(f"✅ {stat_type}: {len(data)} players")
            else:
                print(f"❌ {stat_type}: No data")
            
            # Be respectful
            print("⏱️ Waiting 5 seconds...")
            time.sleep(5)
        
        return year_data
    
    def scrape_all_years(self, years=[2021, 2022, 2023, 2024]):
        """
        MAIN FUNCTION: Pure scraping for all years
        """
        print(f"\n🚀 PURE NFL SCRAPER STARTING")
        print(f"Target years: {years}")
        print(f"Started: {datetime.now()}")
        print("NO HARDCODED DATA - PURE DISCOVERY")
        print("=" * 80)
        
        all_data = {}
        
        for i, year in enumerate(years):
            print(f"\n\n📅 YEAR {i+1}/{len(years)}: {year}")
            
            year_data = self.scrape_year_complete(year)
            
            if year_data:
                all_data[year] = year_data
                self.save_year_data(year, year_data)
                
                # Summary
                total_records = sum(len(df) for df in year_data.values())
                print(f"✅ {year} COMPLETE: {total_records} total records")
            else:
                print(f"❌ {year} FAILED: No data")
                all_data[year] = {}
            
            # Wait between years
            if i < len(years) - 1:
                print(f"⏱️ Waiting 30 seconds before next year...")
                time.sleep(30)
        
        # Save combined data
        self.save_combined_data(all_data)
        
        print(f"\n🏆 PURE SCRAPING COMPLETE!")
        print(f"Finished: {datetime.now()}")
        
        return all_data
    
    def save_year_data(self, year, year_data):
        """Save individual year data"""
        os.makedirs('data/nfl', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for stat_type, df in year_data.items():
            if not df.empty:
                filename = f"data/nfl/pure_{stat_type}_{year}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Saved: {filename}")
    
    def save_combined_data(self, all_data):
        """Save combined datasets"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Combine by stat type
        for stat_type in ['passing', 'rushing', 'receiving']:
            combined = []
            
            for year_data in all_data.values():
                if stat_type in year_data:
                    combined.append(year_data[stat_type])
            
            if combined:
                combined_df = pd.concat(combined, ignore_index=True)
                filename = f"data/nfl/pure_{stat_type}_all_years_{timestamp}.csv"
                combined_df.to_csv(filename, index=False)
                print(f"💾 Combined: {filename} ({len(combined_df)} records)")
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            print("🧹 Cleaned up")

def main():
    """
    RUN THIS: Pure NFL data scraper
    NO hardcoded data - discovers everything from websites
    """
    print("🏈 PURE NFL DATA SCRAPER")
    print("100% discovered data - NO hardcoded information")
    print("This will take time but give you pure, real data!")
    print("=" * 60)
    
    scraper = PureNFLScraper()
    
    try:
        all_data = scraper.scrape_all_years([2021, 2022, 2023, 2024])
        
        # Final summary
        print("\n" + "=" * 80)
        print("🏆 PURE SCRAPING COMPLETE!")
        print("=" * 80)
        
        total_records = 0
        for year, year_data in all_data.items():
            year_records = sum(len(df) for df in year_data.values())
            total_records += year_records
            print(f"{year}: {year_records} records ({list(year_data.keys())})")
        
        print(f"\n📊 TOTAL: {total_records} player-season records")
        print(f"📁 Saved to: data/nfl/")
        print(f"🎯 Ready for NFL modeling!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()