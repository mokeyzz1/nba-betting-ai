#!/usr/bin/env python3
"""
Debug NFL Scraper
"""

import requests
from bs4 import BeautifulSoup

def debug_scraper():
    """Debug the NFL scraper"""
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    # Test QB URL
    url = "https://www.pro-football-reference.com/years/2023/passing.htm"
    print(f"Testing URL: {url}")
    
    response = session.get(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response Length: {len(response.content)}")
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for any tables
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")
        
        for i, table in enumerate(tables):
            table_id = table.get('id', 'no-id')
            print(f"Table {i}: id='{table_id}'")
            
            # If this is the passing table
            if 'passing' in table_id.lower():
                print(f"Found passing table!")
                
                # Check headers
                thead = table.find('thead')
                if thead:
                    headers = []
                    for th in thead.find_all('th'):
                        stat = th.get('data-stat', th.text.strip())
                        headers.append(stat)
                    print(f"Headers: {headers[:10]}...")  # First 10 headers
                
                # Check first few rows
                tbody = table.find('tbody')
                if tbody:
                    rows = tbody.find_all('tr')[:3]  # First 3 rows
                    print(f"Found {len(tbody.find_all('tr'))} total rows")
                    
                    for j, row in enumerate(rows):
                        if 'thead' in row.get('class', []):
                            continue
                        cells = [td.text.strip() for td in row.find_all(['th', 'td'])]
                        print(f"Row {j}: {cells[:5]}...")  # First 5 cells
                
                break
    else:
        print(f"Failed to fetch data. Response headers: {dict(response.headers)}")

if __name__ == "__main__":
    debug_scraper()