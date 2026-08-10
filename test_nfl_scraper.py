#!/usr/bin/env python3
"""
Test NFL Scraper - Quick test run
"""

from data_collection.nfl_scraper import NFLDataScraper
import pandas as pd

def test_scraper():
    print("Testing NFL Data Scraper...")
    
    scraper = NFLDataScraper()
    
    # Test with just 2023 data
    print("Scraping 2023 QB data...")
    qb_data = scraper.scrape_quarterback_stats(2023)
    
    print("Scraping 2023 RB data...")  
    rb_data = scraper.scrape_rushing_stats(2023)
    
    print("Scraping 2023 WR data...")
    wr_data = scraper.scrape_receiving_stats(2023)
    
    # Display results
    print(f"\n--- Results ---")
    print(f"QB players found: {len(qb_data)}")
    if not qb_data.empty:
        print("QB columns:", list(qb_data.columns))
        print("Top 5 QBs by passing yards:")
        if 'pass_yds' in qb_data.columns:
            qb_data['pass_yds_clean'] = pd.to_numeric(qb_data['pass_yds'].str.replace(',', ''), errors='coerce')
            top_qbs = qb_data.nlargest(5, 'pass_yds_clean')[['player', 'pass_yds', 'team']]
            print(top_qbs.to_string(index=False))
    
    print(f"\nRB players found: {len(rb_data)}")
    if not rb_data.empty:
        print("RB columns:", list(rb_data.columns))
        print("Top 5 RBs by rushing yards:")
        if 'rush_yds' in rb_data.columns:
            rb_data['rush_yds_clean'] = pd.to_numeric(rb_data['rush_yds'].str.replace(',', ''), errors='coerce')
            top_rbs = rb_data.nlargest(5, 'rush_yds_clean')[['player', 'rush_yds', 'team']]
            print(top_rbs.to_string(index=False))
    
    print(f"\nWR players found: {len(wr_data)}")
    if not wr_data.empty:
        print("WR columns:", list(wr_data.columns))
        print("Top 5 WRs by receiving yards:")
        if 'rec_yds' in wr_data.columns:
            wr_data['rec_yds_clean'] = pd.to_numeric(wr_data['rec_yds'].str.replace(',', ''), errors='coerce')
            top_wrs = wr_data.nlargest(5, 'rec_yds_clean')[['player', 'rec_yds', 'team']]
            print(top_wrs.to_string(index=False))
    
    # Save test data
    if not qb_data.empty:
        qb_data.to_csv('data/nfl/test_qb_2023.csv', index=False)
        print(f"\nSaved test QB data to data/nfl/test_qb_2023.csv")
    
    if not rb_data.empty:
        rb_data.to_csv('data/nfl/test_rb_2023.csv', index=False)
        print(f"Saved test RB data to data/nfl/test_rb_2023.csv")
        
    if not wr_data.empty:
        wr_data.to_csv('data/nfl/test_wr_2023.csv', index=False)
        print(f"Saved test WR data to data/nfl/test_wr_2023.csv")

if __name__ == "__main__":
    test_scraper()