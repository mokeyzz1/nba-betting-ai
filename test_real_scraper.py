#!/usr/bin/env python3
"""
Test Real NFL Scraper - Just 2023 data
"""

from data_collection.real_nfl_scraper import RealNFLScraper

def test_scraper():
    print("🏈 Testing Real NFL Scraper (2023 only)...")
    
    # Try without Selenium first (faster)
    scraper = RealNFLScraper(use_selenium=False)
    
    # Test just 2023
    data_2023 = scraper.scrape_year(2023)
    
    if data_2023:
        print("\n✅ SUCCESS! Got real data:")
        for stat_type, df in data_2023.items():
            print(f"  {stat_type}: {len(df)} players")
            if not df.empty:
                # Show sample data
                print(f"  Sample {stat_type} data:")
                print(df.head(3).to_string(index=False))
                print()
        
        # Save test data
        scraper.save_data({2023: data_2023})
        
    else:
        print("❌ No data - trying with Selenium...")
        
        # Try with Selenium
        scraper_selenium = RealNFLScraper(use_selenium=True)
        data_2023 = scraper_selenium.scrape_year(2023)
        
        if data_2023:
            print("\n✅ Selenium SUCCESS!")
            scraper_selenium.save_data({2023: data_2023})
        else:
            print("❌ Both methods failed")
        
        scraper_selenium.cleanup()

if __name__ == "__main__":
    test_scraper()