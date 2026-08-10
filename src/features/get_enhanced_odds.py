"""
Enhanced Odds Fetcher - Multiple markets for profitable betting
Supports: Moneyline, Spreads, Totals, Player Props
"""
import requests
import pandas as pd
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from src.utils.config import DATA_DIR
from src.utils.offseason_mode import is_offseason, offseason

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")

if not API_KEY:
    raise ValueError("ODDS_API_KEY environment variable is required")

SPORT = "basketball_nba"
REGION = "us"
BASE_URL = "https://api.the-odds-api.com/v4/sports"

# Market configurations
MARKETS = {
    "h2h": "moneyline",           # Head-to-head (moneyline)
    "spreads": "spreads",         # Point spreads  
    "totals": "totals",           # Over/under totals
    "player_points": "player_points",     # Player points props
    "player_rebounds": "player_rebounds", # Player rebounds props
    "player_assists": "player_assists"    # Player assists props
}

class EnhancedOddsAPI:
    
    def __init__(self):
        self.api_key = API_KEY
        self.cst = ZoneInfo("America/Chicago")
        
    def fetch_all_markets(self, date_str=None):
        """Fetch odds for all available markets"""
        
        if date_str is None:
            date_str = datetime.now(self.cst).strftime('%Y-%m-%d')
        
        print(f"📡 Fetching enhanced NBA odds for {date_str}...")
        
        # Check for offseason
        if is_offseason():
            print("🎭 Offseason mode - generating mock data")
            return self._generate_enhanced_mock_data(date_str)
        
        all_odds = {}
        
        # Fetch each market type
        for market_key, market_name in MARKETS.items():
            print(f"  📊 Fetching {market_name}...")
            
            try:
                odds_data = self._fetch_market(market_key)
                all_odds[market_key] = odds_data
                
                # Rate limiting
                import time
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️ Failed to fetch {market_name}: {e}")
                all_odds[market_key] = []
        
        # Combine and save all markets
        combined_odds = self._combine_markets(all_odds, date_str)
        return combined_odds
    
    def _fetch_market(self, market):
        """Fetch odds for a specific market"""
        
        url = f"{BASE_URL}/{SPORT}/odds"
        params = {
            "regions": REGION,
            "markets": market,
            "apiKey": self.api_key,
            "oddsFormat": "american"
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 429:
            print("⚠️ Rate limit hit. Waiting 60 seconds...")
            import time
            time.sleep(60)
            response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status {response.status_code}")
    
    def _combine_markets(self, all_odds, date_str):
        """Combine all market data into comprehensive dataset"""
        
        games_data = []
        
        # Start with moneyline as base
        h2h_data = all_odds.get('h2h', [])
        
        for game in h2h_data:
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = game['commence_time']
            
            game_info = {
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'game_id': f"{date_str}_{home_team}_{away_team}".replace(" ", "_")
            }
            
            # Add moneyline odds
            self._add_moneyline_odds(game_info, game)
            
            # Add spread odds
            self._add_spread_odds(game_info, all_odds.get('spreads', []), home_team, away_team)
            
            # Add total odds
            self._add_total_odds(game_info, all_odds.get('totals', []), home_team, away_team)
            
            # Add player props (if available)
            self._add_player_props(game_info, all_odds, home_team, away_team)
            
            games_data.append(game_info)
        
        # Save combined data
        df = pd.DataFrame(games_data)
        filename = DATA_DIR / f"enhanced_odds_{date_str}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ Enhanced odds saved to {filename}")
        print(f"📊 Markets: Moneyline, Spreads, Totals + Player Props")
        print(f"🎮 Games: {len(games_data)}")
        
        return df
    
    def _add_moneyline_odds(self, game_info, game_data):
        """Add moneyline odds to game info"""
        
        bookmakers = game_data.get('bookmakers', [])
        if bookmakers:
            outcomes = bookmakers[0]['markets'][0]['outcomes']
            
            for outcome in outcomes:
                if outcome['name'] == game_info['home_team']:
                    game_info['home_ml'] = outcome['price']
                elif outcome['name'] == game_info['away_team']:
                    game_info['away_ml'] = outcome['price']
    
    def _add_spread_odds(self, game_info, spreads_data, home_team, away_team):
        """Add point spread odds"""
        
        for game in spreads_data:
            if game['home_team'] == home_team and game['away_team'] == away_team:
                bookmakers = game.get('bookmakers', [])
                if bookmakers:
                    outcomes = bookmakers[0]['markets'][0]['outcomes']
                    
                    for outcome in outcomes:
                        if outcome['name'] == home_team:
                            game_info['home_spread'] = outcome['point']
                            game_info['home_spread_odds'] = outcome['price']
                        elif outcome['name'] == away_team:
                            game_info['away_spread'] = outcome['point']
                            game_info['away_spread_odds'] = outcome['price']
                break
    
    def _add_total_odds(self, game_info, totals_data, home_team, away_team):
        """Add over/under total odds"""
        
        for game in totals_data:
            if game['home_team'] == home_team and game['away_team'] == away_team:
                bookmakers = game.get('bookmakers', [])
                if bookmakers:
                    outcomes = bookmakers[0]['markets'][0]['outcomes']
                    
                    for outcome in outcomes:
                        if outcome['name'] == 'Over':
                            game_info['total_points'] = outcome['point']
                            game_info['over_odds'] = outcome['price']
                        elif outcome['name'] == 'Under':
                            game_info['under_odds'] = outcome['price']
                break
    
    def _add_player_props(self, game_info, all_odds, home_team, away_team):
        """Add player prop betting data"""
        
        # This would be expanded with actual player prop data
        # For now, add placeholders
        game_info['top_player_points_line'] = None
        game_info['top_player_points_over'] = None
        game_info['top_player_points_under'] = None
    
    def _generate_enhanced_mock_data(self, date_str):
        """Generate comprehensive mock data for testing"""
        
        # Use existing mock odds generator
        base_df = offseason.generate_mock_odds(date_str, num_games=8)
        
        # Enhance with spreads and totals
        enhanced_data = []
        
        for _, row in base_df.iterrows():
            enhanced_game = {
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'commence_time': row['commence_time'],
                'game_id': f"{date_str}_{row['home_team']}_{row['away_team']}".replace(" ", "_"),
                
                # Moneyline
                'home_ml': row['home_odds'],
                'away_ml': row['away_odds'],
                
                # Mock spreads (usually -3.5 to +3.5)
                'home_spread': round(random.uniform(-7.5, 7.5) * 2) / 2,  # Round to 0.5
                'home_spread_odds': random.randint(-115, -105),
                'away_spread_odds': random.randint(-115, -105),
                
                # Mock totals (usually 210-230 for NBA)
                'total_points': round(random.uniform(205, 235) * 2) / 2,
                'over_odds': random.randint(-115, -105),
                'under_odds': random.randint(-115, -105),
                
                # Mock player props
                'top_player_points_line': random.uniform(20, 35),
                'top_player_points_over': random.randint(-120, -100),
                'top_player_points_under': random.randint(-120, -100)
            }
            
            # Calculate away spread
            enhanced_game['away_spread'] = -enhanced_game['home_spread']
            
            enhanced_data.append(enhanced_game)
        
        # Save enhanced mock data
        df = pd.DataFrame(enhanced_data)
        filename = DATA_DIR / f"enhanced_odds_{date_str}.csv"
        df.to_csv(filename, index=False)
        
        print(f"🎭 Generated enhanced mock odds: {len(enhanced_data)} games")
        return df

# Global instance
enhanced_odds_api = EnhancedOddsAPI()

def fetch_enhanced_odds(date_str=None):
    """Main function to fetch all betting markets"""
    return enhanced_odds_api.fetch_all_markets(date_str)

if __name__ == "__main__":
    # Test the enhanced odds fetcher
    import random
    df = fetch_enhanced_odds()
    print("\n📊 Sample enhanced odds:")
    print(df.head())