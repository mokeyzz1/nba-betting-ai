import requests
import pandas as pd
import os
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from dotenv import load_dotenv
from src.utils.config import DATA_DIR

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")

if not API_KEY:
    raise ValueError("ODDS_API_KEY environment variable is required")
SPORT = "basketball_nba"
REGION = "us"
BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

# Support multiple markets
MARKETS = ["h2h", "spreads", "totals"]

def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))

def fetch_odds(markets=None):
    """
    Enhanced odds fetching supporting multiple markets
    """
    print("📡 Fetching enhanced NBA odds from The Odds API...")
    
    if markets is None:
        markets = MARKETS
    
    # Check for offseason
    from src.utils.offseason_mode import is_offseason
    if is_offseason():
        print("🎭 Offseason mode - generating mock data")
        return _generate_mock_enhanced_odds()

    try:
        cst = ZoneInfo("America/Chicago")
        today_cst = datetime.now(cst).date()
        filename = DATA_DIR / f"nba_odds_{today_cst}.csv"

        # === Safeguard: Skip fetch if today's file already exists ===
        if filename.exists():
            print(f"🛑 Odds already exist for {today_cst} — skipping fetch.")
            return

        all_markets_data = {}
        
        # Fetch each market
        for market in markets:
            print(f"  📊 Fetching {market}...")
            
            params = {
                "regions": REGION,
                "markets": market,
                "apiKey": API_KEY,
                "oddsFormat": "american"
            }
            
            response = requests.get(BASE_URL, params=params, timeout=30)
            
            if response.status_code == 429:
                print("⚠️ Rate limit hit. Waiting 60 seconds...")
                import time
                time.sleep(60)
                response = requests.get(BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                all_markets_data[market] = response.json()
            else:
                print(f"❌ Error fetching {market}:", response.status_code)
                all_markets_data[market] = []
            
            # Rate limiting
            import time
            time.sleep(0.5)
        
        # Combine all markets into enhanced odds
        enhanced_odds = _combine_markets_data(all_markets_data, today_cst)
        
        # Save enhanced odds
        enhanced_odds.to_csv(filename, index=False)
        print(f"\n✅ Enhanced odds for {today_cst} saved to {filename}")
        print(f"📊 Markets: {', '.join(markets)} | Games: {len(enhanced_odds)}")
        
        return enhanced_odds
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching odds: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error in fetch_odds: {e}")
        raise

def _combine_markets_data(all_markets_data, today_cst):
    """Combine different market data into single enhanced dataset"""
    
    h2h_data = all_markets_data.get('h2h', [])
    spreads_data = all_markets_data.get('spreads', [])
    totals_data = all_markets_data.get('totals', [])
    
    enhanced_games = []
    
    for game in h2h_data:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']
        
        game_data = {
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': commence_time
        }
        
        # Add moneyline odds
        _add_moneyline_odds(game_data, game)
        
        # Add spreads
        _add_spread_odds(game_data, spreads_data, home_team, away_team)
        
        # Add totals
        _add_total_odds(game_data, totals_data, home_team, away_team)
        
        enhanced_games.append(game_data)
    
    return pd.DataFrame(enhanced_games)

def _add_moneyline_odds(game_data, h2h_game):
    """Add moneyline odds to game data"""
    bookmakers = h2h_game.get('bookmakers', [])
    if bookmakers:
        outcomes = bookmakers[0]['markets'][0]['outcomes']
        for outcome in outcomes:
            if outcome['name'] == game_data['home_team']:
                game_data['home_odds'] = outcome['price']
            elif outcome['name'] == game_data['away_team']:
                game_data['away_odds'] = outcome['price']

def _add_spread_odds(game_data, spreads_data, home_team, away_team):
    """Add spread odds to game data"""
    for game in spreads_data:
        if game['home_team'] == home_team and game['away_team'] == away_team:
            bookmakers = game.get('bookmakers', [])
            if bookmakers:
                outcomes = bookmakers[0]['markets'][0]['outcomes']
                for outcome in outcomes:
                    if outcome['name'] == home_team:
                        game_data['home_spread'] = outcome.get('point', 0)
                        game_data['home_spread_odds'] = outcome['price']
                    elif outcome['name'] == away_team:
                        game_data['away_spread'] = outcome.get('point', 0)
                        game_data['away_spread_odds'] = outcome['price']
            break

def _add_total_odds(game_data, totals_data, home_team, away_team):
    """Add total odds to game data"""
    for game in totals_data:
        if game['home_team'] == home_team and game['away_team'] == away_team:
            bookmakers = game.get('bookmakers', [])
            if bookmakers:
                outcomes = bookmakers[0]['markets'][0]['outcomes']
                for outcome in outcomes:
                    if outcome['name'] == 'Over':
                        game_data['total_points'] = outcome.get('point', 220)
                        game_data['over_odds'] = outcome['price']
                    elif outcome['name'] == 'Under':
                        game_data['under_odds'] = outcome['price']
            break

def _generate_mock_enhanced_odds():
    """Generate mock odds with multiple markets for offseason testing"""
    from src.utils.offseason_mode import offseason
    import numpy as np
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # Generate base moneyline odds
    base_odds = offseason.generate_mock_odds(today_str, num_games=8)
    
    enhanced_games = []
    for _, game in base_odds.iterrows():
        enhanced_game = {
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'commence_time': game['commence_time'],
            'home_odds': game['home_odds'],
            'away_odds': game['away_odds'],
            'home_spread': round(np.random.uniform(-7.5, 7.5) * 2) / 2,
            'home_spread_odds': -110,
            'away_spread_odds': -110,
            'total_points': round(np.random.uniform(210, 230) * 2) / 2,
            'over_odds': -110,
            'under_odds': -110
        }
        enhanced_game['away_spread'] = -enhanced_game['home_spread']
        enhanced_games.append(enhanced_game)
    
    enhanced_df = pd.DataFrame(enhanced_games)
    
    # Save the mock data
    filename = DATA_DIR / f"nba_odds_{today_str}.csv"
    enhanced_df.to_csv(filename, index=False)
    print(f"🎭 Generated enhanced mock odds: {len(enhanced_df)} games")
    
    return enhanced_df

def get_odds_dict():
    today_str = datetime.now(ZoneInfo("America/Chicago")).strftime('%Y-%m-%d')
    odds_path = DATA_DIR / f"nba_odds_{today_str}.csv"
    df = pd.read_csv(odds_path)

    return {
        (row["home_team"], row["away_team"]): {
            "home_odds": row["home_odds"],
            "away_odds": row["away_odds"]
        }
        for _, row in df.iterrows()
    }
