import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from src.utils.config import DATA_DIR

API_KEY = "a2323a916e16fc25d64c84ede464c626"
SPORT = "basketball_nba"
REGION = "us"
MARKET = "h2h"
URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

params = {
    "regions": REGION,
    "markets": MARKET,
    "apiKey": API_KEY,
    "oddsFormat": "decimal"
}

def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))

def fetch_odds():
    print("📡 Fetching NBA odds from The Odds API...")

    # === Get today’s CST date and define filename ===
    cst = ZoneInfo("America/Chicago")
    today_cst = datetime.now(cst).date()
    filename = DATA_DIR / f"nba_odds_{today_cst}.csv"

    # === Safeguard: Skip fetch if today's file already exists ===
    if filename.exists():
        print(f"🛑 Odds already exist for {today_cst} — skipping fetch.")
        return

    response = requests.get(URL, params=params)

    if response.status_code == 200:
        data = response.json()
        rows = []

        for game in data:
            home = game['home_team']
            away = game['away_team']
            time = game['commence_time']

            bookmakers = game.get('bookmakers', [])
            if bookmakers:
                odds = bookmakers[0]['markets'][0]['outcomes']
                home_odds_decimal = None
                away_odds_decimal = None

                for outcome in odds:
                    if outcome['name'] == home:
                        home_odds_decimal = outcome['price']
                    elif outcome['name'] == away:
                        away_odds_decimal = outcome['price']

                if home_odds_decimal and away_odds_decimal:
                    home_odds_american = decimal_to_american(home_odds_decimal)
                    away_odds_american = decimal_to_american(away_odds_decimal)

                    rows.append({
                        "home_team": home,
                        "away_team": away,
                        "home_odds": home_odds_american,
                        "away_odds": away_odds_american,
                        "commence_time": time
                    })

        # --- Create DataFrame ---
        df = pd.DataFrame(rows)
        df['commence_time'] = pd.to_datetime(df['commence_time'], utc=True)
        df['commence_time_cst'] = df['commence_time'].dt.tz_convert(cst)

        # --- Filter for Today's Games ---
        df_today = df[df['commence_time_cst'].dt.date == today_cst].reset_index(drop=True)

        # --- Display and Save ---
        print(df_today[['home_team', 'away_team', 'home_odds', 'away_odds', 'commence_time_cst']])
        df_today.to_csv(filename, index=False)
        print(f"\n✅ Filtered odds for {today_cst} (CST) saved to {filename}")

    else:
        print("❌ Error fetching odds:", response.status_code, response.text)

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
