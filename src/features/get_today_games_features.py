# src/features/get_today_games_features.py

import pandas as pd
from datetime import datetime
from src.features.get_team_stats import get_team_stats
from src.features.get_recent_stats import get_recent_win_pct, get_recent_avg_pts
from src.features.get_odds import get_odds_dict
from src.utils.team_name_mapping import TEAM_NAME_TO_ODDS_NAME
from src.utils.config import DATA_DIR

def build_features():
    today_str = datetime.today().strftime('%Y-%m-%d')
    odds_path = DATA_DIR / f"nba_odds_{today_str}.csv"

    try:
        odds_df = pd.read_csv(odds_path)
    except FileNotFoundError:
        print(f"❌ Odds file not found for today: {odds_path}")
        return

    rows = []

    for _, row in odds_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_clean = home.lower().strip()
        away_clean = away.lower().strip()

        try:
            home_stats = get_team_stats(home_clean)
            away_stats = get_team_stats(away_clean)

            home_recent_win = get_recent_win_pct(home_clean)
            away_recent_win = get_recent_win_pct(away_clean)
            home_recent_pts = get_recent_avg_pts(home_clean)
            away_recent_pts = get_recent_avg_pts(away_clean)

            home_odds = row["home_odds"]
            away_odds = row["away_odds"]

            off_rating_diff = home_stats["OFF_RATING"] - away_stats["DEF_RATING"]
            def_rating_diff = home_stats["DEF_RATING"] - away_stats["OFF_RATING"]
            recent_win_diff = home_recent_win - away_recent_win
            pace_diff = home_stats["PACE"] - away_stats["PACE"]

            odds_diff = home_odds - away_odds
            implied_home_win_pct = 1 / abs(home_odds) if home_odds != 0 else 0.5
            implied_away_win_pct = 1 / abs(away_odds) if away_odds != 0 else 0.5
            implied_win_diff = implied_home_win_pct - implied_away_win_pct

            row_data = {
                "hometeam": home,
                "awayteam": away,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_OFF_RATING": home_stats["OFF_RATING"],
                "away_OFF_RATING": away_stats["OFF_RATING"],
                "home_DEF_RATING": home_stats["DEF_RATING"],
                "away_DEF_RATING": away_stats["DEF_RATING"],
                "home_EFG_PCT": home_stats["EFG_PCT"],
                "away_EFG_PCT": away_stats["EFG_PCT"],
                "home_PACE": home_stats["PACE"],
                "away_PACE": away_stats["PACE"],
                "home_recent_win_pct": home_recent_win,
                "away_recent_win_pct": away_recent_win,
                "home_recent_avg_pts": home_recent_pts,
                "away_recent_avg_pts": away_recent_pts,
                "off_rating_diff": off_rating_diff,
                "def_rating_diff": def_rating_diff,
                "recent_win_diff": recent_win_diff,
                "pace_diff": pace_diff,
                "odds_diff": odds_diff,
                "implied_home_win_pct": implied_home_win_pct,
                "implied_away_win_pct": implied_away_win_pct,
                "implied_win_diff": implied_win_diff
            }

            rows.append(row_data)

        except Exception as e:
            print(f"⚠️ Skipping game {home} vs {away}: {e}")
            continue

    df = pd.DataFrame(rows)
    output_path = DATA_DIR / f"features_today_{today_str}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Features saved to {output_path}")
