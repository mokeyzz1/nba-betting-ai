# src/features/get_today_games_features.py

import pandas as pd
from datetime import datetime
from src.features.get_team_stats import get_team_stats, get_team_four_factors_differential, get_team_situational_stats
from src.features.get_recent_stats import get_recent_win_pct, get_recent_avg_pts
from src.features.get_odds import fetch_odds
from src.utils.team_name_mapping import TEAM_NAME_TO_ODDS_NAME
from src.utils.config import DATA_DIR

def build_features():
    """Enhanced feature building with multiple markets and advanced analytics"""
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    odds_path = DATA_DIR / f"nba_odds_{today_str}.csv"

    # Try to load existing odds, if not found, fetch them
    try:
        odds_df = pd.read_csv(odds_path)
        print(f"📊 Loaded existing odds: {len(odds_df)} games")
    except FileNotFoundError:
        print(f"📡 Odds file not found, fetching fresh data...")
        odds_df = fetch_odds()
        if odds_df is None or len(odds_df) == 0:
            print(f"❌ No odds data available for {today_str}")
            return None

    print(f"🔧 Building enhanced features for {len(odds_df)} games...")
    rows = []

    for _, row in odds_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_clean = home.lower().strip()
        away_clean = away.lower().strip()

        try:
            # Get basic team stats
            home_stats = get_team_stats(home_clean)
            away_stats = get_team_stats(away_clean)

            # Get recent performance  
            home_recent_win = get_recent_win_pct(home_clean)
            away_recent_win = get_recent_win_pct(away_clean)
            home_recent_pts = get_recent_avg_pts(home_clean)
            away_recent_pts = get_recent_avg_pts(away_clean)

            # Get four factors differentials
            home_four_factors = get_team_four_factors_differential(home_clean)
            away_four_factors = get_team_four_factors_differential(away_clean)

            # Get situational stats
            home_situational = get_team_situational_stats(home_clean)
            away_situational = get_team_situational_stats(away_clean)

            # Market data
            home_odds = row.get("home_odds", -110)
            away_odds = row.get("away_odds", 110)
            home_spread = row.get("home_spread", 0)
            total_points = row.get("total_points", 220)

            # Calculate enhanced differentials
            off_rating_diff = home_stats["OFF_RATING"] - away_stats["DEF_RATING"]
            def_rating_diff = home_stats["DEF_RATING"] - away_stats["OFF_RATING"]
            net_rating_diff = home_stats["NET_RATING"] - away_stats["NET_RATING"]
            recent_win_diff = home_recent_win - away_recent_win
            pace_diff = home_stats["PACE"] - away_stats["PACE"]

            # Four factors advantages
            efg_advantage = home_four_factors["efg_advantage"] - away_four_factors["efg_advantage"]
            tov_advantage = home_four_factors["tov_advantage"] - away_four_factors["tov_advantage"]
            oreb_advantage = home_four_factors["oreb_advantage"] - away_four_factors["oreb_advantage"]
            ft_advantage = home_four_factors["ft_advantage"] - away_four_factors["ft_advantage"]

            # Situational advantages
            home_court_adv = (home_situational["home_off_rating"] - home_stats["OFF_RATING"]) - \
                           (away_situational["away_off_rating"] - away_stats["OFF_RATING"])

            # Market features
            odds_diff = home_odds - away_odds
            implied_home_win_pct = _american_to_prob(home_odds)
            implied_away_win_pct = _american_to_prob(away_odds)
            implied_win_diff = implied_home_win_pct - implied_away_win_pct

            row_data = {
                # Basic info
                "hometeam": home,
                "awayteam": away,
                
                # Market data  
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_spread": home_spread,
                "total_points": total_points,
                
                # Team stats
                "home_OFF_RATING": home_stats["OFF_RATING"],
                "away_OFF_RATING": away_stats["OFF_RATING"],
                "home_DEF_RATING": home_stats["DEF_RATING"],
                "away_DEF_RATING": away_stats["DEF_RATING"],
                "home_NET_RATING": home_stats["NET_RATING"],
                "away_NET_RATING": away_stats["NET_RATING"],
                "home_EFG_PCT": home_stats["EFG_PCT"],
                "away_EFG_PCT": away_stats["EFG_PCT"],
                "home_PACE": home_stats["PACE"],
                "away_PACE": away_stats["PACE"],
                
                # Recent performance
                "home_recent_win_pct": home_recent_win,
                "away_recent_win_pct": away_recent_win,
                "home_recent_avg_pts": home_recent_pts,
                "away_recent_avg_pts": away_recent_pts,
                
                # Enhanced differentials (key features)
                "off_rating_diff": off_rating_diff,
                "def_rating_diff": def_rating_diff,
                "net_rating_diff": net_rating_diff,
                "recent_win_diff": recent_win_diff,
                "pace_diff": pace_diff,
                
                # Four factors advantages
                "efg_advantage": efg_advantage,
                "tov_advantage": tov_advantage, 
                "oreb_advantage": oreb_advantage,
                "ft_advantage": ft_advantage,
                
                # Situational
                "home_court_advantage": home_court_adv,
                
                # Market features
                "odds_diff": odds_diff,
                "implied_home_win_pct": implied_home_win_pct,
                "implied_away_win_pct": implied_away_win_pct,
                "implied_win_diff": implied_win_diff,
                
                # Composite scores
                "four_factors_composite": (efg_advantage + tov_advantage + oreb_advantage + ft_advantage) / 4,
                "overall_advantage": (net_rating_diff * 0.4 + efg_advantage * 0.3 + home_court_adv * 0.3)
            }

            rows.append(row_data)

        except Exception as e:
            print(f"⚠️ Skipping game {home} vs {away}: {e}")
            continue

    if not rows:
        print("❌ No valid games processed")
        return None

    df = pd.DataFrame(rows)
    output_path = DATA_DIR / f"features_today_{today_str}.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Enhanced features saved to {output_path}")
    print(f"📊 Features: {len(df.columns)} | Games: {len(df)}")
    print(f"🎯 Key features: Four factors, situational analysis, multiple markets")
    
    return df

def _american_to_prob(odds):
    """Convert American odds to probability"""
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return -odds / (-odds + 100)
