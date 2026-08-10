# Enhanced feature building with advanced NBA metrics
import pandas as pd
from datetime import datetime
from src.features.get_team_stats import get_team_stats
from src.features.get_recent_stats import get_recent_win_pct, get_recent_avg_pts
from src.features.get_advanced_features import (
    get_home_away_splits, 
    get_momentum_features, 
    get_clutch_performance,
    get_team_strength_of_schedule
)
from src.features.get_injury_data import get_injury_report, calculate_injury_impact
from src.features.get_odds import get_odds_dict
from src.utils.config import DATA_DIR

def build_enhanced_features():
    """
    Build comprehensive feature set for NBA predictions
    """
    today_str = datetime.today().strftime('%Y-%m-%d')
    odds_path = DATA_DIR / f"nba_odds_{today_str}.csv"

    try:
        odds_df = pd.read_csv(odds_path)
    except FileNotFoundError:
        print(f"❌ Odds file not found for today: {odds_path}")
        return

    print("🔧 Building enhanced features with advanced metrics...")
    
    # Get injury data once for all teams
    injury_data = get_injury_report()
    
    rows = []

    for _, row in odds_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_clean = home.lower().strip()
        away_clean = away.lower().strip()

        try:
            print(f"📊 Processing {away} @ {home}...")
            
            # Basic team stats
            home_stats = get_team_stats(home_clean)
            away_stats = get_team_stats(away_clean)

            # Recent performance
            home_recent_win = get_recent_win_pct(home_clean)
            away_recent_win = get_recent_win_pct(away_clean)
            home_recent_pts = get_recent_avg_pts(home_clean)
            away_recent_pts = get_recent_avg_pts(away_clean)

            # Advanced features
            home_splits = get_home_away_splits(home_clean)
            away_splits = get_home_away_splits(away_clean)
            
            home_momentum = get_momentum_features(home_clean)
            away_momentum = get_momentum_features(away_clean)
            
            home_clutch = get_clutch_performance(home_clean)
            away_clutch = get_clutch_performance(away_clean)
            
            # Strength of schedule
            home_sos = get_team_strength_of_schedule(home_clean)
            away_sos = get_team_strength_of_schedule(away_clean)
            
            # Injury impact
            injury_differential = calculate_injury_impact(home, away, injury_data)

            # Odds data
            home_odds = row["home_odds"]
            away_odds = row["away_odds"]

            # Basic differentials
            off_rating_diff = home_stats["OFF_RATING"] - away_stats["DEF_RATING"]
            def_rating_diff = home_stats["DEF_RATING"] - away_stats["OFF_RATING"]
            recent_win_diff = home_recent_win - away_recent_win
            pace_diff = home_stats["PACE"] - away_stats["PACE"]

            # Odds features
            odds_diff = home_odds - away_odds
            implied_home_win_pct = 1 / abs(home_odds) if home_odds != 0 else 0.5
            implied_away_win_pct = 1 / abs(away_odds) if away_odds != 0 else 0.5
            implied_win_diff = implied_home_win_pct - implied_away_win_pct

            # Advanced differentials
            home_court_advantage = home_splits['home_win_pct'] - away_splits['away_win_pct']
            momentum_diff = home_momentum['momentum_trend'] - away_momentum['momentum_trend']
            wins_last_5_diff = home_momentum['wins_last_5'] - away_momentum['wins_last_5']
            clutch_diff = home_clutch['clutch_win_pct'] - away_clutch['clutch_win_pct']
            sos_diff = home_sos - away_sos

            row_data = {
                # Basic info
                "hometeam": home,
                "awayteam": away,
                "home_odds": home_odds,
                "away_odds": away_odds,
                
                # Basic team stats
                "home_OFF_RATING": home_stats["OFF_RATING"],
                "away_OFF_RATING": away_stats["OFF_RATING"],
                "home_DEF_RATING": home_stats["DEF_RATING"],
                "away_DEF_RATING": away_stats["DEF_RATING"],
                "home_EFG_PCT": home_stats["EFG_PCT"],
                "away_EFG_PCT": away_stats["EFG_PCT"],
                "home_PACE": home_stats["PACE"],
                "away_PACE": away_stats["PACE"],
                
                # Recent performance
                "home_recent_win_pct": home_recent_win,
                "away_recent_win_pct": away_recent_win,
                "home_recent_avg_pts": home_recent_pts,
                "away_recent_avg_pts": away_recent_pts,
                
                # Home/Away splits
                "home_team_home_win_pct": home_splits['home_win_pct'],
                "away_team_away_win_pct": away_splits['away_win_pct'],
                
                # Momentum features
                "home_wins_last_5": home_momentum['wins_last_5'],
                "away_wins_last_5": away_momentum['wins_last_5'],
                "home_avg_margin": home_momentum['avg_point_margin'],
                "away_avg_margin": away_momentum['avg_point_margin'],
                "home_momentum_trend": home_momentum['momentum_trend'],
                "away_momentum_trend": away_momentum['momentum_trend'],
                
                # Clutch performance
                "home_clutch_win_pct": home_clutch['clutch_win_pct'],
                "away_clutch_win_pct": away_clutch['clutch_win_pct'],
                
                # Strength of schedule
                "home_sos": home_sos,
                "away_sos": away_sos,
                
                # Differentials (key features)
                "off_rating_diff": off_rating_diff,
                "def_rating_diff": def_rating_diff,
                "recent_win_diff": recent_win_diff,
                "pace_diff": pace_diff,
                "odds_diff": odds_diff,
                "implied_home_win_pct": implied_home_win_pct,
                "implied_away_win_pct": implied_away_win_pct,
                "implied_win_diff": implied_win_diff,
                "home_court_advantage": home_court_advantage,
                "momentum_diff": momentum_diff,
                "wins_last_5_diff": wins_last_5_diff,
                "clutch_diff": clutch_diff,
                "sos_diff": sos_diff,
                "injury_differential": injury_differential
            }

            rows.append(row_data)

        except Exception as e:
            print(f"⚠️ Skipping game {home} vs {away}: {e}")
            continue

    if not rows:
        print("❌ No valid games processed")
        return

    df = pd.DataFrame(rows)
    output_path = DATA_DIR / f"enhanced_features_{today_str}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Enhanced features saved to {output_path}")
    print(f"📊 Feature count: {len(df.columns)} features for {len(df)} games")
    
    return df

if __name__ == "__main__":
    build_enhanced_features()