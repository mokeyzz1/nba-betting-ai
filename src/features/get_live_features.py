import pandas as pd
from features.get_odds import get_odds_dict
from get_recent_stats import get_recent_win_pct, get_recent_avg_pts
from utils.team_name_mapping import TEAM_NAME_TO_ODDS_NAME
from get_team_stats import get_team_stats
from nba_api.stats.endpoints import leaguegamefinder
from datetime import datetime

def get_today_games_with_features():
    today_str = datetime.today().strftime("%Y-%m-%d")
    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=today_str,
        date_to_nullable=today_str
    )
    games_df = finder.get_data_frames()[0]
    games = games_df[games_df["SEASON_ID"] >= "22000"]

    if games.empty:
        return pd.DataFrame()

    odds_dict = get_odds_dict()
    rows = []

    for _, row in games.iterrows():
        home = row["HOME_TEAM_NAME"]
        away = row["VISITOR_TEAM_NAME"]
        home_clean = home.lower().strip()
        away_clean = away.lower().strip()

        try:
            home_stats = get_team_stats(home_clean)
            away_stats = get_team_stats(away_clean)

            home_recent_win = get_recent_win_pct(home_clean)
            away_recent_win = get_recent_win_pct(away_clean)
            home_recent_pts = get_recent_avg_pts(home_clean)
            away_recent_pts = get_recent_avg_pts(away_clean)

            home_odds_name = TEAM_NAME_TO_ODDS_NAME.get(home_clean, home_clean)
            away_odds_name = TEAM_NAME_TO_ODDS_NAME.get(away_clean, away_clean)

            home_odds = odds_dict.get(home_odds_name)
            away_odds = odds_dict.get(away_odds_name)

            if None in (home_odds, away_odds, home_stats, away_stats):
                continue

            implied_home = 1 / home_odds
            implied_away = 1 / away_odds
            implied_diff = implied_home - implied_away
            odds_diff = home_odds - away_odds

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
                "PACE": home_stats["PACE"],  # Just use home PACE
                "home_recent_win_pct": home_recent_win,
                "away_recent_win_pct": away_recent_win,
                "home_recent_avg_pts": home_recent_pts,
                "away_recent_avg_pts": away_recent_pts,
                "off_rating_diff": home_stats["OFF_RATING"] - away_stats["DEF_RATING"],
                "def_rating_diff": home_stats["DEF_RATING"] - away_stats["OFF_RATING"],
                "recent_win_diff": home_recent_win - away_recent_win,
                "pace_diff": home_stats["PACE"] - away_stats["PACE"],
                "odds_diff": odds_diff,
                "implied_home_win_pct": implied_home,
                "implied_away_win_pct": implied_away,
                "implied_win_diff": implied_diff
            }

            rows.append(row_data)

        except Exception as e:
            print(f"❌ Error processing {home} vs {away}: {e}")
            continue

    return pd.DataFrame(rows)
