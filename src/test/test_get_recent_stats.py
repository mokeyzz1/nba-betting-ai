from get_recent_stats import get_recent_win_pct_and_points
from nba_api.stats.static import teams

# Pick a team name to test
team_name = "Miami Heat"

# Get the TEAM_ID from the static lookup
team_id = next(team["id"] for team in teams.get_teams() if team["full_name"] == team_name)

# Run the function
win_pct, avg_pts = get_recent_win_pct_and_points(team_id)

# Print the result
print(f"\n📊 Recent Stats for {team_name}")
print(f"Win % (last 5 games): {win_pct}")
print(f"Avg Points (last 5 games): {avg_pts}")
