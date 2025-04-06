# test_team_name_mapping.py
from utils.team_name_mapping import TEAM_NAME_MAPPING

# Sample short names from your data
test_teams = [
    "atlanta", "boston", "brooklyn", "charlotte", "chicago",
    "cleveland", "dallas", "denver", "detroit", "golden state",
    "houston", "indiana", "la clippers", "la lakers", "memphis",
    "miami", "milwaukee", "minnesota", "new jersey", "new orleans",
    "new york", "oklahoma city", "orlando", "philadelphia", "phoenix",
    "portland", "sacramento", "san antonio", "seattle", "toronto",
    "utah", "washington"
]

# Run test
print("🔍 Testing team name mapping...\n")
for short_name in test_teams:
    full_name = TEAM_NAME_MAPPING.get(short_name)
    if full_name:
        print(f"{short_name} ➜ {full_name}")
    else:
        print(f"❌ {short_name} ➜ NOT FOUND")

print("\n✅ Mapping test complete.")


/Users/mk/nba-betting-ai/src/training/test
