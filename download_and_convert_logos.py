import os
import requests
from PIL import Image
from io import BytesIO

# --- CONFIG ---
TEAM_LOGOS = {
    "Atlanta Hawks": "https://loodibee.com/wp-content/uploads/nba-atlanta-hawks-logo.png",
    "Boston Celtics": "https://loodibee.com/wp-content/uploads/nba-boston-celtics-logo.png",
    "Brooklyn Nets": "https://loodibee.com/wp-content/uploads/nba-brooklyn-nets-logo.png",
    "Charlotte Hornets": "https://loodibee.com/wp-content/uploads/nba-charlotte-hornets-logo.png",
    "Chicago Bulls": "https://loodibee.com/wp-content/uploads/nba-chicago-bulls-logo.png",
    "Cleveland Cavaliers": "https://loodibee.com/wp-content/uploads/cleveland-cavaliers-logo-symbol.png",
    "Dallas Mavericks": "https://loodibee.com/wp-content/uploads/dallas-mavericks-logo-symbol.png",
    "Denver Nuggets": "https://loodibee.com/wp-content/uploads/nba-denver-nuggets-logo-2018.png",
    "Detroit Pistons": "https://loodibee.com/wp-content/uploads/nba-detroit-pistons-logo.png",
    "Golden State Warriors": "https://loodibee.com/wp-content/uploads/nba-golden-state-warriors-logo-2020.png",
    "Houston Rockets": "https://loodibee.com/wp-content/uploads/houston-rockets-logo-symbol.png",
    "Indiana Pacers": "https://loodibee.com/wp-content/uploads/indiana-pacers-logo-symbol.png",
    "Los Angeles Clippers": "https://loodibee.com/wp-content/uploads/NBA-LA-Clippers-logo-2024.png",
    "Los Angeles Lakers": "https://loodibee.com/wp-content/uploads/nba-los-angeles-lakers-logo.png",
    "Memphis Grizzlies": "https://loodibee.com/wp-content/uploads/nba-memphis-grizzlies-logo.png",
    "Miami Heat": "https://loodibee.com/wp-content/uploads/nba-miami-heat-logo.png",
    "Milwaukee Bucks": "https://loodibee.com/wp-content/uploads/milwaukee-bucks-logo-symbol.png",
    "Minnesota Timberwolves": "https://loodibee.com/wp-content/uploads/nba-minnesota-timberwolves-logo.png",
    "New Orleans Pelicans": "https://loodibee.com/wp-content/uploads/nba-new-orleans-pelicans-logo.png",
    "New York Knicks": "https://loodibee.com/wp-content/uploads/nba-new-york-knicks-logo.png",
    "Oklahoma City Thunder": "https://loodibee.com/wp-content/uploads/nba-oklahoma-city-thunder-logo.png",
    "Orlando Magic": "https://loodibee.com/wp-content/uploads/orlando-magic-logo-symbol.png",
    "Philadelphia 76ers": "https://loodibee.com/wp-content/uploads/nba-philadelphia-76ers-logo.png",
    "Phoenix Suns": "https://loodibee.com/wp-content/uploads/phoenix-suns-logo-symbol.png",
    "Portland Trail Blazers": "https://loodibee.com/wp-content/uploads/portland-trail-blazers-logo-symbol.png",
    "Sacramento Kings": "https://loodibee.com/wp-content/uploads/nba-sacramento-kings-logo.png",
    "San Antonio Spurs": "https://loodibee.com/wp-content/uploads/san-antonio-spurs-logo-symbol.png",
    "Toronto Raptors": "https://loodibee.com/wp-content/uploads/nba-toronto-raptors-logo-2020.png",
    "Utah Jazz": "https://loodibee.com/wp-content/uploads/nba-utah-jazz-logo.png",
    "Washington Wizards": "https://loodibee.com/wp-content/uploads/nba-washington-wizards-logo.png"
}

output_dir = "static/logos_webp"
os.makedirs(output_dir, exist_ok=True)

for team, url in TEAM_LOGOS.items():
    try:
        print(f"Downloading: {team}")
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        image = image.resize((80, 80))
        image.save(f"{output_dir}/{team}.webp", "WEBP")
    except Exception as e:
        print(f"⚠️ Failed for {team}: {e}")
