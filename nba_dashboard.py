import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import os

# --- CONFIG ---
st.set_page_config(page_title="NBA BETTING Predictions", layout="wide")

# --- GET CSV DATES & LABELS ---
def get_available_dates():
    files = glob("predictions/predictions_*_v4_2.csv")
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)

    date_map = {}
    for file in files:
        try:
            date_str = file.split("_")[1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if file_date == today:
                label = "TODAY"
            elif file_date == yesterday:
                label = "YESTERDAY"
            else:
                label = file_date.strftime("%A %b %d")
            date_map[label] = date_str
        except:
            continue
    return dict(sorted(date_map.items(), key=lambda x: x[1], reverse=True))

# --- SELECT DATE (TOP RIGHT) ---
date_options = get_available_dates()
if not date_options:
    st.error("No prediction files found.")
    st.stop()

with st.container():
    cols = st.columns([6, 1])  # left side empty, dropdown right aligned
    with cols[1]:
        selected_label = st.selectbox(
            label="",
            options=list(date_options.keys()),
            label_visibility="collapsed"
        )

selected_date = date_options[selected_label]
CSV_PATH = f"predictions/predictions_{selected_date}_v4_2.csv"
ODDS_PATH = f"/Users/mk/nba-betting-ai/data/nba_odds_{selected_date}.csv"

# --- LOAD DATA ---
if not os.path.exists(CSV_PATH):
    st.error(f"No predictions found for {selected_date}")
    st.stop()

df = pd.read_csv(CSV_PATH)
odds_df = pd.read_csv(ODDS_PATH) if os.path.exists(ODDS_PATH) else pd.DataFrame()

# --- FORMAT DATA ---
df["win_prob"] = (df["model_win_prob"] * 100).round(1)
df["edge"] = df["value_gap"].apply(lambda x: f"{x * 100:+.1f}%")

# --- MARKET ODDS LOOKUP ---
def get_combined_odds(home, away):
    if odds_df.empty:
        return "N/A"
    match = odds_df[(odds_df["home_team"] == home) & (odds_df["away_team"] == away)]
    if not match.empty:
        row = match.iloc[0]
        return f"{row['home_odds']} / {row['away_odds']}"
    return "N/A"


# --- ABBREVIATIONS ---
TEAM_ABV = {
    "Milwaukee Bucks": "MIL", "Golden State Warriors": "GS", "Brooklyn Nets": "BKN", "Chicago Bulls": "CHI",
    "Miami Heat": "MIA", "New York Knicks": "NY", "Denver Nuggets": "DEN", "Phoenix Suns": "PHX",
    "Toronto Raptors": "TOR", "Cleveland Cavaliers": "CLE", "Boston Celtics": "BOS", "Oklahoma City Thunder": "OKC",
    "Atlanta Hawks": "ATL", "San Antonio Spurs": "SAS", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "Washington Wizards": "WAS", "Indiana Pacers": "IND", "New Orleans Pelicans": "NOP", "Los Angeles Lakers": "LAL",
    "Utah Jazz": "UTA", "Dallas Mavericks": "DAL", "Houston Rockets": "HOU", "Charlotte Hornets": "CHA",
    "Orlando Magic": "ORL", "Minnesota Timberwolves": "MIN", "Philadelphia 76ers": "PHI", "Detroit Pistons": "DET",
    "Memphis Grizzlies": "MEM", "Los Angeles Clippers": "LAC"

}

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

get_logo = lambda team: TEAM_LOGOS.get(team, "")
get_abv = lambda team: TEAM_ABV.get(team, team[:3].upper())

# You can now use get_logo(team_name) and it will give you the correct loodibee URL for all 30 teams.


# --- STYLE ---
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background-color: #0B132B !important;
    color: white !important;
}

h1 {
    font-size: 26px !important;
}

h1, h3, h4, .cell, .table-header {
    color: white !important;
}
            
h2 {
    color: white !important;
    font-size: 36px !important;  /* Bumped from default */
    font-weight: 800;
}

.table-header {
    font-weight: 700;
    background-color: #1B263B;
    padding: 16px;
    border-bottom: 1px solid #2C3E50;
    font-size: 22px;
}

.row {
    display: flex;
    align-items: center;
    padding: 18px 0;
    border-bottom: 1px solid #2C3E50;
}

.cell {
    flex: 1;
    padding: 0 14px;
    text-align: center;
    font-size: 20px;
}

.cell img {
    height: 60px;
    vertical-align: middle;
    margin: 0 6px;
}

/* Winner column highlight */
.winner-cell {
    background-color: rgba(46, 204, 113, 0.20);
    color: white !important;
    font-weight: bold;
    padding: 12px 0;
    border-radius: 6px;
    font-size: 20px;
}

.edge-positive {
    color: #4CAF50;
    font-weight: bold;
    font-size: 20px;
}

.edge-negative {
    color: #E74C3C;
    font-weight: bold;
    font-size: 20px;
}

/* Dropdown */
[data-baseweb="select"] {
    background-color: #1B263B !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: 1px solid #2C3E50 !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] div {
    color: #1B263B !important;
}
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.markdown("""
<style>
.ai-powered-wrapper {
    display: flex;
    gap: 3px;
    margin-top: 4px;
    margin-bottom: -8px;
    padding-left: 2px;
}

.letter {
    font-size: 16px;
    font-weight: 400;
    color: #9aa0a6;
    opacity: 0.4;
    animation: pulseLetter 2s ease-in-out infinite;
    letter-spacing: 1px;
}

.letter:nth-child(1) { animation-delay: 0s; }
.letter:nth-child(2) { animation-delay: 0.1s; }
.letter:nth-child(3) { animation-delay: 0.2s; }
.letter:nth-child(4) { animation-delay: 0.3s; }
.letter:nth-child(5) { animation-delay: 0.4s; }
.letter:nth-child(6) { animation-delay: 0.5s; }
.letter:nth-child(7) { animation-delay: 0.6s; }
.letter:nth-child(8) { animation-delay: 0.7s; }
.letter:nth-child(9) { animation-delay: 0.8s; }
.letter:nth-child(10) { animation-delay: 0.9s; }

@keyframes pulseLetter {
    0%, 100% {
        transform: scale(1);
        opacity: 0.4;
        font-weight: 400;
        color: #9aa0a6;
    }
    50% {
        transform: scale(1.2);
        opacity: 1;
        font-weight: 600;
        color: #ffffff;
    }
}
</style>

<div class="ai-powered-wrapper">
    <span class="letter">A</span>
    <span class="letter">I</span>
    <span class="letter">-</span>
    <span class="letter">P</span>
    <span class="letter">O</span>
    <span class="letter">W</span>
    <span class="letter">E</span>
    <span class="letter">R</span>
    <span class="letter">E</span>
    <span class="letter">D</span>
</div>
""", unsafe_allow_html=True)


# --- /HEADER ---
st.markdown("## NBA BETTING Predictions")

# --- TABLE HEADER ---
st.markdown('<div class="row table-header">'
            '<div class="cell">GAME</div>'
            '<div class="cell">PREDICTED WINNER</div>'
            '<div class="cell">WIN PROB</div>'
            '<div class="cell">ODDS</div>'
            '<div class="cell">EDGE</div>'
            '</div>', unsafe_allow_html=True)

# --- RENDER TABLE ---
for _, row in df.iterrows():
    home = row["hometeam"]
    away = row["awayteam"]
    winner = home if row["prediction"] == "HOME" else away

    away_abv, home_abv = get_abv(away), get_abv(home)
    away_logo = get_logo(away)
    win_prob = f"{row['win_prob']}%"
    edge_class = "edge-positive" if "+" in row["edge"] else "edge-negative"
    odds_display = get_combined_odds(home, away)

    st.markdown(f"""
    <div class="row">
        <div class="cell"><img src="{away_logo}"> {away_abv} <span style="opacity:0.6;">@</span> {home_abv}</div>
        <div class="cell"><div class="winner-cell">{winner}</div></div>
        <div class="cell"><b>{win_prob}</b></div>
        <div class="cell">{odds_display}</div>
        <div class="cell"><span class="{edge_class}">{row['edge']}</span></div>
    </div>
    """, unsafe_allow_html=True)

