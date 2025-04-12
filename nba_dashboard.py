import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import os
import base64

st.set_page_config(page_title="NBA AI Predictions", layout="wide")

st.markdown("""
<link rel="shortcut icon" href="/static/favicon.ico" />
""", unsafe_allow_html=True)

# --- GET DATES ---
def get_available_dates():
    files = glob("predictions/predictions_*_v4_2.csv")
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    date_map = {}
    for file in files:
        try:
            date_str = file.split("_")[1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            label = "TODAY" if file_date == today else "YESTERDAY" if file_date == yesterday else file_date.strftime("%A %b %d")
            date_map[label] = date_str
        except:
            continue
    return dict(sorted(date_map.items(), key=lambda x: x[1], reverse=True))

date_options = get_available_dates()
if not date_options:
    st.error("No prediction files found.")
    st.stop()

with st.container():
    cols = st.columns([6, 1])
    with cols[1]:
        selected_label = st.selectbox(
            label="Select Date",
            options=list(date_options.keys()),
            label_visibility="collapsed"
        )

selected_date = date_options[selected_label]
CSV_PATH = f"predictions/predictions_{selected_date}_v4_2.csv"
ODDS_PATH = f"data/nba_odds_{selected_date}.csv"

@st.cache_data
def load_predictions(path): return pd.read_csv(path)

@st.cache_data
def load_odds(path): return pd.read_csv(path)

if not os.path.exists(CSV_PATH):
    st.error(f"No predictions found for {selected_date}")
    st.stop()

df = load_predictions(CSV_PATH)
odds_df = load_odds(ODDS_PATH) if os.path.exists(ODDS_PATH) else pd.DataFrame()
df["win_prob"] = (df["model_win_prob"] * 100).round(1)
df["edge"] = df["value_gap"].apply(lambda x: f"{x * 100:+.1f}%")

# --- LOGO ---
def get_logo_base64(team_name):
    path = os.path.join("static", "logos_webp", f"{team_name}.webp")
    if os.path.exists(path):
        with open(path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        return f"data:image/webp;base64,{encoded}"
    return ""

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
get_abv = lambda team: TEAM_ABV.get(team, team[:3].upper())

def get_combined_odds(home, away):
    if odds_df.empty:
        return "N/A"
    match = odds_df[(odds_df["home_team"] == home) & (odds_df["away_team"] == away)]
    if not match.empty:
        row = match.iloc[0]
        return f"{row['home_odds']} / {row['away_odds']}"
    return "N/A"

# --- STYLES ---
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background-color: #0B132B !important;
    color: white !important;
}
h2 {
    font-size: 32px;
    font-weight: 800;
    color: white;
}
.table-header {
    font-weight: 700;
    background-color: #1B263B;
    padding: 16px;
    border-bottom: 1px solid #2C3E50;
    font-size: 20px;
}
.row {
    display: flex;
    align-items: center;
    padding: 18px 0;
    border-bottom: 1px solid #2C3E50;
}
.cell {
    flex: 1;
    padding: 0 12px;
    text-align: center;
    font-size: 18px;
}
.cell img {
    height: 60px;
    width: 60px;
    object-fit: contain;
    border-radius: 6px;
}
.winner-cell {
    background-color: rgba(46, 204, 113, 0.2);
    padding: 12px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 18px;
}
.edge-positive { color: #4CAF50; font-weight: bold; }
.edge-negative { color: #E74C3C; font-weight: bold; }

/* TABLE mode on mobile: keep scrolling layout */
@media only screen and (max-width: 600px) {
    .row {
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    .cell {
        min-width: 120px;
        white-space: nowrap;
        font-size: 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# --- BADGE + TITLE ---
st.markdown("""
<div style="display: inline-block; padding: 4px 10px; border-radius: 12px; background: #1B263B; font-size: 12px; color: #8F9AA6; margin-bottom: 8px;">
    ⚡ AI-POWERED MODEL
</div>
""", unsafe_allow_html=True)

st.markdown("## NBA AI Predictions")

# --- MANUAL TOGGLE ---
view_mode = st.radio(
    label="View Mode",
    options=["Desktop", "Mobile"],
    index=0,
    horizontal=True
)

# --- TABLE HEADER (only in desktop) ---
if view_mode == "Desktop":
    st.markdown('<div class="row table-header">'
                '<div class="cell">GAME</div>'
                '<div class="cell">PREDICTED WINNER</div>'
                '<div class="cell">WIN PROB</div>'
                '<div class="cell">ODDS</div>'
                '<div class="cell">EDGE</div>'
                '</div>', unsafe_allow_html=True)

# --- RENDER ---
for _, row in df.iterrows():
    home = row["hometeam"]
    away = row["awayteam"]
    winner = home if row["prediction"] == "HOME" else away
    away_abv = get_abv(away)
    home_abv = get_abv(home)
    away_logo = get_logo_base64(away)
    odds_display = get_combined_odds(home, away)
    win_prob = f"{row['win_prob']}%"
    edge_class = "edge-positive" if "+" in row["edge"] else "edge-negative"

    if view_mode == "Desktop":
        st.markdown(f"""
        <div class="row">
            <div class="cell">
                <img src="{away_logo}" loading="lazy" width="60" height="60">
                {away_abv} <span style="opacity:0.6;">@</span> {home_abv}
            </div>
            <div class="cell"><div class="winner-cell">{winner}</div></div>
            <div class="cell"><b>{win_prob}</b></div>
            <div class="cell">{odds_display}</div>
            <div class="cell"><span class="{edge_class}">{row['edge']}</span></div>
        </div>
        """, unsafe_allow_html=True)
    elif view_mode == "Mobile":
        st.markdown(f"""
        <div style="background-color:#1B263B; border-radius:12px; padding:14px; margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <img src="{away_logo}" loading="lazy" width="48" height="48">
                <span style="font-size:16px;">{away_abv} <span style="opacity:0.5;">@</span> {home_abv}</span>
            </div>
            <div style="margin-top:10px;"><b>Winner:</b> <span style="background-color:rgba(46,204,113,0.2); padding:6px 10px; border-radius:6px;">{winner}</span></div>
            <div><b>Win Prob:</b> {win_prob}</div>
            <div><b>Odds:</b> {odds_display}</div>
            <div><b>Edge:</b> <span class="{edge_class}">{row['edge']}</span></div>
        </div>
        """, unsafe_allow_html=True)
