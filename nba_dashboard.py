import streamlit as st
import pandas as pd
from datetime import datetime
import os

# --- Config ---
st.set_page_config(page_title="NBA AI Predictions", layout="wide")

# --- Define Team Logos Dictionary ---
TEAM_LOGOS = {
    "MIL": "https://loodibee.com/wp-content/uploads/nba-milwaukee-bucks-logo.png",
    "GS": "https://loodibee.com/wp-content/uploads/golden-state-warriors-logo.png",
    "CHI": "https://loodibee.com/wp-content/uploads/chicago-bulls-logo.png",
    "BKN": "https://loodibee.com/wp-content/uploads/brooklyn-nets-logo.png",
    "NY": "https://loodibee.com/wp-content/uploads/new-york-knicks-logo.png",
    "MIA": "https://loodibee.com/wp-content/uploads/miami-heat-logo.png",
    "PHX": "https://loodibee.com/wp-content/uploads/phoenix-suns-logo.png",
    "DEN": "https://loodibee.com/wp-content/uploads/denver-nuggets-logo.png",
    "SAS": "https://loodibee.com/wp-content/uploads/san-antonio-spurs-logo.png",
    "POR": "https://loodibee.com/wp-content/uploads/portland-trail-blazers-logo.png",
    "SAC": "https://loodibee.com/wp-content/uploads/sacramento-kings-logo.png",
    "CLE": "https://loodibee.com/wp-content/uploads/cleveland-cavaliers-logo.png",
    "LAL": "https://loodibee.com/wp-content/uploads/los-angeles-lakers-logo.png",
    "OKC": "https://loodibee.com/wp-content/uploads/oklahoma-city-thunder-logo.png",
    "ATL": "https://loodibee.com/wp-content/uploads/atlanta-hawks-logo.png",
    "BOS": "https://loodibee.com/wp-content/uploads/boston-celtics-logo.png",
    "WAS": "https://loodibee.com/wp-content/uploads/washington-wizards-logo.png",
    "HOU": "https://loodibee.com/wp-content/uploads/houston-rockets-logo.png",
    "TOR": "https://loodibee.com/wp-content/uploads/toronto-raptors-logo.png",
    "ORL": "https://loodibee.com/wp-content/uploads/orlando-magic-logo.png",
    "IND": "https://loodibee.com/wp-content/uploads/indiana-pacers-logo.png",
    "NOP": "https://loodibee.com/wp-content/uploads/new-orleans-pelicans-logo.png",
    "MIN": "https://loodibee.com/wp-content/uploads/minnesota-timberwolves-logo.png",
    "LAC": "https://loodibee.com/wp-content/uploads/los-angeles-clippers-logo.png",
}

# --- Load Prediction Data ---
SELECTED_DATE = st.selectbox("📅 Select Date", options=["2025-04-06", "2025-04-05", "2025-04-04"], index=0)
CSV_PATH = f"predictions/predictions_{SELECTED_DATE}_v4_2.csv"

if not os.path.exists(CSV_PATH):
    st.error(f"No predictions found for {SELECTED_DATE}")
    st.stop()

df = pd.read_csv(CSV_PATH)
df["Win Prob"] = (df["model_win_prob"] * 100).round(1)
df["Edge"] = (df["value_gap"] * 100).round(1)
df["Edge"] = df["Edge"].apply(lambda x: f"{x:+.1f}%")
df["Odds"] = df.get("market_odds", "N/A")
df["AI Odds"] = df.get("ai_line", "N/A")
df["Prediction"] = df["prediction"].str.upper()

# === Custom CSS ===
st.markdown("""
    <style>
        body { background-color: #0e1a2b; }
        .main { background-color: #0e1a2b; color: white; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .header { font-size: 32px; font-weight: bold; color: white; margin-bottom: 30px; }
        .table { width: 100%; border-spacing: 0 10px; }
        .table th { color: #a8b2d1; font-size: 13px; text-align: left; padding-bottom: 10px; }
        .table td { font-size: 16px; padding: 10px 5px; vertical-align: middle; }
        .logo { height: 26px; vertical-align: middle; margin-right: 6px; }
        .edge-pos { color: #4CAF50; font-weight: bold; }
        .edge-neg { color: #E74C3C; font-weight: bold; }
        .game-row:hover { background-color: #1c2b45; }
        .date-header { font-size: 12px; color: #a8b2d1; font-weight: 400; position: absolute; right: 0; top: 0; margin: 20px; }
    </style>
""", unsafe_allow_html=True)

# === Display "Today" or Date ===
today = datetime.today().strftime("%Y-%m-%d")
date_display = "Today" if SELECTED_DATE == today else SELECTED_DATE
st.markdown(f"<div class='date-header'>{date_display}</div>", unsafe_allow_html=True)

# === Header ===
st.markdown(f"<div class='header'>🧠 AI-Powered NBA Betting Predictions</div>", unsafe_allow_html=True)

# === Table Header ===
st.markdown("""
    <table class='table'>
        <thead>
            <tr>
                <th>GAME</th>
                <th>PREDICTED WINNER</th>
                <th>WIN PROBABILITY</th>
                <th>ODDS</th>
                <th>EDGE</th>
            </tr>
        </thead>
        <tbody>
""", unsafe_allow_html=True)

# === Table Rows ===
for _, row in df.iterrows():
    away_team = row["awayteam"]
    home_team = row["hometeam"]
    winner = home_team if row["Prediction"] == "HOME" else away_team
    edge_class = "edge-pos" if "+" in row["Edge"] else "edge-neg"

    st.markdown(f"""
        <tr class="game-row">
            <td>
                <img src="{TEAM_LOGOS.get(away_team, '')}" class="logo"> {away_team}
                <span style="opacity: 0.6;"> at </span>
                {home_team} <img src="{TEAM_LOGOS.get(home_team, '')}" class="logo">
            </td>
            <td><b>{winner}</b></td>
            <td><b>{row["Win Prob"]}%</b></td>
            <td>{row["Odds"]}<br><small>AI {row["AI Odds"]}</small></td>
            <td><span class="{edge_class}">{row["Edge"]}</span></td>
        </tr>
    """, unsafe_allow_html=True)

# === Close Table ===
st.markdown("</tbody></table>", unsafe_allow_html=True)
