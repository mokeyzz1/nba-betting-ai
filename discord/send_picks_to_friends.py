# discord/send_picks_to_friends.py

import pandas as pd
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# === Load secrets from .env file ===
load_dotenv()
WEBHOOK_URL = os.getenv("FRIENDS_WEBHOOK")

# === Date & file ===
today = datetime.today().strftime("%Y-%m-%d")
file_path = f"predictions/predictions_{today}_v4_2.csv"

# === Build message ===
message = f"📅 **{today} — AI Picks**\n\n"

try:
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        prob = float(row["model_win_prob"]) * 100
        message += f"{row['awayteam']} @ {row['hometeam']} → Predicted: **{row['prediction']}** ({prob:.1f}%)\n"
except Exception as e:
    message += f"⚠️ Could not load picks for {today}: {e}"

# === Send to Discord ===
try:
    requests.post(WEBHOOK_URL, json={"content": message})
    print("✅ Sent picks to Friends!")
except Exception as e:
    print(f"❌ Failed to send: {e}")
