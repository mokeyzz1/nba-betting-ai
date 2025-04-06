# discord/send_picks_to_friends.py

import pandas as pd
import requests
from datetime import datetime

WEBHOOK_URL = "https://discord.com/api/webhooks/1355435687947669586/bTak4yW7xjaFnUNK2ZbcQP18hBLUc7DQztf0E6C0X3iZQ0_ijdMneitMbq056JtGb6PB"  # Friends Webhook

today = datetime.today().strftime("%Y-%m-%d")
file_path = f"predictions/predictions_{today}_v4_2.csv"

message = f"📅 **{today} — AI Picks**\n\n"

try:
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        prob = float(row["model_win_prob"]) * 100
        message += f"{row['awayteam']} @ {row['hometeam']} → Predicted: **{row['prediction']}** ({prob:.1f}%)\n"
except Exception as e:
    message += f"⚠️ Could not load picks for {today}: {e}"

try:
    requests.post(WEBHOOK_URL, json={"content": message})
    print("✅ Sent picks to Friends!")
except Exception as e:
    print(f"❌ Failed to send: {e}")