# discord/send_results_to_friends.py

import pandas as pd
import requests
from datetime import datetime, timedelta

WEBHOOK_URL = "https://discord.com/api/webhooks/1355435687947669586/bTak4yW7xjaFnUNK2ZbcQP18hBLUc7DQztf0E6C0X3iZQ0_ijdMneitMbq056JtGb6PB"  # Friends Webhook

yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
file_path = f"predictions/predictions_{yesterday}_v4_2.csv"

message = f"📊 **{yesterday} — Final AI Results**\n\n"

try:
    df = pd.read_csv(file_path)
    if "actual_winner" not in df.columns:
        raise Exception("Missing actual_winner column.")

    for _, row in df.iterrows():
        result = "✅" if row["prediction"] == row["actual_winner"] else "❌"
        prob = float(row["model_win_prob"]) * 100
        message += f"{row['awayteam']} @ {row['hometeam']} → Predicted: **{row['prediction']}** ({prob:.1f}%) → Final: **{row['actual_winner']}** {result}\n"

except Exception as e:
    message += f"⚠️ Could not load results for {yesterday}: {e}"

try:
    requests.post(WEBHOOK_URL, json={"content": message})
    print("✅ Sent results to Friends!")
except Exception as e:
    print(f"❌ Failed to send: {e}")
