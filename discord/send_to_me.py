import pandas as pd
import requests
from datetime import datetime, timedelta

# === Config ===
WEBHOOK_URL = "https://discord.com/api/webhooks/1356827527800164444/CmK3H6HBjxYzm6GM-CSe19ydRi_ZlllvVNnCGHazp8qnwcKX4LznXAwUy87QdxAcEW7G"  # PRIVATE webhook (for you)

# === Dates ===
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
file_path = f"predictions/predictions_{yesterday}_v4_2.csv"

message = f"📊 **{yesterday} — AI Performance Summary**\n\n"

try:
    df = pd.read_csv(file_path)
    if "actual_winner" not in df.columns:
        raise Exception("Missing actual_winner column.")

    # === Calculate accuracy
    df["correct"] = df["prediction"] == df["actual_winner"]
    num_games = len(df)
    num_correct = df["correct"].sum()
    accuracy = num_correct / num_games * 100

    message += f"✅ **Accuracy:** {accuracy:.1f}% ({num_correct}/{num_games})\n\n"

    # === Game-by-game results
    for _, row in df.iterrows():
        prob = float(row["model_win_prob"]) * 100
        result = "✅" if row["prediction"] == row["actual_winner"] else "❌"
        message += f"{row['awayteam']} @ {row['hometeam']} → Predicted: **{row['prediction']}** ({prob:.1f}%) → Final: **{row['actual_winner']}** {result}\n"

except Exception as e:
    message += f"⚠️ Could not load file: {file_path}\nError: {e}"

# === Send to Discord
try:
    requests.post(WEBHOOK_URL, json={"content": message})
    print("✅ Sent AI performance summary to YOU!")
except Exception as e:
    print(f"❌ Failed to send message: {e}")
