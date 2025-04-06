import os
import pandas as pd

# Load your backtested predictions with actual_winner
folder = "predictions"
all_dfs = []

for file in os.listdir(folder):
    if file.endswith(".csv") and "_v4_2" in file:
        path = os.path.join(folder, file)
        df = pd.read_csv(path)
        if "actual_winner" not in df.columns:
            continue
        df["date"] = file.split("_")[1]
        df["correct"] = df["prediction"] == df["actual_winner"]
        df["implied_prob"] = df.apply(
            lambda x: 100 / (100 + x["home_odds"]) if x["prediction"] == "HOME" and x["home_odds"] > 0
            else -x["home_odds"] / (-x["home_odds"] + 100) if x["prediction"] == "HOME"
            else 100 / (100 + x["away_odds"]) if x["away_odds"] > 0
            else -x["away_odds"] / (-x["away_odds"] + 100),
            axis=1
        )
        all_dfs.append(df)

if not all_dfs:
    print("❌ No valid predictions found.")
    exit()

df = pd.concat(all_dfs, ignore_index=True)

# === Diagnosis ===
accuracy = df["correct"].mean()
market_accuracy = (df["implied_prob"] > 0.5).mean()
market_beat = accuracy - market_accuracy

best_bet = df.groupby("prediction")["correct"].mean().idxmax()
worst_day = df.groupby("date")["correct"].mean().idxmin()
best_day = df.groupby("date")["correct"].mean().idxmax()

print(f"""
📊 Model Performance Diagnosis:
- Accuracy: {accuracy:.1%} 
- Market Accuracy: {market_accuracy:.1%}
- Market Outperformance: {market_beat:.1%}
- Best Prediction Side: {best_bet}
- Best Day: {best_day}
- Worst Day: {worst_day}
""")
