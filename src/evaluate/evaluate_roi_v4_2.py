import pandas as pd
import numpy as np
import os
import glob

# === Find the most recent predictions file ===
prediction_files = sorted(glob.glob("predictions/predictions_*_v4_2.csv"))
if not prediction_files:
    raise FileNotFoundError("❌ No prediction files found in /predictions")
latest_file = prediction_files[-1]
print(f"📂 Evaluating latest file: {latest_file}")

# === Load Data ===
df = pd.read_csv(latest_file)

# === Check Columns ===
assert "actual_winner" in df.columns, "❌ Add 'actual_winner' column first."

# === Odds Classification ===
def classify_odds_range(odds):
    if odds < 1.83: return "Heavy Fav (<1.83)"
    elif odds < 2.5: return "Moderate (1.83–2.5)"
    else: return "Underdog (>2.5)"

df["predicted_odds"] = df.apply(lambda x: x["home_odds"] if x["prediction"] == "HOME" else x["away_odds"], axis=1)
df["odds_range"] = df["predicted_odds"].apply(classify_odds_range)

# === Win/ROI Calculation ===
df["won"] = df["prediction"] == df["actual_winner"]
df["roi"] = np.where(df["won"], df["predicted_odds"] - 1, -1)

# === Edge Calculation ===
df["implied_prob"] = 1 / df["predicted_odds"]
df["edge"] = df["model_win_prob"] - df["implied_prob"]

# === Summary Stats ===
summary = df.groupby("odds_range").agg(
    total_bets=("roi", "count"),
    win_rate=("won", "mean"),
    avg_roi=("roi", "mean"),
    avg_edge=("edge", "mean")
).reset_index()

# === Save Results ===
# Extract date from filename
date_str = os.path.basename(latest_file).split("_")[1]
os.makedirs("performance", exist_ok=True)
summary_file = f"performance/{date_str}_summary.csv"
summary.to_csv(summary_file, index=False)

print(f"✅ Summary saved to {summary_file}")
print("\n📊 Odds Range Performance:")
print(summary)
