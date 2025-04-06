import pandas as pd
import numpy as np
import os
import re

pred_dir = "predictions"
perf_dir = "performance"
os.makedirs(perf_dir, exist_ok=True)

# === Step 1: Loop through all prediction files ===
prediction_files = sorted([
    f for f in os.listdir(pred_dir)
    if re.match(r"predictions_\d{4}-\d{2}-\d{2}_v4_2\.csv", f)
])

for file in prediction_files:
    date_str = file.split("_")[1]
    pred_path = os.path.join(pred_dir, file)
    summary_path = os.path.join(perf_dir, f"{date_str}_summary.csv")

    # Skip if summary already exists
    if os.path.exists(summary_path):
        print(f"⏩ Skipping {file} (already evaluated)")
        continue

    try:
        df = pd.read_csv(pred_path)

        # Ensure actual_winner exists
        if "actual_winner" not in df.columns:
            print(f"⚠️ Skipping {file} (missing actual_winner)")
            continue

        # === Process ROI ===
        df["predicted_odds"] = df.apply(lambda x: x["home_odds"] if x["prediction"] == "HOME" else x["away_odds"], axis=1)
        df["odds_range"] = df["predicted_odds"].apply(
            lambda o: "Heavy Fav (<1.83)" if o < 1.83 else ("Moderate (1.83–2.5)" if o < 2.5 else "Underdog (>2.5)")
        )
        df["won"] = df["prediction"] == df["actual_winner"]
        df["roi"] = np.where(df["won"], df["predicted_odds"] - 1, -1)
        df["implied_prob"] = 1 / df["predicted_odds"]
        df["edge"] = df["model_win_prob"] - df["implied_prob"]

        # === Create summary ===
        summary = df.groupby("odds_range").agg(
            total_bets=("roi", "count"),
            win_rate=("won", "mean"),
            avg_roi=("roi", "mean"),
            avg_edge=("edge", "mean")
        ).reset_index()

        # Save it
        summary.to_csv(summary_path, index=False)
        print(f"✅ Evaluated {file} → {summary_path}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
