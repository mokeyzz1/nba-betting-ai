import os
import pandas as pd

predictions_folder = "predictions"
results = []

# Loop through all prediction files
for filename in os.listdir(predictions_folder):
    if filename.endswith(".csv") and "_v4_2" in filename:
        path = os.path.join(predictions_folder, filename)
        df = pd.read_csv(path)

        if "actual_winner" not in df.columns:
            print(f"❌ Skipping {filename} (no actual_winner)")
            continue

        date = filename.split("_")[1]
        df["won"] = df["prediction"] == df["actual_winner"]

        # ROI logic based on American odds
        def get_payout(row):
            odds = row["home_odds"] if row["prediction"] == "HOME" else row["away_odds"]
            if row["won"]:
                return odds / 100 if odds > 0 else 100 / abs(odds)
            else:
                return -1.0

        df["roi"] = df.apply(get_payout, axis=1)
        total_roi = df["roi"].sum()
        avg_roi = df["roi"].mean()
        accuracy = df["won"].mean()

        num_bets = len(df)
        value_bets = df[df["value_flag"] == "👍 Value Bet"]
        traps = df[df["value_flag"] == "⚠️ Trap"]

        results.append({
            "date": date,
            "num_bets": num_bets,
            "accuracy": round(accuracy, 3),
            "total_roi": round(total_roi, 2),
            "avg_roi": round(avg_roi, 2),
            "value_bets": len(value_bets),
            "value_winrate": round(value_bets["won"].mean(), 3) if len(value_bets) > 0 else None,
            "trap_winrate": round(traps["won"].mean(), 3) if len(traps) > 0 else None,
        })

# Final Results
summary = pd.DataFrame(results).sort_values("date")
print("\n📊 Backtest Summary:")
print(summary.to_string(index=False))

# Optional: Save
summary.to_csv("performance/backtest_summary.csv", index=False)
print("\n✅ Backtest summary saved to performance/backtest_summary.csv")
