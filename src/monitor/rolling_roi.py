import pandas as pd
import os
import re

# === Step 1: Locate valid ROI summary files ===
summary_files = sorted([
    f for f in os.listdir("performance")
    if re.match(r"\d{4}-\d{2}-\d{2}_summary\.csv", f)
])

if not summary_files:
    print("❌ No ROI summary files found.")
    exit()

records = []

# === Step 2: Loop through each summary file ===
for file in summary_files:
    date_str = file.split("_")[0]
    path = os.path.join("performance", file)

    try:
        df = pd.read_csv(path)

        # Use pivot to extract ROI by odds range
        roi_by_range = df.pivot_table(values="avg_roi", index=None, columns="odds_range", aggfunc="first")

        # Extract actual float values using .item()
        heavy = roi_by_range.get("Heavy Fav (<1.83)", pd.Series([None])).item()
        moderate = roi_by_range.get("Moderate (1.83–2.5)", pd.Series([None])).item()
        underdog = roi_by_range.get("Underdog (>2.5)", pd.Series([None])).item()

        total_bets = df["total_bets"].sum()
        avg_roi = df["avg_roi"].mean()
        avg_edge = df["avg_edge"].mean()

        records.append({
            "date": date_str,
            "total_bets": total_bets,
            "avg_roi": avg_roi,
            "avg_edge": avg_edge,
            "heavy_fav_roi": heavy,
            "moderate_roi": moderate,
            "underdog_roi": underdog
        })

    except Exception as e:
        print(f"⚠️ Failed to process {file}: {e}")

# === Step 3: Final rolling DataFrame ===
rolling_df = pd.DataFrame(records)
rolling_df["date"] = pd.to_datetime(rolling_df["date"])
rolling_df = rolling_df.sort_values("date").reset_index(drop=True)

# === Step 4: Save final rolling ROI file ===
os.makedirs("performance", exist_ok=True)
rolling_df.to_csv("performance/rolling_roi.csv", index=False)

# === Display ===
print("✅ Rolling ROI saved to performance/rolling_roi.csv")
print(rolling_df[[
    "date", "avg_roi", "heavy_fav_roi", "moderate_roi", "underdog_roi", "avg_edge"
]].to_string(index=False))
