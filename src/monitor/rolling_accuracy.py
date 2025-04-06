import os
import pandas as pd
from datetime import datetime
from src.utils.config import PERFORMANCE_DIR

def update_rolling_accuracy(model_version: str):
    pattern = f"accuracy_{{date}}_{model_version}.csv"

    # Find all matching files
    files = sorted([
        f for f in os.listdir(PERFORMANCE_DIR)
        if f"_{model_version}.csv" in f and f.startswith("accuracy_")
    ])

    if not files:
        print(f"❌ No accuracy files found for model version {model_version}")
        return

    data = []
    for file in files:
        path = PERFORMANCE_DIR / file
        try:
            df = pd.read_csv(path)
            data.append(df)
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")

    if not data:
        print("❌ No valid accuracy data found.")
        return

    full_df = pd.concat(data, ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df = full_df.sort_values('date')
    full_df['rolling_accuracy'] = full_df['accuracy'].rolling(window=5, min_periods=1).mean()

    print("\n📈 Rolling Accuracy Overview:")
    print(full_df[['date', 'model', 'accuracy', 'rolling_accuracy']].to_string(index=False))

    print("\n🔍 Matching file pattern:", f"*_{model_version}.csv")
    print("📁 Files found:", files)

# === CLI Support ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("❌ Usage: python rolling_accuracy.py <model_version>")
        sys.exit()

    model_version = sys.argv[1]
    update_rolling_accuracy(model_version)
