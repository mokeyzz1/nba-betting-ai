# run_pipeline.py (at root)

from src.features.get_odds import fetch_odds
from src.features.get_today_games_features import build_features
from src.prediction.predict_today_api_only_1 import run_predictions
from src.evaluate.evaluate_predictions import evaluate_results
from src.pipeline.fetch_actual_winners import fetch_actual_results
from src.monitor.rolling_accuracy import update_rolling_accuracy

from datetime import datetime, timedelta

def main():
    print("🚀 Starting NBA AI Prediction Pipeline...\n")

    # === Define model version and dates ===
    model_version = "v4_2"
    today = datetime.today()
    yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        # Step 1: Fetch odds for today
        fetch_odds()

        # Step 2: Build features using today’s games
        build_features()

        # Step 3: Run predictions
        run_predictions()

        # Step 4: Fetch yesterday’s results (for evaluation)
        print(f"\n📡 Fetching actual results for {yesterday}...")
        fetch_actual_results(date=yesterday, model_version=model_version)

        # Step 5: Evaluate model performance
        print("\n📊 Evaluating predictions...")
        evaluate_results(date=yesterday, model_version=model_version)

        # Step 6: Update rolling metrics (accuracy, ROI, etc.)
        update_rolling_accuracy(model_version=model_version)

        print("\n✅ Pipeline completed successfully.")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()

    