import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils.config import PREDICTIONS_DIR, PERFORMANCE_DIR

def evaluate_results(date: str, model_version: str):
    pred_file = PREDICTIONS_DIR / f"predictions_{date}_{model_version}.csv"
    save_eval_file = PERFORMANCE_DIR / f"accuracy_{date}_{model_version}.csv"

    print(f"\n📅 Evaluating predictions for: {date} using model {model_version}")

    try:
        df = pd.read_csv(pred_file)
    except FileNotFoundError:
        print(f"❌ Prediction file not found: {pred_file}")
        return

    if "actual_winner" not in df.columns:
        print("❌ 'actual_winner' column missing — cannot evaluate.")
        return

    y_true = df["actual_winner"]
    y_pred = df["prediction"]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["HOME", "AWAY"])

    print(f"\n✅ Model Accuracy: {acc:.2%}\n")
    print("📊 Classification Report:")
    print(pd.DataFrame(report).transpose())
    print("\n📉 Confusion Matrix:")
    print(pd.DataFrame(conf_matrix, index=["Actual HOME", "Actual AWAY"], columns=["Predicted HOME", "Predicted AWAY"]))

    out_df = pd.DataFrame([{
        "date": date,
        "model": model_version,
        "accuracy": round(acc, 4),
        "home_precision": round(report["HOME"]["precision"], 4),
        "away_precision": round(report["AWAY"]["precision"], 4),
        "home_recall": round(report["HOME"]["recall"], 4),
        "away_recall": round(report["AWAY"]["recall"], 4)
    }])

    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_eval_file, index=False)
    print(f"\n💾 Saved evaluation to {save_eval_file}")
