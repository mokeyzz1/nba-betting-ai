from shiny import App, ui, render, reactive, Inputs, Outputs, Session
import pandas as pd
from datetime import datetime, timedelta
import os

# === Load YESTERDAY'S predictions for testing ===
test_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
file_path = f"../predictions/predictions_{test_date}_v4_2.csv"

# === Load Data or fallback ===
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df["Win Prob"] = (df["model_win_prob"] * 100).round(1)

    # Add Confidence label
    def confidence_label(p):
        if p >= 70:
            return "High"
        elif p >= 60:
            return "Medium"
        else:
            return "Low"
    df["Confidence"] = df["Win Prob"].apply(confidence_label)

    # Flag top prediction
    top_idx = df["Win Prob"].idxmax()
    df.loc[top_idx, "🏆 Top"] = "🏆"
    df["🏆 Top"] = df["🏆 Top"].fillna("")

    # Rename for display
    df = df.rename(columns={
        "hometeam": "Home",
        "awayteam": "Away",
        "prediction": "AI Pick",
        "home_odds": "Home Odds",
        "away_odds": "Away Odds",
        "value_flag": "Flag"
    })[["🏆 Top", "Home", "Away", "AI Pick", "Win Prob", "Confidence", "Home Odds", "Away Odds", "Flag"]]

else:
    df = pd.DataFrame(columns=["🏆 Top", "Home", "Away", "AI Pick", "Win Prob", "Confidence", "Home Odds", "Away Odds", "Flag"])

# === UI ===
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 20px; }
        h2 { text-align: center; color: #2c3e50; }
        .dataframe { border-collapse: collapse; width: 100%; }
        .dataframe th, .dataframe td { text-align: center; padding: 10px; border: 1px solid #ccc; }
        .dataframe tr:nth-child(even) { background-color: #f2f2f2; }
        .dataframe tr:hover { background-color: #e0f7fa; }
    """),

    ui.h2(f"🏀 NBA AI Daily Predictions (for {test_date})"),
    ui.input_checkbox("show_value_only", "Show Value Bets Only", value=False),
    ui.output_text("status_msg"),
    ui.output_data_frame("predictions_table")
)

# === SERVER ===
def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def filtered_df():
        if df.empty:
            return df
        if input.show_value_only():
            return df[df["Flag"].str.contains("Value", na=False)].sort_values(by="Win Prob", ascending=False)
        return df.sort_values(by="Win Prob", ascending=False)

    @output
    @render.text
    def status_msg():
        if df.empty:
            return "📭 No predictions available for today."
        if input.show_value_only() and filtered_df().empty:
            return "⚠️ No value bets found today."
        return ""

    @output
    @render.data_frame
    def predictions_table():
        return render.DataGrid(filtered_df())

# === APP ===
app = App(app_ui, server)
