from shiny import App, ui, render, reactive, Inputs, Outputs, Session
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import os
from plotly import graph_objects as go

# === Date options ===
today = datetime.today().date()
yesterday = today - timedelta(days=1)
date_options = [today, yesterday] + [today - timedelta(days=i) for i in range(2, 15)]

def label_for_date(d):
    if d == today:
        return "TODAY"
    elif d == yesterday:
        return "YESTERDAY"
    else:
        return d.strftime("%Y-%m-%d")

# === UI ===
app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent / "www" / "styles.css"),

    ui.h2("🏀 AI Betting Prediction"),

    ui.layout_columns(
        ui.card(
            ui.output_plot("gauge_plot"),
            class_="card"
        ),
        ui.card(
            ui.output_ui("featured_matchup"),
            class_="card"
        ),
        col_widths=(6, 6)
    ),

    ui.input_select(
        id="selected_date",
        label="Select Date",
        choices={label_for_date(d): d.strftime("%Y-%m-%d") for d in date_options},
        selected=today.strftime("%Y-%m-%d")
    ),

    ui.input_checkbox("show_value_only", "Show Value Bets Only", value=False),

    ui.output_text("status_msg", class_="status"),

    ui.card(
        ui.output_data_frame("predictions_table"),
        class_="card"
    )
)

# === SERVER ===
def server(input: Inputs, output: Outputs, session: Session):

    @reactive.Calc
    def raw_df():
        date_str = input.selected_date()
        path = f"../predictions/predictions_{date_str}_v4_2.csv"

        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)
        df["Win Prob"] = (df["model_win_prob"] * 100).round(1)

        def colorize_prob(p):
            if p >= 60:
                return f"<span style='color:green;font-weight:bold;'>{p:.1f}%</span>"
            elif p < 50:
                return f"<span style='color:#e74c3c;font-weight:bold;'>{p:.1f}%</span>"
            else:
                return f"<span style='color:#f39c12;font-weight:bold;'>{p:.1f}%</span>"

        df["Win Prob"] = df["Win Prob"].apply(colorize_prob)

        top_idx = df["model_win_prob"].idxmax()
        df.loc[top_idx, "🏆"] = "🏆"
        df["🏆"] = df["🏆"].fillna("")

        return df.rename(columns={
            "hometeam": "Home",
            "awayteam": "Away",
            "prediction": "AI Pick",
            "home_odds": "Home Odds",
            "away_odds": "Away Odds",
            "value_flag": "Flag"
        })[["🏆", "Home", "Away", "AI Pick", "Win Prob", "Home Odds", "Away Odds", "Flag"]]

    @reactive.Calc
    def filtered_df():
        df = raw_df()
        if df.empty:
            return df
        if input.show_value_only():
            return df[df["Flag"].str.contains("Value", na=False)].sort_values(by="Win Prob", ascending=False)
        return df.sort_values(by="Win Prob", ascending=False)

    @output
    @render.text
    def status_msg():
        df = raw_df()
        if df.empty:
            return "📭 No predictions available for this date."
        if input.show_value_only() and filtered_df().empty:
            return "⚠️ No value bets found for this date."
        return ""

    @output
    @render.data_frame
    def predictions_table():
        return render.DataTable(filtered_df(), width="100%", height="auto")

    @output
    @render.ui
    def featured_matchup():
        return ui.div("📣 Featured matchup coming soon...")

    @output
    @render.plot
    def gauge_plot():
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=74,
            number={'suffix': "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4F9DE2"},
                "bgcolor": "darkblue"
            },
            title={'text': "AI PROBABILITY"}
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
        return fig

# === APP ===
app = App(app_ui, server)
