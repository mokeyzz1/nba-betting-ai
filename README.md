# 🏀 NBA AI Betting System

A fully automated machine learning pipeline that predicts NBA game outcomes, analyzes betting odds, and delivers smart value betting insights — all powered by real-time API data and deployed with cloud automation.

This project combines advanced sports analytics, betting logic, and seamless delivery into one streamlined system.

---

## 🚀 Key Features

### 🧠 AI-Powered Predictions
- Trained **XGBoost** model (v4.2) with calibrated probabilities
- Predicts win probabilities and winners using:
  - Advanced team stats
  - Recent performance metrics
  - Moneyline betting odds
- Supports edge detection and value bet analysis

### ☁️ Full Automation (Google Cloud VM)
- Runs daily via cron on a GCP Ubuntu VM
- Executes full pipeline:
  - Fetch odds and team stats
  - Build features and predict
  - Evaluate previous games
  - Sync results back to local Mac
- Prediction outputs are versioned and timestamped

### 📡 Data Sources
- **NBA API** – Real-time team stats, recent form, and game results
- **Odds API** – Head-to-head betting markets (moneyline odds)
- **Kaggle Datasets (for training only)**:
  - `games.csv` – Historical NBA game outcomes
  - `nba_historical_odds.csv` – Odds data from 2008–2023
  - Cleaned and merged into `merged_games_with_odds.csv`

### 💰 Value Betting Logic
- Calculates implied win probabilities from odds
- Detects positive expected value (EV) bets using a value gap threshold
- Applies fractional **Kelly staking** (capped at 5%)
- Logs ROI by odds range, win % by edge type, and total value bets hit

### 💬 Discord Automation
- Sends formatted predictions to Discord channels daily
- Messages include:
  - 🏆 Top pick
  - 📈 Win probabilities
  - 💰 Odds and value labels
- Supports separate channels for personal and public results
- Welcome message with disclaimer also supported

### 🌐 Streamlit Dashboard
- Live, interactive dashboard deployed to:
  [https://nbaai.moseskoroma.me](https://nbaai.moseskoroma.me)
- Features:
  - 🏆 Top prediction highlight
  - Team logos and abbreviations
  - Color-coded win probabilities
  - Betting odds + model edge
  - Date picker (today, yesterday, or any previous day)

---

## 🛠️ How It Works

1. **Fetch odds** for today’s games from The Odds API
2. **Pull team stats** + recent performance from NBA API
3. **Build features** (e.g., PACE_DIFF, RECENT_WIN_DIFF, EFG_PCT)
4. **Predict winners** using `nba_model_v4_2.pkl`
5. **Compare to odds** to detect value bets
6. **Send predictions** to Discord
7. **Evaluate yesterday’s picks** and log performance
8. **Sync results** from cloud VM to Mac
   
---

## 🧪 Development Workflow

All new logic is first tested in **Jupyter Notebooks** before being added to the production pipeline. This enables:
- Fast iteration on feature engineering and modeling
- Testing betting strategies (e.g., Kelly, thresholds)
- Debugging API pulls or odds mismatches
- Visualizing prediction behavior over time

Once stable, notebook logic is migrated into the `/src/` pipeline modules.

---

## 🧠 Model Info

**Model version:** `v4.2`  
**Engine:** XGBoost  
**Training Data:**  
- Kaggle: `games.csv` + `nba_historical_odds.csv`
- Final merged and engineered dataset: `merged_games_with_odds.csv`

**Features used:**
- `OFF_RATING_DIFF`, `DEF_RATING_DIFF`, `PACE_DIFF`, `EFG_PCT_DIFF`
- `RECENT_WIN_PCT`, `RECENT_AVG_PTS`
- `HOME_ODDS`, `AWAY_ODDS`, `IMPLIED_PROB`, `VALUE_GAP`
- Matchup-style and market-based metrics

**Output:**
- Calibrated win probabilities (`predict_proba`)
- Predicted winner (HOME or AWAY)
- Edge vs market + value bet signal

---


## 🔁 Model Evolution

| Version      | Description |
|--------------|-------------|
| **v3**       | Baseline model using hand-picked features and cleaned CSV data |
| **v4.1**     | Feature-engineered version with odds-based metrics and risk features |
| **v4.2**     | Final calibrated model with hyperparameter tuning and API/odds integration |
| **API-only** | Trained on NBA API-accessible features for full real-time automation |
| **API-blended** | Combines season and recent performance using NBA API |
| **with-odds** | Experimental model incorporating market odds weighting |

---

## 📈 Performance Tracking

The model was trained on 17,000+ historical NBA games and achieved an overall validation accuracy of 68%.

Daily predictions are evaluated post-game and tracked in a rolling log. As of now, the live rolling accuracy is 75%, showing strong performance in real-world use.

Daily predictions and post-game evaluations are logged to `/performance/`, including:
- Accuracy by date and model version
- ROI by odds range
- Value win rate by edge threshold
- Rolling average performance

### 📊 Example:

| Date       | Accuracy | ROI    | Model |
|------------|----------|--------|-------|
| 2025-03-27 | 75.00%   | +12.4% | v4.2  |
| 2025-03-28 | 85.71%   | +18.2% | v4.2  |
| 2025-03-29 | 71.43%   | +9.1%  | v4.2  |

---

## 📁 Project Structure

```bash
nba-betting-ai/
├── models/                 # Trained model files (.pkl)
├── data/                   # Historical data and raw API pulls
├── predictions/            # Daily prediction CSVs
├── performance/            # ROI, edge, and accuracy logs
├── logs/                   # Cron job outputs and pipeline logs
├── src/
│   ├── features/           # Feature engineering scripts
│   ├── evaluate/           # Evaluation + performance tracking
│   ├── utils/              # Configs and helper functions
├── discord/                # Discord webhook integrations
├── nba_dashboard.py        # Streamlit dashboard app
├── run_pipeline.py         # Main orchestrator script (automated daily)
├── sync_from_vm.sh         # SCP script to sync from VM to Mac
└── requirements.txt

### 📊 Example Prediction Output

| 🏆 | Home     | Away     | Pick  | Win Prob | Odds (H/A)   | Edge           | Value   |
|----|----------|----------|-------|----------|--------------|----------------|---------|
| 🏆 | Celtics  | Lakers   | HOME  | 78.4%    | -150 / +130  | +5.2% Edge     | 👍 Yes  |
|    | Suns     | Warriors | AWAY  | 62.3%    | +110 / -125  | –1.1% vs Mkt   | ❌ No   |

---

### 🧪 Tech Stack

- **Python** (3.10+)
- **Machine Learning:** XGBoost, Scikit-learn  
- **Data:** Pandas, NumPy  
- **API Integration:** `nba_api`, `requests`, The Odds API  
- **Automation:** Google Cloud VM (cron), SCP sync to Mac  
- **Dashboard:** Streamlit  
- **Messaging:** Discord Webhooks

---

### 📌 Roadmap (Q2 2025)

✅ Live dashboard (Streamlit)  
✅ Daily automation (Google Cloud VM)  
✅ ROI + Edge tracking  
✅ Discord delivery  
✅ Calibrated prediction model  

🔜 Coming Soon:

- 🧠 Injury data integration  
- 🔁 Back-to-back game tracking  
- 🧮 Round robin betting simulator  
- 📬 Email/SMS alerts for +EV bets  
- 🔍 Model ensemble + confidence boost  
- 🧪 Backtesting + bankroll simulator  
- ☁️ **Migrate full system to Google Cloud** (training, storage, model serving)
