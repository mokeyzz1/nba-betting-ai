import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load your dataset (make sure the path is correct)
df = pd.read_csv("data/your_final_games_data.csv")

# Create balanced win score from recent and season wins
df["home_win_score"] = 0.5 * df["home_recentWins"] + 0.5 * (df["home_seasonWins"] / 82)
df["away_win_score"] = 0.5 * df["away_recentWins"] + 0.5 * (df["away_seasonWins"] / 82)

# Define features
features = [
    'home_win_score', 'away_win_score',
    'home_recentPoints', 'away_recentPoints',
    'E_PACE_home', 'E_DEF_RATING_home', 'E_OFF_RATING_home', 'EFG_PCT_home',
    'E_PACE_away', 'E_DEF_RATING_away', 'E_OFF_RATING_away', 'EFG_PCT_away'
]
target = "home_win"

# Prepare data
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "nba_model_api_blended.pkl")

# Optional: Show feature importance
importances = model.feature_importances_
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")
