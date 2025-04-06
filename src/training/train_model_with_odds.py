# train_model_with_odds.py

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/your_final_games_data.csv")

# --- Feature Engineering: Win Scores ---
df["home_win_score"] = df["home_seasonWins"] * df["home_recentWins"]
df["away_win_score"] = df["away_seasonWins"] * df["away_recentWins"]

# --- Define features including odds ---
features = [
    "home_win_score", "away_win_score",
    "home_recentPoints", "away_recentPoints",
    "E_PACE_home", "E_DEF_RATING_home", "E_OFF_RATING_home", "EFG_PCT_home",
    "E_PACE_away", "E_DEF_RATING_away", "E_OFF_RATING_away", "EFG_PCT_away",
    "home_odds", "away_odds"
]

target = "home_win"

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "nba_model_with_odds.pkl")

# Feature Importance
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance_df)

# Optional: Plot importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance (Model with Betting Odds)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
