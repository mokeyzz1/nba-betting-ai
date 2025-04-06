import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the final dataset
file_path = "data/your_final_games_data.csv"
games_df = pd.read_csv(file_path)

# Select features (ensure these exist in the dataset)
features = [
    'home_seasonWins', 'away_seasonWins', 'home_pointsFastBreak', 'away_pointsFastBreak',
    'home_pointsInThePaint', 'away_pointsInThePaint', 'home_recentWins', 'away_recentWins',
    'home_recentPoints', 'away_recentPoints', 'PACE', 'DEF_RATING', 'OFF_RATING', 'EFG_PCT'
]

# Target variable
target = "home_win"

# Ensure all features exist in the dataframe
missing_features = [f for f in features if f not in games_df.columns]
if missing_features:
    print(f"❌ Missing features in dataset: {missing_features}")
else:
    print("✅ All features present. Proceeding with training.")

# Prepare dataset
X = games_df[features]
y = games_df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🔥 XGBoost Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "xgboost_nba_model.pkl")
print("✅ Model saved as xgboost_nba_model.pkl")

