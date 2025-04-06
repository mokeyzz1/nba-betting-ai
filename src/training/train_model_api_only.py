import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load dataset from the data folder
df = pd.read_csv("data/your_final_games_data.csv")

# Define the target
target = "home_win"

# Define feature set based on API-accessible columns (NO dummy features!)
features = [
    'home_seasonWins', 'away_seasonWins',
    'home_recentWins', 'away_recentWins',
    'home_recentPoints', 'away_recentPoints',
    'E_PACE_home', 'E_DEF_RATING_home', 'E_OFF_RATING_home', 'EFG_PCT_home',
    'E_PACE_away', 'E_DEF_RATING_away', 'E_OFF_RATING_away', 'EFG_PCT_away'
]

# Select features and target
X = df[features]
y = df[target]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model using XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "nba_model_api_only.pkl")

# Show prediction probabilities on test set
y_proba = model.predict_proba(X_test)

# Print first few probabilities
proba_df = pd.DataFrame(y_proba, columns=["Away Win Prob", "Home Win Prob"])
print(proba_df.head())
