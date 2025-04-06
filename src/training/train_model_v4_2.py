import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("data/final_games_full_features.csv", low_memory=False)
df_raw = df.copy()

# === Fill missing values ===
df["home_recent_win_pct"] = df["home_recent_win_pct"].fillna(0.5)
df["away_recent_win_pct"] = df["away_recent_win_pct"].fillna(0.5)
df["home_recent_avg_pts"] = df["home_recent_avg_pts"].fillna(df["home_recent_avg_pts"].median())
df["away_recent_avg_pts"] = df["away_recent_avg_pts"].fillna(df["away_recent_avg_pts"].median())

# === Drop rows with missing core stats or odds ===
required_features = [
    "home_OFF_RATING", "away_OFF_RATING",
    "home_DEF_RATING", "away_DEF_RATING",
    "home_EFG_PCT", "away_EFG_PCT",
    "home_odds", "away_odds"
]
df = df.dropna(subset=required_features)
print(f"\n✅ Training on {len(df)} rows (from original {len(df_raw)})")

# === Engineered features ===
df["off_rating_diff"] = df["home_OFF_RATING"] - df["away_DEF_RATING"]
df["def_rating_diff"] = df["home_DEF_RATING"] - df["away_OFF_RATING"]
df["recent_win_diff"] = df["home_recent_win_pct"] - df["away_recent_win_pct"]
df["pace_diff"] = df["home_PACE"] - df["away_PACE"]
df["odds_diff"] = df["home_odds"] - df["away_odds"]
df["implied_home_win_pct"] = 1 / df["home_odds"]
df["implied_away_win_pct"] = 1 / df["away_odds"]
df["implied_win_diff"] = df["implied_home_win_pct"] - df["implied_away_win_pct"]

# === Select Features & Target ===
features = [
    "off_rating_diff", "def_rating_diff", "recent_win_diff", "pace_diff",
    "home_recent_avg_pts", "away_recent_avg_pts",
    "home_EFG_PCT", "away_EFG_PCT",
    "home_odds", "away_odds", "odds_diff",
    "implied_home_win_pct", "implied_away_win_pct", "implied_win_diff"
]

X = df[features].apply(pd.to_numeric, errors='coerce').dropna()
y = df["home_win"].loc[X.index]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# === GridSearch for best XGBoost model ===
param_grid = {
    'n_estimators': [100, 250],
    'max_depth': [4, 5],
    'learning_rate': [0.05, 0.08],
    'subsample': [0.8, 1.0]
}

base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_
print("\n🔍 Best XGBoost Parameters:")
print(grid.best_params_)

# === Calibrate model ===
calibrated_model = CalibratedClassifierCV(best_xgb, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy (v4.2 tuned + calibrated): {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# === Feature importance from base model ===
importances = best_xgb.feature_importances_
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n📊 Feature Importances:")
print(importance_df)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importances — Model v4.2")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# === Save model ===
joblib.dump(calibrated_model, "nba_model_v4_2.pkl")
print("✅ Model saved as nba_model_v4_2.pkl")
