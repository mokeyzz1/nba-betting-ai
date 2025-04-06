import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Load Data ===
df = pd.read_csv("data/final_games_full_features.csv", low_memory=False)
df_raw = df.copy()

# === Fill recent stats with neutral values ===
df["home_recent_win_pct"] = df["home_recent_win_pct"].fillna(0.5)
df["away_recent_win_pct"] = df["away_recent_win_pct"].fillna(0.5)
df["home_recent_avg_pts"] = df["home_recent_avg_pts"].fillna(df["home_recent_avg_pts"].median())
df["away_recent_avg_pts"] = df["away_recent_avg_pts"].fillna(df["away_recent_avg_pts"].median())

# === Drop rows missing core features (ADV STATS + ODDS) ===
required_features = [
    "home_OFF_RATING", "away_OFF_RATING",
    "home_DEF_RATING", "away_DEF_RATING",
    "home_EFG_PCT", "away_EFG_PCT",
    "home_odds", "away_odds"
]

print("\n🔍 Checking missing values:")
for col in required_features:
    if col not in df.columns:
        print(f"❌ Missing column: {col}")
    else:
        nulls = df[col].isnull().sum()
        print(f"{col}: {nulls} nulls")

existing_required = [col for col in required_features if col in df.columns]
df = df.dropna(subset=existing_required)

print(f"\n✅ Training on {len(df)} rows (from original {len(df_raw)})")

# === Define features and label ===
features = [
    "home_seasonWins", "away_seasonWins",
    "home_recent_win_pct", "away_recent_win_pct",
    "home_recent_avg_pts", "away_recent_avg_pts",
    "home_OFF_RATING", "away_OFF_RATING",
    "home_DEF_RATING", "away_DEF_RATING",
    "home_EFG_PCT", "away_EFG_PCT",
    "home_PACE", "away_PACE",
    "home_odds", "away_odds"
]

X = df[features]
y = df["home_win"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# === Train XGBoost model ===
model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# === Evaluate performance ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# === Save model ===
joblib.dump(model, "nba_model_v3.pkl")
print("✅ Model saved as nba_model_v3.pkl")

import matplotlib.pyplot as plt

# === Feature importance plot
importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\n📊 Feature Importances:")
print(importance_df)

# Optional: plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
