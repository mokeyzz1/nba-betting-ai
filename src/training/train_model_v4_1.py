import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

print("\n🔍 Checking missing values:")
for col in required_features:
    if col not in df.columns:
        print(f"❌ Missing column: {col}")
    else:
        nulls = df[col].isnull().sum()
        print(f"{col}: {nulls} nulls")

df = df.dropna(subset=required_features)
print(f"\n✅ Training on {len(df)} rows (from original {len(df_raw)})")

# === Engineered matchup features ===
df["off_rating_diff"] = df["home_OFF_RATING"] - df["away_DEF_RATING"]
df["def_rating_diff"] = df["home_DEF_RATING"] - df["away_OFF_RATING"]
df["recent_win_diff"] = df["home_recent_win_pct"] - df["away_recent_win_pct"]
df["pace_diff"] = df["home_PACE"] - df["away_PACE"]

# === Vegas-based engineered features ===
df["odds_diff"] = df["home_odds"] - df["away_odds"]
df["implied_home_win_pct"] = 1 / df["home_odds"]
df["implied_away_win_pct"] = 1 / df["away_odds"]
df["implied_win_diff"] = df["implied_home_win_pct"] - df["implied_away_win_pct"]

# === Define features ===
features = [
    "off_rating_diff", "def_rating_diff", "recent_win_diff", "pace_diff",
    "home_recent_avg_pts", "away_recent_avg_pts",
    "home_EFG_PCT", "away_EFG_PCT",
    "home_odds", "away_odds", "odds_diff",
    "implied_home_win_pct", "implied_away_win_pct", "implied_win_diff"
]

X = df[features]
y = df["home_win"]

# === Force numeric and drop rows with NaNs ===
print("\n🧪 Feature dtypes (before):")
print(X.dtypes)

X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = y.loc[X.index]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# === Train model ===
model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy (v4.1 combo model): {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# === Feature importance chart ===
importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\n📊 Feature Importances:")
print(importance_df)

# === Save model ===
joblib.dump(model, "nba_model_v4_1.pkl")
print("✅ Model saved as nba_model_v4_1.pkl")