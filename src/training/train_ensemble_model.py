import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

class NBEnsembleModel:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_base_models(self):
        """Create individual models for ensemble"""
        
        # XGBoost - good for feature interactions
        self.models['xgb'] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Random Forest - good for stability
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Logistic Regression - good baseline
        self.models['lr'] = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # Neural Network - good for complex patterns
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            alpha=0.01,
            random_state=42,
            max_iter=500
        )
        
    def train_ensemble(self, X, y):
        """Train all models and calculate weights"""
        print("🧠 Training ensemble model...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_val_df = pd.DataFrame(X_val_scaled, columns=X.columns)
        
        self.create_base_models()
        
        # Train each model and calculate weights based on validation performance
        val_scores = {}
        predictions = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name in ['lr', 'nn']:
                # Use scaled features for models that benefit from scaling
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
                predictions[name] = model.predict_proba(X_val_scaled)[:, 1]
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                predictions[name] = model.predict_proba(X_val)[:, 1]
            
            score = accuracy_score(y_val, val_pred)
            val_scores[name] = score
            print(f"{name} validation accuracy: {score:.4f}")
        
        # Calculate weights based on performance (softmax of scores)
        scores_array = np.array(list(val_scores.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))
        self.weights = dict(zip(val_scores.keys(), exp_scores / np.sum(exp_scores)))
        
        print("\n📊 Model weights:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.3f}")
        
        # Calibrate ensemble predictions
        ensemble_preds = self._ensemble_predict_proba(X_val, y_val.values)
        
        return val_scores, ensemble_preds
    
    def _ensemble_predict_proba(self, X, y_true=None):
        """Get ensemble predictions"""
        
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        ensemble_proba = np.zeros(len(X))
        
        for name, model in self.models.items():
            if name in ['lr', 'nn']:
                proba = model.predict_proba(X_scaled)[:, 1]
            else:
                proba = model.predict_proba(X)[:, 1]
            
            ensemble_proba += self.weights[name] * proba
        
        return ensemble_proba
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble"""
        return self._ensemble_predict_proba(X)
    
    def predict(self, X):
        """Make predictions using ensemble"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def save_model(self, filepath):
        """Save the entire ensemble"""
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Ensemble model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load ensemble model"""
        model_data = joblib.load(filepath)
        ensemble = cls()
        ensemble.models = model_data['models']
        ensemble.weights = model_data['weights']
        ensemble.scaler = model_data['scaler']
        ensemble.feature_names = model_data['feature_names']
        return ensemble

def train_enhanced_model():
    """Train ensemble model with enhanced features"""
    
    # Load enhanced dataset
    df = pd.read_csv("data/final_games_full_features.csv", low_memory=False)
    print(f"📊 Loaded {len(df)} games for training")
    
    # Enhanced feature set
    enhanced_features = [
        # Basic differentials
        "off_rating_diff", "def_rating_diff", "recent_win_diff", "pace_diff",
        
        # Performance metrics
        "home_recent_avg_pts", "away_recent_avg_pts",
        "home_EFG_PCT", "away_EFG_PCT",
        
        # Odds features
        "home_odds", "away_odds", "odds_diff",
        "implied_home_win_pct", "implied_away_win_pct", "implied_win_diff",
        
        # Advanced features (if available)
        "home_court_advantage", "momentum_diff", "wins_last_5_diff",
        "clutch_diff", "sos_diff"
    ]
    
    # Use available features only
    available_features = [f for f in enhanced_features if f in df.columns]
    
    print(f"🔧 Using {len(available_features)} features:")
    for f in available_features:
        print(f"  - {f}")
    
    # Prepare data
    df_clean = df.dropna(subset=available_features + ['home_win'])
    X = df_clean[available_features]
    y = df_clean['home_win']
    
    print(f"\n✅ Training on {len(X)} clean samples")
    
    # Train ensemble
    ensemble = NBEnsembleModel()
    val_scores, ensemble_preds = ensemble.train_ensemble(X, y)
    
    # Save model
    ensemble.save_model("models/nba_ensemble_model.pkl")
    
    return ensemble, val_scores

if __name__ == "__main__":
    ensemble, scores = train_enhanced_model()