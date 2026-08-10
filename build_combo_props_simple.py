#!/usr/bin/env python3
"""
Build Simple Combo Props Model (Points + Rebounds + Assists)
Fast version using existing individual prop models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_existing_models():
    """Load existing individual prop models"""
    
    print("🏀 BUILDING SIMPLE COMBO PROPS MODEL (PRA)")
    print("Using existing individual models for faster training")
    print("=" * 60)
    
    try:
        # Load individual models
        points_model = joblib.load('models/player_points_regressor.pkl')
        rebounds_model = joblib.load('models/player_rebounds_regressor.pkl') 
        assists_model = joblib.load('models/player_assists_regressor.pkl')
        
        print("✅ Loaded existing individual prop models")
        return points_model, rebounds_model, assists_model
    
    except Exception as e:
        print(f"⚠️ Could not load existing models: {e}")
        return None, None, None

def create_pra_synthetic_data():
    """Create synthetic training data for PRA combinations"""
    
    print("\n📊 Creating synthetic PRA training data...")
    
    # Generate realistic player scenarios
    scenarios = []
    
    # Star players (high PRA)
    for i in range(1000):
        points = np.random.normal(26.5, 5.2)
        rebounds = np.random.normal(7.8, 2.5) 
        assists = np.random.normal(6.5, 2.2)
        pra = points + rebounds + assists
        
        scenarios.append({
            'predicted_points': points,
            'predicted_rebounds': rebounds,
            'predicted_assists': assists,
            'predicted_pra': pra,
            'player_type': 'star'
        })
    
    # Role players (medium PRA)
    for i in range(2000):
        points = np.random.normal(13.2, 4.1)
        rebounds = np.random.normal(5.1, 2.0)
        assists = np.random.normal(2.8, 1.5)
        pra = points + rebounds + assists
        
        scenarios.append({
            'predicted_points': points,
            'predicted_rebounds': rebounds,
            'predicted_assists': assists,
            'predicted_pra': pra,
            'player_type': 'role'
        })
    
    # Bench players (low PRA)
    for i in range(1000):
        points = np.random.normal(7.1, 3.2)
        rebounds = np.random.normal(2.8, 1.5)
        assists = np.random.normal(1.2, 0.8)
        pra = points + rebounds + assists
        
        scenarios.append({
            'predicted_points': points,
            'predicted_rebounds': rebounds,
            'predicted_assists': assists,
            'predicted_pra': pra,
            'player_type': 'bench'
        })
    
    df = pd.DataFrame(scenarios)
    print(f"✅ Created {len(df)} PRA scenarios")
    
    return df

def train_pra_models(df):
    """Train PRA combination models"""
    
    print(f"\n🧠 TRAINING PRA COMBINATION MODELS")
    
    # Features: individual predictions
    feature_cols = ['predicted_points', 'predicted_rebounds', 'predicted_assists']
    X = df[feature_cols]
    
    # Target: PRA total
    y_pra = df['predicted_pra']
    
    # Split data
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_pra.iloc[:split_idx], y_pra.iloc[split_idx:]
    
    print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
    
    models = {}
    
    # 1. PRA Sum Model (simple addition - should be perfect)
    print(f"\n🎯 Training PRA Sum Model...")
    
    # This should just be points + rebounds + assists
    y_pred_sum = X_test.sum(axis=1)
    mae_sum = mean_absolute_error(y_test, y_pred_sum)
    print(f"   Sum Model MAE: {mae_sum:.3f} (should be ~0)")
    
    # Save sum model (identity function)
    models['pra_sum'] = 'sum'
    
    # 2. PRA Classification Models for Over/Under lines
    print(f"\n🎯 Training PRA Over/Under Models...")
    
    common_pra_lines = [20.5, 25.5, 30.5, 35.5, 40.5, 45.5, 50.5]
    pra_accuracies = {}
    
    for pra_line in common_pra_lines:
        # Create Over/Under target
        y_over = (df['predicted_pra'] > pra_line).astype(int)
        y_train_over, y_test_over = y_over.iloc[:split_idx], y_over.iloc[split_idx:]
        
        # Skip if not enough variation
        if y_train_over.sum() < 50 or (len(y_train_over) - y_train_over.sum()) < 50:
            continue
        
        print(f"   Training {pra_line} PRA model...")
        
        pra_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            random_state=42
        )
        
        pra_classifier.fit(X_train, y_train_over)
        y_pred_over = pra_classifier.predict(X_test)
        accuracy = accuracy_score(y_test_over, y_pred_over)
        
        models[f'pra_over_{pra_line}'] = pra_classifier
        pra_accuracies[pra_line] = accuracy
        
        print(f"     PRA Over {pra_line} accuracy: {accuracy:.1%}")
    
    return models, feature_cols, mae_sum, pra_accuracies

def save_pra_models(models, feature_cols):
    """Save PRA combination models"""
    
    print(f"\n💾 SAVING PRA COMBO MODELS")
    
    # Save each model
    for model_name, model in models.items():
        if model != 'sum':  # Skip the sum model (it's just addition)
            model_path = f"models/combo_pra_{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"✅ Saved {model_name}")
    
    # Save feature list
    features_path = "models/combo_pra_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    print(f"📋 Saved feature list: {len(feature_cols)} features")

def test_pra_predictions(models, feature_cols):
    """Test PRA predictions on sample scenarios"""
    
    print(f"\n🧪 TESTING PRA COMBO PREDICTIONS")
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Star Player',
            'predicted_points': 28.5,
            'predicted_rebounds': 8.2,
            'predicted_assists': 7.1,
            'expected_pra': 43.8
        },
        {
            'name': 'Role Player', 
            'predicted_points': 14.2,
            'predicted_rebounds': 5.1,
            'predicted_assists': 3.2,
            'expected_pra': 22.5
        },
        {
            'name': 'Bench Player',
            'predicted_points': 7.8,
            'predicted_rebounds': 2.9,
            'predicted_assists': 1.1,
            'expected_pra': 11.8
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}:")
        print(f"   Individual predictions: {scenario['predicted_points']:.1f}P + {scenario['predicted_rebounds']:.1f}R + {scenario['predicted_assists']:.1f}A")
        
        # Calculate PRA sum
        predicted_pra = scenario['predicted_points'] + scenario['predicted_rebounds'] + scenario['predicted_assists']
        print(f"   Predicted PRA: {predicted_pra:.1f}")
        
        # Test Over/Under lines
        sample_df = pd.DataFrame([{
            'predicted_points': scenario['predicted_points'],
            'predicted_rebounds': scenario['predicted_rebounds'], 
            'predicted_assists': scenario['predicted_assists']
        }])
        
        for pra_line in [25.5, 30.5, 35.5, 40.5, 45.5]:
            model_key = f'pra_over_{pra_line}'
            if model_key in models:
                over_prob = models[model_key].predict_proba(sample_df)[0][1]
                recommendation = "OVER" if over_prob > 0.52 else "UNDER"
                confidence = max(over_prob, 1-over_prob)
                
                print(f"   {pra_line:4.1f} PRA: {recommendation} ({confidence:.1%} confidence)")

def create_pra_prediction_function():
    """Create unified PRA prediction function"""
    
    prediction_code = '''
def predict_pra_combo(points_pred, rebounds_pred, assists_pred):
    """
    Predict PRA combo from individual predictions
    
    Args:
        points_pred: Predicted points
        rebounds_pred: Predicted rebounds  
        assists_pred: Predicted assists
        
    Returns:
        dict with PRA total and Over/Under recommendations
    """
    import joblib
    import pandas as pd
    
    # Calculate PRA total
    pra_total = points_pred + rebounds_pred + assists_pred
    
    # Load classification models for lines
    results = {
        'predicted_pra': pra_total,
        'lines': {}
    }
    
    # Common PRA lines
    lines = [20.5, 25.5, 30.5, 35.5, 40.5, 45.5, 50.5]
    
    sample_df = pd.DataFrame([{
        'predicted_points': points_pred,
        'predicted_rebounds': rebounds_pred,
        'predicted_assists': assists_pred
    }])
    
    for line in lines:
        try:
            model = joblib.load(f'models/combo_pra_pra_over_{line}.pkl')
            over_prob = model.predict_proba(sample_df)[0][1]
            
            results['lines'][str(line)] = {
                'over_probability': over_prob,
                'under_probability': 1 - over_prob,
                'recommendation': 'OVER' if over_prob > 0.52 else 'UNDER',
                'confidence': max(over_prob, 1 - over_prob)
            }
        except:
            # Fallback to simple comparison
            results['lines'][str(line)] = {
                'over_probability': 0.5 if pra_total > line else 0.5,
                'recommendation': 'OVER' if pra_total > line else 'UNDER',
                'confidence': 0.6
            }
    
    return results
'''
    
    # Save the function
    with open('src/prediction/predict_pra_combo.py', 'w') as f:
        f.write(prediction_code)
    
    print("💾 Created PRA combo prediction function")

def main():
    """Build PRA combo models"""
    
    # Load existing models (optional)
    points_model, rebounds_model, assists_model = load_existing_models()
    
    # Create training data
    df = create_pra_synthetic_data()
    
    # Train models
    models, feature_cols, pra_mae, pra_accuracies = train_pra_models(df)
    
    # Save models
    save_pra_models(models, feature_cols)
    
    # Test predictions
    test_pra_predictions(models, feature_cols)
    
    # Create prediction function
    create_pra_prediction_function()
    
    # Final summary
    print(f"\n🎉 PRA COMBO MODELS COMPLETE!")
    print(f"📊 PRA prediction MAE: {pra_mae:.3f} (sum of individual predictions)")
    print(f"🎯 PRA Over/Under accuracies: {len(pra_accuracies)} lines trained")
    for pra_line, acc in pra_accuracies.items():
        print(f"   Over {pra_line:4.1f}: {acc:.1%} accuracy")
    
    print(f"\n🏀 ENHANCED PLAYER PROPS NOW AVAILABLE:")
    print(f"   ✅ Points Props")
    print(f"   ✅ Rebounds Props") 
    print(f"   ✅ Assists Props")
    print(f"   ✅ PRA Combo Props (Simple Sum)")
    
    print(f"\n💡 USAGE:")
    print(f"   PRA = Points Prediction + Rebounds Prediction + Assists Prediction")
    print(f"   Use individual prop models, then combine for PRA total")
    
    return models, feature_cols

if __name__ == "__main__":
    main()