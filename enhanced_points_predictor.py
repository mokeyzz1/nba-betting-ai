
class EnhancedPointsPredictor:
    """Enhanced Points Prediction Interface"""
    
    def __init__(self):
        self.models = self.load_enhanced_models()
        self.features = self.load_features()
        self.config = self.load_config()
    
    def load_enhanced_models(self):
        models = {}
        point_lines = [15.5, 20.5, 25.5, 30.5]
        
        for line in point_lines:
            line_key = f'over_{line}'
            line_models = {}
            
            for algorithm in ['rf', 'gb', 'xgb', 'lr']:
                model_path = f"models/enhanced_points_{line_key}_{algorithm}.pkl"
                try:
                    line_models[algorithm] = joblib.load(model_path)
                except:
                    pass
            
            if line_models:
                models[line_key] = line_models
        
        return models
    
    def predict_enhanced_points(self, player_features):
        """Get enhanced predictions with value calculation"""
        
        predictions = {}
        
        for line_key, line_models in self.models.items():
            line = float(line_key.split('_')[1])
            
            # Get ensemble predictions
            probs = []
            for model in line_models.values():
                prob = model.predict_proba([player_features])[0][1]
                probs.append(prob)
            
            ensemble_prob = np.mean(probs)
            ensemble_confidence = 1 - np.std(probs)
            
            # Calculate value (placeholder for live odds integration)
            market_prob = 0.52  # Placeholder - will use live odds
            edge = ensemble_prob - market_prob
            expected_value = (ensemble_prob * 1.91) - 1
            
            # Betting recommendation
            recommendation = "NO BET"
            if (ensemble_prob > 0.65 and edge > 0.10 and ensemble_confidence > 0.7):
                recommendation = "BET OVER"
            elif (ensemble_prob < 0.35 and edge < -0.10 and ensemble_confidence > 0.7):
                recommendation = "BET UNDER"
            
            predictions[line] = {
                'probability': ensemble_prob,
                'confidence': ensemble_confidence,
                'edge': edge,
                'expected_value': expected_value,
                'recommendation': recommendation
            }
        
        return predictions
