
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
