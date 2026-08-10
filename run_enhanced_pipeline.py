# Enhanced NBA Prediction Pipeline
import sys
import traceback
from datetime import datetime, timedelta
from src.features.get_odds import fetch_odds
from src.features.build_enhanced_features import build_enhanced_features
from src.prediction.predict_enhanced import run_enhanced_predictions
from src.evaluate.evaluate_predictions import evaluate_results
from src.pipeline.fetch_actual_winners import fetch_actual_results
from src.monitor.advanced_analytics import run_analytics
from src.monitor.rolling_accuracy import update_rolling_accuracy

def main():
    """Enhanced NBA AI Prediction Pipeline"""
    
    print("🚀 Starting Enhanced NBA AI Prediction Pipeline...")
    print("=" * 60)
    
    model_version = "enhanced"
    today = datetime.today()
    yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # Step 1: Fetch odds for today's games
        print("\n📡 Step 1: Fetching NBA odds...")
        fetch_odds()
        
        # Step 2: Build enhanced features
        print("\n🔧 Step 2: Building enhanced features...")
        features_df = build_enhanced_features()
        
        if features_df is None or len(features_df) == 0:
            print("⚠️ No games found for today - skipping prediction steps")
        else:
            # Step 3: Run enhanced predictions
            print("\n🧠 Step 3: Running enhanced predictions...")
            predictions_df = run_enhanced_predictions()
            
            print(f"✅ Generated predictions for {len(predictions_df)} games")
        
        # Step 4: Fetch yesterday's actual results
        print(f"\n📊 Step 4: Fetching actual results for {yesterday}...")
        try:
            fetch_actual_results(date=yesterday, model_version=model_version)
        except Exception as e:
            print(f"⚠️ Could not fetch results for {yesterday}: {e}")
        
        # Step 5: Evaluate model performance
        print(f"\n📈 Step 5: Evaluating predictions for {yesterday}...")
        try:
            evaluate_results(date=yesterday, model_version=model_version)
        except Exception as e:
            print(f"⚠️ Could not evaluate {yesterday}: {e}")
        
        # Step 6: Update rolling metrics
        print(f"\n📊 Step 6: Updating rolling performance metrics...")
        try:
            update_rolling_accuracy(model_version=model_version)
        except Exception as e:
            print(f"⚠️ Could not update rolling metrics: {e}")
        
        # Step 7: Run advanced analytics
        print(f"\n📈 Step 7: Running advanced analytics...")
        try:
            run_analytics()
        except Exception as e:
            print(f"⚠️ Analytics failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Enhanced Pipeline completed successfully!")
        
        # Print summary
        if 'predictions_df' in locals() and predictions_df is not None:
            print(f"\n📊 Today's Predictions Summary:")
            print(f"• Total games: {len(predictions_df)}")
            
            if 'confidence_score' in predictions_df.columns:
                high_conf = (predictions_df['confidence_score'] > 0.3).sum()
                print(f"• High confidence picks: {high_conf}")
            
            if 'value_gap' in predictions_df.columns:
                value_bets = (predictions_df['value_gap'] > 0.03).sum()
                print(f"• Value bets identified: {value_bets}")
                
                if value_bets > 0:
                    best_bet = predictions_df.loc[predictions_df['value_gap'].idxmax()]
                    print(f"• Best value bet: {best_bet['awayteam']} @ {best_bet['hometeam']}")
                    print(f"  - Pick: {best_bet['prediction']}")
                    print(f"  - Confidence: {best_bet.get('confidence_score', 0):.2f}")
                    print(f"  - Value gap: {best_bet['value_gap']:+.3f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)

def quick_predictions_only():
    """Quick mode - just run predictions for today"""
    print("⚡ Quick Predictions Mode")
    
    try:
        fetch_odds()
        predictions_df = run_enhanced_predictions()
        
        if predictions_df is not None:
            print(f"\n✅ Quick predictions completed for {len(predictions_df)} games")
        else:
            print("\n⚠️ No games found for today")
            
    except Exception as e:
        print(f"❌ Quick predictions failed: {e}")

if __name__ == "__main__":
    # Check for quick mode
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_predictions_only()
    else:
        main()