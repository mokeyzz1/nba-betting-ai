import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import PERFORMANCE_DIR, PREDICTIONS_DIR
import os

class AdvancedAnalytics:
    
    def __init__(self, model_version="enhanced"):
        self.model_version = model_version
        self.performance_data = None
        self.predictions_data = None
        
    def load_historical_data(self, days_back=30):
        """Load historical performance and prediction data"""
        
        print(f"📊 Loading {days_back} days of historical data...")
        
        # Load performance files
        perf_files = []
        pred_files = []
        
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            
            perf_file = PERFORMANCE_DIR / f"accuracy_{date}_{self.model_version}.csv"
            pred_file = PREDICTIONS_DIR / f"predictions_{date}_{self.model_version}.csv"
            
            if perf_file.exists():
                perf_files.append(pd.read_csv(perf_file))
            
            if pred_file.exists():
                pred_files.append(pd.read_csv(pred_file))
        
        if perf_files:
            self.performance_data = pd.concat(perf_files, ignore_index=True)
            self.performance_data['date'] = pd.to_datetime(self.performance_data['date'])
        
        if pred_files:
            self.predictions_data = pd.concat(pred_files, ignore_index=True)
            
        print(f"✅ Loaded {len(perf_files)} performance files, {len(pred_files)} prediction files")
    
    def calculate_roi_by_confidence(self):
        """Calculate ROI by confidence buckets"""
        
        if self.predictions_data is None:
            print("❌ No prediction data available")
            return None
        
        # Filter for predictions with actual results
        completed = self.predictions_data.dropna(subset=['actual_winner'])
        
        if len(completed) == 0:
            print("❌ No completed predictions found")
            return None
        
        # Create confidence buckets
        completed['confidence_bucket'] = pd.cut(
            completed['confidence_score'], 
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Calculate results
        roi_by_confidence = []
        
        for bucket in completed['confidence_bucket'].unique():
            if pd.isna(bucket):
                continue
                
            bucket_data = completed[completed['confidence_bucket'] == bucket]
            
            # Calculate accuracy
            correct = (bucket_data['prediction'] == bucket_data['actual_winner']).sum()
            total = len(bucket_data)
            accuracy = correct / total if total > 0 else 0
            
            # Calculate ROI (simplified)
            bucket_data['win_amount'] = bucket_data.apply(
                lambda x: abs(x['predicted_odds']) if x['prediction'] == x['actual_winner'] else -100,
                axis=1
            )
            
            total_bet = total * 100  # $100 per bet
            total_return = bucket_data['win_amount'].sum()
            roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
            
            roi_by_confidence.append({
                'confidence_bucket': bucket,
                'games': total,
                'accuracy': accuracy,
                'roi': roi
            })
        
        return pd.DataFrame(roi_by_confidence)
    
    def analyze_value_bet_performance(self):
        """Analyze performance of value bet classifications"""
        
        if self.predictions_data is None:
            return None
        
        completed = self.predictions_data.dropna(subset=['actual_winner'])
        
        if 'value_classification' not in completed.columns:
            print("❌ No value classification data available")
            return None
        
        value_performance = []
        
        for classification in completed['value_classification'].unique():
            class_data = completed[completed['value_classification'] == classification]
            
            if len(class_data) == 0:
                continue
            
            correct = (class_data['prediction'] == class_data['actual_winner']).sum()
            total = len(class_data)
            accuracy = correct / total
            
            # Calculate average value gap
            avg_value_gap = class_data['value_gap'].mean()
            
            value_performance.append({
                'classification': classification,
                'games': total,
                'accuracy': accuracy,
                'avg_value_gap': avg_value_gap
            })
        
        return pd.DataFrame(value_performance)
    
    def plot_performance_trends(self):
        """Create visualizations of performance trends"""
        
        if self.performance_data is None:
            print("❌ No performance data for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy over time
        axes[0, 0].plot(self.performance_data['date'], self.performance_data['accuracy'])
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Rolling accuracy
        if len(self.performance_data) >= 5:
            rolling_acc = self.performance_data['accuracy'].rolling(window=5).mean()
            axes[0, 1].plot(self.performance_data['date'], rolling_acc)
            axes[0, 1].set_title('5-Day Rolling Accuracy')
            axes[0, 1].set_ylabel('Rolling Accuracy')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROI by confidence (if available)
        roi_data = self.calculate_roi_by_confidence()
        if roi_data is not None:
            axes[1, 0].bar(roi_data['confidence_bucket'], roi_data['roi'])
            axes[1, 0].set_title('ROI by Confidence Level')
            axes[1, 0].set_ylabel('ROI')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Value bet performance
        value_data = self.analyze_value_bet_performance()
        if value_data is not None:
            axes[1, 1].bar(value_data['classification'], value_data['accuracy'])
            axes[1, 1].set_title('Accuracy by Value Classification')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(PERFORMANCE_DIR / f"performance_analysis_{datetime.now().strftime('%Y-%m-%d')}.png")
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive performance summary"""
        
        if self.performance_data is None:
            print("❌ No data available for summary")
            return
        
        print("📈 Performance Summary Report")
        print("=" * 50)
        
        # Overall stats
        overall_accuracy = self.performance_data['accuracy'].mean()
        recent_accuracy = self.performance_data.tail(7)['accuracy'].mean()
        accuracy_trend = recent_accuracy - self.performance_data.head(7)['accuracy'].mean()
        
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"Recent 7-day Accuracy: {recent_accuracy:.2%}")
        print(f"Accuracy Trend: {accuracy_trend:+.2%}")
        
        # Best and worst days
        best_day = self.performance_data.loc[self.performance_data['accuracy'].idxmax()]
        worst_day = self.performance_data.loc[self.performance_data['accuracy'].idxmin()]
        
        print(f"\nBest Day: {best_day['date'].strftime('%Y-%m-%d')} ({best_day['accuracy']:.2%})")
        print(f"Worst Day: {worst_day['date'].strftime('%Y-%m-%d')} ({worst_day['accuracy']:.2%})")
        
        # Confidence analysis
        roi_data = self.calculate_roi_by_confidence()
        if roi_data is not None:
            print(f"\n📊 Performance by Confidence:")
            for _, row in roi_data.iterrows():
                print(f"{row['confidence_bucket']}: {row['accuracy']:.2%} accuracy, {row['roi']:+.2%} ROI ({row['games']} games)")
        
        # Value bet analysis
        value_data = self.analyze_value_bet_performance()
        if value_data is not None:
            print(f"\n💰 Value Bet Performance:")
            for _, row in value_data.iterrows():
                print(f"{row['classification']}: {row['accuracy']:.2%} accuracy ({row['games']} games)")

def run_analytics():
    """Run complete analytics suite"""
    analytics = AdvancedAnalytics()
    analytics.load_historical_data(days_back=30)
    analytics.generate_summary_report()
    analytics.plot_performance_trends()

if __name__ == "__main__":
    run_analytics()