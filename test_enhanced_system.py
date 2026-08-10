#!/usr/bin/env python3
"""
Test the Enhanced NBA Prediction System
Run this during offseason to validate all components
"""

import sys
import os
sys.path.append('src')

from features.get_enhanced_odds import fetch_enhanced_odds
from features.get_player_data import get_team_player_impact, get_player_matchups
from features.advanced_nba_analytics import get_advanced_game_analysis
from utils.offseason_mode import is_offseason

def test_enhanced_system():
    """Test all enhanced system components"""
    
    print("🧪 Testing Enhanced NBA Prediction System")
    print("=" * 50)
    
    # Check if we're in offseason mode
    if is_offseason():
        print("🎭 Offseason mode detected - using mock data")
    
    print("\n📡 1. Testing Enhanced Odds Fetching...")
    try:
        odds_df = fetch_enhanced_odds()
        print(f"✅ Fetched odds for {len(odds_df)} games")
        print("Markets available:", [col for col in odds_df.columns if 'odds' in col or 'spread' in col or 'total' in col])
    except Exception as e:
        print(f"❌ Odds fetching failed: {e}")
        return False
    
    if len(odds_df) == 0:
        print("⚠️ No games available - creating mock game for testing")
        # Create a mock game for testing
        test_home = "Boston Celtics"
        test_away = "Los Angeles Lakers"
    else:
        test_home = odds_df.iloc[0]['home_team']
        test_away = odds_df.iloc[0]['away_team']
    
    print(f"\n👨‍💼 2. Testing Player Data Integration...")
    print(f"Analyzing: {test_away} @ {test_home}")
    
    try:
        # Test player impact analysis
        home_impact = get_team_player_impact(test_home)
        away_impact = get_team_player_impact(test_away)
        
        print(f"✅ {test_home} player impact: {home_impact['total_player_impact']:.2f}")
        print(f"✅ {test_away} player impact: {away_impact['total_player_impact']:.2f}")
        
        # Test player matchups
        matchups = get_player_matchups(test_home, test_away)
        print(f"✅ Star power advantage: {matchups['star_power_advantage']:+.2f}")
        
    except Exception as e:
        print(f"❌ Player data failed: {e}")
        return False
    
    print(f"\n🏀 3. Testing Advanced NBA Analytics...")
    
    try:
        # Test comprehensive game analysis
        analysis = get_advanced_game_analysis(test_home, test_away)
        
        differentials = analysis['differentials']
        print(f"✅ Four factors composite: {differentials['four_factors_composite']:+.3f}")
        print(f"✅ Net rating differential: {differentials['net_rating_diff']:+.1f}")
        print(f"✅ Pace differential: {differentials['pace_differential']:+.1f}")
        print(f"✅ Rest advantage: {differentials['rest_advantage']:+.2f}")
        
        # Show key insights
        home_data = analysis['home_team_data']
        away_data = analysis['away_team_data']
        
        print(f"\n📊 Key Insights:")
        print(f"• {test_home} EFG%: {home_data['four_factors']['off_efg_pct']:.1%}")
        print(f"• {test_away} EFG%: {away_data['four_factors']['off_efg_pct']:.1%}")
        print(f"• {test_home} Pace: {home_data['pace_efficiency']['pace']:.1f}")
        print(f"• {test_away} Pace: {away_data['pace_efficiency']['pace']:.1f}")
        
    except Exception as e:
        print(f"❌ Advanced analytics failed: {e}")
        return False
    
    print(f"\n🎯 4. Testing Prediction Features...")
    
    try:
        # Combine all data for prediction features
        prediction_features = {
            # Market data
            'home_ml': odds_df.iloc[0].get('home_ml', -110) if len(odds_df) > 0 else -110,
            'away_ml': odds_df.iloc[0].get('away_ml', +100) if len(odds_df) > 0 else +100,
            'home_spread': odds_df.iloc[0].get('home_spread', -3.5) if len(odds_df) > 0 else -3.5,
            'total_points': odds_df.iloc[0].get('total_points', 220.5) if len(odds_df) > 0 else 220.5,
            
            # Advanced analytics
            'four_factors_composite': differentials['four_factors_composite'],
            'net_rating_diff': differentials['net_rating_diff'],
            'pace_differential': differentials['pace_differential'],
            'rest_advantage': differentials['rest_advantage'],
            
            # Player data
            'star_power_advantage': matchups['star_power_advantage'],
            'depth_advantage': matchups['depth_advantage'],
            'home_availability': home_impact['availability_percentage'],
            'away_availability': away_impact['availability_percentage']
        }
        
        print("✅ Feature engineering successful!")
        print(f"Feature count: {len(prediction_features)}")
        
        # Show some key features
        print(f"\n🔑 Key Prediction Features:")
        for feature, value in list(prediction_features.items())[:6]:
            if isinstance(value, (int, float)):
                print(f"• {feature}: {value:+.3f}")
            else:
                print(f"• {feature}: {value}")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False
    
    print(f"\n🏆 System Test Results:")
    print("=" * 30)
    print("✅ Enhanced odds fetching: PASS")
    print("✅ Player data integration: PASS")  
    print("✅ Advanced NBA analytics: PASS")
    print("✅ Feature engineering: PASS")
    print("\n🎯 Enhanced NBA system is ready for production!")
    
    # Show improvement opportunities
    print(f"\n💡 Next Steps for Production:")
    print("1. Train ensemble model with new features")
    print("2. Implement real-time injury tracking")
    print("3. Add weather data for outdoor arenas")
    print("4. Integrate social sentiment analysis")
    print("5. Create profit tracking dashboard")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_system()
    sys.exit(0 if success else 1)