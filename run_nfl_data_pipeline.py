#!/usr/bin/env python3
"""
NFL Data Collection Pipeline
Automated pipeline to collect and process NFL data for multiple years
"""

import os
import pandas as pd
from datetime import datetime
from data_collection.nfl_api_scraper import NFLAPIScraper

def collect_multi_year_data(years=[2021, 2022, 2023]):
    """Collect NFL data for multiple years"""
    print(f"🏈 Starting NFL Data Collection for years: {years}")
    
    scraper = NFLAPIScraper()
    all_game_logs = []
    all_prop_lines = []
    all_season_data = {}
    
    for year in years:
        print(f"\n--- Processing {year} Season ---")
        
        # Get season data for this year
        season_data = scraper.get_fantasy_data(year)
        all_season_data[year] = season_data
        
        # Generate game logs
        year_game_logs = scraper.generate_game_logs(season_data)
        year_game_logs['year'] = year
        all_game_logs.append(year_game_logs)
        
        # Create prop lines
        year_prop_lines = scraper.create_prop_betting_lines(year_game_logs)
        year_prop_lines['year'] = year
        all_prop_lines.append(year_prop_lines)
        
        print(f"✅ {year}: {len(year_game_logs)} games, {len(year_prop_lines)} prop lines")
    
    # Combine all years
    combined_game_logs = pd.concat(all_game_logs, ignore_index=True)
    combined_prop_lines = pd.concat(all_prop_lines, ignore_index=True)
    
    print(f"\n📊 Total Data Collected:")
    print(f"Game Logs: {len(combined_game_logs)} entries")
    print(f"Prop Lines: {len(combined_prop_lines)} entries")
    
    return combined_game_logs, combined_prop_lines, all_season_data

def save_historical_data(game_logs, prop_lines, season_data):
    """Save all historical data"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save combined datasets
    game_logs.to_csv(f'data/nfl/nfl_historical_game_logs_{timestamp}.csv', index=False)
    prop_lines.to_csv(f'data/nfl/nfl_historical_prop_lines_{timestamp}.csv', index=False)
    
    print(f"💾 Saved historical datasets:")
    print(f"  - data/nfl/nfl_historical_game_logs_{timestamp}.csv")
    print(f"  - data/nfl/nfl_historical_prop_lines_{timestamp}.csv")
    
    # Save season summaries
    for year, year_data in season_data.items():
        for position, df in year_data.items():
            filename = f'data/nfl/season_{position}_{year}_{timestamp}.csv'
            df.to_csv(filename, index=False)
    
    return f'data/nfl/nfl_historical_game_logs_{timestamp}.csv', f'data/nfl/nfl_historical_prop_lines_{timestamp}.csv'

def create_feature_summary():
    """Create a summary of features available for NFL modeling"""
    
    features_summary = {
        'quarterback_features': [
            'avg_passing_yards_last_3',
            'avg_passing_yards_last_5', 
            'passing_yards_trend',
            'games_over_250_yards_rate',
            'games_over_300_yards_rate',
            'home_away_performance_diff',
            'team_strength_rating',
            'opponent_pass_defense_rank'
        ],
        
        'running_back_features': [
            'avg_rushing_yards_last_3',
            'avg_rushing_yards_last_5',
            'rushing_yards_trend',
            'games_over_80_yards_rate',
            'games_over_100_yards_rate',
            'home_away_performance_diff',
            'team_run_blocking_rating',
            'opponent_run_defense_rank'
        ],
        
        'wide_receiver_features': [
            'avg_receiving_yards_last_3',
            'avg_receiving_yards_last_5',
            'receiving_yards_trend',
            'games_over_75_yards_rate',
            'games_over_100_yards_rate',
            'target_share_trend',
            'red_zone_target_rate',
            'opponent_pass_defense_rank'
        ],
        
        'game_context_features': [
            'weather_wind_speed',
            'weather_temperature',
            'dome_vs_outdoor',
            'division_game',
            'prime_time_game',
            'days_rest',
            'injury_report_status',
            'point_spread',
            'total_points_line'
        ]
    }
    
    # Save features summary
    with open('data/nfl/nfl_features_summary.txt', 'w') as f:
        f.write("NFL FEATURES FOR PROP BETTING MODELS\n")
        f.write("=" * 50 + "\n\n")
        
        for position, features in features_summary.items():
            f.write(f"{position.upper().replace('_', ' ')}:\n")
            for feature in features:
                f.write(f"  - {feature}\n")
            f.write("\n")
    
    print("📋 Created NFL features summary: data/nfl/nfl_features_summary.txt")
    return features_summary

def analyze_prop_betting_opportunities(prop_lines):
    """Analyze the prop betting opportunities in the data"""
    print("\n🎯 Analyzing Prop Betting Opportunities:")
    
    # Group by stat type and line
    opportunities = []
    
    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        stat_data = prop_lines[prop_lines['stat_type'] == stat_type]
        
        if not stat_data.empty:
            # Find the most profitable lines (closest to 50/50)
            stat_data['edge'] = abs(stat_data['over_rate'] - 0.5)
            best_lines = stat_data.nsmallest(5, 'edge')
            
            print(f"\n{stat_type.replace('_', ' ').title()} - Best Lines:")
            for _, row in best_lines.iterrows():
                edge_pct = row['edge'] * 100
                print(f"  {row['player']}: {row['line']} ({row['over_rate']:.1%} over rate, {edge_pct:.1f}% edge)")
                
                opportunities.append({
                    'player': row['player'],
                    'stat_type': stat_type,
                    'line': row['line'],
                    'over_rate': row['over_rate'],
                    'edge': row['edge'],
                    'avg_value': row['avg_value']
                })
    
    # Save opportunities
    opps_df = pd.DataFrame(opportunities)
    if not opps_df.empty:
        opps_df.to_csv('data/nfl/nfl_betting_opportunities.csv', index=False)
        print(f"\n💰 Saved betting opportunities: data/nfl/nfl_betting_opportunities.csv")
    
    return opps_df

def main():
    """Main pipeline execution"""
    print("🚀 NFL Data Pipeline Starting...")
    print("=" * 60)
    
    # Create data directory
    os.makedirs('data/nfl', exist_ok=True)
    
    # Step 1: Collect multi-year data
    game_logs, prop_lines, season_data = collect_multi_year_data([2021, 2022, 2023])
    
    # Step 2: Save historical data
    game_logs_file, prop_lines_file = save_historical_data(game_logs, prop_lines, season_data)
    
    # Step 3: Create feature summary
    features = create_feature_summary()
    
    # Step 4: Analyze opportunities
    opportunities = analyze_prop_betting_opportunities(prop_lines)
    
    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("🏆 NFL DATA PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"📈 Game Logs: {len(game_logs)} entries")
    print(f"🎯 Prop Lines: {len(prop_lines)} entries") 
    print(f"💰 Opportunities: {len(opportunities)} identified")
    print(f"📁 Data Location: data/nfl/")
    
    print("\n🔥 NEXT STEPS:")
    print("1. Build feature engineering pipeline (adapt from NBA)")
    print("2. Train prop betting models using historical data")
    print("3. Set up live prediction pipeline for current season")
    print("4. Deploy betting recommendations system")
    
    print("\n⚡ READY TO BUILD NFL PROP MODELS! ⚡")

if __name__ == "__main__":
    main()