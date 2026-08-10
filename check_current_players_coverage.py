#!/usr/bin/env python3
"""
Check Current NBA Players Coverage
See how well the player props system handles current NBA players and rookies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_current_player_coverage():
    """Analyze current NBA player coverage in the database"""
    
    print("🏀 CURRENT NBA PLAYERS COVERAGE ANALYSIS")
    print("Checking how system handles current players + rookies")
    print("=" * 70)
    
    # Load player data
    print("📊 Loading player database...")
    df = pd.read_csv('data/PlayerStatistics.csv', low_memory=False)
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    
    print(f"✅ Loaded {len(df)} total player performances")
    
    # Get recent data (2024-2025 season)
    recent_cutoff = datetime(2024, 10, 1)  # Start of 2024-25 season
    current_season = df[df['gameDate'] >= recent_cutoff]
    
    print(f"📅 Current season data: {len(current_season)} performances")
    print(f"📅 Date range: {current_season['gameDate'].min()} to {current_season['gameDate'].max()}")
    
    # Get unique current players
    current_players = current_season.groupby(['firstName', 'lastName']).agg({
        'gameDate': 'count',
        'points': 'mean',
        'reboundsTotal': 'mean', 
        'assists': 'mean',
        'numMinutes': 'mean',
        'personId': 'first'
    }).round(1)
    
    current_players.columns = ['games_played', 'avg_points', 'avg_rebounds', 'avg_assists', 'avg_minutes', 'person_id']
    current_players = current_players.sort_values('avg_points', ascending=False)
    
    print(f"👥 Current active players: {len(current_players)}")
    
    # Analyze player categories
    print(f"\n📊 PLAYER CATEGORIES ANALYSIS")
    print("-" * 50)
    
    # Stars (15+ PPG, 25+ MPG)
    stars = current_players[
        (current_players['avg_points'] >= 15) & 
        (current_players['avg_minutes'] >= 25) &
        (current_players['games_played'] >= 10)
    ]
    
    print(f"🌟 STAR PLAYERS ({len(stars)}):")
    print(f"   15+ PPG, 25+ MPG, 10+ games")
    if len(stars) > 0:
        print("   Top 10:")
        for i, (name, stats) in enumerate(stars.head(10).iterrows()):
            print(f"   {i+1:2d}. {name[0]} {name[1]:<15} {stats['avg_points']:5.1f} PPG, {stats['avg_rebounds']:4.1f} RPG, {stats['avg_assists']:4.1f} APG")
    
    # Role players (8-15 PPG)
    role_players = current_players[
        (current_players['avg_points'] >= 8) & 
        (current_players['avg_points'] < 15) &
        (current_players['avg_minutes'] >= 15) &
        (current_players['games_played'] >= 5)
    ]
    
    print(f"\n🏀 ROLE PLAYERS ({len(role_players)}):")
    print(f"   8-15 PPG, 15+ MPG, 5+ games")
    
    # Rookies/New players (less than 50 career games in database)
    all_player_games = df.groupby(['firstName', 'lastName']).size()
    rookie_candidates = current_players.merge(
        all_player_games.rename('total_career_games'), 
        left_index=True, 
        right_index=True
    )
    rookies = rookie_candidates[rookie_candidates['total_career_games'] <= 50]
    
    print(f"\n🆕 ROOKIES/NEW PLAYERS ({len(rookies)}):")
    print(f"   ≤50 total games in database")
    if len(rookies) > 0:
        print("   Top rookies by PPG:")
        for i, (name, stats) in enumerate(rookies.head(10).iterrows()):
            print(f"   {i+1:2d}. {name[0]} {name[1]:<15} {stats['avg_points']:5.1f} PPG ({stats['total_career_games']:2.0f} total games)")
    
    # Model readiness analysis
    print(f"\n🎯 MODEL READINESS ANALYSIS")
    print("-" * 50)
    
    # Players with sufficient data for modeling (10+ games this season)
    reliable_players = current_players[current_players['games_played'] >= 10]
    limited_data_players = current_players[
        (current_players['games_played'] >= 3) & 
        (current_players['games_played'] < 10)
    ]
    insufficient_players = current_players[current_players['games_played'] < 3]
    
    print(f"✅ READY FOR PROPS ({len(reliable_players)}):")
    print(f"   10+ games this season - full model confidence")
    
    print(f"\n⚠️ LIMITED DATA ({len(limited_data_players)}):")
    print(f"   3-9 games - reduced confidence, use season averages")
    
    print(f"\n❌ INSUFFICIENT DATA ({len(insufficient_players)}):")
    print(f"   <3 games - avoid or use career averages only")
    
    # High-value prop betting targets
    print(f"\n💰 TOP PLAYER PROPS TARGETS")
    print("-" * 50)
    
    # High-volume players (good for props)
    prop_targets = reliable_players[
        (reliable_players['avg_points'] >= 12) |
        (reliable_players['avg_rebounds'] >= 6) |
        (reliable_players['avg_assists'] >= 4)
    ].copy()
    
    prop_targets['prop_score'] = (
        prop_targets['avg_points'] * 0.4 +
        prop_targets['avg_rebounds'] * 0.3 + 
        prop_targets['avg_assists'] * 0.3
    )
    
    prop_targets = prop_targets.sort_values('prop_score', ascending=False)
    
    print(f"🎯 BEST PROP BETTING CANDIDATES ({len(prop_targets)}):")
    print("   High volume in points, rebounds, or assists:")
    for i, (name, stats) in enumerate(prop_targets.head(15).iterrows()):
        print(f"   {i+1:2d}. {name[0]} {name[1]:<15} {stats['avg_points']:5.1f}P {stats['avg_rebounds']:4.1f}R {stats['avg_assists']:4.1f}A")
    
    # Rookie handling strategy
    print(f"\n🆕 ROOKIE HANDLING STRATEGY")
    print("-" * 50)
    
    promising_rookies = rookies[
        (rookies['avg_minutes'] >= 20) &
        (rookies['games_played'] >= 5)
    ].sort_values('avg_points', ascending=False)
    
    if len(promising_rookies) > 0:
        print(f"🌟 PROMISING ROOKIES FOR PROPS ({len(promising_rookies)}):")
        print("   20+ MPG, 5+ games - can use current season stats")
        for i, (name, stats) in enumerate(promising_rookies.head(10).iterrows()):
            print(f"   {i+1:2d}. {name[0]} {name[1]:<15} {stats['avg_points']:5.1f}P {stats['avg_rebounds']:4.1f}R {stats['avg_assists']:4.1f}A")
    
    print(f"\n📋 ROOKIE STRATEGY:")
    print(f"   • 20+ MPG, 5+ games: Use season averages")
    print(f"   • <5 games: Use college stats or avoid")
    print(f"   • Be conservative with rookie props")
    
    return {
        'total_current_players': len(current_players),
        'stars': len(stars),
        'role_players': len(role_players), 
        'rookies': len(rookies),
        'reliable_for_props': len(reliable_players),
        'prop_targets': len(prop_targets)
    }

def check_specific_players():
    """Check coverage for specific high-profile players"""
    
    print(f"\n🔍 CHECKING SPECIFIC HIGH-PROFILE PLAYERS")
    print("-" * 60)
    
    df = pd.read_csv('data/PlayerStatistics.csv', low_memory=False)
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    
    # Current season
    recent_cutoff = datetime(2024, 10, 1)
    current_season = df[df['gameDate'] >= recent_cutoff]
    
    # Check specific stars
    target_players = [
        ('LeBron', 'James'),
        ('Stephen', 'Curry'),
        ('Nikola', 'Jokic'),
        ('Giannis', 'Antetokounmpo'),
        ('Jayson', 'Tatum'),
        ('Luka', 'Doncic'),
        ('Kevin', 'Durant'),
        ('Anthony', 'Davis'),
        ('Damian', 'Lillard'),
        ('Joel', 'Embiid')
    ]
    
    print("🌟 NBA SUPERSTARS IN DATABASE:")
    found_stars = 0
    
    for first_name, last_name in target_players:
        player_data = current_season[
            (current_season['firstName'].str.contains(first_name, case=False)) &
            (current_season['lastName'].str.contains(last_name, case=False))
        ]
        
        if len(player_data) > 0:
            games = len(player_data)
            avg_points = player_data['points'].mean()
            avg_rebounds = player_data['reboundsTotal'].mean()
            avg_assists = player_data['assists'].mean()
            
            print(f"   ✅ {first_name} {last_name:<15} {games:2d} games, {avg_points:5.1f}P {avg_rebounds:4.1f}R {avg_assists:4.1f}A")
            found_stars += 1
        else:
            print(f"   ❌ {first_name} {last_name:<15} Not found")
    
    print(f"\n📊 Coverage: {found_stars}/{len(target_players)} superstars found")
    
    # Check recent draft picks
    print(f"\n🆕 2024 DRAFT CLASS CHECK:")
    draft_2024 = [
        ('Zaccharie', 'Risacher'),
        ('Alexandre', 'Sarr'), 
        ('Reed', 'Sheppard'),
        ('Stephon', 'Castle'),
        ('Ron', 'Holland')
    ]
    
    found_rookies = 0
    for first_name, last_name in draft_2024:
        player_data = current_season[
            (current_season['firstName'].str.contains(first_name, case=False)) &
            (current_season['lastName'].str.contains(last_name, case=False))
        ]
        
        if len(player_data) > 0:
            games = len(player_data)
            avg_points = player_data['points'].mean()
            print(f"   ✅ {first_name} {last_name:<15} {games:2d} games, {avg_points:5.1f} PPG")
            found_rookies += 1
        else:
            print(f"   ❌ {first_name} {last_name:<15} Not found")
    
    print(f"\n📊 2024 rookies: {found_rookies}/{len(draft_2024)} found")

def main():
    """Run complete current player coverage analysis"""
    
    # Analyze current coverage
    stats = analyze_current_player_coverage()
    
    # Check specific players
    check_specific_players()
    
    # Summary
    print(f"\n🎯 SUMMARY FOR LIVE PLAYER PROPS")
    print("=" * 70)
    print(f"📊 Total current players: {stats['total_current_players']}")
    print(f"🌟 Stars ready for props: {stats['stars']}")
    print(f"🏀 Role players ready: {stats['role_players']}")
    print(f"🆕 Rookies in database: {stats['rookies']}")
    print(f"✅ Reliable for props: {stats['reliable_for_props']}")
    print(f"💰 High-value targets: {stats['prop_targets']}")
    
    print(f"\n💡 SYSTEM READINESS:")
    coverage_rate = stats['reliable_for_props'] / stats['total_current_players'] * 100
    if coverage_rate > 80:
        print(f"   🔥 EXCELLENT - {coverage_rate:.1f}% coverage")
    elif coverage_rate > 60:
        print(f"   ✅ GOOD - {coverage_rate:.1f}% coverage") 
    else:
        print(f"   ⚠️ NEEDS MORE DATA - {coverage_rate:.1f}% coverage")
    
    print(f"\n🚀 READY FOR LIVE PLAYER PROPS BETTING!")

if __name__ == "__main__":
    main()