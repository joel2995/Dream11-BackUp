import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class StrategyEngine:
    def __init__(self):
        self.pitch_conditions = {
            'batting_friendly': ['Mumbai', 'Bengaluru', 'Hyderabad'],
            'bowling_friendly': ['Chennai', 'Delhi', 'Kolkata'],
            'balanced': ['Ahmedabad', 'Pune', 'Lucknow']
        }
        
        self.weather_impact = {
            'clear': {'batting': 1.0, 'bowling': 1.0},
            'cloudy': {'batting': 0.9, 'bowling': 1.1},
            'humid': {'batting': 1.1, 'bowling': 0.9}
        }

    def calculate_pitch_factor(self, venue: str, role: str) -> float:
        """Calculate pitch impact factor based on venue and player role."""
        if venue in self.pitch_conditions['batting_friendly']:
            return 1.2 if role in ['BAT', 'ALL'] else 0.9
        elif venue in self.pitch_conditions['bowling_friendly']:
            return 1.2 if role in ['BOWL', 'ALL'] else 0.9
        return 1.0

    def calculate_form_impact(self, form_metrics: Dict) -> float:
        """Calculate form impact using comprehensive metrics."""
        recent_form = form_metrics.get('Recent Form', 0)
        form_trend = form_metrics.get('Form Trend', 0)
        consistency = form_metrics.get('Consistency Score', 0)
        
        # Weighted combination of form metrics
        form_impact = (
            recent_form * 0.4 +
            max(form_trend * 10, 0) * 0.3 +
            consistency * 0.3
        )
        return form_impact

    def apply_strategy_boosts(self, player_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                            match_no: int, performance_metrics: pd.DataFrame, priority_df: pd.DataFrame) -> pd.DataFrame:
        """Apply strategic boosts using comprehensive analysis."""
        
        # Get match conditions
        match_row = metadata_df[metadata_df['Match Number'] == match_no]
        if not match_row.empty:
            venue = match_row.iloc[0]['Venue'] if 'Venue' in match_row.columns else 'Unknown'
            toss_winner = match_row.iloc[0]['Toss Winner'] if 'Toss Winner' in match_row.columns else 'Unknown'
            toss_decision = match_row.iloc[0]['Decision'] if 'Decision' in match_row.columns else 'Unknown'
            home_team = match_row.iloc[0]['Home Team'] if 'Home Team' in match_row.columns else 'Unknown'
            weather = match_row.iloc[0]['Weather'] if 'Weather' in match_row.columns else 'clear'
        else:
            venue = 'Unknown'
            toss_winner = 'Unknown'
            toss_decision = 'Unknown'
            home_team = 'Unknown'
            weather = 'clear'

        # Merge performance metrics and priority
        player_df = player_df.merge(performance_metrics, on='Player', how='left')
        player_df = player_df.merge(priority_df[['Player', 'Priority']], on='Player', how='left')

        # Ensure 'Role' column exists (map from 'Player Role' if needed)
        if 'Role' not in player_df.columns:
            if 'Player Role' in player_df.columns:
                player_df['Role'] = player_df['Player Role']
            else:
                player_df['Role'] = 'Unknown'

        # Calculate strategic boosts
        for idx, player in player_df.iterrows():
            # 1. Enhanced Form Impact
            form_metrics = {
                'Recent Form': player['Recent Form'],
                'Form Trend': player['Form Trend'],
                'Consistency Score': player['Consistency Score']
            }
            player_df.loc[idx, 'Form Impact'] = self.calculate_form_impact(form_metrics) * 25

            # 2. Pitch and Conditions Impact
            pitch_factor = self.calculate_pitch_factor(venue, player['Role'])
            # Robust mapping for weather_factor
            role_key = player['Role'].upper()
            if role_key in ['BAT', 'ALL']:
                weather_role = 'batting'
            elif role_key == 'BOWL':
                weather_role = 'bowling'
            else:
                weather_role = 'batting'  # Default
            weather_factor = self.weather_impact.get(weather, self.weather_impact['clear'])[weather_role]
            player_df.loc[idx, 'Conditions Impact'] = (pitch_factor * weather_factor - 1) * 20

            # 3. Priority and Experience Boost
            priority_weight = {'High': 20, 'Medium': 10, 'Low': 5}
            player_df.loc[idx, 'Priority Boost'] = priority_weight.get(player['Priority'], 0)

            # 4. Home Advantage
            player_df.loc[idx, 'Home Boost'] = 10 if player['Team'] == home_team else 0

            # 5. Toss Advantage
            toss_boost = 0
            if player['Team'] == toss_winner:
                if toss_decision == 'Bat' and player['Role'] in ['BAT', 'ALL']:
                    toss_boost = 15
                elif toss_decision == 'Bowl' and player['Role'] in ['BOWL', 'ALL']:
                    toss_boost = 15
            player_df.loc[idx, 'Toss Boost'] = toss_boost

        # Calculate final strategic score
        player_df['Strategic Score'] = (
            player_df['Total Points'] +
            player_df['Form Impact'] +
            player_df['Conditions Impact'] +
            player_df['Priority Boost'] +
            player_df['Home Boost'] +
            player_df['Toss Boost']
        )

        return player_df
