import pandas as pd
import logging
from recent_form_generator import PlayerPerformanceAnalyzer
from data_standardizer import DataStandardizer
from fantasy_point_calculator import calculate_total_points
from model_predictor import ModelPredictor
from strategy_engine import StrategyEngine

class TeamSelector:
    def __init__(self, match_data_path, squad_data_path, match_metadata_path, credit_data_path):
        try:
            logging.info("Initializing TeamSelector...")
            
            # Initialize file paths
            self.match_data_path = match_data_path
            self.squad_data_path = squad_data_path
            self.match_metadata_path = match_metadata_path
            self.credit_data_path = credit_data_path
            
            # Load data files
            logging.info("Loading data files...")
            self.match_data = pd.read_excel(self.match_data_path)
            self.squad_data = pd.read_excel(self.squad_data_path, sheet_name=None)  # Load all sheets as a dictionary
            self.match_metadata = pd.read_excel(self.match_metadata_path)
            self.credit_data = pd.read_excel(self.credit_data_path)
            
            # Standardize column names
            self.match_data.rename(columns={
                'Match no': 'Match Number',
                'Player': 'Player',
                'Runs': 'Runs',
                'Balls': 'Balls Faced',
                'Fours': 'Fours',
                'Sixes': 'Sixes',
                'Strike Rate': 'Strike Rate',
                'Overs': 'Overs',
                'Runs Conceded': 'Runs Conceded',
                'Wickets': 'Wickets',
                'Economy': 'Economy',
                'Catches': 'Catches',
                'Stumping': 'Stumpings',
                'Run Out (Direct)': 'Run Outs (Direct)',
                'Run Out (Indirect)': 'Run Outs (Assist)',
                'Player Role': 'Player Role',
                'Team': 'Team',
                'Credits': 'Credits'
            }, inplace=True)

            self.credit_data.rename(columns={
                'Player Name': 'Player',
                'Player Type': 'Player Role',
                'Team': 'Team',
                'Credits': 'Credits'
            }, inplace=True)

            for sheet_name, df in self.squad_data.items():
                self.squad_data[sheet_name].rename(columns={
                    'Player Name': 'Player',
                    'Player Type': 'Player Role',
                    'Team': 'Team',
                    'IsPlaying': 'IsPlaying',
                    'lineupOrder': 'LineupOrder'
                }, inplace=True)

            self.match_metadata.rename(columns={
                'Match No.': 'Match Number',
                'Date': 'Date',
                'Venue': 'Venue',
                'Home Team': 'Home Team',
                'Away Team': 'Away Team',
                'Toss Winner': 'Toss Winner',
                'Toss Decision': 'Toss Decision',
                'Pitch Condition': 'Pitch Condition',
                'Match Result': 'Match Result',
                'Toss Winner is Match Winner': 'Toss Winner is Match Winner',
                'Winning Team': 'Winning Team'
            }, inplace=True)

            logging.info("Data files loaded and standardized successfully.")
            
            # Initialize other components
            logging.info("Initializing components...")
            self.performance_analyzer = PlayerPerformanceAnalyzer()
            self.model_predictor = ModelPredictor()
            self.strategy_engine = StrategyEngine()
            logging.info("Components initialized successfully.")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to initialize TeamSelector: {e}")

    def load_squad_for_match(self, match_id):
        try:
            logging.info(f"Loading squad data for match ID {match_id}...")
            sheet_name = f"Match_{match_id}"
            if sheet_name not in self.squad_data:
                raise ValueError(f"Sheet '{sheet_name}' not found in squad data.")
            return self.squad_data[sheet_name]
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load squad data for match ID {match_id}: {e}")

    def predict(self, match_id, home_team, away_team, venue, toss_winner, toss_decision):
        try:
            logging.info("Starting prediction process...")
            # Try to load squad data for the match
            try:
                squad_df = self.load_squad_for_match(match_id)
                squad_loaded = True
            except Exception as squad_exc:
                logging.warning(f"Squad data for match ID {match_id} not found: {squad_exc}")
                squad_loaded = False
                squad_df = None

            # If squad data is loaded, proceed as before
            if squad_loaded and squad_df is not None and not squad_df.empty:
                # Filter playing XI
                squad_df = squad_df[squad_df['IsPlaying'].str.upper() == "PLAYING"].copy()
            else:
                # Fallback: Use best 11 + 4 backups from home and away teams
                logging.info("Falling back to best 11 + 4 backups from specified home and away teams.")
                combined_df = self.credit_data[(self.credit_data['Team'].str.upper() == str(home_team).upper()) | (self.credit_data['Team'].str.upper() == str(away_team).upper())].copy()
                required_columns = [
                    'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 'Overs',
                    'Runs Conceded', 'Wickets', 'Economy', 'Catches', 'Stumpings',
                    'Run Outs (Direct)', 'Run Outs (Assist)', 'Maidens', 'Runs Given', 'IsPlaying', 'LineupOrder'
                ]
                for col in required_columns:
                    if col not in combined_df.columns:
                        combined_df[col] = 0
                combined_df['IsPlaying'] = 'PLAYING'
                squad_df = combined_df

            # --- Custom logic for X_FACTOR_SUBSTITUTE inclusion ---
            x_factor_players = {
                'MI': 'Rohit Sharma',
                'RCB': 'Devdutt Padikkal',
                'DC': 'Abishek Porel',
                'CSK': 'Shivam Dube',
                'GT': 'Sherfane Rutherford',
                'KKR': 'Angkrish Raghuvanshi',
                'RR': 'Vaibhav Suryavanshi'
            }
            # For each team, if the player is present and IsPlaying is X_FACTOR_SUBSTITUTE, treat as PLAYING
            for team, player in x_factor_players.items():
                mask = (squad_df['Team'].str.upper() == team) & (squad_df['Player'].str.strip().str.upper() == player.upper())
                if mask.any():
                    idx = squad_df[mask & (squad_df['IsPlaying'].str.upper() == 'X_FACTOR_SUBSTITUTE')].index
                    squad_df.loc[idx, 'IsPlaying'] = 'PLAYING'

            # Continue with the rest of the logic
            # Debug information is logged to file but not printed to console
            logging.debug("Squad DataFrame columns: %s", squad_df.columns.tolist())
            logging.debug("Unique Players in squad_df: %s", squad_df['Player'].unique())
            logging.debug("Unique Teams in squad_df: %s", squad_df['Team'].unique())

            match_row = self.match_metadata[self.match_metadata['Match Number'] == match_id]
            if not match_row.empty:
                actual_home_team = match_row.iloc[0]['Home Team'] if 'Home Team' in match_row.columns else home_team
                actual_away_team = match_row.iloc[0]['Away Team'] if 'Away Team' in match_row.columns else away_team
                if (str(home_team).upper() != str(actual_home_team).upper()) or (str(away_team).upper() != str(actual_away_team).upper()):
                    logging.warning(f"User-provided home/away teams do not match metadata. Using actual teams from metadata: {actual_home_team} vs {actual_away_team}")
                    home_team = actual_home_team
                    away_team = actual_away_team

            df = squad_df.merge(
                self.credit_data[['Player', 'Credits', 'Team', 'Priority']],
                on=['Player', 'Team'],
                how='left'
            )
            if 'Credits_y' in df.columns:
                df['Credits'] = df['Credits_y']
                df = df.drop(columns=['Credits_x', 'Credits_y'], errors='ignore')
            if 'Credits' not in df.columns:
                logging.debug("Merged DataFrame preview: %s", df.head())
                raise ValueError("[ERROR] 'Credits' column is missing after merging squad and credit data.")
            df = df.dropna(subset=['Credits'])
            df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce')
            required_columns = [
                'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Strike Rate', 'Overs',
                'Runs Conceded', 'Wickets', 'Economy', 'Catches', 'Stumpings',
                'Run Outs (Direct)', 'Run Outs (Assist)', 'Maidens', 'Runs Given'
            ]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            match_row = self.match_metadata[self.match_metadata['Match Number'] == match_id]
            if not match_row.empty:
                venue_value = match_row.iloc[0]['Venue'] if 'Venue' in match_row.columns else 'Unknown'
                df['Venue'] = venue_value
                if 'Home Team' not in df.columns and 'Home Team' in match_row.columns:
                    df['Home Team'] = match_row.iloc[0]['Home Team']
                if 'Away Team' not in df.columns and 'Away Team' in match_row.columns:
                    df['Away Team'] = match_row.iloc[0]['Away Team']
            else:
                venue_value = 'Unknown'
                df['Venue'] = 'Unknown'
                if 'Home Team' not in df.columns:
                    df['Home Team'] = 'Unknown'
                if 'Away Team' not in df.columns:
                    df['Away Team'] = 'Unknown'

            # --- Team selection constraints logic ---
            # Only consider players marked as PLAYING
            playing_df = df[df['IsPlaying'].str.upper() == 'PLAYING'].copy()
            # Assign player roles
            role_map = {
                'WK': 'wk', 'WICKETKEEPER': 'wk',
                'BAT': 'bat', 'BATSMAN': 'bat',
                'ALL': 'all', 'ALLROUNDER': 'all',
                'BOWL': 'bowl', 'BOWLER': 'bowl'
            }
            playing_df['Role'] = playing_df['Player Role'].str.upper().map(role_map).fillna('bat')
            
            # Merge Priority from credit_data if not already present
            if 'Priority' not in playing_df.columns:
                playing_df = playing_df.merge(
                    self.credit_data[['Player', 'Team', 'Priority']],
                    on=['Player', 'Team'],
                    how='left'
                )
                playing_df['Priority'] = pd.to_numeric(playing_df['Priority'], errors='coerce').fillna(3)
            
            # Group by team
            team_counts = playing_df['Team'].value_counts().to_dict()
            
            # First, select all Priority 1 players from both teams
            priority_1_players = playing_df[playing_df['Priority'] == 1].copy()
            priority_2_players = playing_df[playing_df['Priority'] == 2].copy()
            priority_3_players = playing_df[playing_df['Priority'] == 3].copy()
            
            # Initialize selection lists and counters
            selected = []
            team_player_counts = {team: 0 for team in playing_df['Team'].unique()}
            role_counts = {'wk': 0, 'bat': 0, 'all': 0, 'bowl': 0}
            total_credits = 0
            
            # First add all Priority 1 players
            for _, row in priority_1_players.iterrows():
                team = row['Team']
                role = row['Role']
                credits = row['Credits']
                
                # Check if adding this player violates max per team
                if team_player_counts[team] >= 7:
                    continue
                
                selected.append(row)
                team_player_counts[team] += 1
                role_counts[role] += 1
                total_credits += credits
            
            # Then add Priority 2 players until we reach 11 players
            for _, row in priority_2_players.iterrows():
                if len(selected) >= 11:
                    break
                    
                team = row['Team']
                role = row['Role']
                credits = row['Credits']
                
                # Check if adding this player violates max per team
                if team_player_counts[team] >= 7:
                    continue
                
                selected.append(row)
                team_player_counts[team] += 1
                role_counts[role] += 1
                total_credits += credits
            
            # Finally add Priority 3 players if needed to reach 11 players
            for _, row in priority_3_players.iterrows():
                if len(selected) >= 11:
                    break
                    
                team = row['Team']
                role = row['Role']
                credits = row['Credits']
                
                # Check if adding this player violates max per team
                if team_player_counts[team] >= 7:
                    continue
                
                # Check role constraints for the last player
                if len(selected) == 10:
                    needed_roles = [r for r, c in role_counts.items() if c == 0]
                    if needed_roles and role not in needed_roles:
                        continue
                        
                selected.append(row)
                team_player_counts[team] += 1
                role_counts[role] += 1
                total_credits += credits
            
            # Ensure min 4 per team
            for team in team_player_counts:
                if team_player_counts[team] < 4:
                    # Try to add more from this team if possible
                    candidates = playing_df[(playing_df['Team'] == team) & (~playing_df.index.isin([s.name for s in selected]))]
                    for _, row in candidates.iterrows():
                        if len(selected) >= 11:
                            break
                        selected.append(row)
                        team_player_counts[team] += 1
                        role_counts[row['Role']] += 1
                        total_credits += row['Credits']
            # Final team (main 11)
            final_team = pd.DataFrame(selected).head(11)
            
            # --- Backup selection logic ---
            # Select backups based on Priority values
            backup_candidates = playing_df[~playing_df.index.isin(final_team.index)]
            
            # First select Priority 1 players not already in main team
            priority_1_backups = backup_candidates[backup_candidates['Priority'] == 1]
            # Then Priority 2 players
            priority_2_backups = backup_candidates[backup_candidates['Priority'] == 2]
            # Then Priority 3 players
            priority_3_backups = backup_candidates[backup_candidates['Priority'] == 3]
            
            # Combine in priority order
            backups_by_priority = pd.concat([priority_1_backups, priority_2_backups, priority_3_backups])
            backups = backups_by_priority.head(4).copy()
            backups['RoleFlag'] = 'Backup'  # Explicitly set RoleFlag for backups
            
            # Add RoleFlag to main team
            final_team['RoleFlag'] = 'Main'
            # Combine for output
            output_team = pd.concat([final_team, backups], ignore_index=True)
            # --- Captain and Vice-Captain eligibility filtering ---
            # Only allow C/VC from WK, BAT, ALL roles (not BOWL)
            eligible_roles = ['wk', 'bat', 'all']
            
            # Filter eligible C/VC from main team only, not backups, and not BOWL
            main_eligible = final_team[final_team['Role'].isin(eligible_roles)]
            
            # Further filter to only include players with C=1 in credits dataset for Captain
            # and VC=1 in credits dataset for Vice-Captain
            captain_eligible_players = []
            vice_captain_eligible_players = []
            
            # Create a mapping of player+team to C and VC values from credit_data
            c_vc_mapping = {}
            for _, row in self.credit_data.iterrows():
                player_key = (row['Player'], row['Team'])
                c_vc_mapping[player_key] = {'C': row.get('C', 0), 'VC': row.get('VC', 0)}
            
            # Filter eligible players based on C and VC values
            for _, player in main_eligible.iterrows():
                player_key = (player['Player'], player['Team'])
                if player_key in c_vc_mapping:
                    if c_vc_mapping[player_key]['C'] == 1:
                        captain_eligible_players.append(player)
                    if c_vc_mapping[player_key]['VC'] == 1:
                        vice_captain_eligible_players.append(player)
            
            # Convert to DataFrames
            captain_eligible = pd.DataFrame(captain_eligible_players) if captain_eligible_players else pd.DataFrame()
            vice_captain_eligible = pd.DataFrame(vice_captain_eligible_players) if vice_captain_eligible_players else pd.DataFrame()
            
            # If no eligible players found with C=1 or VC=1, fall back to original logic but still restrict to BAT, WK, ALL
            if captain_eligible.empty or vice_captain_eligible.empty:
                logging.warning("No players found with C=1 or VC=1 in credits dataset. Falling back to role-based filtering.")
                
                # Get home and away team players
                home_team_players = main_eligible[main_eligible['Team'] == home_team]
                away_team_players = main_eligible[main_eligible['Team'] == away_team]
                
                # Prioritize players with Priority 1 for Captain and Vice-Captain
                home_priority_1 = home_team_players[home_team_players['Priority'] == 1]
                away_priority_1 = away_team_players[away_team_players['Priority'] == 1]
                
                # If no Priority 1 players, try Priority 2
                if home_priority_1.empty:
                    home_priority_1 = home_team_players[home_team_players['Priority'] == 2]
                if away_priority_1.empty:
                    away_priority_1 = away_team_players[away_team_players['Priority'] == 2]
                
                # If still no eligible players, use all available players from each team
                if home_priority_1.empty:
                    home_priority_1 = home_team_players
                if away_priority_1.empty:
                    away_priority_1 = away_team_players
            
            # If we have eligible players from C=1 and VC=1 in credits dataset, use them
            if not captain_eligible.empty and not vice_captain_eligible.empty:
                # Sort by credits and performance metrics
                captain_eligible = captain_eligible.sort_values(['Credits', 'Runs', 'Wickets'], ascending=[False, False, False])
                vice_captain_eligible = vice_captain_eligible.sort_values(['Credits', 'Runs', 'Wickets'], ascending=[False, False, False])
                
                # Select captain and vice-captain
                captain = captain_eligible.iloc[0]
                
                # Make sure vice-captain is different from captain
                vice_captain_candidates = vice_captain_eligible[vice_captain_eligible['Player'] != captain['Player']]
                if not vice_captain_candidates.empty:
                    vice_captain = vice_captain_candidates.iloc[0]
                else:
                    # If no other eligible vice-captain, use the next best captain candidate
                    vice_captain = captain_eligible.iloc[1] if len(captain_eligible) > 1 else captain
                
                # Assign C and VC
                output_team['C'] = output_team['Player'] == captain['Player']
                output_team['VC'] = output_team['Player'] == vice_captain['Player']
                
                logging.debug(f"Captain selected from credits dataset C=1: {captain['Player']} ({captain['Team']})")
                logging.debug(f"Vice-Captain selected from credits dataset VC=1: {vice_captain['Player']} ({vice_captain['Team']})")
            
            # Fall back to the original logic if needed
            elif not home_priority_1.empty and not away_priority_1.empty:
                # Sort by credits and performance metrics
                if not home_priority_1.empty:
                    home_priority_1 = home_priority_1.sort_values(['Credits', 'Runs', 'Wickets'], ascending=[False, False, False])
                if not away_priority_1.empty:
                    away_priority_1 = away_priority_1.sort_values(['Credits', 'Runs', 'Wickets'], ascending=[False, False, False])
                
                # Decide which team gets Captain based on Priority 1 player count
                home_p1_count = len(final_team[(final_team['Team'] == home_team) & (final_team['Priority'] == 1)])
                away_p1_count = len(final_team[(final_team['Team'] == away_team) & (final_team['Priority'] == 1)])
                
                if home_p1_count >= away_p1_count:
                    # Home team gets Captain, Away team gets Vice-Captain
                    captain = home_priority_1.iloc[0]
                    vice_captain = away_priority_1.iloc[0]
                else:
                    # Away team gets Captain, Home team gets Vice-Captain
                    captain = away_priority_1.iloc[0]
                    vice_captain = home_priority_1.iloc[0]
                
                # Assign C and VC
                output_team['C'] = output_team['Player'] == captain['Player']
                output_team['VC'] = output_team['Player'] == vice_captain['Player']
                
                logging.debug(f"Captain selected from {captain['Team']}: {captain['Player']} (fallback logic)")
                logging.debug(f"Vice-Captain selected from {vice_captain['Team']}: {vice_captain['Player']} (fallback logic)")
            elif not home_priority_1.empty:
                # Only home team has eligible players
                captain = home_priority_1.iloc[0]
                vice_captain = home_priority_1.iloc[1] if len(home_priority_1) > 1 else captain
                output_team['C'] = output_team['Player'] == captain['Player']
                output_team['VC'] = output_team['Player'] == vice_captain['Player']
                logging.warning(f"Both Captain and Vice-Captain selected from {home_team} due to lack of eligible players from {away_team} (fallback logic)")
            elif not away_priority_1.empty:
                # Only away team has eligible players
                captain = away_priority_1.iloc[0]
                vice_captain = away_priority_1.iloc[1] if len(away_priority_1) > 1 else captain
                output_team['C'] = output_team['Player'] == captain['Player']
                output_team['VC'] = output_team['Player'] == vice_captain['Player']
                logging.warning(f"Both Captain and Vice-Captain selected from {away_team} due to lack of eligible players from {home_team} (fallback logic)")
            else:
                # Fallback: assign C/VC to first two BAT/ALL/WK
                batallwk = final_team[final_team['Role'].isin(['wk','bat','all'])]
                if not batallwk.empty:
                    output_team['C'] = output_team['Player'] == batallwk.iloc[0]['Player']
                    output_team['VC'] = output_team['Player'] == batallwk.iloc[1]['Player'] if len(batallwk) > 1 else False
                    logging.warning("Captain and Vice-Captain assigned based on fallback logic due to lack of eligible players")
                else:
                    output_team['C'] = False
                    output_team['VC'] = False
                    logging.error("No eligible players found for Captain and Vice-Captain roles")
            # Log the summary (removed console print to avoid duplication)
            # The summary will be printed from main.py instead
            logging.info("\n[TEAM SELECTION SUMMARY]")
            logging.info(f"Team composition: {{'wk': {role_counts['wk']}, 'bat': {role_counts['bat']}, 'all': {role_counts['all']}, 'bowl': {role_counts['bowl']}}}")
            logging.info(f"Players from each team: {team_player_counts}")
            logging.info(f"Total credits used: {total_credits}")
            logging.info(f"Final 11: {final_team['Player'].tolist()}")
            logging.info(f"Backups: {backups['Player'].tolist()}")
            logging.info(f"Captain: {output_team[output_team['C']]['Player'].tolist()}")
            logging.info(f"Vice-Captain: {output_team[output_team['VC']]['Player'].tolist()}")
            return output_team.reset_index(drop=True)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise