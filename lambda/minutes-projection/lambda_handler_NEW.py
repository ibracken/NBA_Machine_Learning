"""
NBA Minutes Projection Lambda - REFACTORED
Generates projections for 5 models:
1. Complex Position Overlap (TRUE baseline + 2x multiplier)
2. Direct Position Exchange (TRUE baseline + exact position only)
3. Formula C Baseline (no injury handling)
4. SportsLine Baseline (pull from daily-predictions)
5. DailyFantasyFuel Baseline (DFF fantasy projections)
"""

import pandas as pd
import logging
import pytz
from datetime import datetime

# Import from our new modules
from s3_utils import load_from_s3, send_multi_model_notification, save_to_s3
from actuals_updater import update_actual_minutes
from projection_models import project_minutes_complex, project_minutes_direct, project_minutes_formula_c
from injury_system import (
    transition_beneficiaries_to_ex,
    calculate_team_injury_redistributions
)
from process_model import process_model
from lineup_optimizer import optimize_lineup

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

# Add console handler for local testing (Lambda provides its own handlers)
if not logger.handlers:
    # Console handler - only show INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - capture everything including DEBUG
    file_handler = logging.FileHandler('minutes_projection_debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# ==================== Main Lambda Handler ====================

def lambda_handler(event, context):
    """
    Generate projections for all 5 models OR update actual minutes

    Event parameters:
        - action: 'project' (default) or 'update_actuals'
        - date: Optional date override (YYYY-MM-DD format)
    """
    try:
        # Check event for action type
        action = event.get('action', 'project') if event else 'project'

        # Load common dependencies
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern).date()

        logger.info("Loading player stats from box scores...")
        box_scores = load_from_s3('data/box_scores/current.parquet')

        # Update actual minutes mode
        if action == 'update_actuals':
            target_date = event.get('date') if event and 'date' in event else None
            if target_date:
                target_date = datetime.strptime(target_date, '%Y-%m-%d').date()

            updated_count = update_actual_minutes(box_scores, target_date, today)

            return {
                'statusCode': 200,
                'body': f'Updated {updated_count} actual minutes records'
            }

        # Default: Generate projections
        logger.info(f"Starting minutes projection for {today}")

        if box_scores.empty:
            logger.error("No box scores data available")
            return {'statusCode': 400, 'body': 'No box scores data available'}

        # Get each player's most recent game for current stats
        box_scores['GAME_DATE'] = pd.to_datetime(box_scores['GAME_DATE'])
        box_scores_sorted = box_scores.sort_values(['PLAYER', 'GAME_DATE'], ascending=[True, False])
        player_stats = box_scores_sorted.groupby('PLAYER').first().reset_index()

        # Load LAST season's box scores for players with 0 games this season
        # BUT only for players on the current injury report (filters out free agents/retired players)
        logger.info("Loading previous season box scores for season-long injuries...")
        prev_season_box_scores = load_from_s3('data/box_scores/2024-25.parquet')

        if not prev_season_box_scores.empty:
            prev_season_box_scores['GAME_DATE'] = pd.to_datetime(prev_season_box_scores['GAME_DATE'])
            prev_season_sorted = prev_season_box_scores.sort_values(['PLAYER', 'GAME_DATE'], ascending=[True, False])
            prev_season_stats = prev_season_sorted.groupby('PLAYER').first().reset_index()

            # Mark these as previous season data
            prev_season_stats['FROM_PREV_SEASON'] = True
            player_stats['FROM_PREV_SEASON'] = False

            # Keep only players NOT in current season (season-long injuries)
            prev_season_only = prev_season_stats[~prev_season_stats['PLAYER'].isin(player_stats['PLAYER'])].copy()

            if not prev_season_only.empty:
                logger.info(f"Found {len(prev_season_only)} players with 0 games this season (using last season data)")
                player_stats = pd.concat([player_stats, prev_season_only], ignore_index=True)

        # Map columns to what the projection functions expect
        player_stats = player_stats.rename(columns={
            'Season_MIN_Avg': 'SEASON_AVG_MIN',
            'Last7_MIN_Avg': 'LAST_7_AVG_MIN',
            'MIN': 'PREV_GAME_MIN',
            'Career_MIN_Avg': 'CAREER_AVG_MIN',
            'TEAM_ABBREVIATION': 'TEAM'
        })

        # Calculate games played THIS season (will be 0 for prev season players)
        current_season = box_scores['SEASON'].max() if 'SEASON' in box_scores.columns else None
        if current_season:
            games_by_player = box_scores[box_scores['SEASON'] == current_season].groupby('PLAYER').size()
            player_stats['GAMES_PLAYED'] = player_stats['PLAYER'].map(games_by_player).fillna(0)
        else:
            player_stats['GAMES_PLAYED'] = 0

        # Load injury status from S3 (scraped by injury-scraper lambda)
        injury_data = load_from_s3('data/injuries/current.parquet')

        if not injury_data.empty:
            logger.info(f"Loaded {len(injury_data)} injury records from S3")

            # Merge injury status with player stats
            # ESTIMATED_INJURY_DATE helps with baseline calculations (estimated from last game)
            # RETURN_DATE_DT helps identify players returning today (shouldn't redistribute their minutes)
            merge_cols = ['PLAYER', 'STATUS']
            if 'ESTIMATED_INJURY_DATE' in injury_data.columns:
                merge_cols.append('ESTIMATED_INJURY_DATE')
            if 'RETURN_DATE_DT' in injury_data.columns:
                merge_cols.append('RETURN_DATE_DT')

            player_stats = player_stats.merge(
                injury_data[merge_cols],
                on='PLAYER',
                how='left'
            )

            # Update TEAM for injured players who may have been traded while injured
            # Problem: player_stats['TEAM'] = team from LAST GAME PLAYED (could be old team if traded while out)
            # Solution: Use injury_data['TEAM'] = CURRENT team from ESPN injury report
            # Catches: Season-long injuries + trade, AND mid-season injury + trade while still out
            if 'TEAM' in injury_data.columns:
                # STATUS.notna() = player has injury status (OUT, QUESTIONABLE, etc.) - checks ALL injured players
                injured_mask = player_stats['STATUS'].notna()

                if injured_mask.any():
                    # Get current team for each injured player (drop_duplicates in case player listed twice on injury report)
                    injury_teams = injury_data[['PLAYER', 'TEAM']].drop_duplicates(subset='PLAYER', keep='last')

                    for idx in player_stats[injured_mask].index:
                        player_name = player_stats.loc[idx, 'PLAYER']
                        injury_team_match = injury_teams[injury_teams['PLAYER'] == player_name]

                        if not injury_team_match.empty:
                            old_team = player_stats.loc[idx, 'TEAM']  # Team from last game (could be stale)
                            new_team = injury_team_match['TEAM'].iloc[0]  # .iloc[0] = first row's TEAM value (current team from ESPN)

                            # If mismatch detected → player was traded while injured
                            if old_team != new_team:
                                logger.info(f"{player_name} traded while injured: {old_team} -> {new_team}")
                                player_stats.loc[idx, 'PREV_TEAM'] = old_team  # Store old team before overwriting
                                player_stats.loc[idx, 'TEAM'] = new_team

            # Log injury summary
            injured_players = player_stats[player_stats['STATUS'].notna()]
            if not injured_players.empty:
                status_counts = injured_players['STATUS'].value_counts()
                logger.info(f"Injury status breakdown: {status_counts.to_dict()}")

                # Log estimated injury dates for OUT players
                out_players = injured_players[injured_players['STATUS'] == 'OUT']
                if 'ESTIMATED_INJURY_DATE' in out_players.columns:
                    with_dates = out_players['ESTIMATED_INJURY_DATE'].notna().sum()
                    logger.info(f"OUT players with estimated injury dates: {with_dates}/{len(out_players)}")
        else:
            logger.warning("No injury data available in S3 - all players treated as healthy")
            logger.warning("Make sure injury-scraper lambda has run successfully")

        # Load position data from daily-predictions (has salary/position from DraftKings)
        logger.info("Loading position data from daily predictions...")
        try:
            daily_preds = load_from_s3('data/daily_predictions/current.parquet')
            if not daily_preds.empty and 'POSITION' in daily_preds.columns:
                # Get most recent position for each player
                position_map = daily_preds[['PLAYER', 'POSITION']].drop_duplicates(subset='PLAYER', keep='last')

                # Merge positions
                if 'POSITION' in player_stats.columns:
                    player_stats = player_stats.drop(columns=['POSITION'])

                player_stats = player_stats.merge(position_map, on='PLAYER', how='left')

                # Fill missing with UNKNOWN
                player_stats['POSITION'] = player_stats['POSITION'].fillna('UNKNOWN')

                filled_positions = (player_stats['POSITION'] != 'UNKNOWN').sum()
                logger.info(f"Loaded positions for {filled_positions}/{len(player_stats)} players from daily predictions")
            else:
                logger.warning("Daily predictions unavailable - using UNKNOWN for positions")
                player_stats['POSITION'] = 'UNKNOWN'
        except Exception as e:
            logger.warning(f"Could not load positions from daily predictions: {e}")
            player_stats['POSITION'] = player_stats.get('POSITION', 'UNKNOWN')

        logger.info(f"Loaded stats for {len(player_stats)} players")

        # Filter out free agents: Previous season players NOT on injury report
        # These are retired/unsigned players who shouldn't be projected
        prev_season_mask = player_stats.get('FROM_PREV_SEASON', False) == True
        if prev_season_mask.any():
            if not injury_data.empty:
                injured_players = set(injury_data['PLAYER'].unique())
                free_agent_mask = prev_season_mask & ~player_stats['PLAYER'].isin(injured_players)

                if free_agent_mask.any():
                    free_agents = player_stats[free_agent_mask]['PLAYER'].head(10).tolist()
                    free_agent_count = free_agent_mask.sum()
                    logger.info(f"Filtering out {free_agent_count} free agents/retired players (0 games this season, not on injury report): {free_agents}...")
                    player_stats = player_stats[~free_agent_mask].copy()
            else:
                # No injury data - remove ALL previous season players
                free_agent_count = prev_season_mask.sum()
                player_stats = player_stats[~prev_season_mask].copy()
                logger.warning(f"No injury data - filtered out {free_agent_count} previous season players")

        logger.info(f"After filtering free agents: {len(player_stats)} players remaining")

        # Load injury context (shared by complex and direct models)
        injury_context_complex = load_from_s3('injury_context/complex_position_overlap.parquet')
        injury_context_direct = load_from_s3('injury_context/direct_position_only.parquet')

        # Transition beneficiaries whose injured players have returned: BENEFICIARY → EX_BENEFICIARY
        # If a player was BENEFICIARY_OF someone who's no longer in injury_data, they returned
        if not injury_data.empty:
            currently_injured = set(injury_data['PLAYER'].unique())
        else:
            currently_injured = set()

        injury_context_complex = transition_beneficiaries_to_ex(injury_context_complex, currently_injured, today)
        injury_context_direct = transition_beneficiaries_to_ex(injury_context_direct, currently_injured, today)

        if injury_context_complex.empty:
            injury_context_complex = pd.DataFrame(columns=[
                'PLAYER', 'TEAM', 'STATUS', 'INJURY_DATE', 'RETURN_DATE',
                'TRUE_BASELINE', 'BENEFICIARY_OF', 'UPDATED_DATE'
            ])

        if injury_context_direct.empty:
            injury_context_direct = pd.DataFrame(columns=[
                'PLAYER', 'TEAM', 'STATUS', 'INJURY_DATE', 'RETURN_DATE',
                'TRUE_BASELINE', 'BENEFICIARY_OF', 'UPDATED_DATE'
            ])

        # Load games log (used for confidence tracking after injuries)
        # Games log tracks when players returned from injury
        games_log = box_scores[['PLAYER', 'GAME_DATE']].copy()
        logger.info(f"Loaded games log with {len(games_log)} game records")

        teams = player_stats['TEAM'].unique()

        # ========== Model 1: Complex Position Overlap ==========
        logger.info("Generating projections: Complex Position Overlap")
        projections_complex = []
        injury_ctx_complex = injury_context_complex.copy()
        logged_beneficiaries_complex = set()  # Track (player, injured_player) pairs to avoid spam

        for team in teams:
            team_players = player_stats[player_stats['TEAM'] == team].copy()

            # Calculate injury redistributions ONCE per team (avoids redundant calculations)
            from projection_models import position_overlap_complex
            team_redistributions, injury_ctx_complex = calculate_team_injury_redistributions(
                team_players, injury_ctx_complex, today, box_scores,
                position_overlap_func=position_overlap_complex,
                use_multiplier=True
            )

            for idx, player in team_players.iterrows():
                projected_min, confidence, injury_ctx_complex = project_minutes_complex(
                    player, team_players, injury_ctx_complex, games_log, today, box_scores, injury_data,
                    team_injury_redistributions=team_redistributions
                )
                projections_complex.append({
                    'DATE': today,
                    'PLAYER': player['PLAYER'],
                    'TEAM': player['TEAM'],
                    'POSITION': player.get('POSITION', 'UNKNOWN'),
                    'PROJECTED_MIN': round(projected_min, 1),
                    'ACTUAL_MIN': None,
                    'ACTUAL_FP': None,
                    'CONFIDENCE': confidence
                })

        df_complex, lineup_complex = process_model(
            "Complex Position Overlap",
            "model_comparison/complex_position_overlap",
            projections_complex,
            daily_preds,
            today,
            box_scores,
            injury_context=injury_ctx_complex
        )

        # ========== Model 2: Direct Position Exchange ==========
        logger.info("Generating projections: Direct Position Exchange")
        projections_direct = []
        injury_ctx_direct = injury_context_direct.copy()

        for team in teams:
            team_players = player_stats[player_stats['TEAM'] == team].copy()

            # Calculate injury redistributions ONCE per team (avoids redundant calculations)
            from projection_models import position_overlap_exact
            team_redistributions, injury_ctx_direct = calculate_team_injury_redistributions(
                team_players, injury_ctx_direct, today, box_scores,
                position_overlap_func=position_overlap_exact,
                use_multiplier=False
            )

            for idx, player in team_players.iterrows():
                projected_min, confidence, injury_ctx_direct = project_minutes_direct(
                    player, team_players, injury_ctx_direct, games_log, today, box_scores, injury_data,
                    team_injury_redistributions=team_redistributions
                )
                projections_direct.append({
                    'DATE': today,
                    'PLAYER': player['PLAYER'],
                    'TEAM': player['TEAM'],
                    'POSITION': player.get('POSITION', 'UNKNOWN'),
                    'PROJECTED_MIN': round(projected_min, 1),
                    'ACTUAL_MIN': None,
                    'ACTUAL_FP': None,
                    'CONFIDENCE': confidence
                })

        df_direct, lineup_direct = process_model(
            "Direct Position Exchange",
            "model_comparison/direct_position_only",
            projections_direct,
            daily_preds,
            today,
            box_scores,
            injury_context=injury_ctx_direct
        )

        # ========== Model 3: Formula C Baseline ==========
        logger.info("Generating projections: Formula C Baseline")
        projections_formula_c = []

        for _, player in player_stats.iterrows():
            projected_min, confidence = project_minutes_formula_c(player, today)
            projections_formula_c.append({
                'DATE': today,
                'PLAYER': player['PLAYER'],
                'TEAM': player['TEAM'],
                'POSITION': player.get('POSITION', 'UNKNOWN'),
                'PROJECTED_MIN': round(projected_min, 1),
                'ACTUAL_MIN': None,
                'ACTUAL_FP': None,
                'CONFIDENCE': confidence
            })

        df_formula_c, lineup_formula_c = process_model(
            "Formula C Baseline",
            "model_comparison/formula_c_baseline",
            projections_formula_c,
            daily_preds,
            today,
            box_scores
        )

        # ========== Model 4: SportsLine Baseline ==========
        logger.info("Generating projections: SportsLine Baseline")

        # Use already-loaded daily_preds (contains SportsLine PROJECTED_MIN)
        sportsline_data = daily_preds

        projections_sportsline = []
        if not sportsline_data.empty:
            # Filter for today's games
            todays_sportsline = sportsline_data[sportsline_data['GAME_DATE'] == today].copy()

            logger.info(f"Found {len(todays_sportsline)} SportsLine projections for {today}")

            # Merge with player_stats to get TEAM (daily-predictions doesn't have TEAM column)
            if not player_stats.empty:
                team_mapping = player_stats[['PLAYER', 'TEAM']].drop_duplicates(subset='PLAYER', keep='last')
                todays_sportsline = todays_sportsline.merge(team_mapping, on='PLAYER', how='left')
                todays_sportsline['TEAM'] = todays_sportsline['TEAM'].fillna('UNKNOWN')

                matched = todays_sportsline['TEAM'].notna().sum()
                logger.info(f"Matched {matched}/{len(todays_sportsline)} SportsLine players with teams")
            else:
                todays_sportsline['TEAM'] = 'UNKNOWN'

            # Convert to list of dicts for consistency
            for _, row in todays_sportsline.iterrows():
                projections_sportsline.append({
                    'DATE': today,
                    'PLAYER': row['PLAYER'],
                    'TEAM': row.get('TEAM', 'UNKNOWN'),
                    'POSITION': row.get('POSITION', 'UNKNOWN'),
                    'PROJECTED_MIN': row.get('PROJECTED_MIN', 0),
                    'ACTUAL_MIN': row.get('ACTUAL_MIN'),  # Populated after games
                    'ACTUAL_FP': None,
                    'CONFIDENCE': 'HIGH'  # SportsLine is industry standard
                })

        # Only process if we have SportsLine data
        if projections_sportsline:
            df_sportsline, lineup_sportsline = process_model(
                "SportsLine Baseline",
                "model_comparison/sportsline_baseline",
                projections_sportsline,
                daily_preds,
                today,
                box_scores
            )
        else:
            df_sportsline = pd.DataFrame()
            lineup_sportsline = pd.DataFrame()

        # ========== Model 5: DailyFantasyFuel Baseline ==========
        logger.info("Generating lineup: DailyFantasyFuel Baseline")

        # Use already-loaded daily_preds (contains DFF PPG_PROJECTION)
        dff_data = daily_preds
        lineup_dff = pd.DataFrame()

        if not dff_data.empty:
            # Filter for today's games
            todays_dff = dff_data[dff_data['GAME_DATE'] == today].copy()

            logger.info(f"Found {len(todays_dff)} DailyFantasyFuel projections for {today}")

            # Merge with player_stats to get TEAM (daily-predictions doesn't have TEAM column)
            if not player_stats.empty:
                team_mapping = player_stats[['PLAYER', 'TEAM']].drop_duplicates(subset='PLAYER', keep='last')
                todays_dff = todays_dff.merge(team_mapping, on='PLAYER', how='left')
                todays_dff['TEAM'] = todays_dff['TEAM'].fillna('UNKNOWN')

                matched = todays_dff['TEAM'].notna().sum()
                logger.info(f"Matched {matched}/{len(todays_dff)} DFF players with teams")
            else:
                todays_dff['TEAM'] = 'UNKNOWN'

            # Prepare for lineup optimization - DFF already has fantasy points (PPG_PROJECTION)
            todays_dff['PROJECTED_FP'] = todays_dff['PPG_PROJECTION']  # Use DFF's fantasy projection
            todays_dff['PROJECTED_MIN'] = 0  # DFF doesn't provide minutes - placeholder only

            # Select only required columns (optimize_lineup will merge SALARY from daily_preds)
            # Don't include SALARY here to avoid conflict when optimize_lineup merges it
            dff_projections = todays_dff[['PLAYER', 'TEAM', 'POSITION', 'PROJECTED_MIN', 'PROJECTED_FP']].copy()

            # Optimize lineup using DFF projections
            lineup_dff = optimize_lineup(dff_projections, daily_preds, today)
            if not lineup_dff.empty:
                existing_lineup = load_from_s3('model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet')
                if not existing_lineup.empty:
                    # Convert DATE to date type for proper comparison
                    existing_lineup['DATE'] = pd.to_datetime(existing_lineup['DATE']).dt.date
                    existing_lineup = existing_lineup[existing_lineup['DATE'] != today]
                    lineup_dff = pd.concat([existing_lineup, lineup_dff], ignore_index=True)
                save_to_s3(lineup_dff, 'model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet')
                logger.info(f"DailyFantasyFuel lineup saved: {len(lineup_dff[lineup_dff['DATE'] == today])} players")

        # === NEW: Send Notification ===
        # Collect only today's data for the email
        lineups_to_send = {
            "1. Complex Position Overlap": lineup_complex[lineup_complex['DATE'] == today] if not lineup_complex.empty else pd.DataFrame(),
            "2. Direct Position Only": lineup_direct[lineup_direct['DATE'] == today] if not lineup_direct.empty else pd.DataFrame(),
            "3. Formula C Baseline": lineup_formula_c[lineup_formula_c['DATE'] == today] if not lineup_formula_c.empty else pd.DataFrame(),
            "4. SportsLine Baseline": lineup_sportsline[lineup_sportsline['DATE'] == today] if not lineup_sportsline.empty else pd.DataFrame(),
            "5. DailyFantasyFuel": lineup_dff[lineup_dff['DATE'] == today] if not lineup_dff.empty else pd.DataFrame()
        }

        # Send the email
        send_multi_model_notification(lineups_to_send, today)
        # ==============================

        logger.info(f"Successfully generated projections for all 5 models ({len(projections_complex)} players)")

        return {
            'statusCode': 200,
            'body': f'Generated 5 model projections for {len(projections_complex)} players on {today}'
        }

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}'
        }


# For local testing
if __name__ == "__main__":
    result = lambda_handler({}, None)
    print(result)
