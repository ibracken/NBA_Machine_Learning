"""
NBA Minutes Projection Lambda
Generates projections for 4 models:
1. Complex Position Overlap (TRUE baseline + 2x multiplier)
2. Direct Position Exchange (TRUE baseline + exact position only)
3. Formula C Baseline (no injury handling)
4. SportsLine Baseline (pull from daily-predictions)
"""

import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import io
import pickle
import json
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD

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

# S3 client
s3_client = boto3.client('s3')
# === NEW: Add SNS Client ===
sns_client = boto3.client('sns')
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:349928386418:lineup-optimizer-notifications' 
BUCKET_NAME = 'nba-prediction-ibracken'

# Algorithm constants
BENCH_OPPORTUNITY_CONSTANT = 0.1  # Small boost for deep bench players
EXACT_POSITION_MULTIPLIER = 2.0  # Direct backups get 2x weight (complex model only)
MAX_MINUTES = 37  # Cap individual player minutes
CONFIDENCE_CLEARANCE_GAMES = 3  # Games needed to clear LOW confidence


# ==================== S3 Helper Functions ====================

def load_from_s3(key):
    """Load parquet file from S3"""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(io.BytesIO(response['Body'].read()))
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"File not found: {key}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading {key}: {str(e)}")
        raise


def save_to_s3(df, key):
    """Save parquet file to S3"""
    try:
        # Convert date columns to datetime for parquet compatibility
        df = df.copy()
        date_columns = ['INJURY_DATE', 'RETURN_DATE', 'UPDATED_DATE', 'DATE', 'GAME_DATE']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=buffer.getvalue())
        logger.info(f"Saved to S3: {key}")
    except Exception as e:
        logger.error(f"Error saving {key}: {str(e)}")
        raise


def load_model_from_s3(key):
    """Load pickled model from S3"""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return pickle.loads(response['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Model not found: {key}")
        return None
    except Exception as e:
        logger.error(f"Error loading model {key}: {str(e)}")
        return None

# ==================== Notification Helper ====================

def send_multi_model_notification(lineups_dict, date):
    """
    Send a single email containing lineups from all 5 models
    """
    logger.info("Preparing SNS notification for all models")
    
    try:
        message_lines = [f"🏀 NBA Lineup Projections for {date} 🏀\n"]
        
        # Loop through each model in the dictionary
        for model_name, df in lineups_dict.items():
            message_lines.append(f"\n{'='*20} {model_name} {'='*20}")
            
            if df is None or df.empty:
                message_lines.append("No valid lineup generated for this model.\n")
                continue
                
            # Filter for today's lineup only (in case old ones are in the DF)
            # Ensure we are looking at the right date format
            df_today = df.copy()
            # Handle string vs date object comparison if necessary
            if not df_today.empty:
                # Calculate totals
                total_salary = df_today['SALARY'].sum()
                total_fp = df_today['PROJECTED_FP'].sum()
                
                message_lines.append(f"Salary: ${total_salary:,.0f} | Proj FP: {total_fp:.1f}")
                message_lines.append("-" * 50)
                
                # Sort by slot order for readability
                slot_order = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4, 'G': 5, 'F': 6, 'UTIL': 7}
                df_today['sort'] = df_today['SLOT'].map(slot_order)
                df_today = df_today.sort_values('sort')
                
                for _, row in df_today.iterrows():
                    message_lines.append(
                        f"{row['SLOT']:4} | {row['PLAYER']:<20} | "
                        f"${row['SALARY']:<5,.0f} | {row['PROJECTED_FP']:5.1f} FP | {row['PROJECTED_MIN']:4.1f} Min"
                    )
        
        full_message = "\n".join(message_lines)
        
        # Publish to SNS
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"NBA Projections ({date}): 5 Models Generated",
            Message=full_message
        )
        logger.info("SNS notification sent successfully")
        
    except Exception as e:
        logger.error(f"Failed to send SNS notification: {str(e)}")

# ==================== Position Overlap Functions ====================

def position_overlap_complex(pos1, pos2):
    """
    Complex position overlap: Adjacent positions can substitute
    Used for: Complex Position Overlap model
    """
    position_groups = [
        {'PG', 'SG'},      # Guards can swap
        {'SG', 'SF'},      # Wings can swap
        {'SF', 'PF'},      # Forwards can swap
        {'PF', 'C'},       # Bigs can swap
    ]

    if pos1 == pos2:
        return True

    for group in position_groups:
        if pos1 in group and pos2 in group:
            return True

    return False


def position_overlap_exact(pos1, pos2):
    """
    Exact position match only
    Used for: Direct Position Exchange model
    """
    return pos1 == pos2


# ==================== Fantasy Points Prediction ====================

def calculate_fp_features(box_scores, daily_predictions, today):
    """
    Extract FP features from box scores (matches supervised-learning training pipeline)
    """
    logger.info("Extracting FP features from box scores")

    # Ensure GAME_DATE is datetime
    box_scores = box_scores.copy()
    box_scores['GAME_DATE'] = pd.to_datetime(box_scores['GAME_DATE'])

    # Get most recent game for each player (has latest rolling averages)
    box_scores_sorted = box_scores.sort_values(['PLAYER', 'GAME_DATE'], ascending=[True, False])
    latest_stats = box_scores_sorted.groupby('PLAYER').first().reset_index()

    # Extract FP feature columns (match supervised-learning training)
    fp_cols = ['PLAYER', 'Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
               'Career_FP_Avg', 'Games_Played_Career', 'CLUSTER',
               'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg']

    # Only keep columns that exist
    available_cols = [col for col in fp_cols if col in latest_stats.columns]
    fp_features = latest_stats[available_cols].copy()

    # Calculate rest days from most recent game to today
    if 'GAME_DATE' in latest_stats.columns:
        latest_stats['REST_DAYS'] = (pd.Timestamp(today) - latest_stats['GAME_DATE']).dt.days
        latest_stats['REST_DAYS'] = latest_stats['REST_DAYS'].clip(0, 30)  # Cap at 30 days
        fp_features['REST_DAYS'] = latest_stats['REST_DAYS']
    else:
        fp_features['REST_DAYS'] = 3  # Default

    # Merge with daily predictions to get opponent and home/away info
    if not daily_predictions.empty:
        todays_games = daily_predictions[daily_predictions['GAME_DATE'] == today].copy()

        if not todays_games.empty and 'OPPONENT' in todays_games.columns:
            # Get opponent and home/away - KEEP OPPONENT AS STRING for one-hot encoding
            game_context = todays_games[['PLAYER', 'OPPONENT', 'IS_HOME']].copy()
            game_context['IS_HOME'] = game_context['IS_HOME'].fillna(0).astype(int)

            # Merge with FP features
            fp_features = fp_features.merge(game_context, on='PLAYER', how='left')

    # Fill missing OPPONENT with 'UNKNOWN' (match training)
    if 'OPPONENT' not in fp_features.columns:
        fp_features['OPPONENT'] = 'UNKNOWN'
    else:
        fp_features['OPPONENT'] = fp_features['OPPONENT'].fillna('UNKNOWN')

    # Fill missing IS_HOME with 0
    if 'IS_HOME' not in fp_features.columns:
        fp_features['IS_HOME'] = 0
    else:
        fp_features['IS_HOME'] = fp_features['IS_HOME'].fillna(0).astype(int)

    # Fill missing CLUSTER with 'CLUSTER_NAN' (match training)
    if 'CLUSTER' not in fp_features.columns:
        fp_features['CLUSTER'] = 'CLUSTER_NAN'
    else:
        fp_features['CLUSTER'] = fp_features['CLUSTER'].fillna('CLUSTER_NAN')

    logger.info(f"Extracted FP features for {len(fp_features)} players")

    return fp_features


def predict_fantasy_points(projections_df, box_scores, daily_predictions, today):
    """
    Add PROJECTED_FP column to projections using trained ML model
    Matches supervised-learning training pipeline with one-hot encoding

    Args:
        projections_df: DataFrame with PLAYER, PROJECTED_MIN columns
        box_scores: Historical box scores for feature calculation
        daily_predictions: Today's game context (opponent, home/away)
        today: Current date

    Returns:
        DataFrame with added PROJECTED_FP column
    """
    logger.info("Predicting fantasy points for projections")

    # Load trained FP model
    fp_model = load_model_from_s3('models/RFCluster.sav')

    if fp_model is None:
        logger.warning("FP prediction model not found - setting PROJECTED_FP to 0")
        projections_df['PROJECTED_FP'] = 0
        return projections_df

    # Load expected feature names from S3 (saved during training)
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key='models/RFCluster_feature_names.json')
        feature_names_data = json.loads(response['Body'].read())
        expected_features = feature_names_data['features']
        logger.info(f"Loaded {len(expected_features)} expected features from S3")
    except Exception as e:
        logger.warning(f"Could not load feature names from S3: {e}. Predictions may fail.")
        expected_features = None

    # Calculate FP features from historical data
    fp_features = calculate_fp_features(box_scores, daily_predictions, today)

    # Merge projections with FP features
    df = projections_df.merge(fp_features, on='PLAYER', how='left')

    # Use PROJECTED_MIN as MIN (match training which used actual MIN)
    df['MIN'] = df['PROJECTED_MIN']

    # Prepare features BEFORE one-hot encoding (match training order)
    base_feature_cols = ['Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
                         'Career_FP_Avg', 'Games_Played_Career', 'CLUSTER', 'MIN',
                         'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg',
                         'IS_HOME', 'OPPONENT', 'REST_DAYS']

    # Ensure all feature columns exist before filling
    for col in base_feature_cols:
        if col not in df.columns:
            if col == 'CLUSTER':
                df[col] = 'CLUSTER_NAN'
            elif col == 'OPPONENT':
                df[col] = 'UNKNOWN'
            else:
                df[col] = 0

    # Fill NaN values (match training pipeline)
    df['Last3_FP_Avg'] = df['Last3_FP_Avg'].fillna(0)
    df['Last7_FP_Avg'] = df['Last7_FP_Avg'].fillna(0)
    df['Season_FP_Avg'] = df['Season_FP_Avg'].fillna(0)
    df['Career_FP_Avg'] = df['Career_FP_Avg'].fillna(0)
    df['Games_Played_Career'] = df['Games_Played_Career'].fillna(0)
    df['Last7_MIN_Avg'] = df['Last7_MIN_Avg'].fillna(0)
    df['Season_MIN_Avg'] = df['Season_MIN_Avg'].fillna(0)
    df['Career_MIN_Avg'] = df['Career_MIN_Avg'].fillna(0)
    df['MIN'] = df['MIN'].fillna(0)
    df['IS_HOME'] = df['IS_HOME'].fillna(0).astype(int)
    df['REST_DAYS'] = df['REST_DAYS'].fillna(3).clip(0, 30)
    df['CLUSTER'] = df['CLUSTER'].fillna('CLUSTER_NAN')
    df['OPPONENT'] = df['OPPONENT'].fillna('UNKNOWN')

    # Set PROJECTED_FP to 0 for players with no FP history
    no_history_mask = (df['Season_FP_Avg'] == 0) & (df['Career_FP_Avg'] == 0)
    if no_history_mask.any():
        logger.info(f"Found {no_history_mask.sum()} players with no FP history - will set PROJECTED_FP to 0")

    # Extract features for one-hot encoding
    df_features = df[base_feature_cols].copy()

    # One-hot encode CLUSTER and OPPONENT (match training)
    df_features = pd.get_dummies(df_features, columns=['CLUSTER', 'OPPONENT'], drop_first=False)

    # Align columns with expected features from training
    if expected_features:
        # Add missing columns (fill with 0)
        for col in expected_features:
            if col not in df_features.columns:
                df_features[col] = 0

        # Keep only expected columns in the same order
        df_features = df_features[expected_features]
        logger.info(f"Aligned features: {len(df_features.columns)} columns match training")
    else:
        logger.warning("No expected features loaded - prediction may fail if columns don't match")

    # Make predictions
    X = df_features.values
    df['PROJECTED_FP'] = fp_model.predict(X)

    # Round to 1 decimal
    df['PROJECTED_FP'] = df['PROJECTED_FP'].round(1)

    # Override predictions for players with no history
    df.loc[no_history_mask, 'PROJECTED_FP'] = 0

    # Keep only original projection columns + PROJECTED_FP
    result_cols = [col for col in projections_df.columns] + ['PROJECTED_FP']
    result_df = df[result_cols].copy()

    logger.info(f"Predicted FP for {len(result_df)} players (avg: {result_df['PROJECTED_FP'].mean():.1f})")

    return result_df


# ==================== Lineup Optimization ====================

def is_eligible_for_slot(position, slot):
    """
    Determine if a player's position makes them eligible for a given slot

    Args:
        position: Player's position string (e.g., 'PG', 'PG/SG', 'SF/PF')
        slot: Slot name ('PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL')

    Returns:
        bool: True if eligible, False otherwise
    """
    if pd.isna(position):
        return False

    position = str(position).upper()
    positions = [p.strip() for p in position.split('/')]

    if slot == 'PG':
        return 'PG' in positions
    elif slot == 'SG':
        return 'SG' in positions
    elif slot == 'SF':
        return 'SF' in positions
    elif slot == 'PF':
        return 'PF' in positions
    elif slot == 'C':
        return 'C' in positions
    elif slot == 'G':
        return 'PG' in positions or 'SG' in positions
    elif slot == 'F':
        return 'SF' in positions or 'PF' in positions
    elif slot == 'UTIL':
        return True

    return False


def optimize_lineup(projections_df, daily_predictions, today):
    """
    Optimize DraftKings lineup for a specific model's projections

    Args:
        projections_df: DataFrame with PLAYER, POSITION, PROJECTED_MIN, PROJECTED_FP
        daily_predictions: DataFrame with SALARY information
        today: Current date

    Returns:
        DataFrame with optimal lineup (8 players)
    """
    logger.info(f"Optimizing lineup from {len(projections_df)} players")

    # Merge projections with salary data from daily_predictions
    df = projections_df.copy()

    if not daily_predictions.empty and 'SALARY' in daily_predictions.columns:
        # Filter for only today's games
        todays_games = daily_predictions[daily_predictions['GAME_DATE'] == today]

        if todays_games.empty:
            logger.warning(f"No games scheduled for {today} - cannot optimize lineup")
            return pd.DataFrame()

        salary_data = todays_games[['PLAYER', 'SALARY']].drop_duplicates(subset='PLAYER', keep='last')
        # Use inner merge to only include players with games today
        df = df.merge(salary_data, on='PLAYER', how='inner')
        logger.info(f"Filtered to {len(df)} players with games on {today}")
    else:
        logger.warning("No salary data available - cannot optimize lineup")
        return pd.DataFrame()

    # Filter out players with missing critical data
    df = df.dropna(subset=['PLAYER', 'POSITION', 'SALARY', 'PROJECTED_FP'])
    df = df[df['SALARY'] > 0]
    df = df[df['PROJECTED_FP'] > 0]

    logger.info(f"After filtering: {len(df)} eligible players")

    if len(df) < 8:
        logger.warning(f"Not enough players to create lineup. Only {len(df)} available.")
        return pd.DataFrame()

    # Define slots
    slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    # Create LP problem
    prob = LpProblem("DK_Lineup_Optimizer", LpMaximize)

    # Create decision variables
    player_slot = {}
    for idx, row in df.iterrows():
        position = row['POSITION']
        for slot in slots:
            if is_eligible_for_slot(position, slot):
                player_slot[(idx, slot)] = LpVariable(f"p_{idx}_s_{slot}", cat='Binary')

    if not player_slot:
        logger.warning("No valid player-slot combinations found")
        return pd.DataFrame()

    # Objective: Maximize total projected FP
    prob += lpSum([
        df.loc[idx, 'PROJECTED_FP'] * player_slot[(idx, slot)]
        for (idx, slot) in player_slot.keys()
    ]), "Total_FP"

    # Constraint 1: Exactly one player per slot
    for slot in slots:
        prob += lpSum([
            player_slot[(idx, s)]
            for (idx, s) in player_slot.keys()
            if s == slot
        ]) == 1, f"Slot_{slot}"

    # Constraint 2: Each player used at most once
    for idx in df.index:
        prob += lpSum([
            player_slot[(i, slot)]
            for (i, slot) in player_slot.keys()
            if i == idx
        ]) <= 1, f"Player_{idx}_Once"

    # Constraint 3: Total salary <= $50,000
    prob += lpSum([
        df.loc[idx, 'SALARY'] * player_slot[(idx, slot)]
        for (idx, slot) in player_slot.keys()
    ]) <= 50000, "Salary_Cap"

    # Solve
    solver = PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if prob.status != 1:
        logger.warning(f"Optimization failed with status: {prob.status}")
        return pd.DataFrame()

    # Extract lineup
    lineup_data = []
    total_salary = 0
    total_fp = 0

    for (idx, slot), var in player_slot.items():
        if value(var) == 1:
            player_row = df.loc[idx]
            lineup_data.append({
                'DATE': today,
                'SLOT': slot,
                'PLAYER': player_row['PLAYER'],
                'TEAM': player_row['TEAM'],
                'POSITION': player_row['POSITION'],
                'SALARY': player_row['SALARY'],
                'PROJECTED_MIN': player_row['PROJECTED_MIN'],
                'PROJECTED_FP': player_row['PROJECTED_FP'],
                'ACTUAL_FP': None
            })
            total_salary += player_row['SALARY']
            total_fp += player_row['PROJECTED_FP']

    lineup_df = pd.DataFrame(lineup_data)

    # Sort by slot order
    slot_order = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4, 'G': 5, 'F': 6, 'UTIL': 7}
    lineup_df['slot_order'] = lineup_df['SLOT'].map(slot_order)
    lineup_df = lineup_df.sort_values('slot_order').drop(columns=['slot_order'])

    logger.info(f"Lineup optimized: {len(lineup_df)} players, ${total_salary:.0f} salary, {total_fp:.1f} projected FP")

    return lineup_df


def process_model(model_name, s3_path, projections_data, daily_preds, today, box_scores, injury_context=None):
    """
    Unified model processing pipeline: FP prediction, filtering, lineup optimization, and S3 persistence

    Handles all common operations across the 4 projection models:
    1. Convert projections to DataFrame (if needed)
    2. Add fantasy points predictions
    3. Filter to today's players only
    4. Optimize lineup
    5. Load/merge/save lineup data
    6. Load/merge/save projection data
    7. Save injury context (if provided)

    Args:
        model_name: Display name for logging (e.g., "Complex Position Overlap")
        s3_path: S3 path prefix (e.g., "model_comparison/complex_position_overlap")
        projections_data: Either list of projection dicts or existing DataFrame
        daily_preds: Daily predictions for filtering and lineup optimization
        today: Current date
        box_scores: Box score data for FP predictions
        injury_context: Optional injury context DataFrame to save (for injury-aware models)

    Returns:
        Tuple of (full_projections_df, today_lineup_df) - DataFrames with all dates
    """
    logger.info(f"Processing model: {model_name}")

    # Convert to DataFrame if needed
    if isinstance(projections_data, list):
        df_proj = pd.DataFrame(projections_data)
    else:
        df_proj = projections_data.copy()

    # Add fantasy points projections
    df_proj = predict_fantasy_points(df_proj, box_scores, daily_preds, today)

    # Filter to today's players only
    if not daily_preds.empty:
        todays_players = daily_preds[daily_preds['GAME_DATE'] == today]['PLAYER'].unique()
        df_proj = df_proj[df_proj['PLAYER'].isin(todays_players)]
        logger.info(f"{model_name}: Filtered to {len(df_proj)} players with games on {today}")

    # Optimize lineup
    lineup = optimize_lineup(df_proj, daily_preds, today)

    # Save lineup (merge with existing)
    if not lineup.empty:
        lineup_path = f"{s3_path}/daily_lineups.parquet"
        existing_lineup = load_from_s3(lineup_path)
        if not existing_lineup.empty:
            existing_lineup['DATE'] = pd.to_datetime(existing_lineup['DATE']).dt.date
            existing_lineup = existing_lineup[existing_lineup['DATE'] != today]
            lineup = pd.concat([existing_lineup, lineup], ignore_index=True)
        save_to_s3(lineup, lineup_path)
        logger.info(f"{model_name} lineup saved: {len(lineup[lineup['DATE'] == today])} players")

    # Save projections (merge with existing)
    proj_path = f"{s3_path}/minutes_projections.parquet"
    existing_proj = load_from_s3(proj_path)
    if not existing_proj.empty:
        existing_proj['DATE'] = pd.to_datetime(existing_proj['DATE']).dt.date
        existing_proj = existing_proj[existing_proj['DATE'] != today]
        df_proj = pd.concat([existing_proj, df_proj], ignore_index=True)
    save_to_s3(df_proj, proj_path)

    # Save injury context if provided
    if injury_context is not None:
        context_path = f"injury_context/{s3_path.split('/')[-1]}.parquet"
        save_to_s3(injury_context, context_path)

    return df_proj, lineup


# ==================== Injury Context Helper Functions ====================

def transition_beneficiaries_to_ex(injury_context, currently_injured, today):
    """
    Transition statuses when injured player returns:
    - BENEFICIARY → EX_BENEFICIARY (will revert to pre-injury baseline)
    - ROLE_DECREASED → Remove from tracking (will keep using recent games)

    Args:
        injury_context: DataFrame of injury context to process
        currently_injured: Set of player names who are still injured
        today: Current date (for storing return_date)

    Returns:
        Updated injury_context DataFrame
    """
    if injury_context.empty:
        return injury_context

    # Handle BENEFICIARY transitions
    beneficiaries = injury_context[injury_context['STATUS'] == 'BENEFICIARY']

    for _, beneficiary in beneficiaries.iterrows():
        injured_player = beneficiary['BENEFICIARY_OF']

        if injured_player not in currently_injured:
            # Injured player returned - transition this beneficiary relationship
            logger.info(f"{beneficiary['PLAYER']}: Transitioning BENEFICIARY -> EX_BENEFICIARY ({injured_player} returned)")

            # Create new EX_BENEFICIARY record
            injury_context = add_to_injury_context(
                injury_context,
                player_name=beneficiary['PLAYER'],
                team=beneficiary['TEAM'],
                status='EX_BENEFICIARY',
                true_baseline=beneficiary['TRUE_BASELINE'],
                beneficiary_of=beneficiary['BENEFICIARY_OF'],
                injury_date=beneficiary['INJURY_DATE'],
                return_date=today
            )

            # Remove only the specific BENEFICIARY record for this returned player
            injury_context = injury_context[
                ~((injury_context['PLAYER'] == beneficiary['PLAYER']) &
                  (injury_context['STATUS'] == 'BENEFICIARY') &
                  (injury_context['BENEFICIARY_OF'] == injured_player))
            ].copy()

    # Handle ROLE_DECREASED transitions
    # KEEP ROLE_DECREASED status even when injured player returns
    # These players permanently lost minutes - continue using recent games (don't revert to old baseline)
    # Example: Kennard lost minutes when Trae injured, should stay at reduced role even after Trae returns

    return injury_context


def add_to_injury_context(injury_context, player_name, team, status, true_baseline, beneficiary_of, injury_date, return_date=None):
    """
    Add or update player in injury context

    For EX_BENEFICIARY and BENEFICIARY: Allows multiple rows per player (multiple injury periods or multiple injuries benefited from)
    For RETURNING: Only replaces existing RETURNING rows (keeps BENEFICIARY and EX_BENEFICIARY)
    For other statuses: Replaces existing row
    """
    # For EX_BENEFICIARY and BENEFICIARY, allow multiple rows per player
    # For RETURNING, only remove old RETURNING rows (keep BENEFICIARY and EX_BENEFICIARY)
    # For other statuses, replace existing row
    if status == 'RETURNING':
        # Keep BENEFICIARY and EX_BENEFICIARY, only remove old RETURNING
        injury_context = injury_context[
            ~((injury_context['PLAYER'] == player_name) &
              (injury_context['STATUS'] == 'RETURNING'))
        ].copy()
    elif status not in ['EX_BENEFICIARY', 'BENEFICIARY']:
        injury_context = injury_context[injury_context['PLAYER'] != player_name].copy()

    new_record = pd.DataFrame([{
        'PLAYER': player_name,
        'TEAM': team,
        'STATUS': status,
        'INJURY_DATE': injury_date,
        'RETURN_DATE': return_date,
        'TRUE_BASELINE': true_baseline,
        'BENEFICIARY_OF': beneficiary_of,
        'UPDATED_DATE': datetime.now(pytz.timezone('US/Eastern'))
    }])

    return pd.concat([injury_context, new_record], ignore_index=True)


def remove_from_injury_context(injury_context, player_name):
    """Remove player and their beneficiaries from injury context"""
    injury_context = injury_context[injury_context['PLAYER'] != player_name].copy()
    injury_context = injury_context[injury_context['BENEFICIARY_OF'] != player_name].copy()
    return injury_context


def get_pre_injury_baseline(player_name, box_scores, injury_date):
    """
    Calculate baseline EXCLUDING games played after injury_date

    This prevents the TRUE_BASELINE from being inflated by games where the player
    was already filling in for an injured teammate.

    Args:
        player_name: Name of the player
        box_scores: Full season box scores
        injury_date: Date when teammate got injured (exclude games after this)

    Returns:
        Baseline minutes (float) or None if insufficient data
    """
    player_games = box_scores[box_scores['PLAYER'] == player_name].copy()

    if player_games.empty or 'GAME_DATE' not in player_games.columns:
        return None

    player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])
    injury_date = pd.to_datetime(injury_date)

    # Only use games BEFORE the injury occurred
    pre_injury_games = player_games[player_games['GAME_DATE'] < injury_date].copy()

    if len(pre_injury_games) == 0:
        # No games before injury - can't calculate baseline
        return None

    # Use weighted average (recent games get 2x weight)
    today = pd.Timestamp.now()
    recent_cutoff = today - pd.Timedelta(days=14)
    pre_injury_games['weight'] = pre_injury_games['GAME_DATE'].apply(
        lambda d: 2.0 if d >= recent_cutoff else 1.0
    )

    baseline = (pre_injury_games['MIN'] * pre_injury_games['weight']).sum() / pre_injury_games['weight'].sum()

    logger.info(f"{player_name}: Pre-injury baseline = {baseline:.1f} MPG "
                f"(using {len(pre_injury_games)} games before {injury_date.date()})")
    return baseline


def get_player_baseline(player_row, injury_context, box_scores):
    """
    Get player's TRUE baseline minutes with fallback chain

    For EX_BENEFICIARY players: Excludes inflation periods (games between INJURY_DATE and RETURN_DATE)
    to prevent compounding minutes problem

    Conservative approach: Use 10 MPG fallback when baseline is unreliable (< 4 games, traded players)
    """
    player_name = player_row['PLAYER']

    # Check if player is CURRENTLY benefiting from injury (active beneficiary)
    active_beneficiary = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'] == 'BENEFICIARY')
    ]

    if not active_beneficiary.empty:
        # Return frozen baseline from before they started filling in
        return active_beneficiary['TRUE_BASELINE'].iloc[0]

    # Check for ROLE_DECREASED status (player squeezed out during teammate's injury)
    # These players should NOT revert to pre-injury baseline when injured player returns
    # Use recent games instead to reflect their new, reduced role
    role_decreased = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'] == 'ROLE_DECREASED')
    ]

    if not role_decreased.empty:
        # Use recent performance, not pre-injury baseline
        # This player's role permanently decreased during the injury period
        recent_baseline = player_row.get('LAST_7_AVG_MIN')
        if recent_baseline is not None and recent_baseline > 0:
            logger.info(f"{player_name}: ROLE_DECREASED status - using recent baseline {recent_baseline:.1f} MPG (not reverting to pre-injury)")
            return recent_baseline
        else:
            # Fallback to season average if no recent games
            return player_row.get('SEASON_AVG_MIN', 10)

    # Check if player WAS a beneficiary (ex-beneficiary with historical INJURY_DATE)
    # Can have MULTIPLE rows if player filled in during multiple injury periods
    ex_beneficiary_records = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'] == 'EX_BENEFICIARY')
    ]

    if not ex_beneficiary_records.empty:
        # Ex-beneficiary - calculate baseline excluding ALL inflation periods
        player_games = box_scores[box_scores['PLAYER'] == player_name].copy()

        if not player_games.empty and 'GAME_DATE' in player_games.columns:
            player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])

            # Start with all games valid, then exclude each inflation period
            valid_games_mask = pd.Series([True] * len(player_games), index=player_games.index)

            for _, record in ex_beneficiary_records.iterrows():
                injury_date = pd.to_datetime(record['INJURY_DATE'])
                return_date = pd.to_datetime(record['RETURN_DATE'])

                # Exclude games in this inflation period
                inflation_mask = (
                    (player_games['GAME_DATE'] >= injury_date) &
                    (player_games['GAME_DATE'] < return_date)
                )
                valid_games_mask = valid_games_mask & ~inflation_mask

            valid_games = player_games[valid_games_mask].copy()

            if len(valid_games) >= 1:
                # Weight recent games (last 14 days) at 2.0x for faster convergence
                today = pd.Timestamp.now()
                recent_cutoff = today - pd.Timedelta(days=14)
                valid_games['weight'] = valid_games['GAME_DATE'].apply(
                    lambda d: 2.0 if d >= recent_cutoff else 1.0
                )

                # Weighted average of minutes
                historical_baseline = (valid_games['MIN'] * valid_games['weight']).sum() / valid_games['weight'].sum()
                excluded_count = len(player_games) - len(valid_games)
                recent_games = (valid_games['GAME_DATE'] >= recent_cutoff).sum()

                logger.info(f"{player_name}: Ex-beneficiary baseline = {historical_baseline:.1f} MPG "
                           f"(excluding {excluded_count} inflated games, {recent_games} recent games weighted 2x)")
                return historical_baseline
            else:
                # No valid games - use conservative fallback
                logger.warning(f"{player_name}: No valid games for ex-beneficiary - using conservative 10 MPG")
                return 10

    # Not a beneficiary - normal baseline calculation
    if player_row.get('GAMES_PLAYED', 0) >= 4:
        # Reliable: 4+ games this season
        return player_row.get('SEASON_AVG_MIN', 10)

    elif player_row.get('FROM_PREV_SEASON', False):
        # Season-long injury (0 games this season) - check if same team
        prev_team = player_row.get('PREV_TEAM')
        current_team = player_row.get('TEAM')

        if prev_team == current_team:
            # Same team - use previous season baseline (reliable)
            return player_row.get('SEASON_AVG_MIN', 10)
        else:
            # TRADED - previous season data unreliable (different role on old team)
            logger.warning(f"{player_name}: Season-long injury on new team after trade ({prev_team} -> {current_team}) - using conservative 10 MPG baseline")
            return 10  # Conservative fallback

    else:
        # < 4 games this season - UNRELIABLE
        # Don't use CAREER_AVG_MIN (could be inflated from old team/role)
        logger.debug(f"{player_name}: Insufficient games ({player_row.get('GAMES_PLAYED', 0)}) - using conservative 10 MPG baseline")
        return 10  # Conservative fallback


def update_confidence_status(player_row, injury_context, games_log):
    """
    Update confidence level based on games played since injury

    NOTE: Confidence is METADATA ONLY and does NOT affect projection calculations.
    It's returned alongside projections for monitoring/logging purposes.
    """
    player_name = player_row['PLAYER']

    low_confidence_record = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'].isin(['INJURED', 'BENEFICIARY']))
    ]

    if low_confidence_record.empty:
        return 'HIGH'

    injury_date = pd.to_datetime(low_confidence_record['INJURY_DATE'].iloc[0])
    games_since = games_log[games_log['GAME_DATE'] > injury_date]
    games_played_since = len(games_since)

    if games_played_since >= CONFIDENCE_CLEARANCE_GAMES:
        return 'HIGH'
    elif games_played_since == 0:
        return 'LOW'
    elif games_played_since < CONFIDENCE_CLEARANCE_GAMES:
        return 'MEDIUM'

    return 'HIGH'


# ==================== Injury Impact Analysis ====================

def analyze_injury_impact(teammates, box_scores, injury_date, injury_context, min_games=3):
    """
    Analyze how teammates' minutes changed during an injury period

    Compares pre-injury baseline vs during-injury performance to identify:
    - Players who gained minutes (true beneficiaries)
    - Players who lost minutes (squeezed out by rotation changes)
    - Players unaffected

    Args:
        teammates: DataFrame of teammates to analyze
        box_scores: Full season box scores
        injury_date: Date when injury occurred
        injury_context: Current injury tracking context (to check for existing beneficiaries)
        min_games: Minimum games required in each period for reliable comparison

    Returns:
        role_increased: List of (player_name, pre_baseline, during_avg, delta)
        role_decreased: List of (player_name, pre_baseline, during_avg, delta)
        unchanged: List of player_name
    """
    injury_date = pd.to_datetime(injury_date)
    role_increased = []
    role_decreased = []
    unchanged = []

    for _, player in teammates.iterrows():
        player_name = player['PLAYER']
        player_games = box_scores[box_scores['PLAYER'] == player_name].copy()

        if player_games.empty or 'GAME_DATE' not in player_games.columns:
            unchanged.append(player_name)
            continue

        player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])

        # Split games into pre-injury and during-injury periods
        pre_injury_games = player_games[player_games['GAME_DATE'] < injury_date]
        during_injury_games = player_games[player_games['GAME_DATE'] >= injury_date]

        # Need minimum games in both periods for reliable comparison
        if len(pre_injury_games) < min_games or len(during_injury_games) < min_games:
            unchanged.append(player_name)
            continue

        # Check if player is ALREADY a beneficiary of another injury (stacking injuries)
        # If so, use their frozen TRUE_BASELINE instead of calculating from recent games
        existing_beneficiary = injury_context[
            (injury_context['PLAYER'] == player_name) &
            (injury_context['STATUS'] == 'BENEFICIARY')
        ]

        if not existing_beneficiary.empty:
            # Player already benefiting from another injury - use frozen baseline
            pre_baseline = existing_beneficiary['TRUE_BASELINE'].iloc[0]
            logger.info(f"{player_name}: Already BENEFICIARY of another injury - using frozen baseline {pre_baseline:.1f} MPG (prevents inflation from stacking injuries)")
        else:
            # Calculate baseline from pre-injury games
            pre_baseline = pre_injury_games['MIN'].mean()

        # Calculate post-injury average
        during_avg = during_injury_games['MIN'].mean()
        delta = during_avg - pre_baseline

        # Categorize based on change (threshold: ±3 MPG)
        if delta > 3:
            role_increased.append((player_name, pre_baseline, during_avg, delta))
            logger.info(f"{player_name}: Role increased during injury (+{delta:.1f} MPG: {pre_baseline:.1f} -> {during_avg:.1f})")
        elif delta < -3:
            role_decreased.append((player_name, pre_baseline, during_avg, delta))
            logger.info(f"{player_name}: Role decreased during injury ({delta:.1f} MPG: {pre_baseline:.1f} -> {during_avg:.1f})")
        else:
            unchanged.append(player_name)

    return role_increased, role_decreased, unchanged


# ==================== Injury Redistribution (Complex & Direct Models) ====================

def redistribute_injury_minutes(injured_player, team_players, injury_context, today, box_scores,
                                position_overlap_func, use_multiplier=True):
    """
    Redistribute minutes using TRUE baselines

    Args:
        box_scores: Full season box scores for baseline calculations
        position_overlap_func: Either position_overlap_complex or position_overlap_exact
        use_multiplier: If True, apply 2x multiplier for exact position (complex model)
    """
    # Skip redistribution for season-long injuries (0 games this season, using prev season data)
    # Their teammates' SEASON_AVG_MIN already accounts for their absence all season
    if injured_player.get('GAMES_PLAYED', 1) == 0 and injured_player.get('FROM_PREV_SEASON', False):
        logger.debug(
            f"Season-long injury detected: {injured_player['PLAYER']} (0 games this season). "
            f"Skipping redistribution - teammates' baselines already account for absence."
        )
        return {}, injury_context

    # ========== ONGOING INJURY CHECK (>= 3 games) ==========
    # For ongoing injuries, use post-injury Formula C instead of predictive redistribution
    # This adapts to actual rotation changes (handles positionless basketball)
    injury_date_to_use = injured_player.get('ESTIMATED_INJURY_DATE', today)
    if pd.isna(injury_date_to_use):
        injury_date_to_use = today

    # Count games since injury by finding MAX games played by any teammate
    # This handles cases where some teammates also got injured
    games_since_injury = 0
    for _, teammate in team_players.iterrows():
        if teammate['PLAYER'] == injured_player['PLAYER']:
            continue  # Skip the injured player

        teammate_games = box_scores[
            (box_scores['PLAYER'] == teammate['PLAYER']) &
            (box_scores['GAME_DATE'] >= pd.to_datetime(injury_date_to_use))
        ]

        # Track the maximum - this represents how many games the team has played
        games_since_injury = max(games_since_injury, len(teammate_games))

    if games_since_injury >= 3:
        logger.info(
            f"{injured_player['PLAYER']} out for {games_since_injury} games - using post-injury Formula C "
            f"(calculated from games after injury, adapts to actual rotation)"
        )

        # Find ALL active teammates (not just overlapping positions)
        # This handles positionless basketball - let the data show who actually got minutes
        # Example: McBride (PG) might absorb minutes from OG (SF) if coach goes small
        teammates = team_players[
            (team_players['PLAYER'] != injured_player['PLAYER']) &
            (team_players['STATUS'] != 'OUT')
        ].copy()

        if not teammates.empty:
            # Analyze how teammates' roles actually changed during injury
            role_increased, role_decreased, unchanged = analyze_injury_impact(
                teammates, box_scores, injury_date_to_use, injury_context, min_games=3
            )

            # Track role changes in injury_context (for when injured player returns)
            for player_name, pre_baseline, during_avg, delta in role_increased:
                # Check if already tracked
                existing = injury_context[
                    (injury_context['PLAYER'] == player_name) &
                    (injury_context['STATUS'] == 'BENEFICIARY') &
                    (injury_context['BENEFICIARY_OF'] == injured_player['PLAYER'])
                ]

                if existing.empty:
                    # Mark as BENEFICIARY with clean pre-injury baseline
                    injury_context = add_to_injury_context(
                        injury_context,
                        player_name=player_name,
                        team=injured_player['TEAM'],
                        status='BENEFICIARY',
                        true_baseline=pre_baseline,  # Clean pre-injury baseline
                        beneficiary_of=injured_player['PLAYER'],
                        injury_date=injury_date_to_use
                    )
                    logger.info(f"{player_name}: Tracked as BENEFICIARY (baseline {pre_baseline:.1f} MPG, currently {during_avg:.1f} MPG)")

            # Track role_decreased players so they don't revert to old baseline when injury ends
            for player_name, pre_baseline, during_avg, delta in role_decreased:
                # Check if already tracked
                existing = injury_context[
                    (injury_context['PLAYER'] == player_name) &
                    (injury_context['STATUS'] == 'ROLE_DECREASED') &
                    (injury_context['BENEFICIARY_OF'] == injured_player['PLAYER'])
                ]

                if existing.empty:
                    # Mark as ROLE_DECREASED
                    injury_context = add_to_injury_context(
                        injury_context,
                        player_name=player_name,
                        team=injured_player['TEAM'],
                        status='ROLE_DECREASED',
                        true_baseline=pre_baseline,  # Store original baseline for reference
                        beneficiary_of=injured_player['PLAYER'],
                        injury_date=injury_date_to_use
                    )
                    logger.info(f"{player_name}: Tracked as ROLE_DECREASED (baseline {pre_baseline:.1f} MPG, currently {during_avg:.1f} MPG)")

        # Calculate post-injury Formula C for all teammates
        # This uses ONLY games played after the injury to reflect actual rotation changes
        projections = {}
        for _, player in teammates.iterrows():
            player_name = player['PLAYER']

            # Get post-injury games only
            player_post_injury = box_scores[
                (box_scores['PLAYER'] == player_name) &
                (box_scores['GAME_DATE'] >= pd.to_datetime(injury_date_to_use))
            ].copy()

            if len(player_post_injury) >= 3:
                # Enough post-injury data - use post-injury Formula C
                post_injury_avg = player_post_injury['MIN'].mean()
                post_injury_last_7 = player_post_injury.tail(7)['MIN'].mean()
                post_injury_prev = player_post_injury.iloc[-1]['MIN'] if not player_post_injury.empty else post_injury_avg

                projected = 0.5 * post_injury_avg + 0.3 * post_injury_last_7 + 0.2 * post_injury_prev
                projected = min(projected, MAX_MINUTES)
                projections[player_name] = projected

                logger.debug(f"{player_name}: Post-injury Formula C = {projected:.1f} MPG (avg:{post_injury_avg:.1f}, last7:{post_injury_last_7:.1f}, prev:{post_injury_prev:.1f})")

        return projections, injury_context

    # ========== NEW INJURY (<= 2 games): Use predictive redistribution ==========
    logger.info(f"{injured_player['PLAYER']} recently injured ({games_since_injury} games) - using predictive redistribution")

    # Get injured player's baseline
    if injured_player.get('GAMES_PLAYED', 0) >= 4:
        # Reliable: 4+ games this season
        lost_minutes = injured_player.get('SEASON_AVG_MIN', 5)
    elif injured_player.get('FROM_PREV_SEASON', False):
        # Season-long injury (0 games this season) - check if same team
        prev_team = injured_player.get('PREV_TEAM')
        current_team = injured_player.get('TEAM')

        if prev_team == current_team:
            # Same team - use previous season baseline (reliable)
            lost_minutes = injured_player.get('SEASON_AVG_MIN', 5)
        else:
            # TRADED - previous season data unreliable, use conservative fallback
            logger.warning(f"{injured_player['PLAYER']}: Season-long injury after trade - using conservative 5 MPG baseline")
            lost_minutes = 5
    else:
        # < 4 games this season - UNRELIABLE, use conservative fallback
        logger.debug(f"{injured_player['PLAYER']}: Insufficient games ({injured_player.get('GAMES_PLAYED', 0)}) - using conservative 5 MPG baseline")
        lost_minutes = 5

    # Find teammates at same/overlapping position
    injured_position = injured_player.get('POSITION', 'SF')
    teammates = team_players[
        (team_players['PLAYER'] != injured_player['PLAYER']) &
        (team_players['STATUS'] != 'OUT') &
        (team_players['POSITION'].apply(lambda pos: position_overlap_func(pos, injured_position)))
    ].copy()

    if teammates.empty:
        logger.debug(f"No teammates available to absorb {injured_player['PLAYER']}'s minutes")
        return {}, injury_context

    # Calculate TRUE baselines (with pre-injury logic to prevent inflation)
    # Note: injury_date_to_use already calculated at top of function
    baselines = {}
    for _, player in teammates.iterrows():
        player_name = player['PLAYER']

        # FIRST: Check if player already has a frozen baseline (existing beneficiary)
        active_beneficiary = injury_context[
            (injury_context['PLAYER'] == player_name) &
            (injury_context['STATUS'] == 'BENEFICIARY')
        ]

        if not active_beneficiary.empty:
            # Player already tracked - use frozen baseline
            baselines[player_name] = active_beneficiary['TRUE_BASELINE'].iloc[0]
        else:
            # NEW beneficiary - calculate baseline excluding post-injury games
            pre_injury_baseline = get_pre_injury_baseline(player_name, box_scores, injury_date_to_use)

            if pre_injury_baseline is not None:
                # Successfully calculated clean baseline
                baselines[player_name] = pre_injury_baseline
            else:
                # No pre-injury games - use standard baseline (handles rookies, recent trades)
                baselines[player_name] = get_player_baseline(player, injury_context, box_scores)

    # Apply weighting
    weighted_baselines = {}
    for player_name, baseline in baselines.items():
        base_weight = baseline + BENCH_OPPORTUNITY_CONSTANT
        player_position = teammates[teammates['PLAYER'] == player_name]['POSITION'].iloc[0]

        # Apply 2x multiplier for exact position (only if use_multiplier=True)
        if use_multiplier and player_position == injured_position:
            weighted_baselines[player_name] = base_weight * EXACT_POSITION_MULTIPLIER
        else:
            weighted_baselines[player_name] = base_weight

    total_weighted = sum(weighted_baselines.values())

    # Initial distribution
    projections = {}
    for player_name, weight in weighted_baselines.items():
        distribution_weight = weight / total_weighted
        boost = lost_minutes * distribution_weight
        projected = baselines[player_name] + boost
        projections[player_name] = projected

    # Handle overflow from 36-min cap
    for _ in range(7):
        overflow_minutes = 0
        capped_players = set()

        for player_name, projected in projections.items():
            if projected > MAX_MINUTES:
                overflow_minutes += (projected - MAX_MINUTES)
                projections[player_name] = MAX_MINUTES
                capped_players.add(player_name)

        if overflow_minutes == 0:
            break

        non_capped = [p for p in projections.keys() if p not in capped_players]
        if not non_capped:
            break

        non_capped_total = sum(weighted_baselines[p] for p in non_capped)
        for player_name in non_capped:
            weight = weighted_baselines[player_name] / non_capped_total
            additional_boost = overflow_minutes * weight
            projections[player_name] = min(projections[player_name] + additional_boost, MAX_MINUTES)

    # Cap individual boosts to 50% of lost minutes
    # Prevents one player from absorbing entire injury impact (unrealistic)
    # Example: If OG (30 MPG) is out, no single player should get more than 15 MPG boost
    # Allows multiple injuries to stack (realistic scenario)
    max_individual_boost = lost_minutes * 0.5

    for player_name in list(projections.keys()):
        boost = projections[player_name] - baselines[player_name]
        if boost > max_individual_boost:
            logger.info(f"{player_name}: Capping boost from {injured_player['PLAYER']} injury: {boost:.1f} -> {max_individual_boost:.1f} MPG")
            projections[player_name] = baselines[player_name] + max_individual_boost

    # Track beneficiaries in injury context (only if boost > 3 MPG)
    # Note: injury_date_to_use already calculated at top of function
    for player_name in projections.keys():
        boost = projections[player_name] - baselines[player_name]

        # Only track as BENEFICIARY if boost > 3 MPG
        # Prevents stars from being tracked for minimal boosts (e.g., +1 MPG after 36-min cap)
        if boost > 3:
            # Check if this beneficiary relationship already exists (prevent duplicates)
            existing_beneficiary = injury_context[
                (injury_context['PLAYER'] == player_name) &
                (injury_context['STATUS'] == 'BENEFICIARY') &
                (injury_context['BENEFICIARY_OF'] == injured_player['PLAYER'])
            ]

            if existing_beneficiary.empty:
                injury_context = add_to_injury_context(
                    injury_context,
                    player_name=player_name,
                    team=injured_player['TEAM'],
                    status='BENEFICIARY',
                    true_baseline=baselines[player_name],
                    beneficiary_of=injured_player['PLAYER'],
                    injury_date=injury_date_to_use
                )
                logger.info(f"{player_name}: Marked as BENEFICIARY (+{boost:.1f} MPG from {injured_player['PLAYER']} injury, baseline: {baselines[player_name]:.1f} -> projected: {projections[player_name]:.1f})")
            else:
                logger.debug(f"{player_name}: Already tracked as BENEFICIARY of {injured_player['PLAYER']} (current projection: {projections[player_name]:.1f} MPG)")
        else:
            logger.debug(f"{player_name}: Boost too small (+{boost:.1f} MPG), not tracking as BENEFICIARY")

    return projections, injury_context


# ==================== Shared Projection Logic ====================

def calculate_team_injury_redistributions(team_players, injury_context, today, box_scores,
                                          position_overlap_func, use_multiplier):
    """
    Calculate injury redistributions for ALL injured players on a team ONCE.
    This avoids redundant calculations when looping through each active player.

    Returns:
        tuple: (team_redistributions, updated_injury_context)
            - team_redistributions: dict of {injured_player_name: {player_name: projected_min}}
            - updated_injury_context: injury_context with beneficiaries tracked
    """
    team_redistributions = {}

    # Find teammates currently OUT (exclude those returning today - they're playing!)
    team_injuries = team_players[team_players['STATUS'] == 'OUT']
    if 'RETURN_DATE_DT' in team_injuries.columns:
        teammates_returning_today = team_injuries['RETURN_DATE_DT'] <= today
        team_injuries = team_injuries[~teammates_returning_today]
        excluded_count = teammates_returning_today.sum()
        if excluded_count > 0:
            logger.debug(f"Excluded {excluded_count} teammates marked OUT but returning today (won't redistribute their minutes)")

    if not team_injuries.empty:
        team_injuries_sorted = team_injuries.sort_values('SEASON_AVG_MIN', ascending=False)

        # Calculate redistribution for each injured player ONCE
        for _, injured in team_injuries_sorted.iterrows():
            injury_projections, injury_context = redistribute_injury_minutes(
                injured, team_players, injury_context, today, box_scores,
                position_overlap_func, use_multiplier=use_multiplier
            )
            if injury_projections:  # Only store if there are projections
                team_redistributions[injured['PLAYER']] = injury_projections

    return team_redistributions, injury_context


def project_minutes_with_injuries(player_row, team_players, injury_context, games_log, today, box_scores,
                                   position_overlap_func, use_multiplier, injury_data, team_injury_redistributions=None):
    """
    Unified projection logic for both models with injury handling

    Args:
        team_injury_redistributions: Pre-calculated injury redistributions for the team (optional).
                                     If None, will calculate on-demand (legacy behavior).

    IMPORTANT: This function handles TWO DIFFERENT "returning from injury" scenarios:

    A. RETURNING TODAY (25% reduction):
       - Player still ON injury report (has RETURN_DATE_DT = today)
       - Gets normal projection (redistribution or Formula C) with 25% reduction
       - Applies when player listed as OUT/QUESTIONABLE but expected to play today

    B. RETURNING STATUS (10 MPG for 4 games):
       - Player with season-long injury (0 games, FROM_PREV_SEASON=True)
       - NO LONGER on injury report (STATUS != 'OUT')
       - Gets conservative 10 MPG for first 4 games
       - Applies to stars returning after missing entire season (e.g., Kawhi)

    These scenarios DON'T OVERLAP because:
    - Scenario A requires being ON injury report (RETURN_DATE_DT set)
    - Scenario B requires NOT being on injury report anymore (STATUS != 'OUT')

    Other scenarios:
    3. Active injuries (teammate currently OUT) - Redistribute their minutes
    4. Normal play (no injuries affecting this player) - Use Formula C

    Args:
        position_overlap_func: Either position_overlap_complex or position_overlap_exact
        use_multiplier: If True, apply 2x multiplier for exact position (complex model)
        injury_data: DataFrame of current injury report data (to filter out free agents)

    Returns:
        (projected_minutes, confidence, updated_injury_context)
    """
    player_name = player_row['PLAYER']

    # ========== CHECK: Is player RETURNING TODAY? ==========
    # Scenario A: Player still on injury report (RETURN_DATE_DT = today) but expected to play
    # Will apply 25% reduction later (after redistribution or Formula C calculation)
    # Example: LeBron listed as OUT yesterday, expected to return today
    returning_today = False
    if 'RETURN_DATE_DT' in player_row and pd.notna(player_row.get('RETURN_DATE_DT')):
        return_date = player_row['RETURN_DATE_DT']
        if isinstance(return_date, pd.Timestamp):
            return_date = return_date.date()
        if return_date == today:
            returning_today = True
            logger.info(f"{player_name}: Returning today - will apply 25% minutes reduction")

    # ========== STEP 1: Handle currently injured players (status OUT) ==========
    # If OUT but returning today, treat as playing (with 25% reduction applied later)
    if player_row.get('STATUS') == 'OUT' and not returning_today:
        return 0, 'HIGH', injury_context

    # ========== STEP 2: Check for RETURNING STATUS (Season-Long Returns) ==========
    # Scenario B: Star player (e.g., Kawhi) was out ALL SEASON (0 games), now healthy (STATUS != 'OUT')
    # Both returning star AND affected teammates get conservative 10 MPG for first 4 games
    # NOTE: This is DIFFERENT from "Returning TODAY" (Scenario A) - those players are still on injury report

    # First, check injury_context for ALREADY-TRACKED returning players on this team
    returning_players = injury_context[
        (injury_context['STATUS'] == 'RETURNING') &
        (injury_context['TEAM'] == player_row['TEAM'])
    ]

    # Track if current player is returning OR affected by a returning player
    is_returning_player = False      # True if current player is the one returning
    affected_by_return = False       # True if current player affected by teammate's return
    max_returning_player_games = None  # Games played by returning player (for 4-game threshold)

    # Loop through already-tracked returning players (from previous lambda runs)
    for _, returning_record in returning_players.iterrows():
        returning_player_name = returning_record['PLAYER']
        return_date = pd.to_datetime(returning_record['INJURY_DATE'])

        # Count games played by returning player since return
        returning_games = games_log[
            (games_log['PLAYER'] == returning_player_name) &
            (games_log['GAME_DATE'] >= return_date)
        ]
        games_since_return = len(returning_games)

        if games_since_return >= 4:
            # Cleanup: Remove from RETURNING tracking after 4 games (enough data for baseline)
            logger.info(f"{returning_player_name} has played {games_since_return} games since return - removing from RETURNING tracking")
            injury_context = remove_from_injury_context(injury_context, returning_player_name)
        else:
            # Still in conservative period (< 4 games) - check if current player is affected
            if player_name == returning_player_name:
                is_returning_player = True
                max_returning_player_games = games_since_return
                logger.info(f"{player_name}: Returning player with {games_since_return}/4 games")
            else:
                # Check position overlap with current player
                returning_player_row = team_players[team_players['PLAYER'] == returning_player_name]
                if not returning_player_row.empty:
                    returning_pos = returning_player_row['POSITION'].iloc[0]
                    if position_overlap_func(player_row.get('POSITION', 'SF'), returning_pos):
                        affected_by_return = True
                        # Track the max games of any returning player affecting this player
                        if max_returning_player_games is None or games_since_return > max_returning_player_games:
                            max_returning_player_games = games_since_return
                        logger.info(f"{player_name}: Affected by {returning_player_name} return ({games_since_return}/4 games)")

                        # Mark affected teammate as EX_BENEFICIARY to exclude inflated games from baseline
                        # Check if already tracked (avoid duplicate records)
                        already_ex_beneficiary = injury_context[
                            (injury_context['PLAYER'] == player_name) &
                            (injury_context['STATUS'] == 'EX_BENEFICIARY') &
                            (injury_context['BENEFICIARY_OF'] == returning_player_name)
                        ]

                        if already_ex_beneficiary.empty:
                            # Get injury date from returning player's data
                            returning_player_full = team_players[team_players['PLAYER'] == returning_player_name]
                            if not returning_player_full.empty:
                                injury_date = returning_player_full.iloc[0].get('ESTIMATED_INJURY_DATE')
                                if pd.isna(injury_date) or injury_date is None:
                                    # Estimate as start of season
                                    current_year = pd.Timestamp(today).year
                                    season_start = pd.Timestamp(f"{current_year}-10-15").date()
                                    if pd.Timestamp(today).month < 7:
                                        season_start = pd.Timestamp(f"{current_year - 1}-10-15").date()
                                    injury_date = season_start

                                # Mark as EX_BENEFICIARY to exclude inflated games
                                logger.info(f"{player_name}: Marking as EX_BENEFICIARY of {returning_player_name} (excluding games {injury_date} to {return_date})")
                                injury_context = add_to_injury_context(
                                    injury_context,
                                    player_name=player_name,
                                    team=player_row['TEAM'],
                                    status='EX_BENEFICIARY',
                                    true_baseline=0,
                                    beneficiary_of=returning_player_name,
                                    injury_date=injury_date,
                                    return_date=return_date
                                )
    # Get set of players on injury report (any status - OUT, returning, etc.)
    if not injury_data.empty:
        injured_or_returning_players = set(injury_data['PLAYER'].unique())
    else:
        injured_or_returning_players = set()
    # Detect NEW season-long returns (not yet tracked in injury_context)
    # Example: Kawhi Leonard - played 30 MPG last season, 0 games this season, now healthy
    for _, teammate in team_players.iterrows():
        if (teammate.get('GAMES_PLAYED', 0) == 0 and           # Haven't played this season
            teammate.get('FROM_PREV_SEASON', False) and        # Data is from last season
            teammate.get('STATUS') != 'OUT' and                # Not currently injured (now healthy!)
            teammate.get('CAREER_AVG_MIN', 0) >= 20 and        # Must be significant player (filter out bench warmers)
            teammate['PLAYER'] in injured_or_returning_players):  # ← Must be/was on injury report (filters out free agents)
            # This teammate was out all season but is now returning!
            returning_player_name = teammate['PLAYER']

            # Get injury date (when they first got injured)
            # Use ESTIMATED_INJURY_DATE if available, otherwise estimate as start of season
            injury_date = teammate.get('ESTIMATED_INJURY_DATE')
            if pd.isna(injury_date) or injury_date is None:
                # Estimate as start of current season (October 15th of appropriate year)
                current_year = pd.Timestamp(today).year
                season_start = pd.Timestamp(f"{current_year}-10-15").date()
                if pd.Timestamp(today).month < 7:  # Before July = last year's season
                    season_start = pd.Timestamp(f"{current_year - 1}-10-15").date()
                injury_date = season_start
                logger.info(f"{returning_player_name}: No ESTIMATED_INJURY_DATE, using season start {injury_date}")

            # Check if this return is already tracked
            already_tracked = injury_context[
                (injury_context['PLAYER'] == returning_player_name) &
                (injury_context['STATUS'] == 'RETURNING')
            ]

            if already_tracked.empty:
                # Track this return in injury context
                logger.info(f"NEW season-long return detected: {returning_player_name} (out since {injury_date}, 10 MPG for 4 games)")
                injury_context = add_to_injury_context(
                    injury_context,
                    player_name=returning_player_name,
                    team=player_row['TEAM'],
                    status='RETURNING',
                    true_baseline=0,
                    beneficiary_of=None,
                    injury_date=today
                )

                # Mark if current player is affected
                if player_name == returning_player_name:
                    is_returning_player = True
                    max_returning_player_games = 0  # Just detected, 0 games played
                elif position_overlap_func(player_row.get('POSITION', 'SF'), teammate.get('POSITION', 'SF')):
                    # Mark affected teammate as EX_BENEFICIARY to exclude inflated games from baseline
                    affected_by_return = True
                    max_returning_player_games = 0  # Just detected, 0 games played

                    # Add EX_BENEFICIARY record (skipping BENEFICIARY status)
                    logger.info(f"{player_name}: Marking as EX_BENEFICIARY of {returning_player_name} (excluding games {injury_date} to {today})")
                    injury_context = add_to_injury_context(
                        injury_context,
                        player_name=player_name,
                        team=player_row['TEAM'],
                        status='EX_BENEFICIARY',
                        true_baseline=0,  # Will be calculated from non-inflated games
                        beneficiary_of=returning_player_name,
                        injury_date=injury_date,  # When returning player got injured
                        return_date=today  # When returning player returns
                    )
                    logger.info(f"{player_name}: Affected by {returning_player_name} return (10 MPG for 4 games, then exclude inflated games from baseline)")

    # ========== STEP 3: Apply 10 MPG for RETURNING STATUS (Scenario B) ==========
    # Both returning star (Kawhi) AND affected teammates (Norman Powell) get 10 MPG for first 4 games
    # This prevents overestimation while we collect data on new rotation
    # NOTE: This is NOT for "Returning TODAY" players (Scenario A) - they get 25% reduction instead
    if is_returning_player or affected_by_return:
        if max_returning_player_games is not None and max_returning_player_games < 4:
            logger.info(f"{player_name}: RETURNING STATUS scenario - using conservative 10 MPG (returning player has {max_returning_player_games}/4 games)")
            return 10, 'LOW', injury_context

    # ========== STEP 4: Get baseline and check for ACTIVE injuries ==========
    # At this point, no season-long returns affecting this player
    # Check if any teammates are currently OUT and redistribute their minutes
    confidence = update_confidence_status(player_row, injury_context, games_log)
    baseline = get_player_baseline(player_row, injury_context, box_scores)  # EX_BENEFICIARY logic applied here

    # Use pre-calculated injury redistributions if provided, otherwise calculate on-demand (legacy)
    if team_injury_redistributions is None:
        # Legacy behavior: Calculate on-demand (less efficient)
        team_injuries = team_players[team_players['STATUS'] == 'OUT']
        if 'RETURN_DATE_DT' in team_injuries.columns:
            teammates_returning_today = team_injuries['RETURN_DATE_DT'] <= today
            team_injuries = team_injuries[~teammates_returning_today]
            excluded_count = teammates_returning_today.sum()
            if excluded_count > 0:
                logger.debug(f"Excluded {excluded_count} teammates marked OUT but returning today (won't redistribute their minutes)")

        if not team_injuries.empty:
            team_injuries_sorted = team_injuries.sort_values('SEASON_AVG_MIN', ascending=False)

            # Try to find injury redistribution for this player
            for _, injured in team_injuries_sorted.iterrows():
                injury_projections, injury_context = redistribute_injury_minutes(
                    injured, team_players, injury_context, today, box_scores,
                    position_overlap_func, use_multiplier=use_multiplier
                )
                if player_name in injury_projections:
                    projected_min = injury_projections[player_name]

                    # Apply 25% reduction for RETURNING TODAY (Scenario A)
                    if returning_today:
                        projected_min = projected_min * 0.75
                        logger.info(f"{player_name}: RETURNING TODAY (Scenario A) - Applied 25% reduction: {injury_projections[player_name]:.1f} -> {projected_min:.1f} MPG")

                    return projected_min, confidence, injury_context
    else:
        # Optimized path: Use pre-calculated redistributions (no redundant calls)
        # Check if this player benefits from any injured teammate's minutes
        for _, injury_projections in team_injury_redistributions.items():
            if player_name in injury_projections:
                # Player benefits from this injury - use redistribution
                projected_min = injury_projections[player_name]

                # Apply 25% reduction for RETURNING TODAY (Scenario A)
                # Player still on injury report but expected to play today
                if returning_today:
                    projected_min = projected_min * 0.75
                    logger.info(f"{player_name}: RETURNING TODAY (Scenario A) - Applied 25% reduction: {injury_projections[player_name]:.1f} -> {projected_min:.1f} MPG")

                return projected_min, confidence, injury_context

        # Team has injuries but player doesn't benefit - fall through to Formula C

    # Use Formula C (either no injuries or player not affected by them)
    last_7_avg = player_row.get('LAST_7_AVG_MIN', baseline)
    prev_game = player_row.get('PREV_GAME_MIN', baseline)

    projected = 0.5 * baseline + 0.3 * last_7_avg + 0.2 * prev_game
    projected = min(projected, MAX_MINUTES)

    # Apply 25% reduction for RETURNING TODAY (Scenario A)
    # Player still on injury report but expected to play today
    if returning_today:
        original_proj = projected
        projected = projected * 0.75
        logger.info(f"{player_name}: RETURNING TODAY (Scenario A) - Applied 25% reduction: {original_proj:.1f} -> {projected:.1f} MPG")

    return projected, confidence, injury_context


# ==================== Model 1: Complex Position Overlap ====================

def project_minutes_complex(player_row, team_players, injury_context, games_log, today, box_scores, injury_data, team_injury_redistributions=None):
    """Complex position overlap with 2x multiplier for direct backups"""
    return project_minutes_with_injuries(
        player_row, team_players, injury_context, games_log, today, box_scores,
        position_overlap_func=position_overlap_complex,
        use_multiplier=True,
        injury_data=injury_data,
        team_injury_redistributions=team_injury_redistributions
    )


# ==================== Model 2: Direct Position Exchange ====================

def project_minutes_direct(player_row, team_players, injury_context, games_log, today, box_scores, injury_data, team_injury_redistributions=None):
    """Exact position match only, no 2x multiplier"""
    return project_minutes_with_injuries(
        player_row, team_players, injury_context, games_log, today, box_scores,
        position_overlap_func=position_overlap_exact,
        use_multiplier=False,
        injury_data=injury_data,
        team_injury_redistributions=team_injury_redistributions
    )


# ==================== Model 3: Formula C Baseline (No Injury Handling) ====================

def project_minutes_formula_c(player_row, today):
    """Pure Formula C - no injury redistribution"""
    # Check if player is returning today (still on injury report but expected to play)
    returning_today = False
    if 'RETURN_DATE_DT' in player_row and pd.notna(player_row.get('RETURN_DATE_DT')):
        return_date = player_row['RETURN_DATE_DT']
        if isinstance(return_date, pd.Timestamp):
            return_date = return_date.date()
        if return_date == today:
            returning_today = True

    # If OUT but returning today, treat as playing (with 25% reduction)
    if player_row.get('STATUS') == 'OUT' and not returning_today:
        return 0, 'HIGH'

    # Get baseline with fallbacks (conservative approach)
    if player_row.get('GAMES_PLAYED', 0) >= 4:
        # Reliable: 4+ games this season
        baseline = player_row.get('SEASON_AVG_MIN', 10)
    elif player_row.get('FROM_PREV_SEASON', False):
        # Season-long injury (0 games this season) - check if same team
        prev_team = player_row.get('PREV_TEAM')
        current_team = player_row.get('TEAM')

        if prev_team == current_team:
            # Same team - use previous season baseline (reliable)
            baseline = player_row.get('SEASON_AVG_MIN', 10)
        else:
            # TRADED - previous season data unreliable, use conservative fallback
            baseline = 10
    else:
        # < 4 games this season - UNRELIABLE, use conservative fallback
        baseline = 10

    last_7_avg = player_row.get('LAST_7_AVG_MIN', baseline)
    prev_game = player_row.get('PREV_GAME_MIN', baseline)

    # Formula C: 50% season, 30% last 7, 20% prev game
    projected = 0.5 * baseline + 0.3 * last_7_avg + 0.2 * prev_game
    projected = min(projected, MAX_MINUTES)

    # Apply 25% reduction if returning today
    if returning_today:
        original_proj = projected
        projected = projected * 0.75
        logger.info(f"{player_row['PLAYER']}: Applied 25% reduction (returning today): {original_proj:.1f} -> {projected:.1f} MPG")

    return projected, 'HIGH'


# ==================== Post-Game Actual Minutes Update ====================

def update_actual_minutes(box_scores, target_date=None, today=None):
    """
    Update ACTUAL_MIN and ACTUAL_FP for all 5 models after games complete

    Args:
        box_scores: DataFrame of box score data
        target_date: Date to update (defaults to yesterday based on today)
        today: Current date in Eastern time (used to calculate default target_date)
    """
    if target_date is None:
        if today is None:
            eastern = pytz.timezone('US/Eastern')
            today = datetime.now(eastern).date()
        target_date = today - timedelta(days=1)

    logger.info(f"Updating actual minutes and FP for {target_date}")

    try:
        # Use provided box scores
        df_box = box_scores
        if df_box.empty:
            logger.warning("No box score data available")
            return 0

        # Filter for target date and extract actual minutes + FP
        df_box['GAME_DATE'] = pd.to_datetime(df_box['GAME_DATE']).dt.date
        df_actuals = df_box[df_box['GAME_DATE'] == target_date][['PLAYER', 'GAME_DATE', 'MIN', 'FP']].rename(
            columns={'MIN': 'ACTUAL_MIN', 'FP': 'ACTUAL_FP'}
        )

        if df_actuals.empty:
            logger.warning(f"No box scores found for {target_date}")
            return 0

        logger.info(f"Found {len(df_actuals)} actual results for {target_date}")

        # Update all 5 model files
        models = [
            'complex_position_overlap',
            'direct_position_only',
            'formula_c_baseline',
            'sportsline_baseline',
            'daily_fantasy_fuel_baseline'
        ]

        updated_count = 0
        for model in models:
            # Update minutes projections
            key = f'model_comparison/{model}/minutes_projections.parquet'
            df_proj = load_from_s3(key)

            if not df_proj.empty:
                df_proj['DATE'] = pd.to_datetime(df_proj['DATE']).dt.date

                # Merge actual data
                df_proj = df_proj.merge(
                    df_actuals[['PLAYER', 'ACTUAL_MIN', 'ACTUAL_FP']],
                    on='PLAYER',
                    how='left',
                    suffixes=('', '_new')
                )

                # Update only for target date
                mask = df_proj['DATE'] == target_date
                if 'ACTUAL_MIN_new' in df_proj.columns:
                    df_proj.loc[mask, 'ACTUAL_MIN'] = df_proj.loc[mask, 'ACTUAL_MIN_new']
                    df_proj = df_proj.drop(columns=['ACTUAL_MIN_new'])
                if 'ACTUAL_FP_new' in df_proj.columns:
                    df_proj.loc[mask, 'ACTUAL_FP'] = df_proj.loc[mask, 'ACTUAL_FP_new']
                    df_proj = df_proj.drop(columns=['ACTUAL_FP_new'])

                save_to_s3(df_proj, key)
                updated_count += len(df_proj[mask])
                logger.info(f"Updated {len(df_proj[mask])} projection records for {model}")

            # Update lineups
            lineup_key = f'model_comparison/{model}/daily_lineups.parquet'
            df_lineup = load_from_s3(lineup_key)

            if not df_lineup.empty:
                df_lineup['DATE'] = pd.to_datetime(df_lineup['DATE']).dt.date

                # Merge actual FP for lineup players
                df_lineup = df_lineup.merge(
                    df_actuals[['PLAYER', 'ACTUAL_FP']],
                    on='PLAYER',
                    how='left',
                    suffixes=('', '_new')
                )

                # Update only for target date
                lineup_mask = df_lineup['DATE'] == target_date
                if 'ACTUAL_FP_new' in df_lineup.columns:
                    df_lineup.loc[lineup_mask, 'ACTUAL_FP'] = df_lineup.loc[lineup_mask, 'ACTUAL_FP_new']
                    df_lineup = df_lineup.drop(columns=['ACTUAL_FP_new'])

                save_to_s3(df_lineup, lineup_key)
                logger.info(f"Updated {lineup_mask.sum()} lineup records for {model}")

        return updated_count

    except Exception as e:
        logger.error(f"Error updating actuals: {str(e)}", exc_info=True)
        raise


# ==================== Main Lambda Handler ====================

def lambda_handler(event, context):
    """
    Generate projections for all 4 models OR update actual minutes

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
    print(f"Result: {result}")
