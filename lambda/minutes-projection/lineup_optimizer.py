"""
Fantasy points prediction and lineup optimization
"""

import pandas as pd
import numpy as np
import json
import logging
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD
from s3_utils import load_model_from_s3
from config import s3_client, BUCKET_NAME, MAX_MINUTES

logger = logging.getLogger()


# ==================== Fantasy Points Prediction ====================

def calculate_fp_features(box_scores, daily_predictions, today):
    """
    Extract FP features from box scores (matches supervised-learning training pipeline)
    """
    logger.info("Extracting FP features from box scores")

    # Ensure GAME_DATE is datetime
    box_scores = box_scores.copy()
    box_scores['GAME_DATE'] = pd.to_datetime(box_scores['GAME_DATE'])

    # Calculate current season from today's date (match training logic)
    today_dt = pd.to_datetime(today)
    current_season = f"{today_dt.year}-{str(today_dt.year + 1)[-2:]}" if today_dt.month >= 10 else f"{today_dt.year - 1}-{str(today_dt.year)[-2:]}"

    # Extract season from box scores GAME_DATE
    box_scores['SEASON'] = box_scores['GAME_DATE'].apply(
        lambda x: f"{x.year}-{str(x.year + 1)[-2:]}" if x.month >= 10 else f"{x.year - 1}-{str(x.year)[-2:]}"
    )

    # Count games played this season for each player
    current_season_games = box_scores[box_scores['SEASON'] == current_season].copy()
    current_season_games = current_season_games.sort_values(['PLAYER', 'GAME_DATE'])
    season_game_count = current_season_games.groupby('PLAYER').size().reset_index(name='SEASON_GAMES_PLAYED')

    # Get most recent game for each player (has latest rolling averages)
    box_scores_sorted = box_scores.sort_values(['PLAYER', 'GAME_DATE'], ascending=[True, False])
    latest_stats = box_scores_sorted.groupby('PLAYER').first().reset_index()

    # Merge season game count
    latest_stats = latest_stats.merge(season_game_count, on='PLAYER', how='left')
    latest_stats['SEASON_GAMES_PLAYED'] = latest_stats['SEASON_GAMES_PLAYED'].fillna(0).astype(int)

    # Extract FP feature columns (match supervised-learning training)
    fp_cols = ['PLAYER', 'Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
               'Career_FP_Avg', 'Games_Played_Career', 'CLUSTER',
               'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg', 'SEASON_GAMES_PLAYED']

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
    Add PROJECTED_FP columns to projections using 3 trained ML models
    Matches supervised-learning training pipeline with one-hot encoding

    Args:
        projections_df: DataFrame with PLAYER, PROJECTED_MIN columns
        box_scores: Historical box scores for feature calculation
        daily_predictions: Today's game context (opponent, home/away)
        today: Current date

    Returns:
        DataFrame with added PROJECTED_FP_current, PROJECTED_FP_fp_per_min, PROJECTED_FP_barebones columns
    """
    logger.info("Predicting fantasy points for projections")

    # Load all 3 trained FP models
    model_names = ['current', 'fp_per_min', 'barebones']
    models = {}
    feature_sets = {}

    for model_name in model_names:
        # Load model
        model = load_model_from_s3(f'models/{model_name}.pkl')
        if model is None:
            logger.warning(f"{model_name} model not found - will set predictions to 0")
            continue
        models[model_name] = model

        # Load expected feature names from S3 (saved during training)
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=f'models/{model_name}_feature_names.json')
            feature_names_data = json.loads(response['Body'].read())
            expected_features = feature_names_data['features']
            feature_sets[model_name] = expected_features
            logger.info(f"Loaded {model_name} model with {len(expected_features)} features")
        except Exception as e:
            logger.warning(f"Could not load feature names for {model_name}: {e}")
            feature_sets[model_name] = None

    if not models:
        logger.warning("No FP prediction models found - setting all PROJECTED_FP columns to 0")
        projections_df['PROJECTED_FP_current'] = 0
        projections_df['PROJECTED_FP_fp_per_min'] = 0
        projections_df['PROJECTED_FP_barebones'] = 0
        return projections_df

    # Calculate FP features from historical data
    fp_features = calculate_fp_features(box_scores, daily_predictions, today)

    # Merge projections with FP features
    df = projections_df.merge(fp_features, on='PLAYER', how='left')

    # Use PROJECTED_MIN as MIN (match training which used actual MIN)
    df['MIN'] = df['PROJECTED_MIN']

    # Calculate FP_PER_MIN feature (match training pipeline exactly)
    # First 5 games of season: Use Career_FP_Avg / Career_MIN_Avg
    # After 5 games: Use Season_FP_Avg / Season_MIN_Avg
    df['SEASON_GAMES_PLAYED'] = df['SEASON_GAMES_PLAYED'].fillna(0).astype(int)

    df['FP_PER_MIN'] = np.where(
        df['SEASON_GAMES_PLAYED'] <= 5,
        # First 5 games: Career FP/MIN
        np.where(
            df['Career_MIN_Avg'] > 0,
            df['Career_FP_Avg'] / df['Career_MIN_Avg'],
            0
        ),
        # After 5 games: Season FP/MIN
        np.where(
            df['Season_MIN_Avg'] > 0,
            df['Season_FP_Avg'] / df['Season_MIN_Avg'],
            0
        )
    )
    df['FP_PER_MIN'] = df['FP_PER_MIN'].replace([np.inf, -np.inf], 0).fillna(0)

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

    # Set predictions to 0 for players with no FP history
    no_history_mask = (df['Season_FP_Avg'] == 0) & (df['Career_FP_Avg'] == 0)
    if no_history_mask.any():
        logger.info(f"Found {no_history_mask.sum()} players with no FP history - will set predictions to 0")

    # Define base feature sets for each model (BEFORE one-hot encoding)
    # Note: OPPONENT removed - will be replaced with opponent defensive rating later
    model_base_features = {
        'current': ['Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
                    'Career_FP_Avg', 'Games_Played_Career', 'CLUSTER', 'MIN',
                    'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg',
                    'IS_HOME', 'REST_DAYS'],

        'fp_per_min': ['Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
                       'Career_FP_Avg', 'Games_Played_Career', 'CLUSTER', 'MIN',
                       'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg',
                       'IS_HOME', 'REST_DAYS', 'FP_PER_MIN'],

        'barebones': ['FP_PER_MIN', 'MIN', 'REST_DAYS', 'CLUSTER']
    }

    # Run predictions for each model
    for model_name in model_names:
        if model_name not in models:
            df[f'PROJECTED_FP_{model_name}'] = 0
            continue

        # Get base features for this model
        base_cols = model_base_features[model_name]

        # Extract and prepare features
        df_features = df[base_cols].copy()

        # One-hot encode categorical variables (only if they exist in feature set)
        categorical_cols = []
        if 'CLUSTER' in df_features.columns:
            categorical_cols.append('CLUSTER')

        if categorical_cols:
            df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=False)

        # Align columns with expected features from training
        expected_features = feature_sets.get(model_name)
        if expected_features:
            # Add missing columns (fill with 0)
            for col in expected_features:
                if col not in df_features.columns:
                    df_features[col] = 0

            # Keep only expected columns in the same order
            df_features = df_features[expected_features]
            logger.info(f"{model_name}: Aligned {len(df_features.columns)} features")

        # Make predictions
        X = df_features.values
        predictions = models[model_name].predict(X)
        df[f'PROJECTED_FP_{model_name}'] = predictions.round(1)

        # Override predictions for players with no history
        df.loc[no_history_mask, f'PROJECTED_FP_{model_name}'] = 0

        logger.info(f"{model_name}: Predicted FP (avg: {df[f'PROJECTED_FP_{model_name}'].mean():.1f})")

    # Add default PROJECTED_FP column (use current model for backward compatibility)
    # Other systems can choose which model to use via PROJECTED_FP_current, etc.
    if 'PROJECTED_FP_current' in df.columns:
        df['PROJECTED_FP'] = df['PROJECTED_FP_current']
    elif 'PROJECTED_FP_fp_per_min' in df.columns:
        df['PROJECTED_FP'] = df['PROJECTED_FP_fp_per_min']
    elif 'PROJECTED_FP_barebones' in df.columns:
        df['PROJECTED_FP'] = df['PROJECTED_FP_barebones']
    else:
        df['PROJECTED_FP'] = 0

    # Keep only original projection columns + all PROJECTED_FP columns
    result_cols = [col for col in projections_df.columns] + ['PROJECTED_FP'] + [f'PROJECTED_FP_{name}' for name in model_names]
    result_df = df[result_cols].copy()

    logger.info(f"Predicted FP for {len(result_df)} players using {len(models)} models")

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
