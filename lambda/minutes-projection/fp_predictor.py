"""
Fantasy Points Predictor
Predicts MY_MODEL_PREDICTED_FP using complex-position minutes
"""

import pandas as pd
import logging
import pickle
import json
from s3_utils import load_from_s3, save_to_s3
import boto3
from io import BytesIO

logger = logging.getLogger()
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'


def load_model_from_s3(key):
    """Load sklearn model from S3"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pickle.load(BytesIO(obj['Body'].read()))
    except Exception as e:
        logger.error(f"Error loading model from {key}: {e}")
        raise


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def predict_fantasy_points(daily_preds, box_scores, today):
    """
    Generate ML model predictions for today's playing players

    Args:
        daily_preds: DataFrame with today's players (has PROJECTED_MIN from complex-position model)
        box_scores: DataFrame with historical box scores
        today: Today's date

    Returns:
        Updated daily_preds DataFrame with MY_MODEL_PREDICTED_FP filled in
    """
    logger.info("Starting fantasy points predictions with complex-position minutes")

    try:
        # Load trained model
        logger.info("Loading RF model from S3")
        model = load_model_from_s3('models/RFCluster.sav')

        # Filter for today's games
        todays_preds = daily_preds[daily_preds['GAME_DATE'] == today].copy()

        if todays_preds.empty:
            logger.warning("No predictions for today found in daily_predictions")
            return daily_preds

        # Clean historical data
        box_scores = box_scores[box_scores['MIN'] != 0].copy()
        box_scores = box_scores.dropna(subset=['WL'])
        box_scores['GAME_DATE'] = pd.to_datetime(box_scores['GAME_DATE'])
        box_scores = box_scores.sort_values(by=['PLAYER', 'GAME_DATE'], ascending=[True, True])

        # Get each player's most recent game (for feature extraction)
        player_last_games = box_scores.groupby('PLAYER').tail(1).copy()

        # Calculate rest days from last game to today
        today_ts = pd.Timestamp(today)
        player_last_games['REST_DAYS'] = (today_ts - player_last_games['GAME_DATE']).dt.days
        player_last_games['REST_DAYS'] = player_last_games['REST_DAYS'].clip(0, 30)

        # Filter to only players playing today
        todays_players = todays_preds['PLAYER'].unique()
        todays_player_features = player_last_games[player_last_games['PLAYER'].isin(todays_players)].copy()

        # Save historical MIN as PREV_MIN before merging
        if 'MIN' in todays_player_features.columns:
            todays_player_features['PREV_MIN_HIST'] = todays_player_features['MIN']

        todays_player_features = todays_player_features.drop(columns=['MIN'], errors='ignore')

        # Merge with today's projected minutes (from complex-position model)
        todays_preds_cleaned = todays_preds[['PLAYER', 'PROJECTED_MIN', 'SALARY', 'POSITION', 'TEAM_ABBREVIATION']].copy()
        todays_preds_cleaned = todays_preds_cleaned.rename(columns={'TEAM_ABBREVIATION': 'TEAM'})
        todays_preds_cleaned['MIN'] = pd.to_numeric(todays_preds_cleaned['PROJECTED_MIN'], errors='coerce')
        todays_preds_cleaned['SALARY'] = pd.to_numeric(todays_preds_cleaned['SALARY'], errors='coerce')

        todays_player_features = todays_player_features.merge(todays_preds_cleaned, on='PLAYER', how='left')

        # Set IS_HOME and OPPONENT to defaults (no matchup data available)
        todays_player_features['IS_HOME'] = 0
        todays_player_features['OPPONENT'] = 'ATL'  # Placeholder

        # Fill missing clusters
        todays_player_features['CLUSTER'] = todays_player_features['CLUSTER'].fillna('CLUSTER_NAN')

        # One-hot encode CLUSTER and OPPONENT
        ALL_NBA_TEAMS = [
            'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        ]

        todays_player_features = pd.get_dummies(todays_player_features, columns=['CLUSTER', 'OPPONENT'], drop_first=False)

        # Add missing opponent columns
        for team in ALL_NBA_TEAMS:
            opponent_col = f'OPPONENT_{team}'
            if opponent_col not in todays_player_features.columns:
                todays_player_features[opponent_col] = 0

        # Add missing cluster columns (0-14 plus NAN)
        for cluster_num in range(15):
            cluster_col = f'CLUSTER_{float(cluster_num)}'
            if cluster_col not in todays_player_features.columns:
                todays_player_features[cluster_col] = 0

        if 'CLUSTER_CLUSTER_NAN' not in todays_player_features.columns:
            todays_player_features['CLUSTER_CLUSTER_NAN'] = 0

        # Remove invalid opponent columns
        valid_opponent_columns = [f'OPPONENT_{team}' for team in ALL_NBA_TEAMS]
        invalid_opponent_cols = [col for col in todays_player_features.columns
                                if col.startswith('OPPONENT_') and col not in valid_opponent_columns]

        if invalid_opponent_cols:
            logger.info(f"Removing {len(invalid_opponent_cols)} invalid opponent columns")
            todays_player_features = todays_player_features.drop(columns=invalid_opponent_cols)

        # Get expected features from model
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            # Load from S3
            feature_names_obj = s3.get_object(Bucket=BUCKET_NAME, Key='models/RFCluster_feature_names.json')
            feature_names_data = json.loads(feature_names_obj['Body'].read().decode('utf-8'))
            expected_features = feature_names_data['features']

        logger.info(f"Model expects {len(expected_features)} features")

        # Add missing features as 0
        missing_features = [col for col in expected_features if col not in todays_player_features.columns]
        if missing_features:
            logger.info(f"Adding {len(missing_features)} missing features as 0")
            for col in missing_features:
                todays_player_features[col] = 0

        # Select features in exact order
        model_input_features = todays_player_features[expected_features]

        if model_input_features.empty:
            logger.warning("No data available for prediction")
            return daily_preds

        # Handle NaN values
        nan_counts = model_input_features.isna().sum()
        if nan_counts.any():
            logger.warning(f"Found NaN values in features, filling with 0")
            model_input_features = model_input_features.fillna(0)

        # Generate predictions
        logger.info(f"Making FP predictions for {len(model_input_features)} players")
        predictions = model.predict(model_input_features.to_numpy())

        # Update daily_preds with predictions
        todays_player_features['MY_MODEL_PREDICTED_FP'] = predictions

        # Merge predictions back into daily_preds
        for idx, row in todays_player_features.iterrows():
            player = row['PLAYER']
            prediction = row['MY_MODEL_PREDICTED_FP']

            mask = (daily_preds['PLAYER'] == player) & (daily_preds['GAME_DATE'] == today)
            if mask.any():
                daily_preds.loc[mask, 'MY_MODEL_PREDICTED_FP'] = safe_float(prediction)

                # Also update historical stats if available
                if 'FP' in row and pd.notna(row['FP']):
                    daily_preds.loc[mask, 'PREV_FP'] = safe_float(row['FP'])
                if 'PREV_MIN_HIST' in row and pd.notna(row['PREV_MIN_HIST']):
                    daily_preds.loc[mask, 'PREV_MIN'] = safe_float(row['PREV_MIN_HIST'])
                if 'Season_FP_Avg' in row and pd.notna(row['Season_FP_Avg']):
                    daily_preds.loc[mask, 'SEASON_AVG_FP'] = safe_float(row['Season_FP_Avg'])
                if 'Season_MIN_Avg' in row and pd.notna(row['Season_MIN_Avg']):
                    daily_preds.loc[mask, 'SEASON_AVG_MIN'] = safe_float(row['Season_MIN_Avg'])

        logger.info(f"Successfully predicted FP for {len(todays_player_features)} players")

        return daily_preds
   
    except Exception as e:
        logger.error(f"Error in FP predictions: {str(e)}", exc_info=True)
        return daily_preds
   