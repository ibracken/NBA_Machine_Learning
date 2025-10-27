import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import boto3
import json
import pickle
from io import BytesIO
import logging
from datetime import datetime, timezone, timedelta
import pytz

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'

# S3 utility functions
def save_dataframe_to_s3(df, key):
    """Save DataFrame as Parquet to S3"""
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_buffer.getvalue())
    logger.info(f"Saved {len(df)} records to s3://{BUCKET_NAME}/{key}")

def load_dataframe_from_s3(key):
    """Load DataFrame from S3 Parquet"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except Exception as e:
        logger.error(f"Error loading data from {key}: {e}")
        raise

def save_model_to_s3(model, key):
    """Save sklearn model to S3"""
    model_buffer = BytesIO()
    pickle.dump(model, model_buffer)
    model_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=model_buffer.getvalue())
    logger.info(f"Saved model to s3://{BUCKET_NAME}/{key}")

def calculate_rest_days(df):
    """Calculate rest days for each player based on their previous game"""
    logger.info("Calculating rest days for players")
    
    # Create a copy to avoid modifying the original
    df_sorted = df.copy()
    
    # Ensure GAME_DATE is datetime
    df_sorted['GAME_DATE'] = pd.to_datetime(df_sorted['GAME_DATE'])
    
    # Sort by player and game date (ascending for proper calculation)
    df_sorted = df_sorted.sort_values(['PLAYER', 'GAME_DATE'])
    
    # Calculate days since last game for each player
    df_sorted['PREV_GAME_DATE'] = df_sorted.groupby('PLAYER')['GAME_DATE'].shift(1)
    df_sorted['REST_DAYS'] = (df_sorted['GAME_DATE'] - df_sorted['PREV_GAME_DATE']).dt.days
    
    # For first games (no previous game), set rest days to 3 (reasonable default)
    df_sorted['REST_DAYS'] = df_sorted['REST_DAYS'].fillna(3).astype(int)
    
    # For predictions (current games), calculate rest days from most recent game to today
    # Get current date in ET timezone
    et_tz = pytz.timezone('America/New_York')
    today = datetime.now(et_tz).date()
    
    # Get most recent game for each player
    most_recent_games = df_sorted.groupby('PLAYER')['GAME_DATE'].max().reset_index()
    most_recent_games['MOST_RECENT_DATE'] = most_recent_games['GAME_DATE']
    
    # Merge back and update REST_DAYS for prediction cases
    df_sorted = df_sorted.merge(most_recent_games[['PLAYER', 'MOST_RECENT_DATE']], on='PLAYER', how='left')
    
    # Calculate rest days from most recent game to today for predictions
    current_rest_days = (pd.Timestamp(today) - df_sorted['MOST_RECENT_DATE']).dt.days
    
    # For the most recent games, use the calculated rest days from today
    is_most_recent = df_sorted['GAME_DATE'] == df_sorted['MOST_RECENT_DATE']
    df_sorted.loc[is_most_recent, 'REST_DAYS'] = current_rest_days[is_most_recent]
    
    # Clean up temporary columns
    df_sorted = df_sorted.drop(columns=['PREV_GAME_DATE', 'MOST_RECENT_DATE'])
    
    # Ensure REST_DAYS is non-negative and reasonable (cap at 30 days for outliers)
    df_sorted['REST_DAYS'] = df_sorted['REST_DAYS'].clip(0, 30)
    
    logger.info(f"Rest days calculated: min={df_sorted['REST_DAYS'].min()}, max={df_sorted['REST_DAYS'].max()}, mean={df_sorted['REST_DAYS'].mean():.2f}")
    
    return df_sorted

def run_supervised_learning():
    """Main function to train supervised learning model for fantasy point prediction"""
    logger.info("Starting supervised learning model training")
    
    try:
        # Load box scores data from multiple seasons
        logger.info("Loading box scores data from S3 (multiple seasons)")
        df = load_dataframe_from_s3('data/box_scores/current.parquet')
        df2 = load_dataframe_from_s3('data/box_scores/2024-2025.parquet')
        df3 = load_dataframe_from_s3('data/box_scores/2023-2024.parquet')
        df4 = load_dataframe_from_s3('data/box_scores/2022-2023.parquet')
        df = pd.concat([df, df2, df3, df4])
        logger.info(f"Loaded {len(df)} box score records from multiple seasons")
        
        # Validate box scores data structure
        required_box_score_cols = ['PLAYER', 'GAME_DATE', 'FP', 'Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg', 'MIN', 'MATCHUP']
        missing_cols = [col for col in required_box_score_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns in box scores data: {missing_cols}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # Validate data types and ranges
        if df['FP'].isna().all():
            error_msg = "All FP values are null"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        if len(df) == 0:
            error_msg = "Box scores data is empty"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        logger.info(f"Box scores data validation passed: {len(df)} records with required columns")
        
        # Filter out players with zero minutes (data validation)
        if 'MIN' in df.columns:
            original_count = len(df)
            df = df[df['MIN'] != 0]
            logger.info(f"Filtered from {original_count} to {len(df)} records (MIN != 0)")
        
        # Sort by game date (most recent first)
        df = df.sort_values(by=['GAME_DATE'], ascending=[False])
        
        # Data preprocessing for MIN column (convert to numeric and add noise as in notebook)
        df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
        df['MIN'] = df['MIN'] + np.random.uniform(-5, 5, size=len(df))
        df['MIN'] = df['MIN'].clip(lower=0)
        
        # Handle missing clusters with placeholder
        df['CLUSTER'] = df['CLUSTER'].fillna('CLUSTER_NAN')
        
        # Parse MATCHUP to extract home/away and opponent
        def parse_matchup(matchup_str):
            if pd.isna(matchup_str):
                return 0, 'UNKNOWN'
            matchup_str = str(matchup_str)
            if ' @ ' in matchup_str:
                # Away game: "TOR @ NYK"
                teams = matchup_str.split(' @ ')
                return 0, teams[1] if len(teams) > 1 else 'UNKNOWN'
            elif ' vs. ' in matchup_str:
                # Home game: "TOR vs. PHX" 
                teams = matchup_str.split(' vs. ')
                return 1, teams[1] if len(teams) > 1 else 'UNKNOWN'
            else:
                return 0, 'UNKNOWN'
        
        # Apply parsing
        matchup_parsed = df['MATCHUP'].apply(parse_matchup)
        df['IS_HOME'] = matchup_parsed.apply(lambda x: x[0])
        df['OPPONENT'] = matchup_parsed.apply(lambda x: x[1])
        
        logger.info(f"Parsed MATCHUP: {(df['IS_HOME'] == 1).sum()} home games, {(df['IS_HOME'] == 0).sum()} away games")
        
        # Calculate rest days for each player
        df = calculate_rest_days(df)

        # Handle NaN values in rolling averages (from first games of season)
        logger.info("Handling NaN values in rolling averages")

        # Fill missing rolling averages with season average as fallback
        df['Last3_FP_Avg'] = df['Last3_FP_Avg'].fillna(df['Season_FP_Avg'])
        df['Last5_FP_Avg'] = df['Last5_FP_Avg'].fillna(df['Season_FP_Avg'])
        df['Last7_FP_Avg'] = df['Last7_FP_Avg'].fillna(df['Season_FP_Avg'])

        # For first games with no season average, use 0 (model will learn this represents unknown)
        df['Last3_FP_Avg'] = df['Last3_FP_Avg'].fillna(0)
        df['Last5_FP_Avg'] = df['Last5_FP_Avg'].fillna(0)
        df['Last7_FP_Avg'] = df['Last7_FP_Avg'].fillna(0)
        df['Season_FP_Avg'] = df['Season_FP_Avg'].fillna(0)

        # Log NaN handling results
        logger.info(f"After NaN handling - Last3_FP_Avg nulls: {df['Last3_FP_Avg'].isna().sum()}")
        logger.info(f"After NaN handling - Season_FP_Avg nulls: {df['Season_FP_Avg'].isna().sum()}")

        # Define features and target
        feature_names = ['Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg', 'CLUSTER', 'MIN', 'IS_HOME', 'OPPONENT', 'REST_DAYS']
        label_name = 'FP'
        
        # Validate feature columns exist
        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            error_msg = f"Missing feature columns: {missing_features}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # Prepare feature dataframe
        df_features = df[feature_names].copy()
        df_labels = df[label_name].copy()
        
        # Validate feature data quality
        numeric_features = ['Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg', 'MIN']
        for feature in numeric_features:
            if df_features[feature].isna().all():
                error_msg = f"All values in feature '{feature}' are null"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
        
        # Check for reasonable data ranges
        if (df_features['MIN'] < 0).any():
            logger.warning("Found negative MIN values after preprocessing")
        
        logger.info(f"Feature validation passed: {len(df_features)} records with {len(feature_names)} features")
        
        # One-hot encode categorical features
        logger.info("One-hot encoding categorical features")
        df_features = pd.get_dummies(df_features, columns=['CLUSTER', 'OPPONENT'], drop_first=False)
        
        # Convert to numpy arrays
        features = np.array(df_features)
        labels = np.array(df_labels)
        
        # Validate arrays for training
        if features.shape[0] == 0:
            error_msg = "No features available for training"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        if len(labels) == 0:
            error_msg = "No labels available for training"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        if features.shape[0] != len(labels):
            error_msg = f"Feature-label mismatch: {features.shape[0]} features vs {len(labels)} labels"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # Check for sufficient data for train-test split
        if len(features) < 10:
            error_msg = f"Insufficient data for training: {len(features)} records (need at least 10)"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        logger.info(f"Training data validation passed: {features.shape[0]} samples with {features.shape[1]} features")
        
        # Store additional data for test results
        players = df['PLAYER']
        game_dates = df['GAME_DATE']
        
        # Train-test split
        logger.info("Splitting data for training and testing")
        train, test, train_labels, test_labels, train_players, test_players, train_dates, test_dates = train_test_split(
            features, labels, players, game_dates, 
            test_size=0.25, 
            random_state=30
        )
        
        # Reset indices for test data
        test_players = test_players.reset_index(drop=True)
        test_dates = test_dates.reset_index(drop=True)
        
        # Train RandomForest model
        logger.info("Training RandomForest model")
        logger.info(f"Training set size: {len(train)}, Test set size: {len(test)}")
        logger.info(f"Training data shape: {train.shape}")
        logger.info(f"Training labels shape: {train_labels.shape}")
        logger.info(f"Feature count: {train.shape[1]}")
        logger.info(f"Total records being sent to RandomForest: {train.shape[0]}")

        # Final validation check for any remaining NaN/Inf values
        try:
            # Convert to float to check for NaN/Inf (handles mixed types)
            train_float = train.astype(float)
            if np.isnan(train_float).any() or np.isinf(train_float).any():
                error_msg = "Training data still contains NaN or Inf values after preprocessing"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not validate NaN/Inf due to data types: {e}")

        rf = RandomForestRegressor(random_state=4)
        rf.fit(train, train_labels.ravel())
        
        # Save trained model to S3
        save_model_to_s3(rf, "models/RFCluster.sav")
        
        # Generate predictions
        logger.info("Generating predictions")
        train_predictions = rf.predict(train)
        test_predictions = rf.predict(test)
        
        # Create results dataframe
        logger.info("Creating test results dataframe")
        
        # Get categorical columns for reverse mapping
        cluster_columns = [col for col in df_features.columns if col.startswith('CLUSTER_')]
        opponent_columns = [col for col in df_features.columns if col.startswith('OPPONENT_')]
        
        # Create test results dataframe
        test_df = pd.DataFrame(test, columns=df_features.columns)
        
        # Map back to original cluster and opponent values
        test_df['CLUSTER'] = test_df[cluster_columns].idxmax(axis=1)
        test_df['OPPONENT'] = test_df[opponent_columns].idxmax(axis=1) if opponent_columns else 'UNKNOWN'
        
        def safe_cluster_map(value):
            if isinstance(value, str) and value.startswith('CLUSTER_'):
                try:
                    cluster_value = value.split('_')[-1]
                    return float(cluster_value) if cluster_value != 'NAN' else np.nan
                except ValueError:
                    return np.nan
            return np.nan
        
        def safe_opponent_map(value):
            if isinstance(value, str) and value.startswith('OPPONENT_'):
                return value.split('_', 1)[-1]  # Get everything after first underscore
            return 'UNKNOWN'
        
        test_df['CLUSTER'] = test_df['CLUSTER'].map(safe_cluster_map)
        test_df['OPPONENT'] = test_df['OPPONENT'].map(safe_opponent_map)
        
        # Drop one-hot encoded columns
        test_df = test_df.drop(columns=cluster_columns + opponent_columns)
        
        # Add metadata and predictions
        test_df['PLAYER'] = test_players
        test_df['GAME_DATE'] = test_dates
        test_df['ACTUAL'] = test_labels
        test_df['PREDICTED'] = test_predictions
        test_df['ERROR'] = abs(test_df['ACTUAL'] - test_df['PREDICTED'])
        
        # Reorder columns
        test_df = test_df[['PLAYER'] + [col for col in test_df.columns if col != 'PLAYER']]
        
        # Save test predictions to S3
        save_dataframe_to_s3(test_df, 'data/test_player_predictions/current.parquet')
        
        # Calculate model performance metrics
        train_error = np.mean(abs(train_predictions - train_labels.ravel()))
        test_error = np.mean(abs(test_predictions - test_labels.ravel()))
        
        # Calculate feature importance
        feature_importance = dict(zip(df_features.columns, rf.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        logger.info("Supervised learning model training completed successfully")
        logger.info(f"Training error (MAE): {train_error:.3f}")
        logger.info(f"Test error (MAE): {test_error:.3f}")
        
        return {
            'success': True,
            'total_records': len(df),
            'training_records': len(train),
            'test_records': len(test),
            'train_error_mae': float(train_error),
            'test_error_mae': float(test_error),
            'feature_count': len(df_features.columns),
            'top_features': [(feat, float(imp)) for feat, imp in top_features],
            'players_with_clusters': int((~df['CLUSTER'].str.contains('NAN', na=False)).sum()),
            'players_without_clusters': int((df['CLUSTER'].str.contains('NAN', na=False)).sum())
        }
        
    except Exception as e:
        logger.error(f"Error in supervised learning: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Supervised learning Lambda function started")
    
    try:
        result = run_supervised_learning()
        
        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Supervised learning model training completed successfully',
                    'total_records': result['total_records'],
                    'training_records': result['training_records'],
                    'test_records': result['test_records'],
                    'train_error_mae': result['train_error_mae'],
                    'test_error_mae': result['test_error_mae'],
                    'feature_count': result['feature_count'],
                    'top_features': result['top_features'],
                    'players_with_clusters': result['players_with_clusters'],
                    'players_without_clusters': result['players_without_clusters']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Supervised learning model training failed',
                    'error': result['error']
                })
            }
            
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Lambda execution failed',
                'error': str(e)
            })
        }

# For local testing
if __name__ == "__main__":
    result = run_supervised_learning()
    print(f"Supervised learning result: {result}")