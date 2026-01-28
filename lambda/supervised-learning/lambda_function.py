import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

# Add console handler for local testing (Lambda provides its own handlers)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
    df_sorted = df.copy()
    df_sorted['GAME_DATE'] = pd.to_datetime(df_sorted['GAME_DATE'])
    
    # Sort by player and game date ascending
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
        df2 = load_dataframe_from_s3('data/box_scores/2024-25.parquet')
        df3 = load_dataframe_from_s3('data/box_scores/2023-24.parquet')
        df4 = load_dataframe_from_s3('data/box_scores/2022-23.parquet')
        df = pd.concat([df, df2, df3, df4])
        logger.info(f"Loaded {len(df)} box score records from multiple seasons")
        
        # Check incoming box scores data structure
        required_box_score_cols = ['PLAYER', 'GAME_DATE', 'FP', 'Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
                                   'Career_FP_Avg', 'Games_Played_Career', 'MIN', 'MATCHUP',
                                   'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg']
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

        # Filter out players with zero minutes
        if 'MIN' in df.columns:
            original_count = len(df)
            df = df[df['MIN'] != 0]
            logger.info(f"Filtered from {original_count} to {len(df)} records (MIN != 0)")

        # Data preprocessing for MIN column (convert to numeric and add minutes noise)
        df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
        df['MIN'] = df['MIN'] + np.random.uniform(-8, 8, size=len(df))
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

        # Fill missing FP rolling averages with season average as fallback
        df['Last3_FP_Avg'] = df['Last3_FP_Avg'].fillna(df['Season_FP_Avg'])
        df['Last7_FP_Avg'] = df['Last7_FP_Avg'].fillna(df['Season_FP_Avg'])

        # For first games with no season average, use 0 (model will learn this represents unknown)
        df['Last3_FP_Avg'] = df['Last3_FP_Avg'].fillna(0)
        df['Last7_FP_Avg'] = df['Last7_FP_Avg'].fillna(0)
        df['Season_FP_Avg'] = df['Season_FP_Avg'].fillna(0)

        # For first career game, Career_FP_Avg will be NaN - fill with 0
        df['Career_FP_Avg'] = df['Career_FP_Avg'].fillna(0)

        # Games_Played_Career should never be NaN but handle just in case
        df['Games_Played_Career'] = df['Games_Played_Career'].fillna(0)

        # Handle minutes rolling averages (same pattern as FP)
        df['Last7_MIN_Avg'] = df['Last7_MIN_Avg'].fillna(df['Season_MIN_Avg'])
        df['Last7_MIN_Avg'] = df['Last7_MIN_Avg'].fillna(0)
        df['Season_MIN_Avg'] = df['Season_MIN_Avg'].fillna(0)
        df['Career_MIN_Avg'] = df['Career_MIN_Avg'].fillna(0)

        # Replace any Inf values with 0
        df['Career_FP_Avg'] = df['Career_FP_Avg'].replace([np.inf, -np.inf], 0)
        df['Games_Played_Career'] = df['Games_Played_Career'].replace([np.inf, -np.inf], 0)
        df['Last7_MIN_Avg'] = df['Last7_MIN_Avg'].replace([np.inf, -np.inf], 0)
        df['Season_MIN_Avg'] = df['Season_MIN_Avg'].replace([np.inf, -np.inf], 0)
        df['Career_MIN_Avg'] = df['Career_MIN_Avg'].replace([np.inf, -np.inf], 0)

        # Handle NaN/Inf in REST_DAYS (should be handled by calculate_rest_days, but double-check)
        df['REST_DAYS'] = df['REST_DAYS'].fillna(3)  # Default rest days
        df['REST_DAYS'] = df['REST_DAYS'].replace([np.inf, -np.inf], 3)

        # Handle NaN/Inf in MIN (could have issues from noise addition)
        df['MIN'] = df['MIN'].fillna(0)
        df['MIN'] = df['MIN'].replace([np.inf, -np.inf], 0)

        # Handle NaN/Inf in all rolling averages too
        df['Last3_FP_Avg'] = df['Last3_FP_Avg'].replace([np.inf, -np.inf], 0)
        df['Last7_FP_Avg'] = df['Last7_FP_Avg'].replace([np.inf, -np.inf], 0)
        df['Season_FP_Avg'] = df['Season_FP_Avg'].replace([np.inf, -np.inf], 0)

        # Log NaN handling results
        logger.info(f"After NaN handling - Last3_FP_Avg nulls: {df['Last3_FP_Avg'].isna().sum()}")
        logger.info(f"After NaN handling - Season_FP_Avg nulls: {df['Season_FP_Avg'].isna().sum()}")
        logger.info(f"After NaN handling - Career_FP_Avg nulls: {df['Career_FP_Avg'].isna().sum()}")
        logger.info(f"After NaN handling - Games_Played_Career nulls: {df['Games_Played_Career'].isna().sum()}")
        logger.info(f"After NaN handling - Last7_MIN_Avg nulls: {df['Last7_MIN_Avg'].isna().sum()}")
        logger.info(f"After NaN handling - Season_MIN_Avg nulls: {df['Season_MIN_Avg'].isna().sum()}")
        logger.info(f"After NaN handling - Career_MIN_Avg nulls: {df['Career_MIN_Avg'].isna().sum()}")

        # Calculate FP_PER_MIN feature
        # First 2 games of season: Use Career_FP_Avg / Career_MIN_Avg
        # After 2 games: Use Season_FP_Avg / Season_MIN_Avg
        logger.info("Calculating FP_PER_MIN feature")

        # Extract season from GAME_DATE for season-based grouping
        # NBA seasons span two calendar years (e.g., 2024-25 season starts in Oct 2024)
        # Games from Oct-Dec belong to season starting that year
        # Games from Jan-Sep belong to season starting previous year
        # Note: GAME_DATE is already datetime from calculate_rest_days()
        df['SEASON'] = df['GAME_DATE'].apply(
            lambda x: f"{x.year}-{str(x.year + 1)[-2:]}" if x.month >= 10 else f"{x.year - 1}-{str(x.year)[-2:]}"
        )

        # Sort by PLAYER and GAME_DATE (ascending) to properly number games chronologically
        df = df.sort_values(['PLAYER', 'GAME_DATE'], ascending=[True, True])

        # Determine which games are first 2 of season for each player
        df['SEASON_GAME_NUM'] = df.groupby(['PLAYER', 'SEASON']).cumcount() + 1

        # Calculate FP_PER_MIN based on game number
        df['FP_PER_MIN'] = np.where(
            df['SEASON_GAME_NUM'] <= 2,
            # First 2 games: Career FP/MIN
            np.where(
                df['Career_MIN_Avg'] > 0,
                df['Career_FP_Avg'] / df['Career_MIN_Avg'],
                0
            ),
            # After 2 games: Season FP/MIN
            np.where(
                df['Season_MIN_Avg'] > 0,
                df['Season_FP_Avg'] / df['Season_MIN_Avg'],
                0
            )
        )

        # Handle inf/nan from division
        df['FP_PER_MIN'] = df['FP_PER_MIN'].replace([np.inf, -np.inf], 0)
        df['FP_PER_MIN'] = df['FP_PER_MIN'].fillna(0)

        logger.info(f"Calculated FP_PER_MIN feature - mean: {df['FP_PER_MIN'].mean():.3f}, max: {df['FP_PER_MIN'].max():.3f}")

        # Define three feature sets for three models
        # Note: OPPONENT removed - will be replaced with opponent defensive rating later
        feature_sets = {
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

        label_name = 'FP'

        # Dictionary to store trained models and results
        models_results = {}

        # Train each model variant
        for model_name, feature_names in feature_sets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name} model")
            logger.info(f"{'='*60}")

            # Validate features exist
            missing_features = [col for col in feature_names if col not in df.columns]
            if missing_features:
                logger.error(f"{model_name}: Missing features: {missing_features}")
                continue

            # Prepare data for this model
            features = df[feature_names].copy()
            labels = df[[label_name]].copy()

            # One-hot encode categorical variables (only if they exist in feature set)
            categorical_cols = []
            if 'CLUSTER' in features.columns:
                categorical_cols.append('CLUSTER')

            features_encoded = pd.get_dummies(features, columns=categorical_cols) if categorical_cols else features

            # Get actual feature names after encoding
            feature_names_list = list(features_encoded.columns)

            # Save feature names to S3
            feature_names_json = json.dumps({'features': feature_names_list})
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=f'models/{model_name}_feature_names.json',
                Body=feature_names_json
            )
            logger.info(f"{model_name}: Saved {len(feature_names_list)} feature names")

            # Check for multicollinearity (VIF)
            logger.info(f"{model_name}: Checking multicollinearity...")
            try:
                vif_data = pd.DataFrame()
                vif_data["feature"] = feature_names_list
                vif_data["VIF"] = [variance_inflation_factor(features_encoded.values, i)
                                   for i in range(len(feature_names_list))]

                high_vif = vif_data[vif_data['VIF'] > 10]
                if not high_vif.empty:
                    logger.warning(f"{model_name}: High VIF features:\n{high_vif}")
                else:
                    logger.info(f"{model_name}: No high multicollinearity detected")
            except Exception as e:
                logger.warning(f"{model_name}: Could not calculate VIF: {e}")

            # Split data
            train, test, train_labels, test_labels = train_test_split(
                features_encoded, labels, test_size=0.20, random_state=42
            )

            logger.info(f"{model_name}: Training set: {len(train)}, Test set: {len(test)}")

            # Train model
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1
            )

            logger.info(f"{model_name}: Training GradientBoostingRegressor...")
            model.fit(train, train_labels.values.ravel())

            # Save model to S3
            save_model_to_s3(model, f"models/{model_name}.pkl")

            # Generate predictions
            predictions = model.predict(test)

            # Calculate R² score
            # 0-1 scale, higher is better
            r2 = r2_score(test_labels, predictions)
            logger.info(f"{model_name}: R² Score = {r2:.4f}")

            # Store results
            models_results[model_name] = {
                'model': model,
                'r2_score': r2,
                'feature_count': len(feature_names_list),
                'train_size': len(train),
                'test_size': len(test)
            }

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        for model_name, results in models_results.items():
            logger.info(f"{model_name}: R²={results['r2_score']:.4f}, Features={results['feature_count']}")

        return {
            'success': True,
            'models_trained': list(models_results.keys()),
            'results': {name: {'r2_score': r['r2_score'], 'feature_count': r['feature_count']}
                        for name, r in models_results.items()}
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
                    'message': 'Supervised learning models training completed successfully',
                    'models_trained': result['models_trained'],
                    'results': result['results']
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