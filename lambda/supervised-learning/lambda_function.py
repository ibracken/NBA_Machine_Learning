import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import boto3
import json
import pickle
from io import BytesIO
import logging

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

def run_supervised_learning():
    """Main function to train supervised learning model for fantasy point prediction"""
    logger.info("Starting supervised learning model training")
    
    try:
        # Load box scores data
        logger.info("Loading box scores data from S3")
        df = load_dataframe_from_s3('data/box_scores/current.parquet')
        logger.info(f"Loaded {len(df)} box score records")
        
        # Filter out players with zero minutes (data validation)
        if 'MIN' in df.columns:
            original_count = len(df)
            df = df[df['MIN'] != 0]
            logger.info(f"Filtered from {original_count} to {len(df)} records (MIN != 0)")
        
        # Load player stats and cluster assignments
        logger.info("Loading player stats and cluster data")
        players_df = load_dataframe_from_s3('data/advanced_player_stats/current.parquet')
        clustered_players_df = load_dataframe_from_s3('data/clustered_players/current.parquet')
        
        # Merge cluster assignments with player stats
        dataset_clusters = pd.merge(
            players_df, 
            clustered_players_df[['PLAYER', 'CLUSTER']], 
            on='PLAYER', 
            how='left'
        )
        
        # Create cluster mapping dictionary
        clusterDict = dataset_clusters.set_index('PLAYER')['CLUSTER'].to_dict()
        
        # Map clusters to box score data
        df['CLUSTER'] = df['PLAYER'].map(clusterDict)
        logger.info(f"Mapped clusters to {(~df['CLUSTER'].isna()).sum()} players")
        
        # Sort by game date (most recent first)
        df = df.sort_values(by=['GAME DATE'], ascending=[False])
        
        # Data preprocessing for MIN column (convert to numeric and add noise as in notebook)
        df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
        df['MIN'] = df['MIN'] + np.random.uniform(-3, 3, size=len(df))
        df['MIN'] = df['MIN'].clip(lower=0)
        
        # Handle missing clusters with placeholder
        df['CLUSTER'] = df['CLUSTER'].fillna('CLUSTER_NAN')
        
        # Define features and target
        feature_names = ['Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg', 'CLUSTER', 'MIN']
        label_name = 'FP'
        
        # Prepare feature dataframe
        df_features = df[feature_names].copy()
        df_labels = df[label_name].copy()
        
        # One-hot encode clusters
        logger.info("One-hot encoding cluster features")
        df_features = pd.get_dummies(df_features, columns=['CLUSTER'], drop_first=False)
        
        # Convert to numpy arrays
        features = np.array(df_features)
        labels = np.array(df_labels)
        
        # Store additional data for test results
        players = df['PLAYER']
        game_dates = df['GAME DATE']
        
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
        
        # Get cluster columns for reverse mapping
        cluster_columns = [col for col in df_features.columns if col.startswith('CLUSTER_')]
        
        # Create test results dataframe
        test_df = pd.DataFrame(test, columns=df_features.columns)
        
        # Map back to original cluster numbers
        test_df['CLUSTER'] = test_df[cluster_columns].idxmax(axis=1)
        
        def safe_cluster_map(value):
            if isinstance(value, str) and value.startswith('CLUSTER_'):
                try:
                    cluster_value = value.split('_')[-1]
                    return float(cluster_value) if cluster_value != 'NAN' else np.nan
                except ValueError:
                    return np.nan
            return np.nan
        
        test_df['CLUSTER'] = test_df['CLUSTER'].map(safe_cluster_map)
        
        # Drop one-hot encoded columns
        test_df = test_df.drop(columns=cluster_columns)
        
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