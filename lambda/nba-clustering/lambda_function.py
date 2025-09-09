import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from unidecode import unidecode
import boto3
import json
from io import BytesIO
import logging
from datetime import datetime

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

def run_clustering_analysis():
    """Main function to perform NBA player clustering analysis"""
    logger.info("Starting NBA clustering analysis")
    
    try:
        # Load latest player stats from S3
        logger.info("Loading player stats from S3")
        df = load_dataframe_from_s3('data/advanced_player_stats/current.parquet')
        df2 = load_dataframe_from_s3('data/advanced_player_stats/2024-2025.parquet')
        df3 = load_dataframe_from_s3('data/advanced_player_stats/2023-2024.parquet')
        df4 = load_dataframe_from_s3('data/advanced_player_stats/2022-2023.parquet')
        df = pd.concat([df, df2, df3, df4], ignore_index = True)
        logger.info(f"Loaded {len(df)} player records")
        
        # Set player as index and filter for meaningful playing time
        df.set_index('PLAYER', inplace=True)
        original_count = len(df)
        df = df[df['GP'] >= 10]
        df = df[df['MIN'] >= 12]
        filtered_count = len(df)
        logger.info(f"Filtered from {original_count} to {filtered_count} players (GP>=10, MIN>=15)")
        
        # Drop columns we don't want to use for clustering (updated for new data structure)
        columns_to_drop = ['TEAM', 'W', 'L', 'GP', 'DREB', 'STL', 'BLK', 'STAT_TYPE', 'SCRAPED_DATE', 
                          'PLAYER_ID', 'SOURCE', 'id']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        logger.info(f"Using {len(df.columns)} features for clustering")
        
        # Clean data - log non-numeric conversions
        logger.info("Cleaning data for clustering")
        conversion_count = 0
        for col in df.columns:
            original_series = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            non_numeric_mask = df[col].isna() & original_series.notna()
            if non_numeric_mask.any():
                non_numeric_values = original_series[non_numeric_mask].unique()
                conversion_count += non_numeric_mask.sum()
                logger.info(f"Column '{col}': Found {non_numeric_mask.sum()} non-numeric values: {list(non_numeric_values)}")
        
        logger.info(f"Total non-numeric conversions: {conversion_count}")
        df = df.fillna(0)
        
        # Prepare data for clustering
        dfPlayerCol = df.reset_index()
        features = list(df.columns)
        x = df.loc[:, features].values
        
        # Scale the data
        logger.info("Scaling features")
        x = StandardScaler().fit_transform(x)
        
        # Apply PCA with optimal components (determined from analysis: 16 components)
        logger.info("Applying PCA dimensionality reduction")
        n_components = 16
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(x)
        variance_explained = sum(pca.explained_variance_ratio_)
        logger.info(f"PCA with {n_components} components explains {variance_explained:.3f} of variance")
        
        # Prepare data for clustering
        x_pca = np.column_stack([components[:, i] for i in range(n_components - 1)])
        
        # Apply KMeans with optimal clusters (determined from analysis: 15 clusters)
        logger.info("Performing KMeans clustering")
        n_clusters = 15
        kmeans = KMeans(n_clusters=n_clusters, random_state=4)
        cluster_labels = kmeans.fit_predict(x_pca)
        
        # Calculate silhouette score
        silhouette = silhouette_score(x_pca, cluster_labels)
        logger.info(f"Silhouette score: {silhouette:.3f}")
        
        # Create cluster assignments DataFrame
        df_cluster = pd.DataFrame({
            'PLAYER': dfPlayerCol['PLAYER'],
            'CLUSTER': cluster_labels,
            'TIMESTAMP': pd.Timestamp.now()
        })
        
        # Save cluster assignments to S3
        save_dataframe_to_s3(df_cluster, 'data/clustered_players/current.parquet')
        
        # Generate cluster summary
        cluster_counts = df_cluster['CLUSTER'].value_counts().sort_index()
        cluster_summary = {int(cluster): int(count) for cluster, count in cluster_counts.items()}
        
        logger.info("Clustering analysis completed successfully")
        logger.info(f"Total players clustered: {len(df_cluster)}")
        logger.info(f"Number of clusters: {len(np.unique(cluster_labels))}")
        
        return {
            'success': True,
            'total_players': len(df_cluster),
            'num_clusters': len(np.unique(cluster_labels)),
            'silhouette_score': float(silhouette),
            'variance_explained': float(variance_explained),
            'cluster_summary': cluster_summary
        }
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("NBA clustering Lambda function started")
    
    try:
        result = run_clustering_analysis()
        
        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'NBA clustering analysis completed successfully',
                    'total_players': result['total_players'],
                    'num_clusters': result['num_clusters'],
                    'silhouette_score': result['silhouette_score'],
                    'variance_explained': result['variance_explained'],
                    'cluster_summary': result['cluster_summary']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'NBA clustering analysis failed',
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
    result = run_clustering_analysis()
    print(f"Clustering result: {result}")