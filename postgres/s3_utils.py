import boto3
import pandas as pd
import pickle
from io import BytesIO
import json

# S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'

def save_dataframe_to_s3(df, key):
    """Save DataFrame as Parquet to S3"""
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_buffer.getvalue())
    print(f"Saved {len(df)} records to s3://{BUCKET_NAME}/{key}")

def load_dataframe_from_s3(key):
    """Load DataFrame from S3 Parquet"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except Exception as e:
        print(f"Error loading {key}: {e}")
        return pd.DataFrame()

def save_model_to_s3(model, key):
    """Save sklearn model to S3"""
    model_buffer = BytesIO()
    pickle.dump(model, model_buffer)
    model_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=model_buffer.getvalue())

def load_model_from_s3(key):
    """Load sklearn model from S3"""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pickle.load(BytesIO(obj['Body'].read()))

def save_json_to_s3(data, key):
    """Save JSON data to S3"""
    s3.put_object(
        Bucket=BUCKET_NAME, 
        Key=key, 
        Body=json.dumps(data),
        ContentType='application/json'
    )