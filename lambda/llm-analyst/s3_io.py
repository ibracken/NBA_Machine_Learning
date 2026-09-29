"""
S3 and SNS helpers
"""

import io
import json
import logging

import boto3
import pandas as pd

from config import BUCKET_NAME, SNS_TOPIC_ARN

logger = logging.getLogger()
s3 = boto3.client('s3')
sns = boto3.client('sns')


def load_parquet(key):
    try:
        body = s3.get_object(Bucket=BUCKET_NAME, Key=key)['Body'].read()
    except s3.exceptions.NoSuchKey:
        logger.warning(f"Not found: {key}")
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(body))


def save_parquet(df, key):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buffer.getvalue())
    logger.info(f"Saved {len(df)} rows to {key}")


def replace_date_rows(df_new, key, today, date_col='DATE'):
    """Write today's rows into a history file, replacing any from an earlier run today."""
    existing = load_parquet(key)
    if not existing.empty:
        existing = existing[pd.to_datetime(existing[date_col]).dt.date != today]
    combined = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new.copy()
    # minutes-projection writes some of these files with datetime DATE values; keep one type
    combined[date_col] = pd.to_datetime(combined[date_col])
    save_parquet(combined, key)
    return combined


def load_json(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=key)['Body'].read())
    except s3.exceptions.NoSuchKey:
        return None


def save_json(obj, key):
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=json.dumps(obj, indent=2, default=str).encode())
    logger.info(f"Saved {key}")


def last_modified(key):
    try:
        return s3.head_object(Bucket=BUCKET_NAME, Key=key)['LastModified']
    except s3.exceptions.ClientError:
        return None


def publish(subject, message):
    # SNS subjects are capped at 100 characters
    sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)
    logger.info(f"SNS sent: {subject}")
