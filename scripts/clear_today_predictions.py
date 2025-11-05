"""
Clear today's predictions from daily_predictions/current.parquet
This allows re-running the daily-predictions scraper with fresh data
"""

import pandas as pd
import boto3
from io import BytesIO
import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# S3 setup
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'
KEY = 'data/daily_predictions/current.parquet'

def load_dataframe_from_s3(key):
    """Load DataFrame from S3 Parquet"""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_parquet(BytesIO(obj['Body'].read()))

def save_dataframe_to_s3(df, key):
    """Save DataFrame as Parquet to S3"""
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_buffer.getvalue())
    print(f"Saved {len(df)} records to s3://{BUCKET_NAME}/{key}")

def clear_today_predictions():
    """Remove all predictions for today's date"""
    print("=" * 60)
    print("CLEARING TODAY'S PREDICTIONS")
    print("=" * 60)

    today = datetime.date.today()
    print(f"\nToday's date: {today}")

    # Load current predictions
    print(f"\nLoading predictions from S3...")
    df = load_dataframe_from_s3(KEY)
    print(f"Loaded {len(df)} total predictions")

    # Convert GAME_DATE to date type
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date

    # Show date distribution before
    print(f"\nDate distribution before clearing:")
    date_counts = df['GAME_DATE'].value_counts().sort_index(ascending=False)
    for date, count in date_counts.head(10).items():
        marker = " <- TODAY" if date == today else ""
        print(f"  {date}: {count} predictions{marker}")

    # Count predictions for today
    today_predictions = df[df['GAME_DATE'] == today]
    print(f"\nPredictions for today ({today}): {len(today_predictions)}")

    if len(today_predictions) > 0:
        print(f"\nPlayers with predictions for today:")
        for player in sorted(today_predictions['PLAYER'].unique()):
            print(f"  - {player}")

        # Remove today's predictions
        df_filtered = df[df['GAME_DATE'] != today].copy()

        print(f"\n{len(df) - len(df_filtered)} predictions removed")
        print(f"Remaining predictions: {len(df_filtered)}")

        # Save back to S3
        save_dataframe_to_s3(df_filtered, KEY)

        print(f"\n✓ Successfully cleared today's predictions")
        print(f"  Before: {len(df)} predictions")
        print(f"  After:  {len(df_filtered)} predictions")
    else:
        print(f"\nNo predictions found for today. Nothing to clear.")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    clear_today_predictions()
