"""
Diagnostic script to investigate the 6.2 → 8.0 FP error gap
Focus: Are predictions ignoring recent performance (Last3/Last7)?
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from datetime import datetime, timedelta

# S3 setup
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'

def load_dataframe_from_s3(key):
    """Load DataFrame from S3 Parquet"""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_parquet(BytesIO(obj['Body'].read()))

print("=" * 80)
print("PREDICTION GAP DIAGNOSTIC")
print("=" * 80)

# Load data
print("\nLoading data from S3...")
df_predictions = load_dataframe_from_s3('data/daily_predictions/current.parquet')
df_box = load_dataframe_from_s3('data/box_scores/current.parquet')

print(f"Loaded {len(df_predictions)} predictions")
print(f"Loaded {len(df_box)} box score records")

# Check box score freshness
print("\n" + "=" * 80)
print("BOX SCORE FRESHNESS CHECK")
print("=" * 80)
df_box['GAME_DATE'] = pd.to_datetime(df_box['GAME_DATE'])
most_recent_box_score = df_box['GAME_DATE'].max()
print(f"Most recent box score date: {most_recent_box_score.date()}")
print(f"Today's date: {datetime.now().date()}")
print(f"Lag: {(datetime.now().date() - most_recent_box_score.date()).days} days")

# Get predictions after Oct 30 (exclude early model issues)
df_predictions['GAME_DATE'] = pd.to_datetime(df_predictions['GAME_DATE'])
cutoff_date = pd.to_datetime('2025-10-30')
df_complete = df_predictions[df_predictions['GAME_DATE'] >= cutoff_date].copy()
df_complete = df_complete.dropna(subset=['ACTUAL_FP', 'MY_MODEL_PREDICTED_FP']).copy()
df_complete['ERROR'] = abs(df_complete['MY_MODEL_PREDICTED_FP'] - df_complete['ACTUAL_FP'])

print(f"\nPredictions after Oct 30 with actual results: {len(df_complete)}")
print(f"Date range: {df_complete['GAME_DATE'].min().date()} to {df_complete['GAME_DATE'].max().date()}")
print(f"Average error: {df_complete['ERROR'].mean():.2f} FP")

# KEY ANALYSIS: Find players where prediction >> recent performance
print("\n" + "=" * 80)
print("PREDICTION vs RECENT PERFORMANCE DISCONNECT")
print("=" * 80)
print("Finding players where model prediction is way higher than Last3/Last7 avg...")

# Merge predictions with box scores to get features
disconnect_analysis = []

for idx, row in df_complete.iterrows():
    player = row['PLAYER']
    game_date = pd.to_datetime(row['GAME_DATE'])

    # Find corresponding box score entry
    box_entry = df_box[(df_box['PLAYER'] == player) & (df_box['GAME_DATE'] == game_date)]

    if not box_entry.empty:
        box_entry = box_entry.iloc[0]

        predicted = row['MY_MODEL_PREDICTED_FP']
        actual = row['ACTUAL_FP']
        last3 = box_entry.get('Last3_FP_Avg', np.nan)
        last7 = box_entry.get('Last7_FP_Avg', np.nan)
        season = box_entry.get('Season_FP_Avg', np.nan)
        career = box_entry.get('Career_FP_Avg', np.nan)
        proj_min = row.get('PROJECTED_MIN', np.nan)
        actual_min = row.get('ACTUAL_MIN', np.nan)

        if pd.notna(last3) and pd.notna(last7):
            # Calculate disconnect: how much higher is prediction vs recent performance?
            disconnect_last3 = predicted - last3
            disconnect_last7 = predicted - last7

            disconnect_analysis.append({
                'PLAYER': player,
                'GAME_DATE': game_date.date(),
                'PREDICTED': predicted,
                'ACTUAL': actual,
                'ERROR': row['ERROR'],
                'LAST3_AVG': last3,
                'LAST7_AVG': last7,
                'SEASON_AVG': season,
                'CAREER_AVG': career,
                'DISCONNECT_LAST3': disconnect_last3,
                'DISCONNECT_LAST7': disconnect_last7,
                'PROJ_MIN': proj_min,
                'ACTUAL_MIN': actual_min,
                'MIN_ERROR': abs(proj_min - actual_min) if pd.notna(proj_min) and pd.notna(actual_min) else np.nan
            })

df_disconnect = pd.DataFrame(disconnect_analysis)

# Top players with biggest disconnect (predicted >> Last3)
print("\nTop 20 Players: Prediction Way Higher Than Last3 Average")
print("(Ordered by: Predicted - Last3_Avg)")
print("-" * 80)
top_disconnect = df_disconnect.nlargest(20, 'DISCONNECT_LAST3')
for idx, row in top_disconnect.iterrows():
    print(f"\n{row['PLAYER'].upper()} - {row['GAME_DATE']}")
    print(f"  Predicted: {row['PREDICTED']:6.1f}  |  Actual: {row['ACTUAL']:6.1f}  |  Error: {row['ERROR']:6.1f}")
    print(f"  Last3:     {row['LAST3_AVG']:6.1f}  |  Disconnect: +{row['DISCONNECT_LAST3']:6.1f}")
    print(f"  Last7:     {row['LAST7_AVG']:6.1f}  |  Season: {row['SEASON_AVG']:6.1f}  |  Career: {row['CAREER_AVG']:6.1f}")
    print(f"  Proj MIN:  {row['PROJ_MIN']:6.0f}  |  Actual MIN: {row['ACTUAL_MIN']:6.0f}  |  MIN Error: {row['MIN_ERROR']:6.1f}")

# Check if minutes error correlates with prediction error
print("\n" + "=" * 80)
print("MINUTES ERROR vs PREDICTION ERROR CORRELATION")
print("=" * 80)
df_disconnect_valid = df_disconnect.dropna(subset=['MIN_ERROR', 'ERROR'])
correlation = df_disconnect_valid['MIN_ERROR'].corr(df_disconnect_valid['ERROR'])
print(f"\nCorrelation between minutes error and FP error: {correlation:.3f}")
if correlation > 0.5:
    print("  → Strong correlation: Minutes errors ARE causing FP errors")
elif correlation > 0.3:
    print("  → Moderate correlation: Minutes errors contribute but aren't the main issue")
else:
    print("  → Weak correlation: Minutes errors NOT the main driver of FP errors")

# Specific player checks
print("\n" + "=" * 80)
print("SPECIFIC PLAYER ANALYSIS")
print("=" * 80)

for target_player in ['saddiq bey', 'jordan clarkson']:
    player_data = df_disconnect[df_disconnect['PLAYER'] == target_player]

    if len(player_data) > 0:
        print(f"\n{target_player.upper()}: {len(player_data)} games")
        print("-" * 80)
        for idx, row in player_data.iterrows():
            print(f"\n  {row['GAME_DATE']}")
            print(f"    Predicted: {row['PREDICTED']:5.1f}  |  Actual: {row['ACTUAL']:5.1f}  |  Error: {row['ERROR']:5.1f}")
            print(f"    Last3:     {row['LAST3_AVG']:5.1f}  |  Last7: {row['LAST7_AVG']:5.1f}  |  Season: {row['SEASON_AVG']:5.1f}")
            print(f"    Proj MIN:  {row['PROJ_MIN']:5.0f}  |  Actual MIN: {row['ACTUAL_MIN']:5.0f}  |  Error: {row['MIN_ERROR']:5.1f}")
            print(f"    Disconnect from Last3: +{row['DISCONNECT_LAST3']:5.1f}")
    else:
        print(f"\n{target_player.upper()}: No predictions found after Oct 30")

# Summary stats
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nAverage disconnect (Predicted - Last3): {df_disconnect['DISCONNECT_LAST3'].mean():+.2f}")
print(f"Players with disconnect > +10 FP: {(df_disconnect['DISCONNECT_LAST3'] > 10).sum()} / {len(df_disconnect)} ({(df_disconnect['DISCONNECT_LAST3'] > 10).mean()*100:.1f}%)")
print(f"Players with disconnect > +5 FP:  {(df_disconnect['DISCONNECT_LAST3'] > 5).sum()} / {len(df_disconnect)} ({(df_disconnect['DISCONNECT_LAST3'] > 5).mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
