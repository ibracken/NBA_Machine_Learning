"""
Configuration and constants for minutes projection lambda
"""

import boto3

# S3 client
s3_client = boto3.client('s3')
# === NEW: Add SNS Client ===
sns_client = boto3.client('sns')
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:349928386418:lineup-optimizer-notifications'
BUCKET_NAME = 'nba-prediction-ibracken'

# Algorithm constants
BENCH_OPPORTUNITY_CONSTANT = 0.1  # Small boost for deep bench players
EXACT_POSITION_MULTIPLIER = 2.0  # Direct backups get 2x weight (complex model only)
MAX_MINUTES = 37  # Cap individual player minutes
CONFIDENCE_CLEARANCE_GAMES = 3  # Games needed to clear LOW confidence
