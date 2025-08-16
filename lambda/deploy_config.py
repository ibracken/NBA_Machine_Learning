#!/usr/bin/env python3
"""
Shared deployment configuration using environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# AWS Configuration
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID')
AWS_ROLE_NAME = os.getenv('AWS_ROLE_NAME', 'lambda-execution-role')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET', 'nba-prediction-ibracken')

# Construct role ARN
ROLE_ARN = f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{AWS_ROLE_NAME}'

# Validate configuration
def validate_config():
    """Validate that all required configuration is present"""
    if not AWS_ACCOUNT_ID:
        raise ValueError("AWS_ACCOUNT_ID not set in .env file")
    if AWS_ACCOUNT_ID == "YOUR_ACCOUNT_ID_HERE":
        raise ValueError("Please update AWS_ACCOUNT_ID in .env file with your actual account ID")
    
    print(f"✅ Configuration loaded:")
    print(f"   AWS Account ID: {AWS_ACCOUNT_ID}")
    print(f"   Role ARN: {ROLE_ARN}")
    print(f"   S3 Bucket: {S3_BUCKET}")
    print(f"   Region: {AWS_REGION}")

if __name__ == "__main__":
    validate_config()