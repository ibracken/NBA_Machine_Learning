import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID')

def update_lambda():
    """Update Lambda function with container image"""
    
    lambda_client = boto3.client('lambda')
    ecr_repo = f"{AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/supervised-learning"
    
    try:
        response = lambda_client.update_function_code(
            FunctionName='supervised-learning',
            ImageUri=f"{ecr_repo}:latest"
        )
        print("Lambda function updated successfully!")
        
    except lambda_client.exceptions.ResourceNotFoundException:
        print("Creating new Lambda function...")
        response = lambda_client.create_function(
            FunctionName='supervised-learning',
            Role=f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/lambda-execution-role',
            Code={'ImageUri': f"{ecr_repo}:latest"},
            PackageType='Image',
            Description='NBA Supervised Learning Lambda Function',
            Timeout=900,
            MemorySize=2048,
            Environment={
                'Variables': {
                    'BUCKET_NAME': 'nba-prediction-ibracken'
                }
            }
        )
        print("Lambda function created successfully!")

if __name__ == "__main__":
    update_lambda()