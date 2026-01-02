import boto3
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root .env file
# This script is in lambda/box-score-scraper, so go up 2 levels to root
env_path = Path(__file__).parent.parent.parent / '.env'
print(f"Looking for .env at: {env_path}")
print(f".env exists: {env_path.exists()}")
load_dotenv(env_path)
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID')
print(f"AWS_ACCOUNT_ID loaded: {AWS_ACCOUNT_ID}")

def deploy_container():
    """Deploy Lambda function using container"""
    
    # Build the image
    print("Building Docker image...")
    subprocess.run([
        "docker", "buildx", "build", "--platform", "linux/amd64", "--provenance=false", "--output", "type=docker", "-t", "box-score-scraper", "."
    ], check=True, shell=True)
    
    # Tag for ECR
    ecr_repo = f"{AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/box-score-scraper"
    subprocess.run([
        "docker", "tag", "box-score-scraper:latest", f"{ecr_repo}:latest"
    ], check=True, shell=True)
    
    # Login to ECR
    print("Logging into ECR...")
    login_result = subprocess.run([
        "aws", "ecr", "get-login-password", "--region", "us-east-1"
    ], capture_output=True, text=True, check=True, shell=True)
    
    subprocess.run([
        "docker", "login", "--username", "AWS", "--password-stdin", 
        f"{AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com"
    ], input=login_result.stdout, text=True, check=True, shell=True)
    
    # Push to ECR
    print("Pushing to ECR...")
    subprocess.run([
        "docker", "push", f"{ecr_repo}:latest"
    ], check=True, shell=True)
    
    # Update Lambda function
    print("Updating Lambda function...")
    lambda_client = boto3.client('lambda')
    
    try:
        response = lambda_client.update_function_code(
            FunctionName='box-score-scraper',
            ImageUri=f"{ecr_repo}:latest"
        )
        print("Lambda function updated successfully!")
        
    except lambda_client.exceptions.ResourceNotFoundException:
        print("Creating new Lambda function...")
        response = lambda_client.create_function(
            FunctionName='box-score-scraper',
            Role=f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/lambda-execution-role',
            Code={'ImageUri': f"{ecr_repo}:latest"},
            PackageType='Image',
            Description='NBA Box Score Scraper Lambda Function (API-based)',
            Timeout=900,
            MemorySize=512,  # Less memory needed without Selenium/Chrome
            Environment={
                'Variables': {
                    'BUCKET_NAME': 'nba-prediction-ibracken'
                }
            }
        )
        print("Lambda function created successfully!")

if __name__ == "__main__":
    deploy_container()