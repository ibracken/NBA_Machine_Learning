import boto3
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID')

def deploy_container():
    """Deploy Lambda function using container"""
    
    # Build the image
    print("Building Docker image...")
    subprocess.run([
        "docker", "buildx", "build", "--platform", "linux/amd64", "--output", "type=docker", "-t", "nba-clustering", "."
    ], check=True, shell=True)
    
    # Tag for ECR
    ecr_repo = f"{AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/nba-clustering"
    subprocess.run([
        "docker", "tag", "nba-clustering:latest", f"{ecr_repo}:latest"
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
            FunctionName='nba-clustering',
            ImageUri=f"{ecr_repo}:latest"
        )
        print("Lambda function updated successfully!")
        
    except lambda_client.exceptions.ResourceNotFoundException:
        print("Creating new Lambda function...")
        response = lambda_client.create_function(
            FunctionName='nba-clustering',
            Role=f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/lambda-execution-role',
            Code={'ImageUri': f"{ecr_repo}:latest"},
            PackageType='Image',
            Description='NBA Clustering Analysis Lambda Function',
            Timeout=900,
            MemorySize=1024,  # Less memory than scraper since it's just ML processing
            Environment={
                'Variables': {
                    'BUCKET_NAME': 'nba-prediction-ibracken'
                }
            }
        )
        print("Lambda function created successfully!")

if __name__ == "__main__":
    deploy_container()