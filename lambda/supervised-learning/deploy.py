#!/usr/bin/env python3
"""
Deployment script for Supervised Learning Lambda function
"""

import boto3
import zipfile
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID')

if not AWS_ACCOUNT_ID:
    raise ValueError("AWS_ACCOUNT_ID not found in .env file")

# Configuration
FUNCTION_NAME = 'supervised-learning'
ROLE_ARN = f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/lambda-execution-role'
RUNTIME = 'python3.9'
HANDLER = 'lambda_function.lambda_handler'
TIMEOUT = 900  # 15 minutes
MEMORY_SIZE = 2048  # 2GB for ML model training

def create_deployment_package():
    """Create deployment package for Lambda function"""
    print("Creating deployment package...")
    
    # Create zip file
    zip_path = 'supervised-learning-deployment.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add lambda function
        zipf.write('lambda_function.py')
        zipf.write('requirements.txt')
    
    print(f"Deployment package created: {zip_path}")
    return zip_path

def deploy_lambda_function():
    """Deploy Lambda function to AWS"""
    lambda_client = boto3.client('lambda')
    
    zip_path = create_deployment_package()
    
    # Read zip file
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
    
    try:
        # Try to update existing function
        print(f"Updating Lambda function: {FUNCTION_NAME}")
        response = lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_data
        )
        print("Function updated successfully!")
        
    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new function if it doesn't exist
        print(f"Creating new Lambda function: {FUNCTION_NAME}")
        response = lambda_client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime=RUNTIME,
            Role=ROLE_ARN,
            Handler=HANDLER,
            Code={'ZipFile': zip_data},
            Timeout=TIMEOUT,
            MemorySize=MEMORY_SIZE,
            Environment={
                'Variables': {
                    'S3_BUCKET': 'nba-prediction-ibracken'
                }
            },
            Description='NBA supervised learning model training for fantasy point prediction'
        )
        print("Function created successfully!")
    
    # Clean up zip file
    os.remove(zip_path)
    
    return response

def create_eventbridge_rule():
    """Create EventBridge rule to trigger Lambda after all dependencies complete"""
    events_client = boto3.client('events')
    lambda_client = boto3.client('lambda')
    
    rule_name = f"{FUNCTION_NAME}-trigger"
    
    # Create/update EventBridge rule
    # This should run after clustering (6 AM), box scores (8 AM), so schedule at 10 AM
    print(f"Creating EventBridge rule: {rule_name}")
    events_client.put_rule(
        Name=rule_name,
        ScheduleExpression='cron(0 10 * * ? *)',  # 10 AM UTC daily (after all dependencies)
        Description=f'Daily trigger for {FUNCTION_NAME} Lambda function after data pipeline completion',
        State='ENABLED'
    )
    
    # Add Lambda as target
    function_arn = f"arn:aws:lambda:{boto3.Session().region_name}:{boto3.client('sts').get_caller_identity()['Account']}:function:{FUNCTION_NAME}"
    
    events_client.put_targets(
        Rule=rule_name,
        Targets=[
            {
                'Id': '1',
                'Arn': function_arn
            }
        ]
    )
    
    # Add permission for EventBridge to invoke Lambda
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId=f"{rule_name}-permission",
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=f"arn:aws:events:{boto3.Session().region_name}:{boto3.client('sts').get_caller_identity()['Account']}:rule/{rule_name}"
        )
        print("EventBridge permission added to Lambda function")
    except lambda_client.exceptions.ResourceConflictException:
        print("EventBridge permission already exists")
    
    print(f"EventBridge rule created: {rule_name}")

if __name__ == "__main__":
    print("Deploying Supervised Learning Lambda function...")
    
    # Change to the script directory
    os.chdir(Path(__file__).parent)
    
    # Deploy function
    response = deploy_lambda_function()
    print(f"Lambda function ARN: {response['FunctionArn']}")
    
    # Create EventBridge rule
    create_eventbridge_rule()
    
    print("Deployment completed successfully!")
    print("\nDEPENDENCY CHAIN:")
    print("1. cluster-scraper (6:00 AM) - Scrapes player stats")
    print("2. nba-clustering (6:30 AM) - Creates player clusters") 
    print("3. box-score-scraper (8:00 AM) - Scrapes box scores with cluster enrichment")
    print("4. supervised-learning (10:00 AM) - Trains ML model on enriched data")
    print("\nEnsure all preceding Lambda functions are deployed and scheduled correctly.")