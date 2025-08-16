#!/usr/bin/env python3
"""
Deployment script for Daily Predictions Lambda function
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
FUNCTION_NAME = 'daily-predictions'
ROLE_ARN = f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/lambda-execution-role'
RUNTIME = 'python3.9'
HANDLER = 'lambda_function.lambda_handler'
TIMEOUT = 900  # 15 minutes
MEMORY_SIZE = 1536  # 1.5GB for dual scraping + ML inference

def create_deployment_package():
    """Create deployment package for Lambda function"""
    print("Creating deployment package...")
    
    # Create zip file
    zip_path = 'daily-predictions-deployment.zip'
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
            Layers=[
                'arn:aws:lambda:us-east-1:764866452798:layer:chrome-aws-lambda:31'  # Chrome layer for Selenium
            ],
            Description='NBA daily predictions scraper with ML model inference'
        )
        print("Function created successfully!")
    
    # Clean up zip file
    os.remove(zip_path)
    
    return response

def create_eventbridge_rules():
    """Create EventBridge rules for daily and game-day triggers"""
    events_client = boto3.client('events')
    lambda_client = boto3.client('lambda')
    
    # Rule 1: Daily trigger after supervised learning completes
    daily_rule_name = f"{FUNCTION_NAME}-daily-trigger"
    
    print(f"Creating daily EventBridge rule: {daily_rule_name}")
    events_client.put_rule(
        Name=daily_rule_name,
        ScheduleExpression='cron(0 12 * * ? *)',  # 12 PM UTC daily (after supervised learning at 10 AM)
        Description=f'Daily trigger for {FUNCTION_NAME} Lambda function',
        State='ENABLED'
    )
    
    # Rule 2: Game day triggers (more frequent during NBA season)
    gameday_rule_name = f"{FUNCTION_NAME}-gameday-trigger"
    
    print(f"Creating game day EventBridge rule: {gameday_rule_name}")
    events_client.put_rule(
        Name=gameday_rule_name,
        ScheduleExpression='cron(0 */6 * * ? *)',  # Every 6 hours for game day updates
        Description=f'Game day trigger for {FUNCTION_NAME} Lambda function',
        State='ENABLED'
    )
    
    # Add Lambda as target for both rules
    function_arn = f"arn:aws:lambda:{boto3.Session().region_name}:{boto3.client('sts').get_caller_identity()['Account']}:function:{FUNCTION_NAME}"
    
    for rule_name in [daily_rule_name, gameday_rule_name]:
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
            print(f"EventBridge permission added for {rule_name}")
        except lambda_client.exceptions.ResourceConflictException:
            print(f"EventBridge permission already exists for {rule_name}")
    
    print(f"EventBridge rules created: {daily_rule_name}, {gameday_rule_name}")

if __name__ == "__main__":
    print("Deploying Daily Predictions Lambda function...")
    
    # Change to the script directory
    os.chdir(Path(__file__).parent)
    
    # Deploy function
    response = deploy_lambda_function()
    print(f"Lambda function ARN: {response['FunctionArn']}")
    
    # Create EventBridge rules
    create_eventbridge_rules()
    
    print("Deployment completed successfully!")
    print("\nCOMPLETE PIPELINE EXECUTION ORDER:")
    print("1. cluster-scraper (6:00 AM) - Scrapes player stats")
    print("2. nba-clustering (6:30 AM) - Creates player clusters") 
    print("3. box-score-scraper (8:00 AM) - Scrapes box scores with cluster enrichment")
    print("4. supervised-learning (10:00 AM) - Trains ML model on enriched data")
    print("5. daily-predictions (12:00 PM + every 6 hours) - Daily fantasy predictions")
    print("\nThe daily predictions function also runs every 6 hours for game day updates.")
    print("Ensure all preceding Lambda functions are deployed and working correctly.")