# NBA Prediction Lambda Deployment Guide 🚀

## ✅ Final Status Check

All critical issues have been resolved:

- ✅ **S3 Paths**: Correct in all Lambda functions
- ✅ **AWS Account ID**: Uses .env file with dotenv
- ✅ **Chrome Layers**: Added to Selenium functions
- ✅ **Python Runtime**: Consistent python3.9 across all functions
- ✅ **Dependencies**: All requirements.txt files are correct
- ✅ **Import Statements**: No aws.s3_utils imports remaining

## 📋 Prerequisites

1. **AWS Account** with Lambda and S3 permissions
2. **S3 Bucket**: `nba-prediction-ibracken` (should exist)
3. **IAM Role**: `lambda-execution-role` with S3 access
4. **Python**: 3.9+ with pip
5. **AWS CLI**: Configured with your credentials

## 🔧 Setup Steps

### 1. Install Deployment Dependencies
```bash
cd lambda
pip install python-dotenv boto3
```

### 2. Configure AWS Account ID
Edit `lambda/.env`:
```bash
AWS_ACCOUNT_ID=YOUR_ACTUAL_ACCOUNT_ID
AWS_ROLE_NAME=lambda-execution-role
AWS_REGION=us-east-1
S3_BUCKET=nba-prediction-ibracken
```

### 3. Create IAM Role (if needed)
```bash
# Create role
aws iam create-role --role-name lambda-execution-role --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}'

# Attach policies
aws iam attach-role-policy --role-name lambda-execution-role --policy-arn arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole
aws iam attach-role-policy --role-name lambda-execution-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

## 🚀 Deployment Order

Deploy in this exact order due to dependencies:

### 1. Cluster Scraper (No dependencies)
```bash
cd cluster-scraper
python deploy.py
```

### 2. NBA Clustering (Depends on cluster data)
```bash
cd ../nba-clustering
python deploy.py
```

### 3. Box Score Scraper (Depends on clusters)
```bash
cd ../box-score-scraper
python deploy.py
```

### 4. Supervised Learning (Depends on all previous)
```bash
cd ../supervised-learning
python deploy.py
```

### 5. Daily Predictions (Depends on trained model)
```bash
cd ../daily-predictions
python deploy.py
```

## 📊 Pipeline Schedule

Once deployed, the functions will run automatically:

- **6:00 AM UTC** - Cluster Scraper (player stats)
- **6:30 AM UTC** - NBA Clustering (player groupings)
- **8:00 AM UTC** - Box Score Scraper (game data)
- **10:00 AM UTC** - Supervised Learning (model training)
- **12:00 PM UTC** - Daily Predictions (fantasy projections)
- **Every 6 hours** - Daily Predictions (game day updates)

## 🧪 Testing

### Test Individual Functions (AWS Console)
1. Go to Lambda in AWS Console
2. Select function → Test tab
3. Create test event with `{}`
4. Click Test and check logs

### Test Locally
```bash
cd lambda
python test_locally.py --function cluster
```

### Monitor Logs
```bash
aws logs tail /aws/lambda/cluster-scraper --follow
```

## 📁 Expected S3 Structure

After successful pipeline run:
```
nba-prediction-ibracken/
├── data/
│   ├── advanced_player_stats/current.parquet
│   ├── clustered_players/current.parquet
│   ├── box_scores/current.parquet
│   ├── test_player_predictions/current.parquet
│   └── daily_predictions/current.parquet
└── models/
    └── RFCluster.sav
```

## 🔍 Troubleshooting

### Common Issues:

1. **Chrome Layer Issues**: Chrome layer is region-specific (us-east-1)
2. **Timeout Errors**: Increase memory/timeout in deploy.py
3. **Permission Errors**: Check IAM role has S3 access
4. **Import Errors**: Ensure python-dotenv is installed
5. **Missing Data**: Run functions in dependency order

### Debug Steps:
1. Check CloudWatch logs for detailed error messages
2. Verify S3 bucket exists and has correct structure
3. Test AWS credentials: `aws sts get-caller-identity`
4. Ensure IAM role exists: `aws iam get-role --role-name lambda-execution-role`

## 🎯 Success Indicators

✅ **All functions deploy without errors**  
✅ **S3 bucket contains expected parquet files**  
✅ **CloudWatch logs show successful execution**  
✅ **EventBridge rules are created and enabled**  
✅ **Model generates daily predictions**

## 📞 Support

- Check CloudWatch logs for specific error messages
- Verify IAM permissions for S3 and Lambda
- Ensure Chrome layer is available in your region
- Test functions individually before running full pipeline

---

**Ready to deploy! 🚀**