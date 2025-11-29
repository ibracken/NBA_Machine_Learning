# NBA Injury Scraper Lambda

Scrapes NBA injury data from ESPN and saves to S3.

## Output

**S3 Path:** `s3://nba-prediction-ibracken/data/injuries/current.parquet`

**Columns:**
- `PLAYER` - Normalized player name (lowercase, unidecode)
- `TEAM` - NBA team abbreviation (e.g., 'LAL', 'BOS')
- `STATUS` - Injury status: `OUT`, `QUESTIONABLE`, `DOUBTFUL`, `PROBABLE`
- `INJURY_DESCRIPTION` - Description of injury (e.g., "Knee - Left")
- `DATE_UPDATED` - Date injury data was scraped (YYYY-MM-DD)

## Data Source

Scrapes from: **https://www.espn.com/nba/injuries**

## Local Testing

```bash
# Navigate to directory
cd lambda/injury-scraper

# Run locally (requires AWS credentials)
python lambda_function.py
```

## Deployment

### First Time (Create Lambda + ECR Repository)

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name injury-scraper --region us-east-1

# 2. Deploy container
python deploy-container.py
```

### Code Updates Only

If you only changed `lambda_function.py` (no dependency changes):

```bash
python update-lambda.py
```

### Full Rebuild (Dependency Changes)

If you changed `requirements.txt`:

```bash
python deploy-container.py
```

## CloudWatch Schedule

Set up EventBridge rule to run every 2-4 hours:

```bash
# Create schedule (every 2 hours)
aws events put-rule \
  --name injury-scraper-schedule \
  --schedule-expression "rate(2 hours)"

# Add Lambda as target
aws events put-targets \
  --rule injury-scraper-schedule \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:injury-scraper"

# Grant EventBridge permission to invoke Lambda
aws lambda add-permission \
  --function-name injury-scraper \
  --statement-id injury-scraper-schedule \
  --action 'lambda:InvokeFunction' \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:YOUR_ACCOUNT_ID:rule/injury-scraper-schedule
```

## Integration with Minutes Projection

The `minutes-projection` lambda loads this injury data:

```python
# In minutes-projection lambda_function.py
injury_data = load_from_s3('data/injuries/current.parquet')
player_stats = player_stats.merge(injury_data[['PLAYER', 'STATUS']], on='PLAYER', how='left')
```

Players with `STATUS == 'OUT'` will have their minutes redistributed to teammates.

## Monitoring

- **Logs:** CloudWatch Logs → `/aws/lambda/injury-scraper`
- **Metrics:** Check invocation count, errors, duration
- **S3 Validation:** Check last modified date of `data/injuries/current.parquet`

## Troubleshooting

### No injuries scraped
- ESPN may have changed their HTML structure
- Check CloudWatch logs for parsing errors
- Verify ESPN page is accessible: `curl https://www.espn.com/nba/injuries`

### Player names don't match
- Injury scraper normalizes names (lowercase, unidecode)
- Box scores should also normalize names the same way
- Check `normalize_name()` function consistency

### Lambda timeout
- Current timeout: 300 seconds (5 minutes)
- Increase if needed: `aws lambda update-function-configuration --function-name injury-scraper --timeout 600`
