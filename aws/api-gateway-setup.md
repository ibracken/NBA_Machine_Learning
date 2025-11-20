# API Gateway Setup for Lineup Optimizer

This guide covers setting up an API Gateway endpoint to expose your lineup-optimizer Lambda function.

## Option 1: AWS Console (Quick Setup)

### Step 1: Create API Gateway

1. Go to **AWS Console** → **API Gateway**
2. Click **Create API**
3. Choose **HTTP API** (simpler and cheaper than REST API)
4. Click **Build**

### Step 2: Configure Integration

1. **Add integration**:
   - Integration type: **Lambda**
   - AWS Region: **us-east-1** (or your Lambda's region)
   - Lambda function: Select **lineup-optimizer**
   - Version: **2.0** (recommended)
   - Click **Next**

2. **Configure routes**:
   - Method: **GET**
   - Resource path: `/lineup` (or just `/`)
   - Integration target: Your Lambda function
   - Click **Next**

3. **Configure stages**:
   - Stage name: **prod** (or **dev** for testing)
   - Auto-deploy: **Enabled**
   - Click **Next**

4. **Review and create**:
   - Review settings
   - Click **Create**

### Step 3: Enable CORS

1. In your API Gateway, go to **CORS**
2. Click **Configure**
3. Add these settings:
   - Access-Control-Allow-Origin: `*` (or your specific domain)
   - Access-Control-Allow-Headers: `content-type,x-amz-date,authorization,x-api-key`
   - Access-Control-Allow-Methods: `GET,OPTIONS`
4. Click **Save**

### Step 4: Get Your Endpoint URL

1. Go to **Stages** in your API Gateway
2. Copy the **Invoke URL**
3. Your full endpoint will be: `https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod/lineup`

### Step 5: Update Frontend

Update `frontend/index.html`:
```javascript
const API_ENDPOINT = 'https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod/lineup';
const USE_MOCK_DATA = false;
```

---

## Option 2: AWS CLI (Automated)

Save this script as `setup-api-gateway.sh` and run it:

```bash
#!/bin/bash

# Configuration
LAMBDA_FUNCTION_NAME="lineup-optimizer"
API_NAME="lineup-optimizer-api"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Creating HTTP API Gateway..."

# Create HTTP API
API_ID=$(aws apigatewayv2 create-api \
    --name "$API_NAME" \
    --protocol-type HTTP \
    --target "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$LAMBDA_FUNCTION_NAME" \
    --region $REGION \
    --query 'ApiId' \
    --output text)

echo "API created with ID: $API_ID"

# Enable CORS
aws apigatewayv2 update-api \
    --api-id $API_ID \
    --cors-configuration AllowOrigins="*",AllowMethods="GET,OPTIONS",AllowHeaders="content-type,x-amz-date,authorization,x-api-key" \
    --region $REGION

# Create route
aws apigatewayv2 create-route \
    --api-id $API_ID \
    --route-key 'GET /lineup' \
    --region $REGION

# Add Lambda permission for API Gateway
aws lambda add-permission \
    --function-name $LAMBDA_FUNCTION_NAME \
    --statement-id apigateway-invoke-permission \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT_ID:$API_ID/*" \
    --region $REGION

# Get the endpoint URL
ENDPOINT_URL=$(aws apigatewayv2 get-api --api-id $API_ID --region $REGION --query 'ApiEndpoint' --output text)

echo ""
echo "========================================="
echo "API Gateway setup complete!"
echo "Your endpoint URL is:"
echo "$ENDPOINT_URL/lineup"
echo "========================================="
echo ""
echo "Update your frontend/index.html with:"
echo "const API_ENDPOINT = '$ENDPOINT_URL/lineup';"
```

---

## Option 3: AWS SAM Template (Infrastructure as Code)

Save as `aws/template.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: API Gateway for NBA Lineup Optimizer

Resources:
  LineupOptimizerApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: prod
      CorsConfiguration:
        AllowOrigins:
          - "*"
        AllowHeaders:
          - content-type
          - x-amz-date
          - authorization
          - x-api-key
        AllowMethods:
          - GET
          - OPTIONS

  LineupOptimizerFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: lineup-optimizer
      Handler: lambda_function.lambda_handler
      Runtime: python3.11
      Events:
        GetLineup:
          Type: HttpApi
          Properties:
            ApiId: !Ref LineupOptimizerApi
            Path: /lineup
            Method: GET

Outputs:
  ApiEndpoint:
    Description: "API Gateway endpoint URL"
    Value: !Sub "https://${LineupOptimizerApi}.execute-api.${AWS::Region}.amazonaws.com/prod/lineup"
```

Deploy with:
```bash
sam deploy --guided
```

---

## Testing Your API

### Test with curl:
```bash
curl https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/lineup
```

### Test in browser:
Just paste the URL in your browser - you should see JSON response.

### Expected Response:
```json
{
  "statusCode": 200,
  "body": "{\"message\":\"Lineup optimization completed successfully\",\"lineup_size\":8,...}"
}
```

---

## Troubleshooting

### CORS Errors
If you see CORS errors in browser console:
1. Make sure CORS is enabled in API Gateway
2. Check that Lambda returns proper headers:
   ```python
   return {
       'statusCode': 200,
       'headers': {
           'Access-Control-Allow-Origin': '*',
           'Access-Control-Allow-Headers': 'Content-Type',
           'Access-Control-Allow-Methods': 'GET,OPTIONS'
       },
       'body': json.dumps(...)
   }
   ```

### Lambda Permission Denied
If API Gateway can't invoke Lambda:
```bash
aws lambda add-permission \
    --function-name lineup-optimizer \
    --statement-id apigateway-test \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com
```

### 500 Internal Server Error
Check Lambda logs in CloudWatch:
```bash
aws logs tail /aws/lambda/lineup-optimizer --follow
```

---

## Security Considerations

### For Production:

1. **Use API Key**:
   - Create an API key in API Gateway
   - Require API key for requests
   - Update frontend to include `x-api-key` header

2. **Restrict CORS**:
   - Change AllowOrigins from `*` to your specific domain
   - Example: `https://yourdomain.com`

3. **Add Rate Limiting**:
   - Set up throttling in API Gateway
   - Default: 10,000 requests per second

4. **Enable Logging**:
   - Turn on CloudWatch Logs for API Gateway
   - Monitor for unusual patterns

---

## Cost Estimate

HTTP API Gateway pricing:
- First 1 million requests/month: FREE (AWS Free Tier)
- After that: $1.00 per million requests

For personal use, this should cost nearly nothing.
