import boto3
import json
import time
from datetime import datetime, timedelta, timezone
import logging
import os
import re
from tempfile import mkdtemp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pytz

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_chrome_driver():
    """Create Chrome driver with Lambda-specific configuration"""
    # Check if running in Lambda environment
    is_lambda = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

    if is_lambda:
        chrome_options = Options()
        # Essential Lambda options based on AWS best practices
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--force-device-scale-factor=0.75")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--disable-extensions")

        # Page load strategy - prevent hanging on slow resources
        chrome_options.page_load_strategy = 'eager'  # Don't wait for all resources

        # Lambda-specific arguments - use /tmp for all writable directories
        chrome_options.add_argument(f"--user-data-dir={mkdtemp()}")
        chrome_options.add_argument(f"--data-path={mkdtemp()}")
        chrome_options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        chrome_options.add_argument("--no-first-run")

        chrome_options.binary_location = "/opt/chrome/chrome"
        driver = webdriver.Chrome(
            service=webdriver.chrome.service.Service("/opt/chromedriver"),
            options=chrome_options
        )
        return driver
    else:
        # Local environment - let Selenium automatically manage ChromeDriver
        return webdriver.Chrome()

def scrape_main_slate_time():
    """
    Scrape main slate start time from DailyFantasyFuel
    Returns: datetime object in UTC, or None if not found
    """
    logger.info("Scraping main slate time from DailyFantasyFuel")

    driver = get_chrome_driver()

    try:
        url = "https://www.dailyfantasyfuel.com/nba/projections/draftkings"
        driver.get(url)
        logger.info(f"Navigated to DailyFantasyFuel: {url}")

        # Wait a bit for JavaScript to load
        import time
        time.sleep(3)

        # Try to find the main slate time element
        # XPath: /html/body/div[2]/div[1]/div[1]/div[2]/div[2]/div/div[1]/div[1]/div[2]/div[2]/div/span[1]
        try:
            slate_element = driver.find_element(By.XPATH, "/html/body/div[2]/div[1]/div[1]/div[2]/div[2]/div/div[1]/div[1]/div[2]/div[2]/div/span[1]")
            slate_text = slate_element.text
            logger.info(f"Found slate text: {slate_text}")

            # Parse format: "SUN 6:00PM ET"
            # Extract day and time
            match = re.match(r'(\w+)\s+(\d{1,2}):(\d{2})(AM|PM)\s+ET', slate_text)

            if match:
                day_abbrev, hour, minute, am_pm = match.groups()
                hour = int(hour)
                minute = int(minute)

                # Convert to 24-hour format
                if am_pm == 'PM' and hour != 12:
                    hour += 12
                elif am_pm == 'AM' and hour == 12:
                    hour = 0

                # Get today's date and find the next occurrence of this day
                # Use pytz for proper timezone handling (handles DST automatically)
                et_tz = pytz.timezone('America/New_York')
                now_utc = datetime.now(timezone.utc)
                now_et = now_utc.astimezone(et_tz)

                # Map day abbreviations to weekday numbers
                day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
                target_weekday = day_map.get(day_abbrev.upper())

                if target_weekday is not None:
                    # Calculate days until target weekday
                    current_weekday = now_et.weekday()
                    days_ahead = (target_weekday - current_weekday) % 7

                    # If it's today but the time has passed, assume it's today
                    target_date = now_et.date() + timedelta(days=days_ahead)

                    # Create datetime in ET timezone (pytz handles DST)
                    slate_time_et = et_tz.localize(datetime(
                        target_date.year, target_date.month, target_date.day,
                        hour, minute, 0
                    ))

                    # Convert to UTC (pytz handles DST offset automatically)
                    slate_time_utc = slate_time_et.astimezone(pytz.utc)

                    logger.info(f"Main slate time (ET): {slate_time_et}")
                    logger.info(f"Main slate time (UTC): {slate_time_utc}")
                    return slate_time_utc

            logger.error(f"Could not parse slate time format: {slate_text}")
            return None

        except Exception as e:
            logger.error(f"Could not find slate time element: {e}")
            return None

    except Exception as e:
        logger.error(f"Error scraping main slate time: {e}")
        return None
    finally:
        driver.quit()

def lambda_handler(event, context):
    events_client = boto3.client('events')
    lambda_client = boto3.client('lambda')

    # Check if running locally (mock context)
    is_local = context.invoked_function_arn.startswith("arn:aws:lambda:us-east-1:123456789012")

    # === CLEANUP OLD RULES ===
    if not is_local:
        cleanup_old_rules(events_client, lambda_client)

    # === SCRAPE MAIN SLATE TIME FROM DAILYFANTASYFUEL ===
    main_slate_time = scrape_main_slate_time()

    if not main_slate_time:
        logger.error("Could not scrape main slate time from DailyFantasyFuel")
        return {'statusCode': 500, 'body': 'Failed to scrape main slate time'}

    logger.info(f"Main slate starts at: {main_slate_time}")

    # === LAMBDA FUNCTION PIPELINE (sequential execution with 2-minute delays) ===
    lambda_pipeline = [
        ('cluster-scraper', 0),      # Start at game time - 30 min
        ('nba-clustering', 2),        # +2 minutes
        ('box-score-scraper', 4),     # +4 minutes
        ('supervised-learning', 6),   # +6 minutes
        ('daily-predictions', 9),     # +8 minutes
        ('lineup-optimizer', 11)      # +10 minutes
    ]

    # === CREATE RULES FOR MAIN SLATE ===
    try:
        # Base trigger time: 30 minutes before main slate
        base_trigger_time = main_slate_time - timedelta(minutes=30)

        slate_id = main_slate_time.strftime('%Y%m%d-%H%M')  # e.g., "20251026-1800"
        region = context.invoked_function_arn.split(":")[3]
        account_id = context.invoked_function_arn.split(":")[4]

        if is_local:
            logger.info(f"Running locally - would schedule main slate {slate_id} at {base_trigger_time}")
            logger.info(f"Main slate time: {main_slate_time}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'LOCAL TEST: Would schedule main slate at {main_slate_time.isoformat()}',
                    'slate_id': slate_id,
                    'base_trigger_time': base_trigger_time.isoformat()
                })
            }

        # Create a separate rule for each Lambda function with time offset
        for func_name, delay_minutes in lambda_pipeline:
            trigger_time = base_trigger_time + timedelta(minutes=delay_minutes)

            # Skip if trigger time is in the past
            if trigger_time < datetime.now(timezone.utc):
                logger.info(f"Skipping {func_name} for slate {slate_id} - trigger time in past")
                continue

            rule_name = f"nba-slate-{slate_id}-{func_name}"

            # Create EventBridge cron expression
            cron_expr = (f"cron({trigger_time.minute} {trigger_time.hour} "
                        f"{trigger_time.day} {trigger_time.month} ? {trigger_time.year})")

            # Create the rule
            events_client.put_rule(
                Name=rule_name,
                ScheduleExpression=cron_expr,
                State='ENABLED',
                Description=f"{func_name}: Main Slate at {main_slate_time.strftime('%Y-%m-%d %H:%M UTC')}"
            )

            logger.info(f"Created rule: {rule_name} at {trigger_time}")

            # Add Lambda function as target
            targets = [{
                'Id': '1',
                'Arn': f'arn:aws:lambda:{region}:{account_id}:function:{func_name}',
                'Input': json.dumps({
                    'slateId': slate_id,
                    'slateTime': main_slate_time.isoformat(),
                    'triggerTime': trigger_time.isoformat()
                })
            }]

            # Grant permission to EventBridge to invoke Lambda
            try:
                lambda_client.add_permission(
                    FunctionName=func_name,
                    StatementId=f'EventBridge-{rule_name}',
                    Action='lambda:InvokeFunction',
                    Principal='events.amazonaws.com',
                    SourceArn=f"arn:aws:events:{region}:{account_id}:rule/{rule_name}"
                )
                logger.info(f"Added permission for {func_name}")
            except lambda_client.exceptions.ResourceConflictException:
                pass  # Permission already exists

            # Attach target to the rule
            events_client.put_targets(
                Rule=rule_name,
                Targets=targets
            )

        logger.info(f"Created {len(lambda_pipeline)} rules for slate {slate_id}")

    except Exception as e:
        logger.error(f"Error scheduling main slate: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Failed to schedule main slate: {str(e)}'
            })
        }

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Scheduled main slate with {len(lambda_pipeline)} Lambda functions at {main_slate_time.isoformat()}'
        })
    }


def cleanup_old_rules(events_client, lambda_client):
    """Remove EventBridge rules older than 2 days"""
    try:
        # List all rules with our naming patterns (both old nba-game-* and new nba-slate-*)
        paginator = events_client.get_paginator('list_rules')

        # Clean up both nba-game-* (old) and nba-slate-* (new) rules
        for prefix in ['nba-game-', 'nba-slate-']:
            for page in paginator.paginate(NamePrefix=prefix):
                for rule in page['Rules']:
                    rule_name = rule['Name']

                    # Get all targets for this rule
                    targets_response = events_client.list_targets_by_rule(Rule=rule_name)
                    target_ids = [t['Id'] for t in targets_response['Targets']]

                    try:
                        # Remove targets first
                        if target_ids:
                            events_client.remove_targets(
                                Rule=rule_name,
                                Ids=target_ids
                            )

                        # Delete the rule
                        events_client.delete_rule(Name=rule_name)
                        logger.info(f"Cleaned up old rule: {rule_name}")

                        # Remove Lambda permissions (extract function name from rule_name)
                        # Rule name format: nba-game-{gameId}-{func_name} or nba-slate-{slateId}-{func_name}
                        for func_name in ['cluster-scraper', 'nba-clustering', 'box-score-scraper',
                                         'supervised-learning', 'daily-predictions', 'lineup-optimizer']:
                            if func_name in rule_name:
                                try:
                                    lambda_client.remove_permission(
                                        FunctionName=func_name,
                                        StatementId=f'EventBridge-{rule_name}'
                                    )
                                    logger.info(f"Removed permission for {func_name}")
                                except:
                                    pass  # Permission may not exist
                                break

                    except Exception as e:
                        logger.error(f"Error cleaning up {rule_name}: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# For local testing
if __name__ == "__main__":
    class MockContext:
        invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:game-scheduler"

    result = lambda_handler({}, MockContext())
    print(f"Game scheduler result: {result}")
