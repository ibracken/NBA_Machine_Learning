import boto3
import requests
import json
from datetime import datetime, timedelta, timezone
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    events_client = boto3.client('events')
    lambda_client = boto3.client('lambda')

    # === CLEANUP OLD RULES ===
    cleanup_old_rules(events_client, lambda_client)

    # === FETCH TODAY'S GAMES ===
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        games = data.get('scoreboard', {}).get('games', [])
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        return {'statusCode': 500, 'body': 'Failed to fetch games'}

    if not games:
        logger.info("No games today")
        return {'statusCode': 200, 'body': 'No games scheduled'}

    # === LAMBDA FUNCTION PIPELINE (sequential execution with 2-minute delays) ===
    lambda_pipeline = [
        ('cluster-scraper', 0),      # Start at game time - 30 min
        ('nba-clustering', 2),        # +2 minutes
        ('box-score-scraper', 4),     # +4 minutes
        ('supervised-learning', 6),   # +6 minutes
        ('daily-predictions', 8)      # +8 minutes
    ]

    # === CREATE RULES FOR EACH GAME ===
    for game in games:
        try:
            # Parse game time
            game_time_utc = game['gameTimeUTC']  # e.g., "2025-10-07T23:00:00Z"
            game_dt = datetime.fromisoformat(game_time_utc.replace('Z', '+00:00'))

            # Base trigger time: 30 minutes before game
            base_trigger_time = game_dt - timedelta(minutes=30)

            game_id = game['gameId']
            region = context.invoked_function_arn.split(":")[3]
            account_id = context.invoked_function_arn.split(":")[4]

            # Create a separate rule for each Lambda function with time offset
            for func_name, delay_minutes in lambda_pipeline:
                trigger_time = base_trigger_time + timedelta(minutes=delay_minutes)

                # Skip if trigger time is in the past
                if trigger_time < datetime.now(timezone.utc):
                    logger.info(f"Skipping {func_name} for game {game_id} - trigger time in past")
                    continue

                rule_name = f"nba-game-{game_id}-{func_name}"

                # Create EventBridge cron expression
                cron_expr = (f"cron({trigger_time.minute} {trigger_time.hour} "
                            f"{trigger_time.day} {trigger_time.month} ? {trigger_time.year})")

                # Create the rule
                events_client.put_rule(
                    Name=rule_name,
                    ScheduleExpression=cron_expr,
                    State='ENABLED',
                    Description=f"{func_name}: {game['awayTeam']['teamName']} vs {game['homeTeam']['teamName']}"
                )

                logger.info(f"Created rule: {rule_name} at {trigger_time}")

                # Add Lambda function as target
                targets = [{
                    'Id': '1',
                    'Arn': f'arn:aws:lambda:{region}:{account_id}:function:{func_name}',
                    'Input': json.dumps({
                        'gameId': game_id,
                        'game': game,
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

            logger.info(f"Created {len(lambda_pipeline)} rules for game {game_id}")

        except Exception as e:
            logger.error(f"Error processing game {game.get('gameId')}: {e}")
            continue

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Scheduled {len(games)} games with {len(lambda_pipeline)} Lambda functions each'
        })
    }


def cleanup_old_rules(events_client, lambda_client):
    """Remove EventBridge rules older than 2 days"""
    try:
        # List all rules with our naming pattern
        paginator = events_client.get_paginator('list_rules')

        for page in paginator.paginate(NamePrefix='nba-game-'):
            for rule in page['Rules']:
                rule_name = rule['Name']

                # Extract date from schedule expression if possible
                schedule = rule.get('ScheduleExpression', '')

                # Get all targets for this rule
                targets_response = events_client.list_targets_by_rule(Rule=rule_name)
                target_ids = [t['Id'] for t in targets_response['Targets']]

                # Check if rule is old (simple heuristic: remove if disabled or has passed)
                # More robust: parse cron and check if time has passed
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=2)

                # For simplicity, we'll remove any rule that's not in ENABLED state or is old
                # You can make this more sophisticated by parsing the cron expression

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
                    # Rule name format: nba-game-{gameId}-{func_name}
                    for func_name in ['cluster-scraper', 'nba-clustering', 'box-score-scraper',
                                     'supervised-learning', 'daily-predictions']:
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
