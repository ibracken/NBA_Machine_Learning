import boto3
import pandas as pd
import json
import datetime
import logging
from io import BytesIO
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3 = boto3.client('s3')
sns = boto3.client('sns')
BUCKET_NAME = 'nba-prediction-ibracken'
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:349928386418:lineup-optimizer-notifications'  # Replace with your SNS topic ARN

# S3 utility functions
def load_dataframe_from_s3(key):
    """Load DataFrame from S3 Parquet"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except Exception as e:
        logger.error(f"Error loading data from {key}: {e}")
        raise

def save_dataframe_to_s3(df, key):
    """Save DataFrame as Parquet to S3"""
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_buffer.getvalue())
    logger.info(f"Saved {len(df)} records to s3://{BUCKET_NAME}/{key}")

def send_sns_notification(lineup_df, total_salary, total_fp, remaining_salary):
    """Send SNS notification with optimal lineup"""
    try:
        # Format the message with player names and details
        message_lines = ["🏀 Optimal DraftKings Lineup Generated 🏀\n"]
        message_lines.append(f"Total Salary: ${total_salary:,.0f} / $50,000")
        message_lines.append(f"Remaining: ${remaining_salary:,.0f}")
        message_lines.append(f"Projected FP: {total_fp:.2f}\n")
        message_lines.append("=" * 50)

        for _, row in lineup_df.iterrows():
            message_lines.append(
                f"{row['SLOT']:4} | {row['PLAYER']:20} | "
                f"${row['SALARY']:5,.0f} | {row['PREDICTED_FP']:5.2f} FP"
            )

        message = "\n".join(message_lines)

        # Create a simple subject with just player names
        player_names = lineup_df['PLAYER'].tolist()
        subject = f"DK Lineup: {', '.join(player_names[:3])}..."  # First 3 names to keep subject short

        # Publish to SNS
        response = sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )

        logger.info(f"SNS notification sent successfully. MessageId: {response['MessageId']}")
        return True

    except Exception as e:
        logger.error(f"Failed to send SNS notification: {str(e)}")
        return False

def is_eligible_for_slot(position, slot):
    """
    Determine if a player's position makes them eligible for a given slot.

    Args:
        position: Player's position string (e.g., 'PG', 'PG/SG', 'SF/PF')
        slot: Slot name ('PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL')

    Returns:
        bool: True if eligible, False otherwise
    """
    if pd.isna(position):
        return False

    position = str(position).upper()
    positions = [p.strip() for p in position.split('/')]

    if slot == 'PG':
        return 'PG' in positions
    elif slot == 'SG':
        return 'SG' in positions
    elif slot == 'SF':
        return 'SF' in positions
    elif slot == 'PF':
        return 'PF' in positions
    elif slot == 'C':
        return 'C' in positions
    elif slot == 'G':
        # Guard: PG or SG
        return 'PG' in positions or 'SG' in positions
    elif slot == 'F':
        # Forward: SF or PF
        return 'SF' in positions or 'PF' in positions
    elif slot == 'UTIL':
        # Utility: anyone
        return True

    return False

def optimize_lineup(df):
    """
    Optimize DraftKings lineup using linear programming.

    Args:
        df: DataFrame with columns PLAYER, POSITION, SALARY, MY_MODEL_PREDICTED_FP

    Returns:
        DataFrame with optimal lineup
    """
    logger.info(f"Optimizing lineup from {len(df)} available players")

    # Filter out players with missing critical data
    # Am I filtering out by Date?!?!
    df = df.dropna(subset=['PLAYER', 'POSITION', 'SALARY', 'MY_MODEL_PREDICTED_FP'])
    df = df[df['SALARY'] > 0]  # Remove players with invalid salary

    logger.info(f"After filtering: {len(df)} eligible players")

    if len(df) < 8:
        logger.error(f"Not enough players to create a lineup. Only {len(df)} available.")
        return pd.DataFrame()

    # Define slots
    slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    # Create the LP problem
    prob = LpProblem("DK_Lineup_Optimizer", LpMaximize)

    # Create decision variables: player_slot[(player_idx, slot)] = 1 if player assigned to slot
    player_slot = {}
    for idx, row in df.iterrows():
        player = row['PLAYER']
        position = row['POSITION']
        for slot in slots:
            if is_eligible_for_slot(position, slot):
                player_slot[(idx, slot)] = LpVariable(f"player_{idx}_slot_{slot}", cat='Binary')

    logger.info(f"Created {len(player_slot)} decision variables")

    # Objective: Maximize total predicted fantasy points
    prob += lpSum([
        df.loc[idx, 'MY_MODEL_PREDICTED_FP'] * player_slot[(idx, slot)]
        for (idx, slot) in player_slot.keys()
    ]), "Total_Fantasy_Points"

    # Constraint 1: Exactly one player per slot
    for slot in slots:
        prob += lpSum([
            player_slot[(idx, s)]
            for (idx, s) in player_slot.keys()
            if s == slot
        ]) == 1, f"One_Player_In_{slot}"

    # Constraint 2: Each player can be used at most once
    for idx in df.index:
        prob += lpSum([
            player_slot[(i, slot)]
            for (i, slot) in player_slot.keys()
            if i == idx
        ]) <= 1, f"Player_{idx}_Used_Once"

    # Constraint 3: Total salary <= $50,000
    prob += lpSum([
        df.loc[idx, 'SALARY'] * player_slot[(idx, slot)]
        for (idx, slot) in player_slot.keys()
    ]) <= 50000, "Salary_Cap"

    # Solve the problem
    logger.info("Solving linear programming problem...")
    solver = PULP_CBC_CMD(msg=0)  # msg=0 suppresses solver output
    prob.solve(solver)

    # Check if solution is optimal
    if prob.status != 1:
        logger.error(f"Optimization failed with status: {prob.status}")
        return pd.DataFrame()

    logger.info(f"Optimization successful! Optimal fantasy points: {value(prob.objective):.2f}")

    # Extract the lineup
    lineup_data = []
    total_salary = 0
    total_fp = 0

    for (idx, slot), var in player_slot.items():
        if value(var) == 1:  # Player is selected for this slot
            player_row = df.loc[idx]
            lineup_data.append({
                'SLOT': slot,
                'PLAYER': player_row['PLAYER'],
                'POSITION': player_row['POSITION'],
                'SALARY': player_row['SALARY'],
                'PREDICTED_FP': player_row['MY_MODEL_PREDICTED_FP'],
                'GAME_DATE': player_row['GAME_DATE']
            })
            total_salary += player_row['SALARY']
            total_fp += player_row['MY_MODEL_PREDICTED_FP']

    lineup_df = pd.DataFrame(lineup_data)

    # Sort by slot order
    slot_order = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4, 'G': 5, 'F': 6, 'UTIL': 7}
    lineup_df['slot_order'] = lineup_df['SLOT'].map(slot_order)
    lineup_df = lineup_df.sort_values('slot_order').drop(columns=['slot_order'])

    logger.info(f"Lineup created: {len(lineup_df)} players, ${total_salary:.0f} salary, {total_fp:.2f} projected FP")

    return lineup_df

def generate_optimal_lineup():
    """Main function to generate optimal DraftKings lineup"""
    logger.info("Starting lineup optimization")

    try:
        # Load today's predictions
        df_predictions = load_dataframe_from_s3('data/daily_predictions/current.parquet')

        logger.info(f"Loaded {len(df_predictions)} total predictions")

        # Filter for today's date only
        today = datetime.date.today()
        df_predictions['GAME_DATE'] = pd.to_datetime(df_predictions['GAME_DATE']).dt.date
        df_today = df_predictions[df_predictions['GAME_DATE'] == today].copy()

        logger.info(f"Filtered to {len(df_today)} predictions for today ({today})")

        if df_today.empty:
            logger.warning("No predictions found for today")
            return {
                'success': False,
                'error': 'No predictions available for today'
            }

        # Check required columns
        required_cols = {'PLAYER', 'POSITION', 'SALARY', 'MY_MODEL_PREDICTED_FP', 'GAME_DATE'}
        missing_cols = required_cols - set(df_today.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return {
                'success': False,
                'error': f'Missing required columns: {missing_cols}'
            }

        # Optimize lineup
        lineup_df = optimize_lineup(df_today)

        if lineup_df.empty:
            logger.error("Failed to generate optimal lineup")
            return {
                'success': False,
                'error': 'Failed to generate optimal lineup'
            }
        
        if 'GAME_DATE' in lineup_df.columns and not lineup_df['GAME_DATE'].empty and isinstance(lineup_df['GAME_DATE'].iloc[0], datetime.date):
            lineup_df['GAME_DATE'] = lineup_df['GAME_DATE'].apply(lambda x: x.isoformat())
        # Save lineup to S3
        save_dataframe_to_s3(lineup_df, 'data/daily_lineups/current.parquet')

        # Calculate summary stats
        total_salary = lineup_df['SALARY'].sum()
        total_fp = lineup_df['PREDICTED_FP'].sum()
        remaining_salary = 50000 - total_salary

        # Send SNS notification
        send_sns_notification(lineup_df, total_salary, total_fp, remaining_salary)

        logger.info("Lineup optimization completed successfully")

        return {
            'success': True,
            'lineup_size': len(lineup_df),
            'total_salary': float(total_salary),
            'remaining_salary': float(remaining_salary),
            'total_predicted_fp': float(total_fp),
            'players': lineup_df.to_dict('records')
        }

    except Exception as e:
        logger.error(f"Error in lineup optimization: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Lineup optimizer Lambda function started")

    try:
        result = generate_optimal_lineup()

        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Lineup optimization completed successfully',
                    'lineup_size': result['lineup_size'],
                    'total_salary': result['total_salary'],
                    'remaining_salary': result['remaining_salary'],
                    'total_predicted_fp': result['total_predicted_fp'],
                    'players': result['players']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Lineup optimization failed',
                    'error': result['error']
                })
            }

    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Lambda execution failed',
                'error': str(e)
            })
        }

# For local testing
if __name__ == "__main__":
    result = generate_optimal_lineup()
    print(f"Lineup optimization result: {json.dumps(result, indent=2)}")
