"""
S3 and SNS utility functions
"""

import pandas as pd
import io
import pickle
import logging
from config import s3_client, sns_client, SNS_TOPIC_ARN, BUCKET_NAME

logger = logging.getLogger()


# ==================== S3 Helper Functions ====================

def load_from_s3(key):
    """Load parquet file from S3"""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(io.BytesIO(response['Body'].read()))
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"File not found: {key}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading {key}: {str(e)}")
        raise


def save_to_s3(df, key):
    """Save parquet file to S3"""
    try:
        # Convert date columns to datetime for parquet compatibility
        df = df.copy()
        date_columns = ['INJURY_DATE', 'RETURN_DATE', 'UPDATED_DATE', 'DATE', 'GAME_DATE']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=buffer.getvalue())
        logger.info(f"Saved to S3: {key}")
    except Exception as e:
        logger.error(f"Error saving {key}: {str(e)}")
        raise


def load_model_from_s3(key):
    """Load pickled model from S3"""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return pickle.loads(response['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Model not found: {key}")
        return None
    except Exception as e:
        logger.error(f"Error loading model {key}: {str(e)}")
        return None

# ==================== Notification Helper ====================

def send_multi_model_notification(lineups_dict, date):
    """
    Send a single email containing lineups from all 5 models
    """
    logger.info("Preparing SNS notification for all models")

    try:
        message_lines = [f"🏀 NBA Lineup Projections for {date} 🏀\n"]

        # Loop through each model in the dictionary
        for model_name, df in lineups_dict.items():
            message_lines.append(f"\n{'='*20} {model_name} {'='*20}")

            if df is None or df.empty:
                message_lines.append("No valid lineup generated for this model.\n")
                continue

            # Filter for today's lineup only (in case old ones are in the DF)
            # Ensure we are looking at the right date format
            df_today = df.copy()
            # Handle string vs date object comparison if necessary
            if not df_today.empty:
                # Calculate totals
                total_salary = df_today['SALARY'].sum()
                total_fp = df_today['PROJECTED_FP'].sum()

                message_lines.append(f"Salary: ${total_salary:,.0f} | Proj FP: {total_fp:.1f}")
                message_lines.append("-" * 50)

                # Sort by slot order for readability
                slot_order = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4, 'G': 5, 'F': 6, 'UTIL': 7}
                df_today['sort'] = df_today['SLOT'].map(slot_order)
                df_today = df_today.sort_values('sort')

                for _, row in df_today.iterrows():
                    message_lines.append(
                        f"{row['SLOT']:4} | {row['PLAYER']:<20} | "
                        f"${row['SALARY']:<5,.0f} | {row['PROJECTED_FP']:5.1f} FP | {row['PROJECTED_MIN']:4.1f} Min"
                    )

        full_message = "\n".join(message_lines)

        # Publish to SNS
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"NBA Projections ({date}): {len(lineups_dict)} Lineups Generated",
            Message=full_message
        )
        logger.info("SNS notification sent successfully")

    except Exception as e:
        logger.error(f"Failed to send SNS notification: {str(e)}")
