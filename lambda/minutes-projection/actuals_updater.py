"""
Post-game actual minutes and fantasy points updater
"""

import pandas as pd
import logging
import pytz
from datetime import datetime, timedelta
from s3_utils import load_from_s3, save_to_s3

logger = logging.getLogger()


# ==================== Post-Game Actual Minutes Update ====================

def update_actual_minutes(box_scores, target_date=None, today=None):
    """
    Update ACTUAL_MIN and ACTUAL_FP for all 4 models after games complete

    Args:
        box_scores: DataFrame of box score data
        target_date: DEPRECATED - function now updates ALL missing actuals
        today: DEPRECATED - not used anymore
    """
    logger.info(f"Updating actual minutes and FP for ALL games with available box scores")

    try:
        # Use provided box scores
        df_box = box_scores
        if df_box.empty:
            logger.warning("No box score data available")
            return 0

        # Extract ALL actual minutes + FP from box scores (don't filter by date)
        df_box['GAME_DATE'] = pd.to_datetime(df_box['GAME_DATE']).dt.date
        df_actuals = df_box[['PLAYER', 'GAME_DATE', 'MIN', 'FP']].rename(
            columns={'MIN': 'ACTUAL_MIN', 'FP': 'ACTUAL_FP', 'GAME_DATE': 'DATE'}
        )

        if df_actuals.empty:
            logger.warning(f"No box scores found")
            return 0

        logger.info(f"Found {len(df_actuals)} actual results across all dates")

        # Update all 4 model files
        models = [
            'complex_position_overlap',
            'direct_position_only',
            'formula_c_baseline',
            'daily_fantasy_fuel_baseline'
        ]

        updated_count = 0
        for model in models:
            # Update minutes projections
            key = f'model_comparison/{model}/minutes_projections.parquet'
            df_proj = load_from_s3(key)

            if not df_proj.empty:
                df_proj['DATE'] = pd.to_datetime(df_proj['DATE']).dt.date

                # Track rows with missing actuals before update
                missing_before = df_proj['ACTUAL_MIN'].isna().sum()

                # Drop existing ACTUAL_MIN and ACTUAL_FP to avoid conflicts
                if 'ACTUAL_MIN' in df_proj.columns:
                    df_proj = df_proj.drop(columns=['ACTUAL_MIN'])
                if 'ACTUAL_FP' in df_proj.columns:
                    df_proj = df_proj.drop(columns=['ACTUAL_FP'])

                # Merge actual data on (PLAYER, DATE) to match correct game
                df_proj = df_proj.merge(
                    df_actuals[['PLAYER', 'DATE', 'ACTUAL_MIN', 'ACTUAL_FP']],
                    on=['PLAYER', 'DATE'],
                    how='left'
                )

                # Track rows with missing actuals after update
                missing_after = df_proj['ACTUAL_MIN'].isna().sum()
                updated_this_model = missing_before - missing_after

                save_to_s3(df_proj, key)
                updated_count += updated_this_model
                logger.info(f"Updated {updated_this_model} projection records for {model} (was {missing_before} missing, now {missing_after} missing)")

            # Update lineups
            lineup_key = f'model_comparison/{model}/daily_lineups.parquet'
            df_lineup = load_from_s3(lineup_key)

            if not df_lineup.empty:
                df_lineup['DATE'] = pd.to_datetime(df_lineup['DATE']).dt.date

                # Track rows with missing actuals before update
                lineup_missing_before = df_lineup['ACTUAL_FP'].isna().sum()

                # Drop existing ACTUAL_FP to avoid conflicts
                if 'ACTUAL_FP' in df_lineup.columns:
                    df_lineup = df_lineup.drop(columns=['ACTUAL_FP'])

                # Merge actual FP on (PLAYER, DATE) to match correct game
                df_lineup = df_lineup.merge(
                    df_actuals[['PLAYER', 'DATE', 'ACTUAL_FP']],
                    on=['PLAYER', 'DATE'],
                    how='left'
                )

                # Track rows with missing actuals after update
                lineup_missing_after = df_lineup['ACTUAL_FP'].isna().sum()
                lineup_updated = lineup_missing_before - lineup_missing_after

                save_to_s3(df_lineup, lineup_key)
                logger.info(f"Updated {lineup_updated} lineup records for {model} (was {lineup_missing_before} missing, now {lineup_missing_after} missing)")

        return updated_count

    except Exception as e:
        logger.error(f"Error updating actuals: {str(e)}", exc_info=True)
        raise
