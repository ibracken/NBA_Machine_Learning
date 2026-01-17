"""
Model processing pipeline - unified FP prediction, filtering, and lineup optimization
"""

import pandas as pd
import logging
from lineup_optimizer import predict_fantasy_points, optimize_lineup
from s3_utils import load_from_s3, save_to_s3

logger = logging.getLogger()


def process_model(model_name, s3_path, projections_data, daily_preds, today, box_scores,
                  injury_context=None, fp_model_names=None):
    """
    Unified model processing pipeline: FP prediction, filtering, lineup optimization, and S3 persistence

    Handles all common operations across the 4 projection models:
    1. Convert projections to DataFrame (if needed)
    2. Add fantasy points predictions
    3. Filter to today's players only
    4. Optimize lineup
    5. Load/merge/save lineup data
    6. Load/merge/save projection data
    7. Save injury context (if provided)

    Args:
        model_name: Display name for logging (e.g., "Complex Position Overlap")
        s3_path: S3 path prefix (e.g., "model_comparison/complex_position_overlap")
        projections_data: Either list of projection dicts or existing DataFrame
        daily_preds: Daily predictions for filtering and lineup optimization
        today: Current date
        box_scores: Box score data for FP predictions
        injury_context: Optional injury context DataFrame to save (for injury-aware models)

    Returns:
        Tuple of (full_projections_df, today_lineup_df) - DataFrames with all dates
    """
    logger.info(f"Processing model: {model_name}")

    # Convert to DataFrame if needed
    if isinstance(projections_data, list):
        df_proj = pd.DataFrame(projections_data)
    else:
        df_proj = projections_data.copy()

    # Add fantasy points projections
    df_proj = predict_fantasy_points(df_proj, box_scores, daily_preds, today)

    # Filter to today's players only
    if not daily_preds.empty:
        todays_players = daily_preds[daily_preds['GAME_DATE'] == today]['PLAYER'].unique()
        df_proj = df_proj[df_proj['PLAYER'].isin(todays_players)]
        logger.info(f"{model_name}: Filtered to {len(df_proj)} players with games on {today}")

    # Optimize lineup(s) for each FP model variant
    lineups_by_fp = {}
    fp_model_names = fp_model_names or ['current']

    for fp_model in fp_model_names:
        fp_col = f'PROJECTED_FP_{fp_model}'
        if fp_col not in df_proj.columns:
            logger.warning(f"{model_name}: Missing {fp_col} - skipping lineup")
            lineups_by_fp[fp_model] = pd.DataFrame()
            continue

        df_lineup = df_proj.copy()
        df_lineup['PROJECTED_FP'] = df_lineup[fp_col]

        lineup = optimize_lineup(df_lineup, daily_preds, today)

        # Save lineup (merge with existing) in FP-specific folder
        if not lineup.empty:
            lineup_path = f"{s3_path}/fp_{fp_model}/daily_lineups.parquet"
            existing_lineup = load_from_s3(lineup_path)
            if not existing_lineup.empty:
                existing_lineup['DATE'] = pd.to_datetime(existing_lineup['DATE']).dt.date
                existing_lineup = existing_lineup[existing_lineup['DATE'] != today]
                lineup = pd.concat([existing_lineup, lineup], ignore_index=True)
            save_to_s3(lineup, lineup_path)
            logger.info(f"{model_name} ({fp_model}) lineup saved: {len(lineup[lineup['DATE'] == today])} players")

        lineups_by_fp[fp_model] = lineup

    # Save projections (merge with existing)
    proj_path = f"{s3_path}/minutes_projections.parquet"
    existing_proj = load_from_s3(proj_path)
    if not existing_proj.empty:
        existing_proj['DATE'] = pd.to_datetime(existing_proj['DATE']).dt.date
        existing_proj = existing_proj[existing_proj['DATE'] != today]
        df_proj = pd.concat([existing_proj, df_proj], ignore_index=True)
    save_to_s3(df_proj, proj_path)

    # Save injury context if provided
    if injury_context is not None:
        context_path = f"injury_context/{s3_path.split('/')[-1]}.parquet"
        save_to_s3(injury_context, context_path)

    return df_proj, lineups_by_fp
