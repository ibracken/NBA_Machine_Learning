"""
Analyze and compare performance of 5 NBA prediction models

This script loads the daily lineups from all 5 models and compares their performance
by analyzing median FP, mean FP, and top-performing lineups.
"""

import pandas as pd
import boto3
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'

def load_dataframe_from_s3(key):
    """Load DataFrame from S3 Parquet"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except Exception as e:
        logger.error(f"Error loading data from {key}: {e}")
        return pd.DataFrame()

def load_all_model_lineups():
    """Load daily lineups from all 5 models"""
    models = {
        'Complex Position Overlap': 'model_comparison/complex_position_overlap/daily_lineups.parquet',
        'Direct Position Only': 'model_comparison/direct_position_only/daily_lineups.parquet',
        'Formula C Baseline': 'model_comparison/formula_c_baseline/daily_lineups.parquet',
        'SportsLine Baseline': 'model_comparison/sportsline_baseline/daily_lineups.parquet',
        'DailyFantasyFuel Baseline': 'model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet'
    }

    all_lineups = {}
    for model_name, s3_key in models.items():
        logger.info(f"Loading {model_name}...")
        df = load_dataframe_from_s3(s3_key)

        if not df.empty:
            # Add model name column
            df['MODEL'] = model_name
            all_lineups[model_name] = df
            logger.info(f"  Loaded {len(df)} lineup entries for {model_name}")
        else:
            logger.warning(f"  No data found for {model_name}")

    return all_lineups

def calculate_daily_lineup_totals(lineups_dict):
    """
    Calculate total ACTUAL_FP for each daily lineup across all models
    Returns a DataFrame with columns: MODEL, DATE, TOTAL_ACTUAL_FP, PLAYER_COUNT
    """
    daily_totals = []

    for model_name, df in lineups_dict.items():
        # Filter to only lineups with ACTUAL_FP (games that have been played)
        df_with_actual = df[df['ACTUAL_FP'].notna()].copy()

        if df_with_actual.empty:
            logger.warning(f"{model_name}: No lineups with ACTUAL_FP data yet")
            continue

        # Group by DATE and sum ACTUAL_FP for each daily lineup
        daily = df_with_actual.groupby('DATE').agg({
            'ACTUAL_FP': 'sum',
            'PLAYER': 'count'
        }).reset_index()

        daily.columns = ['DATE', 'TOTAL_ACTUAL_FP', 'PLAYER_COUNT']
        daily['MODEL'] = model_name

        daily_totals.append(daily)
        logger.info(f"{model_name}: {len(daily)} complete daily lineups")

    if not daily_totals:
        logger.error("No models have actual FP data yet!")
        return pd.DataFrame()

    # Combine all models
    return pd.concat(daily_totals, ignore_index=True)

def print_model_summary_stats(daily_totals_df):
    """Print median and mean FP for each model"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)

    for model in daily_totals_df['MODEL'].unique():
        model_data = daily_totals_df[daily_totals_df['MODEL'] == model]

        median_fp = model_data['TOTAL_ACTUAL_FP'].median()
        mean_fp = model_data['TOTAL_ACTUAL_FP'].mean()
        count = len(model_data)
        min_fp = model_data['TOTAL_ACTUAL_FP'].min()
        max_fp = model_data['TOTAL_ACTUAL_FP'].max()

        print(f"\n{model}")
        print(f"  Total Lineups: {count}")
        print(f"  Median FP:     {median_fp:.2f}")
        print(f"  Mean FP:       {mean_fp:.2f}")
        print(f"  Min FP:        {min_fp:.2f}")
        print(f"  Max FP:        {max_fp:.2f}")

def print_top_lineups(daily_totals_df, lineups_dict, top_n=2):
    """Print top N lineups for each model with player details"""
    print("\n" + "="*80)
    print(f"TOP {top_n} LINEUPS BY MODEL")
    print("="*80)

    for model in daily_totals_df['MODEL'].unique():
        model_data = daily_totals_df[daily_totals_df['MODEL'] == model]

        # Get top N lineups by TOTAL_ACTUAL_FP
        top_lineups = model_data.nlargest(top_n, 'TOTAL_ACTUAL_FP')

        print(f"\n{model}")
        print("-" * 80)

        for idx, (_, row) in enumerate(top_lineups.iterrows(), 1):
            date = row['DATE']
            total_fp = row['TOTAL_ACTUAL_FP']

            print(f"\n  #{idx} - {date} - Total: {total_fp:.2f} FP")
            print("  " + "-" * 76)

            # Get player details for this lineup
            lineup_details = lineups_dict[model][
                (lineups_dict[model]['DATE'] == date) &
                (lineups_dict[model]['ACTUAL_FP'].notna())
            ].sort_values('SLOT')

            for _, player_row in lineup_details.iterrows():
                slot = player_row['SLOT']
                player = player_row['PLAYER'].title()
                team = player_row['TEAM']
                actual_fp = player_row['ACTUAL_FP']
                projected_fp = player_row['PROJECTED_FP']
                diff = actual_fp - projected_fp

                print(f"    {slot:4s} {player:25s} ({team:3s})  Actual: {actual_fp:5.1f}  Proj: {projected_fp:5.1f}  Diff: {diff:+6.1f}")

def print_median_lineups(daily_totals_df, lineups_dict):
    """Print the lineup closest to median FP for each model"""
    print("\n" + "="*80)
    print("MEDIAN FP LINEUPS (lineup closest to median total FP)")
    print("="*80)

    for model in daily_totals_df['MODEL'].unique():
        model_data = daily_totals_df[daily_totals_df['MODEL'] == model]

        median_fp = model_data['TOTAL_ACTUAL_FP'].median()

        # Find the lineup closest to median
        model_data['diff_from_median'] = abs(model_data['TOTAL_ACTUAL_FP'] - median_fp)
        median_lineup = model_data.nsmallest(1, 'diff_from_median').iloc[0]

        date = median_lineup['DATE']
        total_fp = median_lineup['TOTAL_ACTUAL_FP']

        print(f"\n{model}")
        print("-" * 80)
        print(f"  Median FP: {median_fp:.2f}")
        print(f"  Closest Lineup: {date} - Total: {total_fp:.2f} FP (diff: {abs(total_fp - median_fp):.2f})")
        print("  " + "-" * 76)

        # Get player details for this lineup
        lineup_details = lineups_dict[model][
            (lineups_dict[model]['DATE'] == date) &
            (lineups_dict[model]['ACTUAL_FP'].notna())
        ].sort_values('SLOT')

        for _, player_row in lineup_details.iterrows():
            slot = player_row['SLOT']
            player = player_row['PLAYER'].title()
            team = player_row['TEAM']
            actual_fp = player_row['ACTUAL_FP']
            projected_fp = player_row['PROJECTED_FP']
            diff = actual_fp - projected_fp

            print(f"    {slot:4s} {player:25s} ({team:3s})  Actual: {actual_fp:5.1f}  Proj: {projected_fp:5.1f}  Diff: {diff:+6.1f}")

def print_mean_lineups(daily_totals_df, lineups_dict):
    """Print the lineup closest to mean FP for each model"""
    print("\n" + "="*80)
    print("MEAN FP LINEUPS (lineup closest to mean total FP)")
    print("="*80)

    for model in daily_totals_df['MODEL'].unique():
        model_data = daily_totals_df[daily_totals_df['MODEL'] == model]

        mean_fp = model_data['TOTAL_ACTUAL_FP'].mean()

        # Find the lineup closest to mean
        model_data['diff_from_mean'] = abs(model_data['TOTAL_ACTUAL_FP'] - mean_fp)
        mean_lineup = model_data.nsmallest(1, 'diff_from_mean').iloc[0]

        date = mean_lineup['DATE']
        total_fp = mean_lineup['TOTAL_ACTUAL_FP']

        print(f"\n{model}")
        print("-" * 80)
        print(f"  Mean FP: {mean_fp:.2f}")
        print(f"  Closest Lineup: {date} - Total: {total_fp:.2f} FP (diff: {abs(total_fp - mean_fp):.2f})")
        print("  " + "-" * 76)

        # Get player details for this lineup
        lineup_details = lineups_dict[model][
            (lineups_dict[model]['DATE'] == date) &
            (lineups_dict[model]['ACTUAL_FP'].notna())
        ].sort_values('SLOT')

        for _, player_row in lineup_details.iterrows():
            slot = player_row['SLOT']
            player = player_row['PLAYER'].title()
            team = player_row['TEAM']
            actual_fp = player_row['ACTUAL_FP']
            projected_fp = player_row['PROJECTED_FP']
            diff = actual_fp - projected_fp

            print(f"    {slot:4s} {player:25s} ({team:3s})  Actual: {actual_fp:5.1f}  Proj: {projected_fp:5.1f}  Diff: {diff:+6.1f}")

def main():
    """Main analysis function"""
    logger.info("Starting model performance analysis")

    # Load all model lineups
    lineups_dict = load_all_model_lineups()

    if not lineups_dict:
        logger.error("No lineup data loaded. Exiting.")
        return

    # Calculate daily lineup totals
    daily_totals_df = calculate_daily_lineup_totals(lineups_dict)

    if daily_totals_df.empty:
        logger.error("No daily totals calculated. Need lineups with ACTUAL_FP data.")
        return

    # Print all analyses
    print_model_summary_stats(daily_totals_df)
    print_median_lineups(daily_totals_df, lineups_dict)
    print_mean_lineups(daily_totals_df, lineups_dict)
    print_top_lineups(daily_totals_df, lineups_dict, top_n=2)

    print("\n" + "="*80)
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
