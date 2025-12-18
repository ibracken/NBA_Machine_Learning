# Import all buckets and take their heads hehe
import pandas as pd
from aws.s3_utils import load_dataframe_from_s3, load_model_from_s3
import sys
import os
import json
import boto3

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_s3_data():
    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None) # Show all columns
    pd.set_option('display.width', None) # Make sure the width is not restricted
    pd.set_option('display.max_colwidth', None) # Show full content of each column
    """Test all S3 data files and show their structure"""

    with open('s3_bucket_test_results.txt', 'w') as f:
        print("=" * 80, file=f)
        print("S3 BUCKET DATA TEST - NBA PREDICTION MODEL", file=f)
        print("=" * 80, file=f)

        # ========================================================================
        # INPUT DATA (from scrapers and daily-predictions)
        # ========================================================================
        print(f"\n\n{'#' * 80}", file=f)
        print("# INPUT DATA - Raw scraped data used by all models", file=f)
        print(f"{'#' * 80}\n", file=f)

        input_data_keys = [
            ('data/box_scores/current.parquet', 'Box scores with stats and rolling averages'),
            ('data/daily_predictions/current.parquet', 'DraftKings + DFF projections'),
            ('data/advanced_player_stats/current.parquet', 'Advanced NBA stats'),
            ('data/daily_lineups/current.parquet', 'Historical daily lineups'),
            ('data/injuries/current.parquet', 'Scraped injury data')
        ]

        for key, description in input_data_keys:
            test_file(f, key, description)

        # ========================================================================
        # TRAINED MODELS
        # ========================================================================
        print(f"\n\n{'#' * 80}", file=f)
        print("# TRAINED MODELS - Fantasy points prediction", file=f)
        print(f"{'#' * 80}\n", file=f)

        model_keys = [
            # ('models/RFCluster.sav', 'GradientBoosting fantasy points model'),
            ('models/RFCluster_feature_names.json', 'Expected features for prediction')
        ]

        for key, description in model_keys:
            test_file(f, key, description)

        # ========================================================================
        # MODEL COMPARISON - Complex Position Overlap (injury-aware, 2x multiplier)
        # ========================================================================
        print(f"\n\n{'#' * 80}", file=f)
        print("# MODEL 1: COMPLEX POSITION OVERLAP", file=f)
        print("# - Injury-aware minute redistribution", file=f)
        print("# - 2x multiplier for exact position matches", file=f)
        print(f"{'#' * 80}\n", file=f)

        complex_keys = [
            ('model_comparison/complex_position_overlap/daily_lineups.parquet', 'Optimized 8-player DK lineups'),
            ('model_comparison/complex_position_overlap/minutes_projections.parquet', 'Minutes projections for all players')
        ]

        for key, description in complex_keys:
            test_file(f, key, description)

        # ========================================================================
        # MODEL COMPARISON - Direct Position Only (injury-aware, exact position)
        # ========================================================================
        print(f"\n\n{'#' * 80}", file=f)
        print("# MODEL 2: DIRECT POSITION EXCHANGE", file=f)
        print("# - Injury-aware minute redistribution", file=f)
        print("# - Only exact position matches get injured player's minutes", file=f)
        print(f"{'#' * 80}\n", file=f)

        direct_keys = [
            ('model_comparison/direct_position_only/daily_lineups.parquet', 'Optimized 8-player DK lineups'),
            ('model_comparison/direct_position_only/minutes_projections.parquet', 'Minutes projections for all players')
        ]

        for key, description in direct_keys:
            test_file(f, key, description)

        # ========================================================================
        # MODEL COMPARISON - Formula C Baseline (no injury handling)
        # ========================================================================
        print(f"\n\n{'#' * 80}", file=f)
        print("# MODEL 3: FORMULA C BASELINE", file=f)
        print("# - No injury handling", file=f)
        print("# - Simple baseline formula", file=f)
        print(f"{'#' * 80}\n", file=f)

        formula_c_keys = [
            ('model_comparison/formula_c_baseline/daily_lineups.parquet', 'Optimized 8-player DK lineups'),
            ('model_comparison/formula_c_baseline/minutes_projections.parquet', 'Minutes projections for all players')
        ]

        for key, description in formula_c_keys:
            test_file(f, key, description)

        # ========================================================================
        # MODEL COMPARISON - DailyFantasyFuel Baseline
        # ========================================================================
        print(f"\n\n{'#' * 80}", file=f)
        print("# MODEL 4: DAILYFANTASYFUEL BASELINE", file=f)
        print("# - Uses DailyFantasyFuel fantasy points projections", file=f)
        print("# - Industry standard comparison", file=f)
        print(f"{'#' * 80}\n", file=f)

        dff_keys = [
            ('model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet', 'Optimized 8-player DK lineups (using DFF FP projections)')
        ]

        for key, description in dff_keys:
            test_file(f, key, description)

        # ========================================================================
        # INJURY CONTEXT - State tracking for injury-aware models
        # ========================================================================
        print(f"\n\n{'#' * 80}", file=f)
        print("# INJURY CONTEXT - Stateful injury tracking", file=f)
        print("# - Tracks BENEFICIARY, EX_BENEFICIARY, RETURNING statuses", file=f)
        print("# - Persisted across lambda runs", file=f)
        print(f"{'#' * 80}\n", file=f)

        injury_keys = [
            ('injury_context/complex_position_overlap.parquet', 'Complex model injury state'),
            ('injury_context/direct_position_only.parquet', 'Direct model injury state')
        ]

        for key, description in injury_keys:
            test_file(f, key, description, show_all_rows=True)

        print(f"\n\n{'=' * 80}", file=f)
        print("S3 BUCKET TEST COMPLETE", file=f)
        print(f"{'=' * 80}", file=f)


def test_file(f, key, description, show_all_rows=False):
    """Test a single S3 file and write results to file handle"""
    print(f"\n{'-' * 80}", file=f)
    print(f"FILE: {key}", file=f)
    print(f"DESC: {description}", file=f)
    print(f"{'-' * 80}", file=f)

    try:
        if key.endswith('.sav'):
            # Test model loading
            print(f"Loading model from S3...", file=f)
            model = load_model_from_s3(key)
            print(f" Model loaded successfully!", file=f)
            print(f"Model type: {type(model).__name__}", file=f)
            if hasattr(model, 'feature_importances_'):
                print(f"Number of features: {len(model.feature_importances_)}", file=f)
                print(f"Top 5 features:", file=f)
                # This would require feature names, skip for now
            if hasattr(model, 'n_estimators'):
                print(f"Number of estimators: {model.n_estimators}", file=f)

        elif key.endswith('.json'):
            # Test JSON loading
            print(f"Loading JSON from S3...", file=f)
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket='nba-prediction-ibracken', Key=key)
            data = json.loads(response['Body'].read())
            print(f" JSON loaded successfully!", file=f)

            if 'features' in data:
                print(f"Feature count: {len(data['features'])}", file=f)
                print(f"First 20 features: {data['features'][:20]}", file=f)
            else:
                print(f"JSON keys: {list(data.keys())}", file=f)

        else:
            # Test DataFrame loading
            print(f"Loading DataFrame from S3...", file=f)
            df = load_dataframe_from_s3(key)

            if df.empty:
                print(f" DataFrame is EMPTY", file=f)
            else:
                print(f" DataFrame loaded successfully!", file=f)
                print(f"Shape: {df.shape}", file=f)
                print(f"Columns: {list(df.columns)}", file=f)

                # Show most recent data by appropriate date column
                date_col = None
                for col in ['DATE', 'GAME_DATE']:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df_sorted = df.sort_values(date_col, ascending=False)
                    if show_all_rows:
                        print(f"\nAll rows (sorted by {date_col}):", file=f)
                        print(df_sorted, file=f)
                    else:
                        print(f"\nMost recent 20 rows (sorted by {date_col}):", file=f)
                        print(df_sorted.head(20), file=f)
                else:
                    if show_all_rows:
                        print(f"\nAll rows:", file=f)
                        print(df, file=f)
                    else:
                        print(f"\nFirst 20 rows:", file=f)
                        print(df.head(20), file=f)

                # Show some basic stats
                print(f"\nBasic info:", file=f)
                print(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB", file=f)

                null_counts = df.isnull().sum()
                if null_counts.sum() > 0:
                    print(f"- Null values per column:", file=f)
                    for col, count in null_counts[null_counts > 0].items():
                        print(f"  {col}: {count}", file=f)
                else:
                    print(f"- No null values", file=f)

    except Exception as e:
        print(f"Error loading {key}: {e}", file=f)


if __name__ == "__main__":
    test_s3_data()
