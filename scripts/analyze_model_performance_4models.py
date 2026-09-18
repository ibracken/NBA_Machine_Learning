"""
Analyze and compare performance of 4 NBA lineup models.

Models:
1. Complex Position Overlap
2. Direct Position Only
3. Formula C Baseline
4. DailyFantasyFuel Baseline
"""

import logging
from io import BytesIO

import boto3
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
BUCKET_NAME = "nba-prediction-ibracken"


def load_dataframe_from_s3(key):
    """Load a parquet DataFrame from S3. Return empty DataFrame on failure."""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except Exception as exc:
        logger.error("Error loading data from %s: %s", key, exc)
        return pd.DataFrame()


def list_s3_keys(prefix):
    """List all S3 keys under a prefix."""
    keys = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": BUCKET_NAME, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        continuation_token = resp.get("NextContinuationToken")
    return keys


def load_archive_lineups_for_model(model_prefix):
    """
    Load and combine old pre-FP archive lineup files for a model.

    Expected naming pattern:
      model_comparison/<model>/archive/daily_lineups_pre_fp_models_YYYY-MM-DD.parquet
    """
    archive_prefix = f"{model_prefix}/archive/"
    keys = list_s3_keys(archive_prefix)
    keys = [k for k in keys if "daily_lineups_pre_fp_models_" in k and k.endswith(".parquet")]

    if not keys:
        return pd.DataFrame(), []

    frames = []
    loaded_keys = []
    for key in sorted(keys):
        df = load_dataframe_from_s3(key)
        if not df.empty:
            frames.append(df)
            loaded_keys.append(key)

    if not frames:
        return pd.DataFrame(), []

    combined = pd.concat(frames, ignore_index=True)
    return combined, loaded_keys


def load_all_model_lineups():
    """Load daily lineups from 4 models."""
    model_paths = {
        "Complex Position Overlap": [
            "model_comparison/complex_position_overlap/archive/daily_lineups.parquet",
            "model_comparison/complex_position_overlap/daily_lineups.parquet",
        ],
        "Direct Position Only": [
            "model_comparison/direct_position_only/archive/daily_lineups.parquet",
            "model_comparison/direct_position_only/daily_lineups.parquet",
        ],
        "Formula C Baseline": [
            "model_comparison/formula_c_baseline/archive/daily_lineups.parquet",
            "model_comparison/formula_c_baseline/daily_lineups.parquet",
        ],
        "DailyFantasyFuel Baseline": [
            "model_comparison/daily_fantasy_fuel_baseline/archive/daily_lineups.parquet",
            "model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet",
        ],
    }

    all_lineups = {}
    for model_name, candidate_keys in model_paths.items():
        logger.info("Loading %s...", model_name)
        model_prefix = candidate_keys[-1].rsplit("/", 1)[0]

        # First, try archive pattern files (daily_lineups_pre_fp_models_*)
        archive_df, archive_keys = load_archive_lineups_for_model(model_prefix)
        if not archive_df.empty:
            archive_df["MODEL"] = model_name
            all_lineups[model_name] = archive_df
            logger.info(
                "  Loaded %s lineup entries for %s from %s archive files",
                len(archive_df),
                model_name,
                len(archive_keys),
            )
            continue

        # Fallback to single-file paths
        loaded = False
        for s3_key in candidate_keys:
            df = load_dataframe_from_s3(s3_key)
            if df.empty:
                continue
            df["MODEL"] = model_name
            all_lineups[model_name] = df
            logger.info("  Loaded %s lineup entries for %s from %s", len(df), model_name, s3_key)
            loaded = True
            break

        if not loaded:
            logger.warning("  No data found for %s in archive or non-archive paths", model_name)

    return all_lineups


def calculate_daily_lineup_totals(lineups_dict):
    """
    Calculate total ACTUAL_FP for each daily lineup across all models.
    Returns DataFrame: MODEL, DATE, TOTAL_ACTUAL_FP, PLAYER_COUNT
    """
    daily_totals = []

    for model_name, df in lineups_dict.items():
        df_with_actual = df[df["ACTUAL_FP"].notna()].copy()
        if df_with_actual.empty:
            logger.warning("%s: No lineups with ACTUAL_FP data yet", model_name)
            continue

        daily = (
            df_with_actual.groupby("DATE")
            .agg({"ACTUAL_FP": "sum", "PLAYER": "count"})
            .reset_index()
        )
        daily.columns = ["DATE", "TOTAL_ACTUAL_FP", "PLAYER_COUNT"]
        daily["MODEL"] = model_name
        daily_totals.append(daily)
        logger.info("%s: %s complete daily lineups", model_name, len(daily))

    if not daily_totals:
        logger.error("No models have actual FP data yet")
        return pd.DataFrame()

    return pd.concat(daily_totals, ignore_index=True)


def print_model_summary_stats(daily_totals_df):
    """Print median/mean/min/max FP for each model."""
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)

    for model in daily_totals_df["MODEL"].unique():
        model_data = daily_totals_df[daily_totals_df["MODEL"] == model]
        print(f"\n{model}")
        print(f"  Total Lineups: {len(model_data)}")
        print(f"  Median FP:     {model_data['TOTAL_ACTUAL_FP'].median():.2f}")
        print(f"  Mean FP:       {model_data['TOTAL_ACTUAL_FP'].mean():.2f}")
        print(f"  Min FP:        {model_data['TOTAL_ACTUAL_FP'].min():.2f}")
        print(f"  Max FP:        {model_data['TOTAL_ACTUAL_FP'].max():.2f}")


def print_top_lineups(daily_totals_df, lineups_dict, top_n=2):
    """Print top N lineups by total ACTUAL_FP for each model."""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} LINEUPS BY MODEL")
    print("=" * 80)

    for model in daily_totals_df["MODEL"].unique():
        model_data = daily_totals_df[daily_totals_df["MODEL"] == model]
        top_lineups = model_data.nlargest(top_n, "TOTAL_ACTUAL_FP")

        print(f"\n{model}")
        print("-" * 80)

        for idx, (_, row) in enumerate(top_lineups.iterrows(), 1):
            date = row["DATE"]
            total_fp = row["TOTAL_ACTUAL_FP"]
            print(f"\n  #{idx} - {date} - Total: {total_fp:.2f} FP")
            print("  " + "-" * 76)

            lineup_details = lineups_dict[model][
                (lineups_dict[model]["DATE"] == date)
                & (lineups_dict[model]["ACTUAL_FP"].notna())
            ].sort_values("SLOT")

            for _, player_row in lineup_details.iterrows():
                slot = player_row["SLOT"]
                player = player_row["PLAYER"].title()
                team = player_row["TEAM"]
                actual_fp = player_row["ACTUAL_FP"]
                projected_fp = player_row["PROJECTED_FP"]
                diff = actual_fp - projected_fp
                print(
                    f"    {slot:4s} {player:25s} ({team:3s})  "
                    f"Actual: {actual_fp:5.1f}  Proj: {projected_fp:5.1f}  Diff: {diff:+6.1f}"
                )


def main():
    """Main analysis runner."""
    logger.info("Starting 4-model performance analysis")

    lineups_dict = load_all_model_lineups()
    if not lineups_dict:
        logger.error("No lineup data loaded. Exiting.")
        return

    daily_totals_df = calculate_daily_lineup_totals(lineups_dict)
    if daily_totals_df.empty:
        logger.error("No daily totals calculated. Need lineups with ACTUAL_FP data.")
        return

    print_model_summary_stats(daily_totals_df)
    print_top_lineups(daily_totals_df, lineups_dict, top_n=2)

    print("\n" + "=" * 80)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
