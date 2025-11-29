"""
Wipe injury context and model comparison data from S3

This script deletes:
- Injury context files (complex and direct models)
- All 5 model minutes projections
- All 5 model daily lineups

Run this after fixing bugs to ensure clean data rebuild.
"""

import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = 'nba-prediction-ibracken'
s3_client = boto3.client('s3')

# Files to delete
FILES_TO_DELETE = [
    # Injury context (2 files)
    'injury_context/complex_position_overlap.parquet',
    'injury_context/direct_position_only.parquet',

    # Model 1: Complex Position Overlap
    'model_comparison/complex_position_overlap/minutes_projections.parquet',
    'model_comparison/complex_position_overlap/daily_lineups.parquet',

    # Model 2: Direct Position Only
    'model_comparison/direct_position_only/minutes_projections.parquet',
    'model_comparison/direct_position_only/daily_lineups.parquet',

    # Model 3: Formula C Baseline
    'model_comparison/formula_c_baseline/minutes_projections.parquet',
    'model_comparison/formula_c_baseline/daily_lineups.parquet',

    # Model 4: SportsLine Baseline
    'model_comparison/sportsline_baseline/minutes_projections.parquet',
    'model_comparison/sportsline_baseline/daily_lineups.parquet',

    # Model 5: DailyFantasyFuel Baseline
    'model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet',
]

def delete_s3_files(bucket, files):
    """Delete list of S3 files"""
    deleted_count = 0
    not_found_count = 0
    error_count = 0

    print(f"Deleting {len(files)} files from s3://{bucket}/...\n")

    for file_key in files:
        try:
            s3_client.head_object(Bucket=bucket, Key=file_key)
            s3_client.delete_object(Bucket=bucket, Key=file_key)
            print(f"✓ Deleted: {file_key}")
            deleted_count += 1
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"⊘ Not found (skipping): {file_key}")
                not_found_count += 1
            else:
                print(f"✗ Error deleting {file_key}: {str(e)}")
                error_count += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Deleted: {deleted_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")

    return deleted_count, not_found_count, error_count

if __name__ == "__main__":
    print("="*60)
    print("WIPE MODEL DATA - S3 Cleanup Script")
    print("="*60)
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Files to delete: {len(FILES_TO_DELETE)}")
    print("="*60)

    response = input("\nAre you sure you want to delete these files? (yes/no): ")

    if response.lower() == 'yes':
        deleted, not_found, errors = delete_s3_files(BUCKET_NAME, FILES_TO_DELETE)

        if errors == 0:
            print("\n✓ Cleanup completed successfully!")
            print("\nNext steps:")
            print("1. Deploy updated lambda code")
            print("2. Run lambda to rebuild data from scratch")
            print("3. Verify injury context row counts are reasonable (~40-60 rows)")
        else:
            print(f"\n⚠ Cleanup completed with {errors} errors")
    else:
        print("\nCanceled - no files were deleted")
