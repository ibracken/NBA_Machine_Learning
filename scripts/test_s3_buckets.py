# Import all buckets and take their heads hehe
import pandas as pd
from aws.s3_utils import load_dataframe_from_s3, load_model_from_s3
import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_s3_data():
    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None) # Show all columns
    pd.set_option('display.width', None) # Make sure the width is not restricted
    pd.set_option('display.max_colwidth', None) # Show full content of each column
    """Test all S3 data files and show their structure"""
    
    with open('output.txt', 'w') as f:
        print("=" * 60, file=f)
        print("S3 BUCKET DATA TEST", file=f)
        print("=" * 60, file=f)
        
        # List of all S3 keys to test
        s3_keys = [
            'data/box_scores/current.parquet',
            'data/daily_predictions/current.parquet',
            'data/advanced_player_stats/current.parquet',
            'data/daily_lineups/current.parquet'
        ]
        
        for key in s3_keys:
            print(f"\n{'='*50}", file=f)
            print(f"TESTING: {key}", file=f)
            print(f"{'='*50}", file=f)
            
            try:
                if key.endswith('.sav'):
                    # Test model loading
                    print(f"Loading model from S3...", file=f)
                    model = load_model_from_s3(key)
                    print(f" Model loaded successfully!", file=f)
                    print(f"Model type: {type(model)}", file=f)
                    if hasattr(model, 'feature_importances_'):
                        print(f"Number of features: {len(model.feature_importances_)}", file=f)
                    
                else:
                    # Test DataFrame loading
                    print(f"Loading DataFrame from S3...", file=f)
                    df = load_dataframe_from_s3(key)
                    
                    if df.empty:
                        print(f"⚠️  DataFrame is empty", file=f)
                    else:
                        print(f" DataFrame loaded successfully!", file=f)
                        print(f"Shape: {df.shape}", file=f)
                        print(f"Columns: {list(df.columns)}", file=f)

                        # Show most recent data by GAME_DATE if available
                        if 'GAME_DATE' in df.columns:
                            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                            df_sorted = df.sort_values('GAME_DATE', ascending=False)
                            print(f"\nMost recent 20 rows (sorted by GAME_DATE):", file=f)
                            print(df_sorted.head(50), file=f)
                        else:
                            print(f"\nLast 20 rows:", file=f)
                            print(df.tail(50), file=f)
                        
                        # Show some basic stats
                        print(f"\nBasic info:", file=f)
                        print(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB", file=f)
                        print(f"- Null values per column:", file=f)
                        null_counts = df.isnull().sum()
                        for col, count in null_counts[null_counts > 0].items():
                            print(f"  {col}: {count}", file=f)
                            
            except Exception as e:
                print(f" Error loading {key}: {e}", file=f)
        
        print(f"\n{'='*60}", file=f)
        print("S3 BUCKET TEST COMPLETE", file=f)
        print(f"{'='*60}", file=f)

if __name__ == "__main__":
    test_s3_data()