# Import all buckets and take their heads hehe
import pandas as pd
from aws.s3_utils import load_dataframe_from_s3, load_model_from_s3
import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_s3_data():
    """Test all S3 data files and show their structure"""
    
    print("=" * 60)
    print("S3 BUCKET DATA TEST")
    print("=" * 60)
    
    # List of all S3 keys to test
    s3_keys = [
        'data/box_scores/current.parquet',
        'data/daily_predictions/current.parquet', 
        'data/advanced_player_stats/current.parquet',
        'models/RFCluster.sav'
    ]
    
    for key in s3_keys:
        print(f"\n{'='*50}")
        print(f"TESTING: {key}")
        print(f"{'='*50}")
        
        try:
            if key.endswith('.sav'):
                # Test model loading
                print(f"Loading model from S3...")
                model = load_model_from_s3(key)
                print(f"✅ Model loaded successfully!")
                print(f"Model type: {type(model)}")
                if hasattr(model, 'feature_importances_'):
                    print(f"Number of features: {len(model.feature_importances_)}")
                
            else:
                # Test DataFrame loading
                print(f"Loading DataFrame from S3...")
                df = load_dataframe_from_s3(key)
                
                if df.empty:
                    print(f"⚠️  DataFrame is empty")
                else:
                    print(f"✅ DataFrame loaded successfully!")
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    print(f"\nFirst 5 rows:")
                    print(df.head())
                    
                    # Show some basic stats
                    print(f"\nBasic info:")
                    print(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                    print(f"- Null values per column:")
                    null_counts = df.isnull().sum()
                    for col, count in null_counts[null_counts > 0].items():
                        print(f"  {col}: {count}")
                        
        except Exception as e:
            print(f"❌ Error loading {key}: {e}")
    
    print(f"\n{'='*60}")
    print("S3 BUCKET TEST COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_s3_data()