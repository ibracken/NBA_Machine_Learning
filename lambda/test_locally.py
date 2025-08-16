#!/usr/bin/env python3
"""
Local testing script for Lambda functions
Run this to test function logic without deploying to AWS
"""

import sys
import os
from pathlib import Path

def test_cluster_scraper():
    """Test cluster scraper locally"""
    print("=== Testing Cluster Scraper ===")
    sys.path.append(str(Path("cluster-scraper")))
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("lambda_function", "cluster-scraper/lambda_function.py")
        lambda_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lambda_module)
        from lambda_module import run_cluster_scraper
        result = run_cluster_scraper()
        print(f"Result: {result}")
        return result['success'] if result else False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_nba_clustering():
    """Test NBA clustering locally"""
    print("=== Testing NBA Clustering ===")
    sys.path.append(str(Path("nba-clustering")))
    
    try:
        from nba_clustering.lambda_function import run_clustering_analysis
        result = run_clustering_analysis()
        print(f"Result: {result}")
        return result['success'] if result else False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_box_score_scraper():
    """Test box score scraper locally"""
    print("=== Testing Box Score Scraper ===")
    sys.path.append(str(Path("box-score-scraper")))
    
    try:
        from box_score_scraper.lambda_function import scrape_box_scores
        result = scrape_box_scores()
        print(f"Result: {result}")
        return result['success'] if result else False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_supervised_learning():
    """Test supervised learning locally"""
    print("=== Testing Supervised Learning ===")
    sys.path.append(str(Path("supervised-learning")))
    
    try:
        from supervised_learning.lambda_function import run_supervised_learning
        result = run_supervised_learning()
        print(f"Result: {result}")
        return result['success'] if result else False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_daily_predictions():
    """Test daily predictions locally"""
    print("=== Testing Daily Predictions ===")
    sys.path.append(str(Path("daily-predictions")))
    
    try:
        from daily_predictions.lambda_function import run_daily_predictions_scraper
        result = run_daily_predictions_scraper()
        print(f"Result: {result}")
        return result['success'] if result else False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_pipeline():
    """Test the complete pipeline in order"""
    print("🚀 Starting NBA Prediction Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("Cluster Scraper", test_cluster_scraper),
        ("NBA Clustering", test_nba_clustering), 
        ("Box Score Scraper", test_box_score_scraper),
        ("Supervised Learning", test_supervised_learning),
        ("Daily Predictions", test_daily_predictions)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n⏳ Running {name}...")
        try:
            success = test_func()
            results[name] = "✅ PASS" if success else "❌ FAIL"
            print(f"{results[name]}")
        except Exception as e:
            results[name] = f"❌ ERROR: {e}"
            print(f"{results[name]}")
    
    print("\n" + "=" * 50)
    print("📊 PIPELINE TEST RESULTS:")
    for name, result in results.items():
        print(f"{name}: {result}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NBA Lambda functions locally')
    parser.add_argument('--function', choices=['cluster', 'clustering', 'boxscore', 'supervised', 'predictions', 'pipeline'], 
                       default='pipeline', help='Which function to test')
    
    args = parser.parse_args()
    
    if args.function == 'cluster':
        test_cluster_scraper()
    elif args.function == 'clustering':
        test_nba_clustering()
    elif args.function == 'boxscore':
        test_box_score_scraper()
    elif args.function == 'supervised':
        test_supervised_learning()
    elif args.function == 'predictions':
        test_daily_predictions()
    else:
        test_pipeline()