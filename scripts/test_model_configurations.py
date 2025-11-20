import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
        raise

def prepare_data(df):
    """
    Prepare data for modeling - same as lambda function
    """
    logger.info("Preparing data for modeling...")

    # Filter out zero minutes
    df = df[df['MIN'] != 0].copy()

    # Sort by game date
    df = df.sort_values(by=['GAME_DATE'], ascending=[False])

    # Add noise to MIN
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    df['MIN'] = df['MIN'] + np.random.uniform(-5, 5, size=len(df))
    df['MIN'] = df['MIN'].clip(lower=0)

    # Handle missing clusters
    df['CLUSTER'] = df['CLUSTER'].fillna('CLUSTER_NAN')

    # Parse MATCHUP
    def parse_matchup(matchup_str):
        if pd.isna(matchup_str):
            return 0, 'UNKNOWN'
        matchup_str = str(matchup_str)
        if ' @ ' in matchup_str:
            teams = matchup_str.split(' @ ')
            return 0, teams[1] if len(teams) > 1 else 'UNKNOWN'
        elif ' vs. ' in matchup_str:
            teams = matchup_str.split(' vs. ')
            return 1, teams[1] if len(teams) > 1 else 'UNKNOWN'
        else:
            return 0, 'UNKNOWN'

    matchup_parsed = df['MATCHUP'].apply(parse_matchup)
    df['IS_HOME'] = matchup_parsed.apply(lambda x: x[0])
    df['OPPONENT'] = matchup_parsed.apply(lambda x: x[1])

    # Handle NaN values in rolling averages
    df['Last3_FP_Avg'] = df['Last3_FP_Avg'].fillna(df['Season_FP_Avg']).fillna(0)
    df['Last7_FP_Avg'] = df['Last7_FP_Avg'].fillna(df['Season_FP_Avg']).fillna(0)
    df['Season_FP_Avg'] = df['Season_FP_Avg'].fillna(0)
    df['Career_FP_Avg'] = df['Career_FP_Avg'].fillna(0)
    df['Games_Played_Career'] = df['Games_Played_Career'].fillna(0)

    # Handle minutes rolling averages
    df['Last7_MIN_Avg'] = df['Last7_MIN_Avg'].fillna(df['Season_MIN_Avg']).fillna(0)
    df['Season_MIN_Avg'] = df['Season_MIN_Avg'].fillna(0)
    df['Career_MIN_Avg'] = df['Career_MIN_Avg'].fillna(0)

    # Replace Inf values
    numeric_cols = ['Career_FP_Avg', 'Games_Played_Career', 'Last7_MIN_Avg',
                    'Season_MIN_Avg', 'Career_MIN_Avg', 'MIN', 'Last3_FP_Avg',
                    'Last7_FP_Avg', 'Season_FP_Avg']
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    # Define features
    feature_names = ['Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
                    'Career_FP_Avg', 'Games_Played_Career', 'CLUSTER', 'MIN',
                    'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg',
                    'IS_HOME', 'OPPONENT']

    df_features = df[feature_names].copy()
    df_labels = df['FP'].copy()

    # One-hot encode categorical features
    df_features = pd.get_dummies(df_features, columns=['CLUSTER', 'OPPONENT'], drop_first=False)

    # Convert to numpy arrays
    features = np.array(df_features)
    labels = np.array(df_labels)

    logger.info(f"Prepared {len(features)} samples with {features.shape[1]} features")

    return features, labels, df_features.columns.tolist()

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, iteration=None):
    """
    Train a model and evaluate it, returning metrics and timing info
    """
    if iteration is not None:
        logger.info(f"  Iteration {iteration}...")
    else:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*80}")

    # Time the training
    start_time = time.time()
    model.fit(X_train, y_train.ravel())
    training_time = time.time() - start_time

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'train_time_seconds': training_time,
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred)
    }

    # Log results only if single iteration
    if iteration is None:
        logger.info(f"Training Time: {training_time:.2f} seconds")
        logger.info(f"Train MAE: {metrics['train_mae']:.3f}")
        logger.info(f"Test MAE:  {metrics['test_mae']:.3f}")
        logger.info(f"Train RMSE: {metrics['train_rmse']:.3f}")
        logger.info(f"Test RMSE:  {metrics['test_rmse']:.3f}")
        logger.info(f"Train R²: {metrics['train_r2']:.3f}")
        logger.info(f"Test R²:  {metrics['test_r2']:.3f}")

    return model, metrics, test_pred

def plot_comparison(results_df, all_predictions, y_test):
    """Create comprehensive comparison visualizations with error bars"""

    # Set up the plot style
    sns.set_style("whitegrid")

    # Figure 1: Metrics Comparison with error bars
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Model Performance Comparison (Mean ± Std)', fontsize=16, fontweight='bold')

    # Test MAE
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results_df))
    ax1.bar(x_pos, results_df['test_mae'], yerr=results_df['test_mae_std'],
            capsize=5, alpha=0.8, color=sns.color_palette('viridis', len(results_df)))
    ax1.set_title('Test MAE (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    for i, (mean, std) in enumerate(zip(results_df['test_mae'], results_df['test_mae_std'])):
        ax1.text(i, mean + std + 0.1, f'{mean:.3f}±{std:.3f}', ha='center', fontweight='bold', fontsize=9)

    # Test RMSE
    ax2 = axes[0, 1]
    ax2.bar(x_pos, results_df['test_rmse'], yerr=results_df['test_rmse_std'],
            capsize=5, alpha=0.8, color=sns.color_palette('viridis', len(results_df)))
    ax2.set_title('Test RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Root Mean Squared Error')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    for i, (mean, std) in enumerate(zip(results_df['test_rmse'], results_df['test_rmse_std'])):
        ax2.text(i, mean + std + 0.1, f'{mean:.3f}±{std:.3f}', ha='center', fontweight='bold', fontsize=9)

    # Test R²
    ax3 = axes[1, 0]
    ax3.bar(x_pos, results_df['test_r2'], yerr=results_df['test_r2_std'],
            capsize=5, alpha=0.8, color=sns.color_palette('viridis', len(results_df)))
    ax3.set_title('Test R² (Higher is Better)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_ylabel('R² Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    for i, (mean, std) in enumerate(zip(results_df['test_r2'], results_df['test_r2_std'])):
        ax3.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', ha='center', fontweight='bold', fontsize=9)

    # Training Time
    ax4 = axes[1, 1]
    ax4.bar(x_pos, results_df['train_time_seconds'], yerr=results_df['train_time_std'],
            capsize=5, alpha=0.8, color=sns.color_palette('rocket', len(results_df)))
    ax4.set_title('Training Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_ylabel('Seconds')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    for i, (mean, std) in enumerate(zip(results_df['train_time_seconds'], results_df['train_time_std'])):
        ax4.text(i, mean + std + 0.5, f'{mean:.1f}±{std:.1f}s', ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("Saved model_performance_comparison.png")
    plt.close()

    # Figure 2: Prediction Scatter Plots
    n_models = len(all_predictions)
    fig2, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig2.suptitle('Actual vs Predicted Fantasy Points', fontsize=16, fontweight='bold')

    for idx, (model_name, predictions) in enumerate(all_predictions.items()):
        ax = axes[idx]

        # Scatter plot
        ax.scatter(y_test, predictions, alpha=0.3, s=10)

        # Perfect prediction line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Calculate metrics for annotation
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Add metrics box
        textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Actual FP', fontsize=12)
        ax.set_ylabel('Predicted FP', fontsize=12)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig('model_predictions_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("Saved model_predictions_comparison.png")
    plt.close()

    # Figure 3: Error Distribution
    fig3, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig3.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')

    for idx, (model_name, predictions) in enumerate(all_predictions.items()):
        ax = axes[idx]

        errors = predictions - y_test

        # Histogram
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')

        # Add mean and median lines
        mean_error = errors.mean()
        median_error = np.median(errors)
        ax.axvline(x=mean_error, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}')
        ax.axvline(x=median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}')

        ax.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig('model_error_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Saved model_error_distribution.png")
    plt.close()

def main():
    """Main function to test different model configurations"""
    logger.info("="*80)
    logger.info("MODEL CONFIGURATION TESTING")
    logger.info("Testing: Default RF vs Optimized RF vs GradientBoosting")
    logger.info("="*80)

    # Number of iterations for each model
    N_ITERATIONS = 5

    try:
        # Load data
        logger.info("\n1. Loading box score data from S3...")
        df1 = load_dataframe_from_s3('data/box_scores/current.parquet')
        df2 = load_dataframe_from_s3('data/box_scores/2024-25.parquet')
        df3 = load_dataframe_from_s3('data/box_scores/2023-24.parquet')
        df4 = load_dataframe_from_s3('data/box_scores/2022-23.parquet')
        df = pd.concat([df1, df2, df3, df4])
        logger.info(f"Loaded {len(df)} records from 4 seasons")

        # Prepare data
        logger.info("\n2. Preparing data...")
        X, y, feature_names = prepare_data(df)

        # Define models to test
        model_configs = {
            '1. Default RF (Current)': lambda: RandomForestRegressor(
                random_state=None  # Different random state each iteration
            ),
            '2. Optimized RF': lambda: RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=None,
                n_jobs=-1
            ),
            '3. GradientBoosting': lambda: GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=None,
                verbose=0
            )
        }

        # Store all iteration results
        logger.info(f"\n3. Training and evaluating models ({N_ITERATIONS} iterations each)...")
        all_iteration_results = {model_name: [] for model_name in model_configs.keys()}
        final_models = {}
        final_predictions = {}

        for model_name, model_factory in model_configs.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*80}")

            iteration_metrics = []

            for i in range(N_ITERATIONS):
                # Use different random state for train/test split each iteration
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42 + i
                )

                # Create new model instance
                model = model_factory()

                # Train and evaluate
                trained_model, metrics, predictions = train_and_evaluate(
                    model, model_name, X_train, X_test, y_train, y_test, iteration=i+1
                )

                iteration_metrics.append(metrics)

                # Store last iteration's model and predictions for visualization
                if i == N_ITERATIONS - 1:
                    final_models[model_name] = trained_model
                    final_predictions[model_name] = predictions
                    final_y_test = y_test

            all_iteration_results[model_name] = iteration_metrics

            # Calculate and log statistics for this model
            metrics_df = pd.DataFrame(iteration_metrics)
            logger.info(f"\n{model_name} - Summary across {N_ITERATIONS} iterations:")
            logger.info(f"  Test MAE:  {metrics_df['test_mae'].mean():.3f} ± {metrics_df['test_mae'].std():.3f}")
            logger.info(f"  Test RMSE: {metrics_df['test_rmse'].mean():.3f} ± {metrics_df['test_rmse'].std():.3f}")
            logger.info(f"  Test R²:   {metrics_df['test_r2'].mean():.3f} ± {metrics_df['test_r2'].std():.3f}")
            logger.info(f"  Train Time: {metrics_df['train_time_seconds'].mean():.1f}s ± {metrics_df['train_time_seconds'].std():.1f}s")

        # Create aggregated results dataframe
        results = []
        for model_name, iteration_metrics in all_iteration_results.items():
            metrics_df = pd.DataFrame(iteration_metrics)

            aggregated = {
                'model_name': model_name,
                'test_mae': metrics_df['test_mae'].mean(),
                'test_mae_std': metrics_df['test_mae'].std(),
                'test_rmse': metrics_df['test_rmse'].mean(),
                'test_rmse_std': metrics_df['test_rmse'].std(),
                'test_r2': metrics_df['test_r2'].mean(),
                'test_r2_std': metrics_df['test_r2'].std(),
                'train_mae': metrics_df['train_mae'].mean(),
                'train_mae_std': metrics_df['train_mae'].std(),
                'train_rmse': metrics_df['train_rmse'].mean(),
                'train_rmse_std': metrics_df['train_rmse'].std(),
                'train_r2': metrics_df['train_r2'].mean(),
                'train_r2_std': metrics_df['train_r2'].std(),
                'train_time_seconds': metrics_df['train_time_seconds'].mean(),
                'train_time_std': metrics_df['train_time_seconds'].std()
            }
            results.append(aggregated)

        results_df = pd.DataFrame(results)

        # Display comparison table
        logger.info("\n" + "="*80)
        logger.info(f"RESULTS SUMMARY (Average of {N_ITERATIONS} iterations)")
        logger.info("="*80)

        # Create a nice display table
        display_df = results_df[['model_name', 'test_mae', 'test_mae_std',
                                  'test_rmse', 'test_rmse_std', 'test_r2', 'test_r2_std',
                                  'train_time_seconds', 'train_time_std']].copy()
        logger.info("\n" + display_df.to_string(index=False))

        # Calculate improvements
        logger.info("\n" + "="*80)
        logger.info("IMPROVEMENTS OVER DEFAULT RF")
        logger.info("="*80)

        baseline_mae = results_df[results_df['model_name'] == '1. Default RF (Current)']['test_mae'].values[0]
        baseline_rmse = results_df[results_df['model_name'] == '1. Default RF (Current)']['test_rmse'].values[0]
        baseline_r2 = results_df[results_df['model_name'] == '1. Default RF (Current)']['test_r2'].values[0]

        for idx, row in results_df.iterrows():
            if row['model_name'] == '1. Default RF (Current)':
                continue

            mae_improvement = ((baseline_mae - row['test_mae']) / baseline_mae) * 100
            rmse_improvement = ((baseline_rmse - row['test_rmse']) / baseline_rmse) * 100
            r2_improvement = ((row['test_r2'] - baseline_r2) / abs(baseline_r2)) * 100

            logger.info(f"\n{row['model_name']}:")
            logger.info(f"  MAE:  {mae_improvement:+.2f}% {'✓' if mae_improvement > 0 else '✗'}")
            logger.info(f"  RMSE: {rmse_improvement:+.2f}% {'✓' if rmse_improvement > 0 else '✗'}")
            logger.info(f"  R²:   {r2_improvement:+.2f}% {'✓' if r2_improvement > 0 else '✗'}")

        # Feature importance for best model (from last iteration)
        best_model_name = results_df.loc[results_df['test_mae'].idxmin(), 'model_name']
        best_model = final_models[best_model_name]

        logger.info("\n" + "="*80)
        logger.info(f"TOP 15 FEATURES - {best_model_name}")
        logger.info("="*80)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in importance_df.head(15).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.6f}")

        # Generate visualizations
        logger.info("\n4. Generating visualizations...")
        plot_comparison(results_df, final_predictions, final_y_test)

        # Recommendation
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATION")
        logger.info("="*80)

        best_idx = results_df['test_mae'].idxmin()
        best_name = results_df.loc[best_idx, 'model_name']
        best_mae = results_df.loc[best_idx, 'test_mae']
        best_mae_std = results_df.loc[best_idx, 'test_mae_std']
        best_time = results_df.loc[best_idx, 'train_time_seconds']
        best_time_std = results_df.loc[best_idx, 'train_time_std']

        logger.info(f"\nBest Model: {best_name}")
        logger.info(f"  Test MAE: {best_mae:.3f} ± {best_mae_std:.3f} (across {N_ITERATIONS} iterations)")
        logger.info(f"  Training Time: {best_time:.1f}s ± {best_time_std:.1f}s")
        logger.info("\nUpdate your lambda function with the best performing model configuration!")
        logger.info("="*80)

        return {
            'success': True,
            'results': results_df.to_dict('records'),
            'best_model': best_name
        }

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    result = main()
    if result['success']:
        print("\n✓ Model comparison completed successfully!")
        print("Check the generated PNG files for visualizations.")
    else:
        print(f"\n✗ Analysis failed: {result['error']}")
