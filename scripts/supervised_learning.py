"""
Supervised learning training pipeline.

Purpose:
    Train three GradientBoosting models for fantasy point prediction using
    S3-hosted box scores and engineered features.

Notes:
    - Includes data validation, feature engineering, and VIF diagnostics.
    - Saves models and feature lists back to S3 for inference-time alignment.
"""

import boto3
import json
import logging
import pickle
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class ModelTrainer:
    """Trains the three model variants and returns metrics."""

    def __init__(self, save_model_fn, save_feature_names_fn) -> None:
        self.save_model = save_model_fn
        self.save_feature_names = save_feature_names_fn

    def train(self, df: pd.DataFrame, feature_sets: dict) -> dict:
        models_results = {}

        for model_name, feature_names in feature_sets.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {model_name} model")
            logger.info(f"{'=' * 60}")

            missing_features = [col for col in feature_names if col not in df.columns]
            if missing_features:
                logger.error(f"{model_name}: Missing features: {missing_features}")
                continue

            features = df[feature_names].copy()
            labels = df[["FP"]].copy()

            categorical_cols = ["CLUSTER"] if "CLUSTER" in features.columns else []
            features_encoded = pd.get_dummies(features, columns=categorical_cols) if categorical_cols else features
            feature_names_list = list(features_encoded.columns)

            # Save feature names to S3 for inference-time alignment
            self.save_feature_names(feature_names_list, f"models/{model_name}_feature_names.json")

            # Check for multicollinearity (VIF)
            logger.info(f"{model_name}: Checking multicollinearity...")
            try:
                vif_data = pd.DataFrame()
                vif_data["feature"] = feature_names_list
                vif_data["VIF"] = [
                    variance_inflation_factor(features_encoded.values, i)
                    for i in range(len(feature_names_list))
                ]

                high_vif = vif_data[vif_data["VIF"] > 10]
                if not high_vif.empty:
                    logger.warning(f"{model_name}: High VIF features:\n{high_vif}")
                else:
                    logger.info(f"{model_name}: No high multicollinearity detected")
            except Exception as exc:
                logger.warning(f"{model_name}: Could not calculate VIF: {exc}")

            train, test, train_labels, test_labels = train_test_split(
                features_encoded, labels, test_size=0.20, random_state=42
            )
            logger.info(f"{model_name}: Training set: {len(train)}, Test set: {len(test)}")

            # Train model
            # More estimators help capture weaker patterns; 0.1 learning_rate keeps boosting stable.
            # max_depth=5 allows interaction depth; can overfit, but useful when feature impact is uneven.
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1,
            )
            logger.info(f"{model_name}: Training GradientBoostingRegressor...")
            model.fit(train, train_labels.values.ravel())

            # Save model to S3
            self.save_model(model, f"models/{model_name}.pkl")

            # Generate predictions + metrics
            predictions = model.predict(test)
            r2 = r2_score(test_labels, predictions)
            logger.info(f"{model_name}: R2 Score = {r2:.4f}")

            models_results[model_name] = {
                "model": model,
                "r2_score": r2,
                "feature_count": len(feature_names_list),
                "train_size": len(train),
                "test_size": len(test),
            }

        return models_results


class SupervisedLearningPipeline:
    """End-to-end training pipeline for the supervised FP models."""

    def __init__(self, bucket_name: str = "nba-prediction-ibracken") -> None:
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name

    # -------------------- S3 helpers --------------------
    def load_dataframe_from_s3(self, key: str) -> pd.DataFrame:
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return pd.read_parquet(BytesIO(obj["Body"].read()))
        except Exception as exc:
            logger.error(f"Error loading data from {key}: {exc}")
            raise

    def save_model_to_s3(self, model: GradientBoostingRegressor, key: str) -> None:
        model_buffer = BytesIO()
        pickle.dump(model, model_buffer)
        model_buffer.seek(0)
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=model_buffer.getvalue(),
        )
        logger.info(f"Saved model to s3://{self.bucket_name}/{key}")

    def save_feature_names(self, feature_names: list[str], key: str) -> None:
        payload = json.dumps({"features": feature_names})
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=payload,
        )
        logger.info(f"Saved {len(feature_names)} feature names")

    # ------------------------ Core feature logic ------------------------
    def calculate_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rest days for each player based on their previous game."""
        logger.info("Calculating rest days for players")
        df_sorted = df.copy()
        df_sorted["GAME_DATE"] = pd.to_datetime(df_sorted["GAME_DATE"])

        # Use game chronology per player to derive time since prior game.
        df_sorted = df_sorted.sort_values(["PLAYER", "GAME_DATE"])
        df_sorted["PREV_GAME_DATE"] = df_sorted.groupby("PLAYER")["GAME_DATE"].shift(1)
        df_sorted["REST_DAYS"] = (df_sorted["GAME_DATE"] - df_sorted["PREV_GAME_DATE"]).dt.days

        # First game per player gets a reasonable default.
        df_sorted["REST_DAYS"] = df_sorted["REST_DAYS"].fillna(3).astype(int)

        et_tz = pytz.timezone("America/New_York")
        today = datetime.now(et_tz).date()

        # For the most recent game row, override rest days relative to today.
        most_recent_games = df_sorted.groupby("PLAYER")["GAME_DATE"].max().reset_index()
        most_recent_games["MOST_RECENT_DATE"] = most_recent_games["GAME_DATE"]

        df_sorted = df_sorted.merge(
            most_recent_games[["PLAYER", "MOST_RECENT_DATE"]],
            on="PLAYER",
            how="left",
        )

        current_rest_days = (pd.Timestamp(today) - df_sorted["MOST_RECENT_DATE"]).dt.days
        is_most_recent = df_sorted["GAME_DATE"] == df_sorted["MOST_RECENT_DATE"]
        df_sorted.loc[is_most_recent, "REST_DAYS"] = current_rest_days[is_most_recent]

        df_sorted = df_sorted.drop(columns=["PREV_GAME_DATE", "MOST_RECENT_DATE"])
        df_sorted["REST_DAYS"] = df_sorted["REST_DAYS"].clip(0, 30)

        logger.info(
            f"Rest days calculated: min={df_sorted['REST_DAYS'].min()}, "
            f"max={df_sorted['REST_DAYS'].max()}, mean={df_sorted['REST_DAYS'].mean():.2f}"
        )
        return df_sorted

    # ---------------------------- Training ----------------------------
    def run(self) -> dict:
        """Main training flow (same behavior as Lambda)."""
        logger.info("Starting supervised learning model training")

        try:
            # Load box scores data from multiple seasons
            logger.info("Loading box scores data from S3 (multiple seasons)")
            df = self.load_dataframe_from_s3("data/box_scores/current.parquet")
            df2 = self.load_dataframe_from_s3("data/box_scores/2024-25.parquet")
            df3 = self.load_dataframe_from_s3("data/box_scores/2023-24.parquet")
            df4 = self.load_dataframe_from_s3("data/box_scores/2022-23.parquet")
            df = pd.concat([df, df2, df3, df4])
            logger.info(f"Loaded {len(df)} box score records from multiple seasons")

            # Validate input structure and basic integrity
            required_cols = [
                "PLAYER",
                "GAME_DATE",
                "FP",
                "Last3_FP_Avg",
                "Last7_FP_Avg",
                "Season_FP_Avg",
                "Career_FP_Avg",
                "Games_Played_Career",
                "MIN",
                "MATCHUP",
                "Last7_MIN_Avg",
                "Season_MIN_Avg",
                "Career_MIN_Avg",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns in box scores data: {missing_cols}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if df["FP"].isna().all():
                error_msg = "All FP values are null"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if len(df) == 0:
                error_msg = "Box scores data is empty"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            logger.info(f"Box scores data validation passed: {len(df)} records with required columns")

            # Filter out players with zero minutes
            if "MIN" in df.columns:
                original_count = len(df)
                df = df[df["MIN"] != 0]
                logger.info(f"Filtered from {original_count} to {len(df)} records (MIN != 0)")

            # Data preprocessing for MIN (convert to numeric + add minutes noise)
            df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
            df["MIN"] = df["MIN"] + np.random.uniform(-8, 8, size=len(df))
            df["MIN"] = df["MIN"].clip(lower=0)

            # Handle missing clusters with placeholder
            df["CLUSTER"] = df["CLUSTER"].fillna("CLUSTER_NAN")

            # Parse MATCHUP into home/away and opponent
            def parse_matchup(matchup_str):
                if pd.isna(matchup_str):
                    return 0, "UNKNOWN"
                matchup_str = str(matchup_str)
                if " @ " in matchup_str:
                    teams = matchup_str.split(" @ ")
                    return 0, teams[1] if len(teams) > 1 else "UNKNOWN"
                if " vs. " in matchup_str:
                    teams = matchup_str.split(" vs. ")
                    return 1, teams[1] if len(teams) > 1 else "UNKNOWN"
                return 0, "UNKNOWN"

            matchup_parsed = df["MATCHUP"].apply(parse_matchup)
            df["IS_HOME"] = matchup_parsed.apply(lambda x: x[0])
            df["OPPONENT"] = matchup_parsed.apply(lambda x: x[1])
            logger.info(f"Parsed MATCHUP: {(df['IS_HOME'] == 1).sum()} home games, {(df['IS_HOME'] == 0).sum()} away games")

            # Calculate rest days for each player
            df = self.calculate_rest_days(df)

            # Handle NaN values in rolling averages (first games, missing data)
            logger.info("Handling NaN values in rolling averages")
            df["Last3_FP_Avg"] = df["Last3_FP_Avg"].fillna(df["Season_FP_Avg"])
            df["Last7_FP_Avg"] = df["Last7_FP_Avg"].fillna(df["Season_FP_Avg"])

            df["Last3_FP_Avg"] = df["Last3_FP_Avg"].fillna(0)
            df["Last7_FP_Avg"] = df["Last7_FP_Avg"].fillna(0)
            df["Season_FP_Avg"] = df["Season_FP_Avg"].fillna(0)
            df["Career_FP_Avg"] = df["Career_FP_Avg"].fillna(0)
            df["Games_Played_Career"] = df["Games_Played_Career"].fillna(0)

            df["Last7_MIN_Avg"] = df["Last7_MIN_Avg"].fillna(df["Season_MIN_Avg"])
            df["Last7_MIN_Avg"] = df["Last7_MIN_Avg"].fillna(0)
            df["Season_MIN_Avg"] = df["Season_MIN_Avg"].fillna(0)
            df["Career_MIN_Avg"] = df["Career_MIN_Avg"].fillna(0)

            df["Career_FP_Avg"] = df["Career_FP_Avg"].replace([np.inf, -np.inf], 0)
            df["Games_Played_Career"] = df["Games_Played_Career"].replace([np.inf, -np.inf], 0)
            df["Last7_MIN_Avg"] = df["Last7_MIN_Avg"].replace([np.inf, -np.inf], 0)
            df["Season_MIN_Avg"] = df["Season_MIN_Avg"].replace([np.inf, -np.inf], 0)
            df["Career_MIN_Avg"] = df["Career_MIN_Avg"].replace([np.inf, -np.inf], 0)

            df["REST_DAYS"] = df["REST_DAYS"].fillna(3)
            df["REST_DAYS"] = df["REST_DAYS"].replace([np.inf, -np.inf], 3)

            df["MIN"] = df["MIN"].fillna(0)
            df["MIN"] = df["MIN"].replace([np.inf, -np.inf], 0)

            df["Last3_FP_Avg"] = df["Last3_FP_Avg"].replace([np.inf, -np.inf], 0)
            df["Last7_FP_Avg"] = df["Last7_FP_Avg"].replace([np.inf, -np.inf], 0)
            df["Season_FP_Avg"] = df["Season_FP_Avg"].replace([np.inf, -np.inf], 0)

            logger.info(f"After NaN handling - Last3_FP_Avg nulls: {df['Last3_FP_Avg'].isna().sum()}")
            logger.info(f"After NaN handling - Season_FP_Avg nulls: {df['Season_FP_Avg'].isna().sum()}")
            logger.info(f"After NaN handling - Career_FP_Avg nulls: {df['Career_FP_Avg'].isna().sum()}")
            logger.info(f"After NaN handling - Games_Played_Career nulls: {df['Games_Played_Career'].isna().sum()}")
            logger.info(f"After NaN handling - Last7_MIN_Avg nulls: {df['Last7_MIN_Avg'].isna().sum()}")
            logger.info(f"After NaN handling - Season_MIN_Avg nulls: {df['Season_MIN_Avg'].isna().sum()}")
            logger.info(f"After NaN handling - Career_MIN_Avg nulls: {df['Career_MIN_Avg'].isna().sum()}")

            # Calculate FP_PER_MIN
            # First 2 games: Career FP/MIN; after that: Season FP/MIN
            logger.info("Calculating FP_PER_MIN feature")
            df["SEASON"] = df["GAME_DATE"].apply(
                lambda x: f"{x.year}-{str(x.year + 1)[-2:]}" if x.month >= 10 else f"{x.year - 1}-{str(x.year)[-2:]}"
            )
            df = df.sort_values(["PLAYER", "GAME_DATE"], ascending=[True, True])
            df["SEASON_GAME_NUM"] = df.groupby(["PLAYER", "SEASON"]).cumcount() + 1

            df["FP_PER_MIN"] = np.where(
                df["SEASON_GAME_NUM"] <= 2,
                np.where(df["Career_MIN_Avg"] > 0, df["Career_FP_Avg"] / df["Career_MIN_Avg"], 0),
                np.where(df["Season_MIN_Avg"] > 0, df["Season_FP_Avg"] / df["Season_MIN_Avg"], 0),
            )

            df["FP_PER_MIN"] = df["FP_PER_MIN"].replace([np.inf, -np.inf], 0)
            df["FP_PER_MIN"] = df["FP_PER_MIN"].fillna(0)

            logger.info(
                f"Calculated FP_PER_MIN feature - mean: {df['FP_PER_MIN'].mean():.3f}, "
                f"max: {df['FP_PER_MIN'].max():.3f}"
            )

            # Define three feature sets for three models
            feature_sets = {
                "current": [
                    "Last3_FP_Avg",
                    "Last7_FP_Avg",
                    "Season_FP_Avg",
                    "Career_FP_Avg",
                    "Games_Played_Career",
                    "CLUSTER",
                    "MIN",
                    "Last7_MIN_Avg",
                    "Season_MIN_Avg",
                    "Career_MIN_Avg",
                    "IS_HOME",
                    "REST_DAYS",
                ],
                "fp_per_min": [
                    "Last3_FP_Avg",
                    "Last7_FP_Avg",
                    "Season_FP_Avg",
                    "Career_FP_Avg",
                    "Games_Played_Career",
                    "CLUSTER",
                    "MIN",
                    "Last7_MIN_Avg",
                    "Season_MIN_Avg",
                    "Career_MIN_Avg",
                    "IS_HOME",
                    "REST_DAYS",
                    "FP_PER_MIN",
                ],
                "barebones": ["FP_PER_MIN", "MIN", "REST_DAYS", "CLUSTER"],
            }

            # Train each model variant
            trainer = ModelTrainer(self.save_model_to_s3, self.save_feature_names)
            models_results = trainer.train(df, feature_sets)

            logger.info(f"\n{'=' * 60}")
            logger.info("TRAINING SUMMARY")
            logger.info(f"{'=' * 60}")
            for model_name, results in models_results.items():
                logger.info(f"{model_name}: R2={results['r2_score']:.4f}, Features={results['feature_count']}")

            return {
                "success": True,
                "models_trained": list(models_results.keys()),
                "results": {
                    name: {"r2_score": res["r2_score"], "feature_count": res["feature_count"]}
                    for name, res in models_results.items()
                },
            }

        except Exception as exc:
            logger.error(f"Error in supervised learning: {exc}")
            return {"success": False, "error": str(exc)}


def run_training_pipeline() -> dict:
    """Convenience entrypoint for local runs."""
    pipeline = SupervisedLearningPipeline()
    return pipeline.run()

if __name__ == "__main__":
    result = run_training_pipeline()
    print(result)
