import pandas as pd
import requests
import json
import boto3
import logging
from datetime import datetime
from io import BytesIO
from unidecode import unidecode

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add console handler for local testing (Lambda provides its own handlers)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'

# S3 utility functions
def save_dataframe_to_s3(df, key):
    """Save DataFrame as Parquet to S3"""
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_buffer.getvalue())
    logger.info(f"Saved {len(df)} records to s3://{BUCKET_NAME}/{key}")

def load_dataframe_from_s3(key):
    """Load DataFrame from S3 Parquet"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except Exception as e:
        logger.error(f"Error loading data from {key}: {e}")
        raise

def get_current_nba_season():
    """Calculate current NBA season based on date"""
    now = datetime.now()
    year = now.year
    # NBA season runs Oct-June, so if before July, use previous year
    if now.month <= 6:  # Jan-June = current season started previous Oct
        return f"{year-1}-{str(year)[2:]}"
    else:  # July-Dec = new season starting in Oct
        return f"{year}-{str(year+1)[2:]}"

def normalize_name(name):
    """Normalize player name for matching"""
    return unidecode(name.strip().lower())

def fetch_box_scores_from_api(season=None):
    """Fetch box scores from NBA API for a specific season"""
    if season is None:
        season = get_current_nba_season()

    logger.info(f"Starting NBA API box score fetch for season {season}")

    url = "https://stats.nba.com/stats/leaguegamelog"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nba.com/stats/players/advanced',
        'Origin': 'https://www.nba.com',
        'Host': 'stats.nba.com',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }

    params = {
        'Counter': '1000',
        'DateFrom': '',
        'DateTo': '',
        'Direction': 'DESC',
        'ISTRound': '',
        'LeagueID': '00',
        'PlayerOrTeam': 'P',
        'Season': season,
        'SeasonType': 'Regular Season',
        'Sorter': 'DATE'
    }

    # Use proxy for NBA API requests (NBA blocks AWS IPs)
    proxy_url = "http://smart-b0ibmkjy90uq_area-US_state-Northcarolina:sU8CQmV8LDmh2mXj@proxy.smartproxy.net:3120"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }

    try:
        logger.info(f"Making request to NBA API for season {season}...")
        response = requests.get(url, headers=headers, params=params, proxies=proxies, timeout=60, verify=False)

        if response.status_code == 200:
            data = response.json()

            if 'resultSets' in data and len(data['resultSets']) > 0:
                result_set = data['resultSets'][0]
                headers_list = result_set['headers']
                rows = result_set['rowSet']

                logger.info(f"Successfully fetched {len(rows)} box score records for season {season}")

                # Create DataFrame
                df = pd.DataFrame(rows, columns=headers_list)
                return df
            else:
                logger.error(f"No resultSets found in NBA API response for season {season}")
                return None
        else:
            logger.error(f"NBA API request failed with status {response.status_code}: {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error fetching from NBA API for season {season}: {str(e)}")
        return None

def process_box_scores(df):
    """Process and enrich box score data"""
    logger.info("Processing box score data")
    
    try:
        # Rename columns first to match existing structure
        df = df.rename(columns={'FANTASY_PTS': 'FP', 'TEAM_NAME': 'PLAYER TEAM'})
        
        # Data preprocessing - normalize player names and drop original
        df['PLAYER'] = df['PLAYER_NAME'].apply(normalize_name)
        df = df.drop(columns=['PLAYER_NAME'])  # Remove duplicate column
        
        # Convert data types
        df['FP'] = pd.to_numeric(df['FP'], errors='coerce')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%Y-%m-%d')
        
        # Sort by player and game date
        df = df.sort_values(by=['PLAYER', 'GAME_DATE'], ascending=[True, True])
        
        logger.info(f"Processed {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error processing box scores: {str(e)}")
        return None

def enrich_with_clusters(df):
    """Add cluster information to box score data"""
    logger.info("Adding cluster information")

    try:
        # Get cluster data from S3
        cluster_df = load_dataframe_from_s3('data/clustered_players/current.parquet')

        # Normalize player names for matching
        cluster_df['PLAYER'] = cluster_df['PLAYER'].apply(normalize_name)

        # Join on (PLAYER, SEASON) to get the correct cluster for each season
        # This prevents the bug where all seasons got the last season's cluster
        df = df.merge(
            cluster_df[['PLAYER', 'SEASON', 'CLUSTER']],
            on=['PLAYER', 'SEASON'],
            how='left'
        )

        # Log cluster matching stats
        matched_records = df['CLUSTER'].notna().sum()
        total_records = len(df)
        unique_player_seasons = df[['PLAYER', 'SEASON']].drop_duplicates()
        matched_player_seasons = df[df['CLUSTER'].notna()][['PLAYER', 'SEASON']].drop_duplicates()

        logger.info(f"Matched clusters for {matched_records}/{total_records} records")
        logger.info(f"Matched {len(matched_player_seasons)}/{len(unique_player_seasons)} unique player-season combinations")

        # Log some examples to verify correct matching
        logger.info("Sample cluster assignments (verification):")
        for season in df['SEASON'].unique()[:2]:
            season_sample = df[df['SEASON'] == season][['PLAYER', 'SEASON', 'CLUSTER']].drop_duplicates().head(3)
            for _, row in season_sample.iterrows():
                cluster_val = row['CLUSTER'] if pd.notna(row['CLUSTER']) else 'NO MATCH'
                logger.info(f"  {row['PLAYER']} ({row['SEASON']}): Cluster {cluster_val}")

        return df

    except Exception as e:
        logger.error(f"Error adding cluster information: {str(e)}")
        # Continue without clusters if cluster data unavailable
        df['CLUSTER'] = None
        return df

def calculate_rolling_averages(df):
    """Calculate rolling averages for fantasy points"""
    logger.info("Calculating rolling averages and career features")

    try:
        # Calculate rolling averages for different windows (RESET EACH SEASON)
        # This prevents contamination from previous season when player changes teams/roles
        windows = [3, 7]
        for window in windows:
            df[f'Last{window}_FP_Avg'] = df.groupby(['PLAYER', 'SEASON'])['FP'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Calculate season average FP (WITHIN CURRENT SEASON ONLY)
        # Previously this was grouping by PLAYER only, causing Season = Career bug
        df['Season_FP_Avg'] = df.groupby(['PLAYER', 'SEASON'])['FP'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # Calculate career average FP across ALL seasons (all historical games)
        # This provides the long-term baseline across different teams/roles
        df['Career_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # Track cumulative games played for each player (across all seasons)
        df['Games_Played_Career'] = df.groupby('PLAYER').cumcount()

        # Calculate minutes rolling averages (same pattern as FP features)
        # Last 7 games minutes average (within season)
        df['Last7_MIN_Avg'] = df.groupby(['PLAYER', 'SEASON'])['MIN'].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )

        # Season average minutes (within current season only)
        df['Season_MIN_Avg'] = df.groupby(['PLAYER', 'SEASON'])['MIN'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # Career average minutes (across all seasons)
        df['Career_MIN_Avg'] = df.groupby('PLAYER')['MIN'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        logger.info("Successfully calculated rolling averages and career features (FP + Minutes)")

        # Log sample values to verify Season != Career
        sample_players = df['PLAYER'].unique()[:3]
        for player in sample_players:
            player_latest = df[df['PLAYER'] == player].tail(1)
            if not player_latest.empty:
                season_fp = player_latest['Season_FP_Avg'].values[0]
                career_fp = player_latest['Career_FP_Avg'].values[0]
                season_min = player_latest['Season_MIN_Avg'].values[0]
                career_min = player_latest['Career_MIN_Avg'].values[0]
                season = player_latest['SEASON'].values[0] if 'SEASON' in player_latest.columns else 'N/A'
                logger.info(f"Verification - {player} ({season}): Season_FP={season_fp:.2f}, Career_FP={career_fp:.2f}, Season_MIN={season_min:.1f}, Career_MIN={career_min:.1f}")

        return df

    except Exception as e:
        logger.error(f"Error calculating rolling averages: {str(e)}")
        return df

def run_box_score_scraper():
    """Main function to run the box score scraping process"""
    logger.info("Starting NBA box score API scraper")

    try:
        # Define seasons to fetch (past 3 years + current season)
        current_season = get_current_nba_season()
        seasons = ['2022-23', '2023-24', '2024-25', current_season]
        # Remove duplicates in case current_season is already in the list
        seasons = list(dict.fromkeys(seasons))

        logger.info(f"Current NBA season: {current_season}")
        logger.info(f"Will fetch seasons: {seasons}")

        season_dataframes = {}

        # Fetch data for each season
        for season in seasons:
            logger.info(f"Fetching data for season {season}")
            df = fetch_box_scores_from_api(season)
            if df is None:
                logger.warning(f"Failed to fetch data for season {season}, skipping")
                continue

            # Process the data
            df = process_box_scores(df)
            if df is None:
                logger.warning(f"Failed to process data for season {season}, skipping")
                continue

            # Store season data with season identifier
            df['SEASON'] = season
            season_dataframes[season] = df
            logger.info(f"Successfully fetched {len(df)} records for season {season}")

        # Check if we have any data
        if not season_dataframes:
            return {
                'success': False,
                'error': 'Failed to fetch data for any season'
            }

        # Combine all seasons for feature calculation
        logger.info("Combining all seasons for career feature calculation")
        combined_df = pd.concat(season_dataframes.values(), ignore_index=True)

        # Sort by player and game date (critical for career features)
        combined_df = combined_df.sort_values(by=['PLAYER', 'GAME_DATE'], ascending=[True, True])
        logger.info(f"Combined dataset has {len(combined_df)} records from {len(combined_df['PLAYER'].unique())} unique players")

        # Add cluster information to combined dataset
        combined_df = enrich_with_clusters(combined_df)

        # Calculate rolling averages and career features on combined dataset
        # This ensures Career_FP_Avg and Games_Played_Career span across all seasons
        combined_df = calculate_rolling_averages(combined_df)

        # Split back into individual seasons and save
        results = {}
        for season in seasons:
            season_df = combined_df[combined_df['SEASON'] == season].copy()

            if len(season_df) > 0:

                # Determine S3 key
                if season == seasons[-1]:  # Latest season is "current"
                    s3_key = 'data/box_scores/current.parquet'
                else:
                    s3_key = f'data/box_scores/{season}.parquet'

                # Save to S3
                save_dataframe_to_s3(season_df, s3_key)

                # Store results
                results[season] = {
                    'records': len(season_df),
                    'unique_players': len(season_df['PLAYER'].unique()),
                    'date_range': f"{season_df['GAME_DATE'].min().strftime('%Y-%m-%d')} to {season_df['GAME_DATE'].max().strftime('%Y-%m-%d')}"
                }

                logger.info(f"Saved {len(season_df)} records for season {season} to {s3_key}")

        # Calculate overall summary statistics
        total_records = len(combined_df)
        unique_players = len(combined_df['PLAYER'].unique())
        date_range = f"{combined_df['GAME_DATE'].min().strftime('%Y-%m-%d')} to {combined_df['GAME_DATE'].max().strftime('%Y-%m-%d')}"

        logger.info("Box score scraping completed successfully")
        logger.info(f"Total records across all seasons: {total_records}")
        logger.info(f"Unique players: {unique_players}")
        logger.info(f"Date range: {date_range}")

        return {
            'success': True,
            'total_records': total_records,
            'unique_players': unique_players,
            'date_range': date_range,
            'season_details': results
        }

    except Exception as e:
        logger.error(f"Error in box score scraper: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Box score scraper Lambda function started")

    try:
        result = run_box_score_scraper()

        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Box score scraping completed successfully',
                    'total_records': result['total_records'],
                    'unique_players': result['unique_players'],
                    'date_range': result['date_range'],
                    'season_details': result.get('season_details', {})
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Box score scraping failed',
                    'error': result['error']
                })
            }

    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Lambda execution failed',
                'error': str(e)
            })
        }

# For local testing
if __name__ == "__main__":
    result = run_box_score_scraper()
    print(f"Box score scraper result: {result}")