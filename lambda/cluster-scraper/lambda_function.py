import requests
import pandas as pd
from datetime import datetime
import logging
import boto3
import json
from io import BytesIO
import os
from dotenv import load_dotenv

# Load .env file for local testing (Lambda uses environment variables)
load_dotenv()

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
        logger.info(f"No existing data found at {key}: {e}")
        return pd.DataFrame()

# Column mapping from API response to original clusterScraper format
# Based on your original numeric_columns mapping
api_to_cluster_mapping = {
    # Core identifiers
    'PLAYER_ID': 'PLAYER_ID',  # Keep this new field
    'PLAYER_NAME': 'PLAYER',
    'TEAM_ABBREVIATION': 'TEAM',
    
    # Basic stats (from Advanced)
    'AGE': 'AGE',
    'GP': 'GP', 
    'W': 'W',
    'L': 'L',
    'MIN': 'MIN',
    
    # Advanced stats
    'OFF_RATING': 'OFFRTG',
    'DEF_RATING': 'DEFRTG', 
    'NET_RATING': 'NETRTG',
    'AST_PCT': 'AST_PERCENT',
    'AST_TO': 'AST_TO',
    'AST_RATIO': 'AST_RATIO',
    'OREB_PCT': 'OREB_PERCENT',
    'DREB_PCT': 'DREB_PERCENT',
    'REB_PCT': 'REB_PERCENT',
    'TM_TOV_PCT': 'TO_RATIO',  # Turnover ratio
    'EFG_PCT': 'EFG_PERCENT',
    'TS_PCT': 'TS_PERCENT',
    'USG_PCT': 'USG_PERCENT',
    'PACE': 'PACE',
    'PIE': 'PIE',
    'POSS': 'POSS',
    
    # Scoring stats (will come from Scoring endpoint)
    'PCT_FGA_2PT': 'FGA2P_PERCENT',
    'PCT_FGA_3PT': 'FGA3P_PERCENT', 
    'PCT_PTS_2PT': 'PTS2P_PERCENT',
    'PCT_PTS_2PT_MR': 'PTS2P_MR_PERCENT',
    'PCT_PTS_3PT': 'PTS3P_PERCENT',
    'PCT_PTS_FB': 'PTSFBPS_PERCENT',
    'PCT_PTS_FT': 'PTSFT_PERCENT',
    'PCT_PTS_OFF_TOV': 'PTS_OFFTO_PERCENT',
    'PCT_PTS_PAINT': 'PTSPITP_PERCENT',
    'PCT_AST_2PM': 'FG2M_AST_PERCENT',
    'PCT_UAST_2PM': 'FG2M_UAST_PERCENT',
    'PCT_AST_3PM': 'FG3M_AST_PERCENT', 
    'PCT_UAST_3PM': 'FG3M_UAST_PERCENT',
    'PCT_AST_FGM': 'FGM_AST_PERCENT',
    'PCT_UAST_FGM': 'FGM_UAST_PERCENT',
    
    # Defense stats (will come from Defense endpoint)
    'PCT_DREB': 'DREB_PERCENT_TEAM',  # Team's DREB% (player's DREB% is DREB_PCT above)
    'PCT_STL': 'STL_PERCENT',  # Defense API uses PCT_STL instead of STL_PCT
    'PCT_BLK': 'BLK_PERCENT',  # Defense API uses PCT_BLK instead of BLK_PCT
    'OPP_PTS_OFF_TOV': 'OPP_PTS_OFFTO',
    'OPP_PTS_2ND_CHANCE': 'OPP_PTS_2ND_CHANCE',
    'OPP_PTS_FB': 'OPP_PTS_FB',
    'OPP_PTS_PAINT': 'OPP_PTS_PAINT',
    'DEF_WS': 'DEFWS',
}

def get_current_nba_season():
    """Calculate current NBA season based on date"""
    now = datetime.now()
    year = now.year
    # NBA season runs Oct-June, so if before July, use previous year
    if now.month <= 6:  # Jan-June = current season started previous Oct
        return f"{year-1}-{str(year)[2:]}"
    else:  # July-Dec = new season starting in Oct
        return f"{year}-{str(year+1)[2:]}"

def safe_float(value):
    """Convert value to float safely"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def normalize_name(name):
    """Normalize player name to match original format"""
    try:
        from unidecode import unidecode
        return unidecode(str(name).strip().lower())
    except ImportError:
        return str(name).strip().lower()

def transform_api_data_to_cluster_format(df):
    """Transform API response DataFrame to match format expected by nba-clustering"""
    logger.info("Transforming API data to cluster format")
    
    # Create new DataFrame with mapped columns
    cluster_df = pd.DataFrame()
    
    # Define numeric columns that should be converted to float
    numeric_columns_list = [
        'AGE', 'GP', 'W', 'L', 'MIN', 'OFFRTG', 'DEFRTG', 'NETRTG',
        'AST_PERCENT', 'AST_TO', 'AST_RATIO', 'OREB_PERCENT', 'DREB_PERCENT',
        'REB_PERCENT', 'TO_RATIO', 'EFG_PERCENT', 'TS_PERCENT', 'USG_PERCENT',
        'PACE', 'PIE', 'POSS', 'FGA2P_PERCENT', 'FGA3P_PERCENT', 'PTS2P_PERCENT',
        'PTS2P_MR_PERCENT', 'PTS3P_PERCENT', 'PTSFBPS_PERCENT', 'PTSFT_PERCENT',
        'PTS_OFFTO_PERCENT', 'PTSPITP_PERCENT', 'FG2M_AST_PERCENT', 'FG2M_UAST_PERCENT',
        'FG3M_AST_PERCENT', 'FG3M_UAST_PERCENT', 'FGM_AST_PERCENT', 'FGM_UAST_PERCENT',
        'DREB_PERCENT_TEAM', 'STL_PERCENT', 'BLK_PERCENT',
        'OPP_PTS_OFFTO', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'DEFWS'
    ]
    
    # Map columns that exist in both API and cluster format
    missing_columns = []
    mapped_columns = []
    conversion_failures = {}

    for api_col, cluster_col in api_to_cluster_mapping.items():
        if api_col in df.columns:
            if cluster_col in numeric_columns_list:
                # Apply safe_float to numeric columns and track failures
                original_series = df[api_col].copy()
                cluster_df[cluster_col] = df[api_col].apply(safe_float)

                # Check for conversion failures
                failed_mask = cluster_df[cluster_col].isna() & original_series.notna()
                if failed_mask.any():
                    failed_values = original_series[failed_mask].unique()
                    conversion_failures[cluster_col] = {
                        'count': failed_mask.sum(),
                        'values': list(failed_values)[:5]  # Show first 5 unique failed values
                    }
                    logger.warning(f"Column '{cluster_col}': {failed_mask.sum()} numeric conversion failures. Sample values: {list(failed_values)[:5]}")
            else:
                # Keep as string for PLAYER_ID, PLAYER, TEAM
                cluster_df[cluster_col] = df[api_col]
            mapped_columns.append(f"{api_col}->{cluster_col}")
        else:
            missing_columns.append(f"{api_col}->{cluster_col}")
            logger.warning(f"Missing API column: '{api_col}' (expected for '{cluster_col}')")

    # Log summary
    logger.info(f"Successfully mapped {len(mapped_columns)} columns")
    if missing_columns:
        logger.warning(f"Missing {len(missing_columns)} expected columns: {missing_columns}")
    if conversion_failures:
        total_failures = sum(f['count'] for f in conversion_failures.values())
        logger.warning(f"Total numeric conversion failures: {total_failures} across {len(conversion_failures)} columns")
    
    # Normalize player names to match original format
    if 'PLAYER' in cluster_df.columns:
        cluster_df['PLAYER'] = cluster_df['PLAYER'].apply(normalize_name)
    
    # Add id column to match clusterScraper format
    if len(cluster_df) > 0:
        cluster_df.insert(0, 'id', range(1, len(cluster_df) + 1))
    
    # Add metadata columns to match original
    cluster_df['STAT_TYPE'] = 'advanced'
    cluster_df['SCRAPED_DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cluster_df['SOURCE'] = 'NBA_API'
    
    logger.info(f"Transformed data shape: {cluster_df.shape}")
    # logger.info(f"Transformed columns: {list(cluster_df.columns)}")
    
    return cluster_df

def scrape_single_api_endpoint(session, measure_type, stat_type):
    """Scrape a single NBA API endpoint"""
    logger.info(f"Scraping {stat_type} stats (MeasureType={measure_type})")
    
    base_url = "https://stats.nba.com/stats/leaguedashplayerstats"
    params = {
        'College': '',
        'Conference': '',
        'Country': '',
        'DateFrom': '',
        'DateTo': '',
        'Division': '',
        'DraftPick': '',
        'DraftYear': '',
        'GameScope': '',
        'GameSegment': '',
        'Height': '',
        'ISTRound': '',
        'LastNGames': '0',
        'LeagueID': '00',
        'Location': '',
        'MeasureType': measure_type,
        'Month': '0',
        'OpponentTeamID': '0',
        'Outcome': '',
        'PORound': '0',
        'PaceAdjust': 'N',
        'PerMode': 'PerGame',
        'Period': '0',
        'PlayerExperience': '',
        'PlayerPosition': '',
        'PlusMinus': 'N',
        'Rank': 'N',
        'Season': get_current_nba_season(),
        'SeasonSegment': '',
        'SeasonType': 'Regular Season',
        'ShotClockRange': '',
        'StarterBench': '',
        'TeamID': '0',
        'VsConference': '',
        'VsDivision': '',
        'Weight': ''
    }
    
    try:
        response = session.get(base_url, params=params, timeout=30, verify=False)
        
        if response.status_code == 200:
            logger.info(f"{stat_type} API request successful")
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                result_set = data['resultSets'][0]
                headers_list = result_set['headers']
                rows = result_set['rowSet']
                
                logger.info(f"Retrieved {len(rows)} players with {len(headers_list)} columns for {stat_type}")
                
                # Create DataFrame
                df = pd.DataFrame(rows, columns=headers_list)
                df['STAT_TYPE'] = stat_type
                
                return df
            else:
                logger.error(f"No resultSets found in {stat_type} API response")
                return pd.DataFrame()
        else:
            logger.error(f"{stat_type} API request failed with status {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error scraping {stat_type}: {str(e)}")
        return pd.DataFrame()

def merge_api_dataframes(advanced_df, scoring_df, defense_df):
    """Merge all API DataFrames together"""
    logger.info("Merging API DataFrames")
    
    if advanced_df.empty:
        logger.error("Advanced stats DataFrame is empty - cannot proceed")
        return pd.DataFrame()
    
    # Start with advanced stats as base (has most complete player list)
    merged_df = advanced_df.copy()
    logger.info(f"Base advanced stats: {len(merged_df)} players")
    
    # Merge scoring stats
    if not scoring_df.empty:
        merged_df = merged_df.merge(
            scoring_df, 
            on=['PLAYER_ID', 'PLAYER_NAME'], 
            how='left', 
            suffixes=('', '_scoring')
        )
        # Drop duplicate columns from scoring merge
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_scoring')]
        logger.info(f"After scoring merge: {len(merged_df)} players")
    
    # Merge defense stats  
    if not defense_df.empty:
        merged_df = merged_df.merge(
            defense_df, 
            on=['PLAYER_ID', 'PLAYER_NAME'], 
            how='left', 
            suffixes=('', '_defense')
        )
        # Drop duplicate columns from defense merge
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_defense')]
        logger.info(f"After defense merge: {len(merged_df)} players")
    
    return merged_df

def scrape_nba_api():
    """Scrape NBA stats using all three API endpoints"""
    logger.info("Starting comprehensive NBA API scraper")

    # Get proxy URL from environment variable
    # For Lambda: Set PROXY_URL in Lambda environment variables
    # For local: Set in .env file or environment
    PROXY_URL = os.environ.get('PROXY_URL')
    if not PROXY_URL:
        raise ValueError("PROXY_URL environment variable must be set")

    proxies = {
        'http': PROXY_URL,
        'https': PROXY_URL
    }
    
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
    
    session = requests.Session()
    session.headers.update(headers)
    session.proxies.update(proxies)
    
    try:
        # Scrape all three stat types
        logger.info("=== SCRAPING ADVANCED STATS ===")
        advanced_df = scrape_single_api_endpoint(session, 'Advanced', 'advanced')
        
        logger.info("=== SCRAPING SCORING STATS ===") 
        scoring_df = scrape_single_api_endpoint(session, 'Scoring', 'scoring')
        
        logger.info("=== SCRAPING DEFENSE STATS ===")
        defense_df = scrape_single_api_endpoint(session, 'Defense', 'defense')
        
        # Merge all DataFrames
        if not advanced_df.empty:
            merged_df = merge_api_dataframes(advanced_df, scoring_df, defense_df)
            
            if not merged_df.empty:
                # Transform to cluster format
                cluster_df = transform_api_data_to_cluster_format(merged_df)
                
                # Save to S3
                save_dataframe_to_s3(cluster_df, 'data/advanced_player_stats/current.parquet')
                logger.info(f"Successfully saved {len(cluster_df)} player records to S3")
                
                return {
                    'success': True,
                    'records_scraped': len(cluster_df),
                    'columns_count': len(cluster_df.columns)
                }
            else:
                logger.error("Merged DataFrame is empty")
                return {
                    'success': False,
                    'error': 'Merged data is empty'
                }
        else:
            logger.error("No advanced stats data - cannot proceed")
            return {
                'success': False,
                'error': 'No advanced stats data'
            }
            
    except Exception as e:
        logger.error(f"Error in comprehensive NBA API scraper: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Lambda function started - NBA API scraper")
    
    try:
        result = scrape_nba_api()
        
        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'NBA API scraper completed successfully',
                    'records_scraped': result['records_scraped'],
                    'columns_count': result['columns_count']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'NBA API scraper failed',
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
    
if __name__ == "__main__":
    result = scrape_nba_api()
    print(f"Box score scraper result: {result}")