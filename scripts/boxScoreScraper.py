import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import requests
import logging
from datetime import datetime
from aws.s3_utils import save_dataframe_to_s3, load_dataframe_from_s3
from unidecode import unidecode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def safe_float(value):
    """Convert value to float safely"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def fetch_box_scores_from_api():
    """Fetch box scores from NBA API"""
    logger.info("Starting NBA API box score fetch")
    
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
        'Season': get_current_nba_season(),
        # 'Season' : '2022-23',
        'SeasonType': 'Regular Season',
        'Sorter': 'DATE'
    }
    
    # Use proxy for NBA API requests (NBA blocks some IPs)
    proxy_url = "http://smart-b0ibmkjy90uq_area-US_state-Northcarolina_life-15_session-0Ve35bhsUr:sU8CQmV8LDmh2mXj@proxy.smartproxy.net:3120"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    
    try:
        logger.info("Making request to NBA API...")
        response = requests.get(url, headers=headers, params=params, proxies=proxies, timeout=60, verify=False)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                result_set = data['resultSets'][0]
                headers_list = result_set['headers']
                rows = result_set['rowSet']
                
                logger.info(f"Successfully fetched {len(rows)} box score records")
                
                # Create DataFrame
                df = pd.DataFrame(rows, columns=headers_list)
                return df
            else:
                logger.error("No resultSets found in NBA API response")
                return None
        else:
            logger.error(f"NBA API request failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching from NBA API: {str(e)}")
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
        
        # Create cluster mapping dictionary
        cluster_df['PLAYER'] = cluster_df['PLAYER'].apply(normalize_name)
        cluster_dict = cluster_df.set_index('PLAYER')['CLUSTER'].to_dict()
        
        # Add cluster information
        df['CLUSTER'] = df['PLAYER'].map(cluster_dict)
        
        # Log cluster matching stats
        matched_players = df['CLUSTER'].notna().sum()
        total_players = len(df['PLAYER'].unique())
        logger.info(f"Matched clusters for {matched_players}/{len(df)} records from {total_players} unique players")
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding cluster information: {str(e)}")
        # Continue without clusters if cluster data unavailable
        df['CLUSTER'] = None
        return df

def calculate_rolling_averages(df):
    """Calculate rolling averages for fantasy points"""
    logger.info("Calculating rolling averages")
    
    try:
        # Calculate rolling averages for different windows
        windows = [3, 5, 7]
        for window in windows:
            df[f'Last{window}_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Calculate season average FP
        df['Season_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(
            lambda x: x.expanding(min_periods=1).mean()
        )
        
        logger.info("Successfully calculated rolling averages")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating rolling averages: {str(e)}")
        return df

def scrapeBoxScores():
    """Main function to scrape box scores using NBA API (replaces web scraping)"""
    logger.info("Starting API-based box score scraper")
    
    try:
        # Fetch data from NBA API
        df = fetch_box_scores_from_api()
        if df is None:
            logger.error("Failed to fetch data from NBA API")
            return
        
        # Process the data
        df = process_box_scores(df)
        if df is None:
            logger.error("Failed to process box score data")
            return
        
        # Add cluster information
        df = enrich_with_clusters(df)
        
        # Calculate rolling averages
        df = calculate_rolling_averages(df)
        
        # Save to S3
        save_dataframe_to_s3(df, 'data/box_scores/current.parquet')
        logger.info(f"Saved columns: {list(df.columns)}")
        logger.info(f"Column count: {len(df.columns)}")
        
        # Calculate summary statistics
        total_records = len(df)
        unique_players = len(df['PLAYER'].unique())
        date_range = f"{df['GAME_DATE'].min().strftime('%Y-%m-%d')} to {df['GAME_DATE'].max().strftime('%Y-%m-%d')}"
        
        logger.info("Box score scraping completed successfully")
        logger.info(f"Total records: {total_records}")
        logger.info(f"Unique players: {unique_players}")
        logger.info(f"Date range: {date_range}")
        
        return {
            'success': True,
            'total_records': total_records,
            'unique_players': unique_players,
            'date_range': date_range
        }
        
    except Exception as e:
        logger.error(f"Error in box score scraper: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def run_scrape_box_scores():
    """
    Scrapes box scores using NBA API and saves to S3.
    """
    logger.info("Starting box score scraping process")
    result = scrapeBoxScores()
    
    if result and result.get('success'):
        logger.info("Box score scraping completed successfully")
    else:
        logger.error("Box score scraping failed")
        if result and 'error' in result:
            logger.error(f"Error: {result['error']}")

if __name__ == "__main__":
    run_scrape_box_scores()