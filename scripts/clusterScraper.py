from seleniumwire import webdriver
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
import time
from datetime import datetime
from unidecode import unidecode
import logging
import sys
import os
from aws.s3_utils import save_dataframe_to_s3, save_json_to_s3, load_dataframe_from_s3


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

numeric_columns = {
    "AGE": "AGE",
    "GP": "GP",
    "W": "W",
    "L": "L",
    "MIN": "MIN",
    "OFFRTG": "OFFRTG",
    "DEFRTG": "DEFRTG",
    "NETRTG": "NETRTG",
    "AST%": "AST_PERCENT",
    "AST/TO": "AST_TO",
    "AST RATIO": "AST_RATIO",
    "OREB%": "OREB_PERCENT",
    "DREB%": "DREB_PERCENT",
    "REB%": "REB_PERCENT",
    "TO RATIO": "TO_RATIO",
    "EFG%": "EFG_PERCENT",
    "TS%": "TS_PERCENT",
    "USG%": "USG_PERCENT",
    "PACE": "PACE",
    "PIE": "PIE",
    "POSS": "POSS",
    "%FGA2PT": "FGA2P_PERCENT",
    "%FGA3PT": "FGA3P_PERCENT",
    "%PTS2PT": "PTS2P_PERCENT",
    "%PTS2PT MR": "PTS2P_MR_PERCENT",
    "%PTS3PT": "PTS3P_PERCENT",
    "%PTSFBPS": "PTSFBPS_PERCENT",
    "%PTSFT": "PTSFT_PERCENT",
    "%PTSOFFTO": "PTS_OFFTO_PERCENT",
    "%PTSPITP": "PTSPITP_PERCENT",
    "2FGM%AST": "FG2M_AST_PERCENT",
    "2FGM%UAST": "FG2M_UAST_PERCENT",
    "3FGM%AST": "FG3M_AST_PERCENT",
    "3FGM%UAST": "FG3M_UAST_PERCENT",
    "FGM%AST": "FGM_AST_PERCENT",
    "FGM%UAST": "FGM_UAST_PERCENT",
    "DEF RTG": "DEF_RTG",
    "DREB": "DREB",
    "DREB%TEAM": "DREB_PERCENT_TEAM",
    "STL": "STL",
    "STL%": "STL_PERCENT",
    "BLK": "BLK",
    "%BLK": "BLK_PERCENT",
    "OPP PTSOFF TOV": "OPP_PTS_OFFTO",
    "OPP PTS2ND CHANCE": "OPP_PTS_2ND_CHANCE",
    "OPP PTSFB": "OPP_PTS_FB",
    "OPP PTSPAINT": "OPP_PTS_PAINT",
    "DEFWS": "DEFWS",
}

string_columns = {
    'TEAM': 'TEAM',
}

def safe_float(value):
    try:
        return float(value)
    except ValueError:
        return None

def normalize_name(name):
    return unidecode(name.strip().lower())

def scrape_stats_from_url(url, stat_type):
    """Scrape stats from a single URL and return DataFrame"""
    logger.info(f"Starting scrape for {stat_type} from: {url}")
    
    proxy_url = "http://smart-b0ibmkjy90uq_area-US_state-Northcarolina_life-15_session-0Ve35bhsUr:sU8CQmV8LDmh2mXj@proxy.smartproxy.net:3120"
    
    seleniumwire_options = {
        'proxy': {
            'http': proxy_url,
            'https': proxy_url,
            'no_proxy': 'localhost,127.0.0.1'
        },
        'connection_timeout': 30,
        'verify_ssl': False
    }
    
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--ignore-certificate-errors-spki-list')
    chrome_options.add_argument('--allow-running-insecure-content')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--force-device-scale-factor=0.75")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = None
    try:
        logger.info("Initializing Chrome driver with proxy...")
        driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=seleniumwire_options)
        logger.info("Chrome driver initialized successfully")
        
        driver.get(url)
        logger.info(f"Navigated to {url}")
        
        # Wait for page to load completely
        wait = WebDriverWait(driver, 20)
        logger.info("Waiting for page elements to load...")
        
        # Wait for and select Regular Season
        logger.info("Looking for season selector dropdown...")
        season_dropdown = wait.until(
            EC.element_to_be_clickable((By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[1]/div/div/div[2]/label/div/select"))
        )
        select = Select(season_dropdown)
        
        # Log available options before selecting
        options = [option.text for option in select.options]
        logger.info(f"Season dropdown options: {options}")
        
        # Select Regular Season (index 1)
        select.select_by_index(1)
        selected_option = select.first_selected_option.text
        logger.info(f"Selected season: '{selected_option}'")
        
        # Wait for page to update
        time.sleep(3)
        
        # Wait for and select All
        logger.info("Looking for team selector dropdown...")
        team_dropdown = wait.until(
            EC.element_to_be_clickable((By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))
        )
        select = Select(team_dropdown)
        
        # Log available options before selecting
        team_options = [option.text for option in select.options]
        logger.info(f"Team dropdown options: {team_options}")
        
        select.select_by_index(0)
        selected_team = select.first_selected_option.text
        logger.info(f"Selected team filter: '{selected_team}'")
        
        # Wait for table to load
        time.sleep(3)
        logger.info("Waiting for data table to load...")
        
        src = driver.page_source 
        parser = BeautifulSoup(src, 'lxml')
        table = parser.find("div", attrs = {"class": "Crom_container__C45Ti crom-container"})
        
        if not table or isinstance(table, NavigableString) or not isinstance(table, Tag):
            logger.error(f"Could not find valid table on {url}")
            return pd.DataFrame()
        
        headers = table.findAll('th')
        headerlist = [h.text.strip().upper() for h in headers[1:]]
        headerlist = [a for a in headerlist if not 'RANK' in a]
        logger.info(f"Found headers: {headerlist}")
        
        # Special handling for defense stats
        for i, header in enumerate(headerlist):
            if header == "DREB%" and url == "https://www.nba.com/stats/players/defense":
                headerlist[i] = "DREB%TEAM"
                break
        
        rows = table.findAll('tr')[1:]
        player_stats = [[td.getText().strip() for td in rows[i].findAll('td')[1:]] for i in range(len(rows))]
        logger.info(f"Scraped {len(player_stats)} player rows")
        
        if url == r"https://www.nba.com/stats/players/advanced":
            headerlist = headerlist[:-5]
        
        stats_df = pd.DataFrame(player_stats, columns=headerlist)
        
        # Normalize player names
        stats_df['PLAYER'] = stats_df['PLAYER'].apply(normalize_name)
        
        # Add stat type for tracking
        stats_df['STAT_TYPE'] = stat_type
        stats_df['SCRAPED_DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Created DataFrame with shape: {stats_df.shape}")
        return stats_df
        
    except Exception as e:
        logger.error(f"Error scraping {stat_type}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame()
    finally:
        if driver:
            try:
                logger.info("Cleaning up Chrome driver...")
                driver.quit()
                logger.info("Chrome driver closed successfully")
            except Exception as cleanup_error:
                logger.warning(f"Error during driver cleanup: {cleanup_error}")

def merge_stats_dataframes(advanced_df, scoring_df, defense_df):
    """Merge all stats DataFrames into one comprehensive DataFrame"""
    logger.info("Merging stats DataFrames")
    
    # Start with advanced stats as base
    merged_df = advanced_df.copy()
    logger.info(f"Base advanced stats: {len(merged_df)} players")
    
    # Merge scoring stats
    if not scoring_df.empty:
        merged_df = merged_df.merge(scoring_df, on='PLAYER', how='left', suffixes=('', '_scoring'))
        # Drop duplicate columns from scoring merge
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_scoring')]
    
    # Merge defense stats
    if not defense_df.empty:
        merged_df = merged_df.merge(defense_df, on='PLAYER', how='left', suffixes=('', '_defense'))
        # Drop duplicate columns from defense merge
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_defense')]
    
    # Convert numeric columns
    for scraped_col, db_col in numeric_columns.items():
        if scraped_col in merged_df.columns:
            merged_df[db_col] = merged_df[scraped_col].apply(safe_float)
    
    # Handle string columns
    for scraped_col, db_col in string_columns.items():
        if scraped_col in merged_df.columns:
            merged_df[db_col] = merged_df[scraped_col]
    
    logger.info(f"Final merged DataFrame shape: {merged_df.shape}")
    logger.info(f"Final columns: {list(merged_df.columns)}")
    return merged_df

def run_cluster_scraper():
    """Main function to scrape all player stats and save to S3"""
    logger.info("Starting S3 cluster scraper")
    
    try:
        # Load existing data from S3 (if any)
        existing_df = load_dataframe_from_s3('data/advanced_player_stats/current.parquet')
        logger.info(f"Loaded existing data: {len(existing_df)} records")
        
        # Scrape Advanced Stats
        logger.info("=== SCRAPING ADVANCED STATS ===")
        advanced_df = scrape_stats_from_url(
            url="https://www.nba.com/stats/players/advanced",
            stat_type="advanced"
        )
        
        # Scrape Scoring Stats
        logger.info("=== SCRAPING SCORING STATS ===")
        scoring_df = scrape_stats_from_url(
            url="https://www.nba.com/stats/players/scoring",
            stat_type="scoring"
        )
        
        # Scrape Defense Stats
        logger.info("=== SCRAPING DEFENSE STATS ===")
        defense_df = scrape_stats_from_url(
            url="https://www.nba.com/stats/players/defense",
            stat_type="defense"
        )
        
        # Merge all stats
        if not advanced_df.empty:
            merged_df = merge_stats_dataframes(advanced_df, scoring_df, defense_df)
            
            # Save current version only
            save_dataframe_to_s3(merged_df, 'data/advanced_player_stats/current.parquet')
            logger.info(f"Successfully saved {len(merged_df)} player records to S3")
            logger.info(f"Data shape: {merged_df.shape}")
            
            # Check for duplicates
            duplicate_check = merged_df[merged_df.duplicated(subset=['PLAYER'], keep=False)]
            if not duplicate_check.empty:
                logger.warning(f"Found {len(duplicate_check)} duplicate player entries")
                logger.warning(f"Duplicate players: {duplicate_check['PLAYER'].tolist()}")
            
            return {
                'success': True,
                'records_scraped': len(merged_df),
                'columns_count': len(merged_df.columns)
            }
        else:
            logger.error("No data scraped - advanced stats DataFrame is empty")
            return {
                'success': False,
                'error': 'No data scraped'
            }
            
    except Exception as e:
        logger.error(f"Error in cluster scraper: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# Lambda handler function
def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Lambda function started")
    
    try:
        result = run_cluster_scraper()
        
        if result['success']:
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Cluster scraper completed successfully',
                    'records_scraped': result['records_scraped'],
                    'columns_count': result['columns_count']
                }
            }
        else:
            return {
                'statusCode': 500,
                'body': {
                    'message': 'Cluster scraper failed',
                    'error': result['error']
                }
            }
            
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'message': 'Lambda execution failed',
                'error': str(e)
            }
        }

# For local testing
if __name__ == "__main__":
    result = run_cluster_scraper()
    print(f"Scraper result: {result}")