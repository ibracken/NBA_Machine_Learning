from selenium import webdriver
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
import boto3
import json
from io import BytesIO

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

# Column mappings
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
    "AST RATIO": "AST_RATIO",
    "OREB%": "OREB_PERCENT",
    "DREB%": "DREB_PERCENT",
    "REB%": "REB_PERCENT",
    "TO RATIO": "TO_RATIO",
    "EFG%": "EFG_PERCENT",
    "TS%": "TS_PERCENT",
    "USG%": "USG_PERCENT",
    "PACE": "PACE",
    "PIE": "PIE",
    "POSS": "POSS",
    "%FGA2PT": "FGA2P_PERCENT",
    "%FGA3PT": "FGA3P_PERCENT",
    "%PTS2PT": "PTS2P_PERCENT",
    "%PTS2PT MR": "PTS2P_MR_PERCENT",
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
    "DEF RTG": "DEF_RTG",
    "DREB": "DREB",
    "DREB%TEAM": "DREB_PERCENT_TEAM",
    "STL": "STL",
    "STL%": "STL_PERCENT",
    "BLK": "BLK",
    "%BLK": "BLK_PERCENT",
    "OPP PTSOFF TOV": "OPP_PTS_OFFTO",
    "OPP PTS2ND CHANCE": "OPP_PTS_2ND_CHANCE",
    "OPP PTSFB": "OPP_PTS_FB",
    "OPP PTSPAINT": "OPP_PTS_PAINT",
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

def take_screenshot(driver, filename_prefix, stat_type):
    """Take screenshot and save to S3 for debugging"""
    try:
        screenshot = driver.get_screenshot_as_png()
        s3.put_object(
            Bucket=BUCKET_NAME, 
            Key=f'debug/screenshots/{filename_prefix}_{stat_type}.png', 
            Body=screenshot,
            ContentType='image/png'
        )
        logger.info(f"Screenshot saved: {filename_prefix}_{stat_type}.png")
    except Exception as e:
        logger.error(f"Failed to take screenshot {filename_prefix}: {e}")

def setup_browser_options():
    """Configure Chrome with exact working options from local scraper + proxy"""
    chrome_options = Options()
    
    # Exact options from working local scraper
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # chrome_options.add_argument("--disable-gpu")  # Commented out like in working version
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--allow-running-insecure-content')
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--force-device-scale-factor=0.75")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Lambda-specific requirement
    chrome_options.add_argument("--single-process")
    
    # Decodo residential proxy configuration with auth (LA location)
    proxy_with_auth = "http://spa71pky7e:amCuowkaGq3n07_L8O@us.decodo.com:10001"
    chrome_options.add_argument(f"--proxy-server={proxy_with_auth}")
    
    # Additional proxy authentication flags to handle 407 errors
    chrome_options.add_argument("--proxy-bypass-list=<-loopback>")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    
    return chrome_options

def scrape_stats_from_url(url, stat_type):
    """Scrape stats from a single URL and return DataFrame"""
    logger.info(f"Starting scrape for {stat_type} from: {url}")
    
    chrome_options = setup_browser_options()
    
    # Lambda-specific Chrome binary path
    chrome_options.binary_location = "/opt/chrome/chrome"
    
    driver = webdriver.Chrome(
        service=webdriver.chrome.service.Service("/opt/chromedriver"),
        options=chrome_options
    )
    
    try:
        # Proxy warm-up: Make a simple request to establish authentication
        logger.info("Warming up proxy connection...")
        driver.set_page_load_timeout(15)
        
        try:
            # Simple page to warm up the proxy auth
            driver.get("https://httpbin.org/ip")
            logger.info(f"Proxy warm-up successful. Response length: {len(driver.page_source)}")
            time.sleep(1)  # Give auth time to settle
        except Exception as warmup_error:
            logger.warning(f"Proxy warm-up failed (continuing anyway): {warmup_error}")
        
        logger.info(f"Attempting to navigate to NBA.com: {url}")
        
        # Set longer timeout for actual scraping
        driver.set_page_load_timeout(30)
        
        try:
            driver.get(url)
            logger.info(f"Successfully navigated to {url}")
        except Exception as nav_error:
            logger.error(f"Navigation failed: {nav_error}")
            # Take screenshot of what we have
            take_screenshot(driver, "00_navigation_failed", stat_type)
            raise
            
        logger.info(f"Page title: '{driver.title}'")
        logger.info(f"Current URL after navigation: {driver.current_url}")
        logger.info(f"Page source length: {len(driver.page_source)}")
        
        # Take initial screenshot
        take_screenshot(driver, "01_initial_load", stat_type)
        
        # Wait for basic page load
        wait = WebDriverWait(driver, 20)
        time.sleep(5)
        
        # Look for data table with simplified approach
        logger.info("Waiting for data table...")
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".Crom_container__C45Ti")))
            logger.info("Data table found!")
        except:
            logger.warning("Data table not found, continuing anyway...")
        
        # Take screenshot after waiting
        take_screenshot(driver, "02_after_wait", stat_type)
        
        # Try multiple selectors for Regular Season dropdown
        season_selectors = [
            r"/html/body/div[1]/div[2]/div[2]/div[3]/section[1]/div/div/div[2]/label/div/select",
            "select[name='Season']",
            "select:first-of-type",
            "//select[contains(@class,'DropDown')]"
        ]
        
        season_selected = False
        for selector in season_selectors:
            try:
                if selector.startswith("//") or selector.startswith("/html"):
                    select_element = wait.until(EC.presence_of_element_located((By.XPATH, selector)))
                else:
                    select_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                logger.info(f"Found Regular Season dropdown with selector: {selector}")
                select = Select(select_element)
                select.select_by_index(1)
                logger.info("Selected 'Regular Season' from dropdown")
                season_selected = True
                time.sleep(5)  # Wait for page to update after selection
                
                # Take screenshot after season selection
                take_screenshot(driver, "03_after_season_selection", stat_type)
                # Wait for any loading indicators to disappear
                try:
                    wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".loading, .spinner")))
                except:
                    pass  # No loading indicator found, continue
                break
            except Exception as e:
                logger.warning(f"Selector {selector} failed: {e}")
                continue
        
        if not season_selected:
            logger.error("Could not find Regular Season dropdown with any selector")
            # Save page source for debugging
            s3.put_object(Bucket=BUCKET_NAME, Key=f'debug/page_source_{stat_type}.html', Body=driver.page_source.encode('utf-8'))
            # Take screenshot when dropdown not found
            take_screenshot(driver, "04_dropdown_not_found", stat_type)
        
        # Focus on the original selector that works locally - likely just needs more time
        all_xpath = r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"
        
        all_selected = False
        logger.info("Waiting for 'All' dropdown to appear after first selection...")
        try:
            # Give it more time since page updates after first dropdown
            select_element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, all_xpath))
            )
            logger.info("Found 'All' dropdown")
            select = Select(select_element)
            select.select_by_index(0)
            logger.info("Selected 'All' from dropdown")
            all_selected = True
            time.sleep(2)
            
            # Take screenshot after 'All' selection
            take_screenshot(driver, "05_after_all_selection", stat_type)
        except Exception as e:
            logger.error(f"Could not find 'All' dropdown: {e}")
            # Save page HTML after first dropdown selection for debugging
            s3.put_object(Bucket=BUCKET_NAME, Key=f'debug/page_after_first_dropdown_{stat_type}.html', Body=driver.page_source.encode('utf-8'))
            # Take screenshot when second dropdown not found
            take_screenshot(driver, "06_all_dropdown_not_found", stat_type)
        
        if not all_selected:
            logger.error("Could not find 'All' dropdown with any selector")
        
        src = driver.page_source 
        logger.info(f"Final page source length: {len(src)}")
        
        # Take final screenshot before parsing
        take_screenshot(driver, "07_final_before_parsing", stat_type)
        
        parser = BeautifulSoup(src, 'lxml')
        table = parser.find("div", attrs = {"class": "Crom_container__C45Ti crom-container"})
        
        if not table or isinstance(table, NavigableString) or not isinstance(table, Tag):
            logger.error(f"Could not find valid table on {url}")
            # Take screenshot when table not found
            take_screenshot(driver, "08_table_not_found", stat_type)
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
        return pd.DataFrame()
    finally:
        driver.quit()

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

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Lambda function started")
    
    try:
        result = run_cluster_scraper()
        
        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Cluster scraper completed successfully',
                    'records_scraped': result['records_scraped'],
                    'columns_count': result['columns_count']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Cluster scraper failed',
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