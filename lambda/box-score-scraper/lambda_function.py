from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
import time
import boto3
import json
from io import BytesIO
from unidecode import unidecode
import logging

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
        logger.error(f"Error loading data from {key}: {e}")
        raise

def normalize_name(name):
    return unidecode(name.strip().lower())

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def scrape_box_scores():
    """Main function to scrape NBA box scores and save to S3"""
    logger.info("Starting box score scraper")
    
    try:
        # Initialize Chrome WebDriver with Lambda-specific options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--force-device-scale-factor=0.75")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--no-first-run")
        
        # Lambda-specific Chrome binary path
        chrome_options.binary_location = "/opt/chrome/chrome"
        
        driver = webdriver.Chrome(
            service=webdriver.chrome.service.Service("/opt/chromedriver"),
            options=chrome_options
        )
        
        # Optimized timeout settings for Lambda
        driver.set_page_load_timeout(120)  # 2 minutes for page load
        driver.implicitly_wait(20)  # 20 seconds for finding elements
        
        url_advanced = "https://www.nba.com/stats/players/boxscores"
        driver.get(url_advanced)
        logger.info(f"Navigated to {url_advanced}")
        time.sleep(5)

        # Handle cookies popup
        try:
            cookies = driver.find_element(By.XPATH, "/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[1]")
            cookies.click()
            logger.info("Clicked cookies acceptance button")
        except Exception as e:
            logger.warning(f"Could not find or click cookies button: {e}")
        
        time.sleep(2)

        # Select Regular Season with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                select = Select(driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div[3]/section[1]/div/div/div[2]/label/div/select"))
                select.select_by_index(1)
                logger.info("Selected 'Regular Season' from dropdown")
                time.sleep(3)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to select Regular Season dropdown after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(3)
        
        # Select All players with retry logic
        for attempt in range(max_retries):
            try:
                select = Select(driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))
                select.select_by_index(0)
                logger.info("Selected 'All' players from dropdown")
                time.sleep(3)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to select All players dropdown after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(3)

        time.sleep(10)  # Reduced wait time for Lambda efficiency

        # Parse the page content
        src = driver.page_source 
        parser = BeautifulSoup(src, 'lxml')
        table = parser.find("div", attrs={"class": "Crom_container__C45Ti crom-container"})
        
        if not table or isinstance(table, NavigableString) or not isinstance(table, Tag):
            logger.error("Could not find valid table")
            raise Exception("Box score table not found")
            
        # Extract headers and data
        headers = table.findAll('th')
        headerlist = [h.text.strip() for h in headers]
        headerlist1 = [a for a in headerlist if not 'RANK' in a]
        logger.info(f"Found headers: {headerlist1}")

        rows = table.findAll('tr')[1:]
        player_box_scores = [[td.getText().strip() for td in rows[i].findAll('td')] for i in range(len(rows))]
        df = pd.DataFrame(data=player_box_scores, columns=headerlist1)
        logger.info(f"Scraped {len(df)} box score records")

        # Data preprocessing
        logger.info("Processing scraped data")
        df['PLAYER'] = df['PLAYER'].apply(normalize_name)
        df['FP'] = pd.to_numeric(df['FP'], errors='coerce')
        df['GAME DATE'] = pd.to_datetime(df['GAME DATE'], format='%m/%d/%Y')
        df = df.sort_values(by=['PLAYER', 'GAME DATE'], ascending=[True, True])

        # Get cluster data from S3
        logger.info("Loading cluster assignments from S3")
        try:
            cluster_df = load_dataframe_from_s3('data/clustered_players/current.parquet')
            clusterDict = cluster_df.set_index('PLAYER')['CLUSTER'].to_dict()
            df['CLUSTER'] = df['PLAYER'].map(clusterDict)
            
            # Count players with missing cluster assignments
            missing_clusters = df['CLUSTER'].isna().sum()
            if missing_clusters > 0:
                logger.warning(f"{missing_clusters} players missing cluster assignments")
                
        except Exception as e:
            logger.error(f"Failed to load cluster data: {e}")
            # Continue without cluster assignments - set all to None
            df['CLUSTER'] = None

        # Calculate rolling averages
        logger.info("Calculating rolling averages")
        windows = [3, 5, 7]
        for window in windows:
            df[f'Last{window}_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # Calculate Season Average FP
        df['Season_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(
            lambda x: x.expanding(min_periods=1).mean()
        )

        # Save to S3
        save_dataframe_to_s3(df, 'data/box_scores/current.parquet')
        logger.info(f"Successfully saved {len(df)} box score records to S3")
        
        driver.quit()
        
        return {
            'success': True,
            'records_scraped': len(df),
            'unique_players': df['PLAYER'].nunique(),
            'date_range': {
                'earliest': df['GAME DATE'].min().strftime('%Y-%m-%d'),
                'latest': df['GAME DATE'].max().strftime('%Y-%m-%d')
            },
            'players_with_clusters': int((~df['CLUSTER'].isna()).sum()),
            'players_missing_clusters': int(df['CLUSTER'].isna().sum())
        }
        
    except Exception as e:
        logger.error(f"Error in box score scraper: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Box score scraper Lambda function started")
    
    try:
        result = scrape_box_scores()
        
        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Box score scraping completed successfully',
                    'records_scraped': result['records_scraped'],
                    'unique_players': result['unique_players'],
                    'date_range': result['date_range'],
                    'players_with_clusters': result['players_with_clusters'],
                    'players_missing_clusters': result['players_missing_clusters']
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
    result = scrape_box_scores()
    print(f"Box score scraper result: {result}")