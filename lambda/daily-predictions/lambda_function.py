from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import boto3
import json
from io import BytesIO
from unidecode import unidecode
import logging
import pytz
import os
from dotenv import load_dotenv

# Load .env file for local testing (Lambda uses environment variables)
load_dotenv()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add console handler for local testing
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'

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

def load_model_from_s3(key):
    """Load sklearn model from S3"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pickle.load(BytesIO(obj['Body'].read()))
    except Exception as e:
        logger.error(f"Error loading model from {key}: {e}")
        raise

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def normalize_name(name):
    return unidecode(name.strip().lower())

def get_chrome_driver():
    """Create Chrome driver with Lambda-specific configuration"""
    import os
    from tempfile import mkdtemp

    # Check if running in Lambda environment
    is_lambda = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

    if is_lambda:
        chrome_options = Options()
        # Essential Lambda options based on AWS best practices
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--force-device-scale-factor=0.75")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--disable-extensions")

        # Page load strategy - prevent hanging on slow resources
        chrome_options.page_load_strategy = 'eager'  # Don't wait for all resources

        # Lambda-specific arguments - use /tmp for all writable directories
        chrome_options.add_argument(f"--user-data-dir={mkdtemp()}")
        chrome_options.add_argument(f"--data-path={mkdtemp()}")
        chrome_options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        chrome_options.add_argument("--no-first-run")

        chrome_options.binary_location = "/opt/chrome/chrome"
        driver = webdriver.Chrome(
            service=webdriver.chrome.service.Service("/opt/chromedriver"),
            options=chrome_options
        )
        # Set page load and script timeouts
        driver.set_page_load_timeout(30)  # 30 second page load timeout
        driver.set_script_timeout(30)      # 30 second script timeout
        return driver
    else:
        # Local environment - let Selenium automatically manage ChromeDriver (no options, like script)
        return webdriver.Chrome()


def scrape_starting_lineup(driver):
    """
    Scrape starting lineup indicators from DailyFantasyFuel
    Returns a dict mapping player names to starter status ('CONFIRMED' or 'EXPECTED')
    """
    try:
        src = driver.page_source
        parser = BeautifulSoup(src, 'lxml')

        table = parser.find('table', class_="col-pad-lg-left-5 col-pad-lg-right-5 col-pad-md-left-3 col-pad-md-right-3 text-black row-pad-lg-top-2 row-pad-md-top-2 row-pad-sm-top-2 col-12 row-pad-5 row-pad-xs-1")

        starters = {}

        if table:
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')

                for row in rows:
                    cells = row.find_all('td')

                    # td[5] (cells[4] in 0-indexed) contains starting indicator
                    if len(cells) >= 5:
                        starting_indicator = cells[4].get_text(strip=True)

                        # Check for both confirmed (YES) and expected (EXP.) starters
                        if starting_indicator in ["YES", "EXP."]:
                            # Get player name from data-name attribute
                            data_name = row.get('data-name')
                            if data_name:
                                normalized_name = normalize_name(data_name)
                                status = 'CONFIRMED' if starting_indicator == "YES" else 'EXPECTED'
                                starters[normalized_name] = status
                                logger.info(f"Detected {status.lower()} starter: {normalized_name}")

        confirmed_count = sum(1 for s in starters.values() if s == 'CONFIRMED')
        expected_count = sum(1 for s in starters.values() if s == 'EXPECTED')
        logger.info(f"Total starters detected: {len(starters)} ({confirmed_count} confirmed, {expected_count} expected)")
        return starters

    except Exception as e:
        logger.error(f"Failed to scrape starting lineup: {e}")
        return {}

def scrape_projections_data():
    """Scrape fantasy projections from DailyFantasyFuel and merge with minutes"""
    logger.info("Starting DailyFantasyFuel projections scraping")

    driver = get_chrome_driver()

    try:
        url = "https://www.dailyfantasyfuel.com/nba/projections/draftkings"
        driver.get(url)
        logger.info(f"Navigated to DailyFantasyFuel: {url}")

        # Try to click span element if present
        try:
            span_element = driver.find_element(By.XPATH, "/html/body/div[2]/div[1]/div[2]/div/div/div[3]/div/div/span")
            ActionChains(driver).move_to_element(span_element).click().perform()
            logger.info("Clicked span element")
        except Exception as e:
            logger.warning(f"Could not find or click span element: {e}")

        # Scrape starting lineup before parsing main table
        starters = scrape_starting_lineup(driver)

        # Parse the page
        src = driver.page_source
        parser = BeautifulSoup(src, 'lxml')

        table = parser.find('table', class_="col-pad-lg-left-5 col-pad-lg-right-5 col-pad-md-left-3 col-pad-md-right-3 text-black row-pad-lg-top-2 row-pad-md-top-2 row-pad-sm-top-2 col-12 row-pad-5 row-pad-xs-1")

        player_data = {}

        if table:
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                logger.info(f"Found {len(rows)} rows in DailyFantasyFuel table")

                for row in rows:
                    data_name = row.get('data-name')
                    data_name = normalize_name(data_name)
                    data_ppg_proj = row.get('data-ppg_proj')
                    data_salary = row.get('data-salary')
                    data_pos = row.get('data-pos')

                    if data_name and data_ppg_proj and data_ppg_proj != "0.0":
                        player_data[data_name] = {
                            'ppg_proj': data_ppg_proj,
                            'salary': data_salary,
                            'position': data_pos
                        }

                logger.info(f"Extracted {len(player_data)} players with valid projections from DailyFantasyFuel")
            else:
                logger.warning("No tbody found in table")
        else:
            logger.warning("No table found")

        # Create DataFrame with projections
        df = pd.DataFrame([
            {'Player': player, 'PPG Projection': data['ppg_proj'], 'SALARY': safe_float(data['salary']), 'POSITION': data.get('position')}
            for player, data in player_data.items()
        ])

        # Set MIN to None - will be filled later by minutes-projection lambda
        df['MIN'] = None

        # Add starter status from scraped data
        df['STARTER_STATUS'] = df['Player'].map(starters)

        logger.info(f"Final dataset: {len(df)} players with projections (minutes will be set by minutes-projection lambda)")
        logger.info(f"Starter status: {df['STARTER_STATUS'].value_counts().to_dict()}")
        return df

    except Exception as e:
        logger.error(f"Error scraping projections: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame(columns=['Player', 'PPG Projection', 'MIN', 'SALARY', 'POSITION', 'STARTER_STATUS'])
    finally:
        driver.quit()


def run_daily_predictions_scraper():
    """
    Main function to scrape DFF projections and save to S3
    Note: FP predictions are now handled by minutes-projection lambda
    """
    logger.info("Starting daily predictions scraper")

    try:
        # Scrape DFF projections (SALARY, POSITION, PPG_PROJECTION)
        logger.info("=== SCRAPING DFF PROJECTIONS ===")
        df = scrape_projections_data()

        if df.empty:
            logger.error("No projection data scraped")
            return {
                'success': False,
                'error': 'No projection data scraped'
            }

        # Prepare data for S3
        # Set today's date in ET timezone (Lambda runs in UTC)
        et_tz = pytz.timezone('America/New_York')
        current_date = datetime.datetime.now(et_tz).date()

        # Create records for daily_predictions
        new_records = []
        for _, row in df.iterrows():
            new_records.append({
                "PLAYER": row['Player'],
                "GAME_DATE": current_date,
                "PPG_PROJECTION": safe_float(row['PPG Projection']),
                "PROJECTED_MIN": None,  # Will be filled by minutes-projection
                "SALARY": safe_float(row['SALARY']),
                "POSITION": row['POSITION'],
                "STARTER_STATUS": row.get('STARTER_STATUS'),  # CONFIRMED, EXPECTED, or None
                "ACTUAL_FP": None,
                "ACTUAL_MIN": None
            })

        # Load existing daily predictions from S3
        try:
            df_fp = load_dataframe_from_s3('data/daily_predictions/current.parquet')
            # Ensure GAME_DATE is date type
            df_fp['GAME_DATE'] = pd.to_datetime(df_fp['GAME_DATE']).dt.date
        except:
            logger.info("No existing daily predictions found, creating new DataFrame")
            df_fp = pd.DataFrame(columns=['PLAYER', 'GAME_DATE', 'PPG_PROJECTION', 'PROJECTED_MIN',
                                         'ACTUAL_MIN', 'ACTUAL_FP', 'SALARY', 'POSITION', 'STARTER_STATUS'])

        if df_fp.empty:
            df_fp = pd.DataFrame(columns=['PLAYER', 'GAME_DATE', 'PPG_PROJECTION', 'PROJECTED_MIN',
                                         'ACTUAL_MIN', 'ACTUAL_FP', 'SALARY', 'POSITION', 'STARTER_STATUS'])

        # Update or add new records
        for record in new_records:
            mask = (df_fp['PLAYER'] == record['PLAYER']) & (df_fp['GAME_DATE'] == record['GAME_DATE'])
            if mask.any():
                # Update existing record (keep existing MY_MODEL_PREDICTED_FP and PROJECTED_MIN if present)
                df_fp.loc[mask, 'PPG_PROJECTION'] = record['PPG_PROJECTION']
                df_fp.loc[mask, 'SALARY'] = record['SALARY']
                df_fp.loc[mask, 'POSITION'] = record['POSITION']
                df_fp.loc[mask, 'STARTER_STATUS'] = record['STARTER_STATUS']
            else:
                # Add new record - convert to DataFrame first to avoid FutureWarning
                new_row_df = pd.DataFrame([record])
                df_fp = pd.concat([df_fp, new_row_df], ignore_index=True)

        # Save to S3
        save_dataframe_to_s3(df_fp, 'data/daily_predictions/current.parquet')

        logger.info(f"Daily predictions scraper completed successfully: {len(new_records)} players")

        return {
            'success': True,
            'scraped_players': len(new_records)
        }

    except Exception as e:
        logger.error(f"Error in daily predictions scraper: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Daily predictions Lambda function started")

    try:
        result = run_daily_predictions_scraper()

        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'DFF scraping completed successfully (FP predictions will be done by minutes-projection)',
                    'scraped_players': result.get('scraped_players')
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'DFF scraping failed',
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
    result = run_daily_predictions_scraper()
    print(f"Daily predictions result: {result}")