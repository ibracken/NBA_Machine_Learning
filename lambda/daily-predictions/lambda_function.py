from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import pickle
import time
import datetime
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

# Use proxy for NBA API requests (NBA blocks some IPs)
proxy_url = "http://smart-b0ibmkjy90uq_area-US_state-Northcarolina:sU8CQmV8LDmh2mXj@proxy.smartproxy.net:3120"
proxies = {
    'http': proxy_url,
    'https': proxy_url
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

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--allow-running-insecure-content')
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--force-device-scale-factor=0.75")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-extensions")

    # Check if running in Lambda environment
    is_lambda = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

    if is_lambda:
        # Lambda-specific arguments
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-first-run")

        # Configure proxy for Lambda
        chrome_options.add_argument(f"--proxy-server={proxy_url}")

        chrome_options.binary_location = "/opt/chrome/chrome"
        return webdriver.Chrome(
            service=webdriver.chrome.service.Service("/opt/chromedriver"),
            options=chrome_options
        )
    else:
        # Local environment - let Selenium automatically manage ChromeDriver
        return webdriver.Chrome(options=chrome_options)

def scrape_minutes_projection():
    """Scrape projected minutes from FanDuel"""
    logger.info("Starting FanDuel minutes projection scraping")
    
    driver = get_chrome_driver()
    
    try:
        url = "https://www.fanduel.com/research/nba/fantasy/dfs-projections"
        driver.get(url)
        logger.info(f"Navigated to FanDuel: {url}")

        # Wait for the scrollable container
        try:
            scrollable_container = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="virtuoso-scroller"]'))
            )
            logger.info("Found scrollable container")
        except Exception as e:
            logger.error(f"Could not find scrollable container: {e}")
            return {}

        player_minutes = {}
        seen_rows = set()
        scroll_pause_time = 1
        max_scrolls = 50  # Prevent infinite loops in Lambda

        scroll_count = 0
        while scroll_count < max_scrolls:
            # Get currently visible HTML
            html = driver.execute_script("return document.documentElement.outerHTML;")
            parser = BeautifulSoup(html, 'lxml')
            tbody = parser.find('tbody', class_='tableStyles_vtbody__Tj_Pq')

            if not tbody:
                logger.warning("No table body found")
                break

            rows = tbody.find_all('tr')
            new_rows_found = False

            for row in rows:
                # Skip spacer rows
                if 'height' in row.get('style', ''):
                    continue

                row_id = str(row)  # Unique identifier
                if row_id in seen_rows:
                    continue

                seen_rows.add(row_id)
                new_rows_found = True

                td_elements = row.find_all('td', class_="tableStyles_vtd__HAZr4")
                if len(td_elements) >= 5:
                    try:
                        name_td = td_elements[0]
                        player_name = name_td.find('a', class_='link_link__fHWk6').find('div', class_='PlayerCell_nameLinkText__P3INe').text.strip()
                        player_name = normalize_name(player_name)
                        minutes = td_elements[4].text.strip()
                        player_minutes[player_name] = minutes
                        logger.debug(f"Found player: {player_name} - Minutes: {minutes}")
                    except (AttributeError, IndexError) as e:
                        logger.debug(f"Error processing row: {e}")
                        continue

            # Scroll down to load more rows
            driver.execute_script("arguments[0].scrollBy(0, 500);", scrollable_container)
            time.sleep(scroll_pause_time)
            scroll_count += 1

            # Break if no new rows found
            if not new_rows_found:
                logger.info("No new rows found, stopping scroll")
                break

        logger.info(f"Scraped minutes for {len(player_minutes)} players from FanDuel")
        return player_minutes

    except Exception as e:
        logger.error(f"Error scraping FanDuel minutes: {e}")
        return {}
    finally:
        driver.quit()

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

        # Parse the page
        src = driver.page_source 
        parser = BeautifulSoup(src, 'lxml')

        table = parser.find('table', class_="col-pad-lg-left-5 col-pad-lg-right-5 col-pad-md-left-3 col-pad-md-right-3 text-black row-pad-lg-top-2 row-pad-md-top-2 row-pad-sm-top-2 col-12 row-pad-5 row-pad-xs-1")

        player_data = {}
        
        if table:
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')

                for row in rows:
                    data_name = row.get('data-name')
                    data_name = normalize_name(data_name)
                    data_ppg_proj = row.get('data-ppg_proj')

                    if data_name and data_ppg_proj and data_ppg_proj != "0.0":
                        player_data[data_name] = data_ppg_proj
                        
                logger.info(f"Scraped projections for {len(player_data)} players")
            else:
                logger.warning("No tbody found in table")
        else:
            logger.warning("No table found")

        # Create DataFrame with projections
        df = pd.DataFrame(list(player_data.items()), columns=['Player', 'PPG Projection'])
        
        # Get minutes projections
        minutes_dict = scrape_minutes_projection()
        
        # Merge minutes data
        df['MIN'] = df['Player'].map(minutes_dict)
        df = df.dropna(subset=['MIN'])
        
        logger.info(f"Final dataset: {len(df)} players with both projections and minutes")
        return df

    except Exception as e:
        logger.error(f"Error scraping projections: {e}")
        return pd.DataFrame(columns=['Player', 'PPG Projection', 'MIN'])
    finally:
        driver.quit()

def run_model_predictions(df):
    """Run ML model predictions using scraped data"""
    logger.info("Starting model predictions")
    
    try:
        # Load the trained Random Forest model from S3
        logger.info("Loading trained model from S3")
        rf = load_model_from_s3('models/RFCluster.sav')
        
        # Load box scores from S3
        logger.info("Loading box scores data")
        dataset = load_dataframe_from_s3('data/box_scores/current.parquet')
        
        # Validate required columns
        required_cols = {'MIN', 'W/L', 'FP', 'PLAYER', 'GAME_DATE'}
        missing_cols = required_cols - set(dataset.columns)
        if missing_cols:
            logger.error(f"Missing required columns {missing_cols} in box scores data")
            return
        
        # Data preprocessing
        dataset = dataset[dataset['MIN'] != 0]
        dataset = dataset.dropna(subset=['W/L'])
        dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE'])
        
        # Sort by player and game date
        dataset_sorted = dataset.sort_values(by=['PLAYER', 'GAME_DATE'], ascending=[True, True])
        
        # Calculate rolling averages
        logger.info("Calculating rolling averages")
        windows = [3, 5, 7]
        for window in windows:
            dataset_sorted[f'Last{window}_FP_Avg'] = (
                dataset_sorted.groupby('PLAYER')['FP']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
        
        # Calculate season average
        dataset_sorted['Season_FP_Avg'] = (
            dataset_sorted.groupby('PLAYER')['FP']
            .transform(lambda x: x.expanding(min_periods=1).mean())
        )
        
        # One-hot encode clusters
        dataset_sorted = pd.get_dummies(dataset_sorted, columns=['CLUSTER'], drop_first=False)
        all_cluster_columns = [col for col in dataset_sorted.columns if col.startswith('CLUSTER_')]
        
        # Get most recent games for each player
        most_recent_games = dataset_sorted.groupby('PLAYER').tail(1)
        
        # Filter to players in scraped dataset
        refined_most_recent_games = most_recent_games[most_recent_games['PLAYER'].isin(df['Player'].unique())]
        refined_most_recent_games = refined_most_recent_games.merge(
            df.rename(columns={'Player': 'PLAYER', 'MIN': 'PRED_MIN'}), 
            on='PLAYER', 
            how='left'
        )
        
        # Ensure all cluster columns exist
        for col in all_cluster_columns:
            if col not in refined_most_recent_games.columns:
                refined_most_recent_games[col] = False
        
        # Prepare features for model
        feature_columns = ['Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg', 'PRED_MIN'] + all_cluster_columns
        refined_most_recent_games_model = refined_most_recent_games[feature_columns]
        
        if refined_most_recent_games_model.empty:
            logger.warning("No data available for prediction")
            return
        
        # Make predictions
        logger.info(f"Making predictions for {len(refined_most_recent_games_model)} players")
        features = refined_most_recent_games_model.to_numpy()
        predictions = rf.predict(features)
        
        # Add predictions to DataFrame
        refined_most_recent_games['My Model Predicted FP'] = predictions
        
        # Prepare final results
        final_columns = ['PLAYER', 'PPG Projection', 'My Model Predicted FP', 'GAME_DATE']
        refined_most_recent_games = refined_most_recent_games[final_columns]
        
        # Set current date for predictions
        current_date = datetime.date.today()
        refined_most_recent_games['GAME_DATE'] = current_date
        
        # Load existing daily predictions
        try:
            df_fp = load_dataframe_from_s3('data/daily_predictions/current.parquet')
        except:
            logger.info("No existing daily predictions found, creating new DataFrame")
            df_fp = pd.DataFrame(columns=['PLAYER', 'GAME_DATE', 'PPG_PROJECTION', 'MY_MODEL_PREDICTED_FP', 'ACTUAL_FP', 'MY_MODEL_CLOSER_PREDICTION'])
        
        # Update predictions
        logger.info("Updating daily predictions")
        for _, row in refined_most_recent_games.iterrows():
            mask = (df_fp['PLAYER'] == row['PLAYER']) & (df_fp['GAME_DATE'] == row['GAME_DATE'])
            if mask.any():
                df_fp.loc[mask, 'PPG_PROJECTION'] = safe_float(row['PPG Projection'])
                df_fp.loc[mask, 'MY_MODEL_PREDICTED_FP'] = safe_float(row['My Model Predicted FP'])
            else:
                new_record = {
                    "PLAYER": row['PLAYER'],
                    "GAME_DATE": row['GAME_DATE'],
                    "PPG_PROJECTION": safe_float(row['PPG Projection']),
                    "MY_MODEL_PREDICTED_FP": safe_float(row['My Model Predicted FP']),
                }
                df_fp = pd.concat([df_fp, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save updated predictions
        save_dataframe_to_s3(df_fp, 'data/daily_predictions/current.parquet')
        logger.info("Model predictions completed and saved")
        
        return len(refined_most_recent_games)

    except Exception as e:
        logger.error(f"Error in model predictions: {e}")
        raise

def check_scores_for_accuracy():
    """Check prediction accuracy against actual scores"""
    logger.info("Checking prediction accuracy")
    
    try:
        df_box = load_dataframe_from_s3('data/box_scores/current.parquet')
        df_fp = load_dataframe_from_s3('data/daily_predictions/current.parquet')
        
        # Validate required columns
        required_box_cols = {'PLAYER', 'GAME_DATE', 'FP'}
        missing_box_cols = required_box_cols - set(df_box.columns)
        if missing_box_cols:
            logger.error(f"Missing required columns {missing_box_cols} in box scores")
            return
        
        if df_fp.empty:
            logger.warning("No daily predictions found")
            return
        
        # Merge actual scores
        df_box = df_box[['PLAYER', 'GAME_DATE', 'FP']].rename(columns={'FP': 'ACTUAL_FP'})
        df_fp = df_fp.drop(columns=['ACTUAL_FP'], errors='ignore')
        df_fp = df_fp.merge(df_box, on=['PLAYER', 'GAME_DATE'], how='left')
        
        if 'MY_MODEL_CLOSER_PREDICTION' not in df_fp.columns:
            df_fp['MY_MODEL_CLOSER_PREDICTION'] = pd.Series(dtype=bool)
        
        # Calculate accuracy
        false_count = 0
        true_count = 0
        
        for index, row in df_fp.iterrows():
            actual_fp = safe_float(row['ACTUAL_FP'])
            ppg_projection = safe_float(row['PPG_PROJECTION'])
            model_predicted_fp = safe_float(row['MY_MODEL_PREDICTED_FP'])
            
            if not pd.isnull(actual_fp):
                ppg_diff = abs(ppg_projection - actual_fp) if not pd.isnull(ppg_projection) else float('inf')
                model_diff = abs(model_predicted_fp - actual_fp) if not pd.isnull(model_predicted_fp) else float('inf')
                
                if ppg_diff < model_diff:
                    df_fp.at[index, 'MY_MODEL_CLOSER_PREDICTION'] = False
                    false_count += 1
                else:
                    df_fp.at[index, 'MY_MODEL_CLOSER_PREDICTION'] = True
                    true_count += 1
        
        # Save updated predictions with accuracy
        save_dataframe_to_s3(df_fp, 'data/daily_predictions/current.parquet')
        
        # Calculate accuracy ratio
        total_predictions = true_count + false_count
        accuracy_ratio = true_count / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"Accuracy check completed: {true_count}/{total_predictions} = {accuracy_ratio:.3f}")
        return accuracy_ratio

    except Exception as e:
        logger.error(f"Error checking accuracy: {e}")
        return None

def run_daily_predictions_scraper():
    """Main function to run daily predictions pipeline"""
    logger.info("Starting daily predictions scraper")
    
    try:
        # Step 1: Scrape projections and minutes
        logger.info("=== STEP 1: SCRAPING PROJECTIONS ===")
        df = scrape_projections_data()
        
        if df.empty:
            logger.error("No projection data scraped")
            return {
                'success': False,
                'error': 'No projection data scraped'
            }
        
        # Step 2: Run model predictions
        logger.info("=== STEP 2: MODEL PREDICTIONS ===")
        predictions_count = run_model_predictions(df)
        
        # Step 3: Check accuracy
        logger.info("=== STEP 3: ACCURACY CHECK ===")
        accuracy_ratio = check_scores_for_accuracy()
        
        logger.info("Daily predictions scraper completed successfully")
        
        return {
            'success': True,
            'scraped_players': len(df),
            'predictions_generated': predictions_count,
            'accuracy_ratio': accuracy_ratio
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
                    'message': 'Daily predictions completed successfully',
                    'scraped_players': result['scraped_players'],
                    'predictions_generated': result['predictions_generated'],
                    'accuracy_ratio': result['accuracy_ratio']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Daily predictions failed',
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