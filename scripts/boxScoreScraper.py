import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
import time
from aws.s3_utils import save_dataframe_to_s3, load_dataframe_from_s3
from unidecode import unidecode
import logging


logger = logging.getLogger(__name__)

def normalize_name(name):
    return unidecode(name.strip().lower())

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def scrapeBoxScores():
    logger.info("Starting box score scraper")
    
    try:
        # Initialize Chrome WebDriver with headless options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument("--window-size=1920x1080")  # Wider viewport
        chrome_options.add_argument("--force-device-scale-factor=0.75")  # Zoom out
        chrome_options.add_argument("--start-maximized")  # Start maximized
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        driver = webdriver.Chrome(options=chrome_options)
        # Add timeout settings
        driver.set_page_load_timeout(300)  # 5 minutes for page load
        driver.implicitly_wait(30)  # 30 seconds for finding elements
        url_advanced = r"https://www.nba.com/stats/players/boxscores"
        driver.get(url_advanced)
        time.sleep(5)

        try:
            cookies = driver.find_element(By.XPATH, r"/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[1]")
            cookies.click()
        except Exception as e:
            logger.warning(f"Could not find or click cookies button: {e}")
        
        time.sleep(2)

        try:
            # Select Regular Season first with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[1]/div/div/div[2]/label/div/select"))
                    select.select_by_index(1)
                    time.sleep(3)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to select Regular Season dropdown after {max_retries} attempts: {e}")
                        driver.quit()
                        return
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(3)
            
            # Then select All players with retry logic
            for attempt in range(max_retries):
                try:
                    select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))
                    select.select_by_index(0)
                    time.sleep(3)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to select All players dropdown after {max_retries} attempts: {e}")
                        driver.quit()
                        return
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(3)
        except Exception as e:
            logger.error(f"Failed to select dropdown: {e}")
            driver.quit()
            return

        time.sleep(15)

        # Parse the page content
        src = driver.page_source 
        parser = BeautifulSoup(src, 'lxml')
        table = parser.find("div", attrs={"class": "Crom_container__C45Ti crom-container"})
        
        if not table or isinstance(table, NavigableString) or not isinstance(table, Tag):
            logger.error("Could not find valid table")
            driver.quit()
            return
            
        # Extract headers and data
        headers = table.findAll('th')
        headerlist = [h.text.strip() for h in headers]
        headerlist1 = [a for a in headerlist if not 'RANK' in a]

        rows = table.findAll('tr')[1:]
        player_box_scores = [[td.getText().strip() for td in rows[i].findAll('td')] for i in range(len(rows))]
        df = pd.DataFrame(data=player_box_scores, columns=headerlist1)
        print(df.head())

        # Data preprocessing
        df['PLAYER'] = df['PLAYER'].apply(normalize_name)
        df['FP'] = pd.to_numeric(df['FP'], errors='coerce')
        df['GAME DATE'] = pd.to_datetime(df['GAME DATE'], format='%m/%d/%Y')
        df = df.sort_values(by=['PLAYER', 'GAME DATE'], ascending=[True, True])

        # Get cluster data from S3
        cluster_df = load_dataframe_from_s3('data/clustered_players/current.parquet')
        clusterDict = cluster_df.set_index('PLAYER')['CLUSTER'].to_dict()
        df['CLUSTER'] = df['PLAYER'].map(clusterDict)

        # Calculate rolling averages
        windows = [3, 5, 7]
        for window in windows:
            df[f'Last{window}_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        # Calculate Season Average FP
        df['Season_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(lambda x: x.expanding(min_periods=1).mean())

        # Save to S3
        save_dataframe_to_s3(df, 'data/box_scores/current.parquet')
        logger.info(f"Successfully saved {len(df)} box score records to S3")
        
        driver.quit()
        
    except Exception as e:
        logger.error(f"Error in box score scraper: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        raise

def run_scrape_box_scores():
    """
    Scrapes box scores and saves to S3.
    """
    logger.info("Starting box score scraping process")
    scrapeBoxScores()

if __name__ == "__main__":
    run_scrape_box_scores()