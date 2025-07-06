import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
import time
from postgres.config import SessionLocal
from postgres.models import BoxScores, ClusteredPlayers
from unidecode import unidecode
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('boxscore_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
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
        # Initialize Chrome WebDriver
        driver = webdriver.Chrome()
        url_advanced = r"https://www.nba.com/stats/players/boxscores"
        driver.get(url_advanced)
        time.sleep(2)

        try:
            cookies = driver.find_element(By.XPATH, r"/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[1]")
            cookies.click()
        except Exception as e:
            logger.warning(f"Could not find or click cookies button: {e}")
        
        time.sleep(2)

        try:
            # Select Regular Season first
            select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[1]/div/div/div[2]/label/div/select"))
            select.select_by_index(1)
            time.sleep(1)
            
            # Then select All players
            select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))
            select.select_by_index(0)
        except Exception as e:
            logger.error(f"Failed to select dropdown: {e}")
            driver.quit()
            return

        time.sleep(10)

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

        # Data preprocessing
        df['PLAYER'] = df['PLAYER'].apply(normalize_name)
        df['FP'] = pd.to_numeric(df['FP'], errors='coerce')
        df['GAME DATE'] = pd.to_datetime(df['GAME DATE'], format='%m/%d/%Y')
        df = df.sort_values(by=['PLAYER', 'GAME DATE'], ascending=[True, True])

        # Get cluster data from database
        session = SessionLocal()
        players = session.query(ClusteredPlayers).all()
        data = [{column.name: getattr(player, column.name) for column in player.__table__.columns} for player in players]
        dataset_clusters = pd.DataFrame(data=data).drop('id', axis=1)
        clusterDict = dataset_clusters.set_index('PLAYER')['CLUSTER'].to_dict()
        df['CLUSTER'] = df['PLAYER'].map(clusterDict)

        # Calculate rolling averages
        windows = [3, 5, 7]
        for window in windows:
            df[f'Last{window}_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        # Calculate Season Average FP
        df['Season_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(lambda x: x.expanding(min_periods=1).mean())

        updates = inserts = errors = skipped = 0
        
        for index, row in df.iterrows():
            try:
                player_name = str(row['PLAYER'])
                game_date = row['GAME DATE']
                
                if pd.isnull(row['FP']):
                    logger.debug(f"Skipping {player_name} on {game_date} - no FP data")
                    skipped += 1
                    continue

                # Check for existing record
                existing_record = (
                    session.query(BoxScores)
                    .filter_by(PLAYER=player_name, GAME_DATE=game_date)
                    .one_or_none()
                )
                
                record_data = {
                    'PLAYER': str(row['PLAYER']),
                    'TEAM': str(row['TEAM']),
                    'MATCH_UP': str(row['MATCH UP']),
                    'GAME_DATE': row['GAME DATE'],
                    'WL': str(row['W/L']),
                    'MIN': safe_float(row['MIN']),
                    'PTS': safe_float(row['PTS']),
                    'FGM': safe_float(row['FGM']),
                    'FGA': safe_float(row['FGA']),
                    'FG_PERCENT': safe_float(row['FG%']),
                    'THREE_PM': safe_float(row['3PM']),
                    'THREE_PA': safe_float(row['3PA']),
                    'THREE_PERCENT': safe_float(row['3P%']),
                    'FTM': safe_float(row['FTM']),
                    'FTA': safe_float(row['FTA']),
                    'FT_PERCENT': safe_float(row['FT%']),
                    'OREB': safe_float(row['OREB']),
                    'DREB': safe_float(row['DREB']),
                    'REB': safe_float(row['REB']),
                    'AST': safe_float(row['AST']),
                    'STL': safe_float(row['STL']),
                    'BLK': safe_float(row['BLK']),
                    'TOV': safe_float(row['TOV']),
                    'PF': safe_float(row['PF']),
                    'PLUS_MINUS': safe_float(row['+/-']),
                    'FP': safe_float(row['FP']),
                    'CLUSTER': str(row['CLUSTER']) if not pd.isnull(row['CLUSTER']) else None,
                    'Last3_FP_Avg': safe_float(row['Last3_FP_Avg']),
                    'Last5_FP_Avg': safe_float(row['Last5_FP_Avg']),
                    'Last7_FP_Avg': safe_float(row['Last7_FP_Avg']),
                    'Season_FP_Avg': safe_float(row['Season_FP_Avg'])
                }

                if existing_record:
                    # Update existing record
                    for key, value in record_data.items():
                        setattr(existing_record, key, value)
                    updates += 1
                else:
                    # Create new record
                    new_record = BoxScores(**record_data)
                    session.add(new_record)
                    inserts += 1

                if isinstance(index, int) and index % 100 == 0:
                    logger.info(f"Processed {index}/{len(df)} records (Updates: {updates}, Inserts: {inserts}, Errors: {errors}, Skipped: {skipped})")
                    
            except Exception as e:
                logger.error(f"Error processing row {index} for player {row.get('PLAYER', 'Unknown')}: {str(e)}")
                errors += 1
        
        logger.info(f"Database operations completed - Updates: {updates}, Inserts: {inserts}, Errors: {errors}, Skipped: {skipped}")
        session.commit()
        session.close()
        driver.quit()
        
    except Exception as e:
        logger.error(f"Error in box score scraper: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        if 'session' in locals():
            session.close()
        raise

def run_scrape_box_scores():
    """
    Scrapes box scores and updates the database.
    """
    logger.info("Starting box score scraping process")
    scrapeBoxScores()

if __name__ == "__main__":
    run_scrape_box_scores()