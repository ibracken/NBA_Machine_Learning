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
import requests
from io import BytesIO
from unidecode import unidecode
import logging
import pytz

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

def normalize_team_abbrev(team_abbrev):
    """
    Normalize SportsLine team abbreviations to NBA official abbreviations
    SportsLine uses different abbreviations for some teams
    """
    # Mapping from SportsLine abbreviations to NBA official abbreviations
    # Keep adding to this
    sportsline_to_nba = {
        'SA': 'SAS',    # San Antonio Spurs
        'NO': 'NOP',    # New Orleans Pelicans
        'NY': 'NYK',    # New York Knicks
        'GS': 'GSW',    # Golden State Warriors
        'PHO': 'PHX',   # Phoenix Suns
    }

    team_abbrev = str(team_abbrev).strip().upper()
    return sportsline_to_nba.get(team_abbrev, team_abbrev)

def parse_matchup(matchup_str, player_team_abbrev):
    """
    Parse matchup string like 'UTA@SAC' or 'ATL@ORL'
    Returns: (is_home, opponent)
    player_team_abbrev is needed to determine if player's team is home or away
    """
    if pd.isna(matchup_str) or not matchup_str:
        return 0, 'UNKNOWN'

    matchup_str = str(matchup_str).strip()

    if '@' in matchup_str:
        # Away @ Home format
        parts = matchup_str.split('@')
        if len(parts) == 2:
            away_team = normalize_team_abbrev(parts[0].strip())
            home_team = normalize_team_abbrev(parts[1].strip())

            # Normalize player's team abbreviation too
            player_team_abbrev_normalized = normalize_team_abbrev(player_team_abbrev)

            # Determine if player's team is home or away
            if player_team_abbrev_normalized == home_team:
                return 1, away_team  # Player is on home team, opponent is away team
            elif player_team_abbrev_normalized == away_team:
                return 0, home_team  # Player is on away team, opponent is home team

    return 0, 'UNKNOWN'

def scrape_minutes_projection():
    """Scrape minutes projections and matchup data from SportsLine"""
    logger.info("Starting SportsLine minutes projection scraping")

    url = "https://www.sportsline.com/nba/expert-projections/simulation/"

    try:
        response = requests.get(url, timeout=30, proxies=proxies)
        response.raise_for_status()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'lxml')

        # Find the table using the xpath structure
        # XPath: /html/body/div[1]/div[5]/div/section[1]/div/main/div/section/section/section/table/tbody
        table = soup.find('table')

        if not table:
            logger.error("Could not find table on SportsLine page")
            return {}, {}

        tbody = table.find('tbody')
        if not tbody:
            logger.error("Could not find tbody in table")
            return {}, {}

        rows = tbody.find_all('tr')
        logger.info(f"Found {len(rows)} rows in SportsLine table")

        player_minutes = {}
        player_matchups = {}
        for row in rows:
            cells = row.find_all('td')

            # td[1] has player name, td[4] has matchup, td[10] has minutes
            if len(cells) >= 10:
                player_name = cells[0].get_text(strip=True)  # td[1] is cells[0] (0-indexed)
                matchup = cells[3].get_text(strip=True)      # td[4] is cells[3] (0-indexed)
                minutes = cells[9].get_text(strip=True)      # td[10] is cells[9] (0-indexed)

                if player_name and minutes:
                    # Normalize the player name to match the format used elsewhere
                    normalized_name = normalize_name(player_name)
                    player_minutes[normalized_name] = str(minutes)
                    player_matchups[normalized_name] = matchup
                    logger.debug(f"Found player: {normalized_name} - Minutes: {minutes} - Matchup: {matchup}")

        logger.info(f"Successfully scraped minutes for {len(player_minutes)} players from SportsLine")
        return player_minutes, player_matchups

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from SportsLine: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {}, {}
    except Exception as e:
        logger.error(f"Failed to parse SportsLine page: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {}, {}

def scrape_starting_lineup(driver):
    """
    Scrape starting lineup indicators from DailyFantasyFuel
    Returns a set of normalized player names who are listed as starters
    """
    try:
        src = driver.page_source
        parser = BeautifulSoup(src, 'lxml')

        table = parser.find('table', class_="col-pad-lg-left-5 col-pad-lg-right-5 col-pad-md-left-3 col-pad-md-right-3 text-black row-pad-lg-top-2 row-pad-md-top-2 row-pad-sm-top-2 col-12 row-pad-5 row-pad-xs-1")

        starters = set()

        if table:
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')

                for row in rows:
                    cells = row.find_all('td')

                    # td[5] (cells[4] in 0-indexed) contains starting indicator
                    # If there's any text/content in that cell, they're likely starting
                    if len(cells) >= 5:
                        starting_indicator = cells[4].get_text(strip=True)

                        # If there's any text in the starting column
                        if starting_indicator == "YES":
                            # Get player name from data-name attribute
                            data_name = row.get('data-name')
                            if data_name:
                                normalized_name = normalize_name(data_name)
                                starters.add(normalized_name)
                                logger.info(f"Detected starter: {normalized_name} (indicator: '{starting_indicator}')")

        logger.info(f"Total starters detected: {len(starters)}")
        return starters

    except Exception as e:
        logger.error(f"Failed to scrape starting lineup: {e}")
        return set()

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

        # Get minutes projections and matchups
        minutes_dict, matchup_dict = scrape_minutes_projection()

        # Merge minutes and matchup data
        df['MIN'] = df['Player'].map(minutes_dict)
        df['MATCHUP'] = df['Player'].map(matchup_dict)

        # Apply baseline minutes for starters with low projected minutes
        STARTER_BASELINE_MINUTES = 24

        for idx, row in df.iterrows():
            player_name = row['Player']
            current_minutes = pd.to_numeric(row['MIN'], errors='coerce')

            # If player is a starter and their projected minutes are below baseline
            if player_name in starters:
                if pd.isna(current_minutes) or current_minutes < STARTER_BASELINE_MINUTES:
                    logger.info(f"Applying baseline {STARTER_BASELINE_MINUTES} min for starter: {player_name} (was: {current_minutes})")
                    df.at[idx, 'MIN'] = str(STARTER_BASELINE_MINUTES)

        df = df.dropna(subset=['MIN'])

        logger.info(f"Final dataset: {len(df)} players with both projections and minutes")
        return df

    except Exception as e:
        logger.error(f"Error scraping projections: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame(columns=['Player', 'PPG Projection', 'MIN', 'MATCHUP', 'SALARY', 'POSITION'])
    finally:
        driver.quit()

def run_model_predictions(todays_scraped_projections):
    """
    Generate ML model predictions for today's playing players.

    This function:
    1. Loads each player's most recent historical game to extract features
    2. Merges with today's scraped data (minutes, matchup, salary, position)
    3. Encodes features and generates predictions
    4. Saves predictions to S3

    Args:
        todays_scraped_projections: DataFrame with today's players
                                   (columns: Player, PPG Projection, MIN, MATCHUP, SALARY, POSITION)

    Returns:
        Number of predictions generated
    """
    logger.info("Starting model predictions")

    try:
        # Load trained model
        logger.info("Loading trained model from S3")
        model = load_model_from_s3('models/RFCluster.sav')

        # Load historical box scores (to extract features from each player's last game)
        logger.info("Loading historical box scores data")
        historical_box_scores = load_dataframe_from_s3('data/box_scores/current.parquet')

        # Validate required columns
        required_cols = {'MIN', 'WL', 'FP', 'PLAYER', 'GAME_DATE', 'TEAM_ABBREVIATION'}
        missing_cols = required_cols - set(historical_box_scores.columns)
        if missing_cols:
            logger.error(f"Missing required columns {missing_cols} in box scores data")
            return None

        # Clean historical data
        historical_box_scores = historical_box_scores[historical_box_scores['MIN'] != 0]
        historical_box_scores = historical_box_scores.dropna(subset=['WL'])
        historical_box_scores['GAME_DATE'] = pd.to_datetime(historical_box_scores['GAME_DATE'])
        historical_box_scores = historical_box_scores.sort_values(by=['PLAYER', 'GAME_DATE'], ascending=[True, True])

        # Get each player's most recent game (for feature extraction like Last3_FP_Avg, Career_FP_Avg, etc.)
        logger.info("Using pre-calculated rolling averages and career features from box score data")
        player_last_games = historical_box_scores.groupby('PLAYER').tail(1).copy()

        # Calculate rest days from last game to today
        today = pd.Timestamp(datetime.date.today())
        player_last_games['REST_DAYS'] = (today - player_last_games['GAME_DATE']).dt.days
        player_last_games['REST_DAYS'] = player_last_games['REST_DAYS'].clip(0, 30)

        # Filter to only players playing today
        todays_players = todays_scraped_projections['Player'].unique()
        todays_player_features = player_last_games[player_last_games['PLAYER'].isin(todays_players)].copy()

        # Save historical MIN as PREV_MIN_HIST before dropping (we'll use today's projected MIN instead)
        if 'MIN' in todays_player_features.columns:
            todays_player_features['PREV_MIN_HIST'] = todays_player_features['MIN']

        todays_player_features = todays_player_features.drop(columns=['MIN'], errors='ignore')

        # Merge today's scraped data (projected minutes, matchup, salary, position)
        todays_projections_cleaned = todays_scraped_projections.rename(columns={
            'Player': 'PLAYER',
            'MATCHUP': 'MATCHUP_TODAY'
        })
        todays_projections_cleaned['MIN'] = pd.to_numeric(todays_projections_cleaned['MIN'], errors='coerce')
        todays_projections_cleaned['SALARY'] = pd.to_numeric(todays_projections_cleaned['SALARY'], errors='coerce')

        todays_player_features = todays_player_features.merge(todays_projections_cleaned, on='PLAYER', how='left')

        # Parse matchup to extract home/away and opponent team
        matchup_parsed = todays_player_features.apply(
            lambda row: parse_matchup(row['MATCHUP_TODAY'], row['TEAM_ABBREVIATION']), axis=1
        )
        todays_player_features['IS_HOME'] = matchup_parsed.apply(lambda x: x[0])
        todays_player_features['OPPONENT'] = matchup_parsed.apply(lambda x: x[1])

        # Fill missing clusters with placeholder
        todays_player_features['CLUSTER'] = todays_player_features['CLUSTER'].fillna('CLUSTER_NAN')

        # One-hot encode CLUSTER and OPPONENT
        # All 30 NBA teams for opponent encoding
        ALL_NBA_TEAMS = [
            'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        ]

        todays_player_features = pd.get_dummies(todays_player_features, columns=['CLUSTER', 'OPPONENT'], drop_first=False)

        # Add missing opponent columns (teams not playing today)
        for team in ALL_NBA_TEAMS:
            opponent_col = f'OPPONENT_{team}'
            if opponent_col not in todays_player_features.columns:
                todays_player_features[opponent_col] = 0

        # Add missing cluster columns (0-14 plus NAN)
        for cluster_num in range(15):
            cluster_col = f'CLUSTER_{float(cluster_num)}'
            if cluster_col not in todays_player_features.columns:
                todays_player_features[cluster_col] = 0

        if 'CLUSTER_CLUSTER_NAN' not in todays_player_features.columns:
            todays_player_features['CLUSTER_CLUSTER_NAN'] = 0

        # Remove invalid opponent columns (e.g., OPPONENT_UNKNOWN)
        valid_opponent_columns = [f'OPPONENT_{team}' for team in ALL_NBA_TEAMS]
        invalid_opponent_cols = [col for col in todays_player_features.columns
                                if col.startswith('OPPONENT_') and col not in valid_opponent_columns]

        if invalid_opponent_cols:
            logger.info(f"Removing {len(invalid_opponent_cols)} invalid opponent columns: {invalid_opponent_cols}")
            todays_player_features = todays_player_features.drop(columns=invalid_opponent_cols)

        # Get expected features from model
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            logger.info(f"Model expects {len(expected_features)} features")

            # Add any missing features as 0
            missing_features = [col for col in expected_features if col not in todays_player_features.columns]
            if missing_features:
                logger.info(f"Adding {len(missing_features)} missing features as 0")
                for col in missing_features:
                    todays_player_features[col] = 0

            # Select features in exact order expected by model
            model_input_features = todays_player_features[expected_features]
        else:
            # Fallback for older sklearn without feature_names_in_
            logger.info("Model doesn't have feature_names_in_, reconstructing feature order...")

            base_features = ['Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg',
                           'Career_FP_Avg', 'Games_Played_Career', 'MIN',
                           'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg',
                           'IS_HOME', 'REST_DAYS']

            cluster_columns = sorted([col for col in todays_player_features.columns if col.startswith('CLUSTER_')])
            opponent_columns = sorted([col for col in todays_player_features.columns if col.startswith('OPPONENT_')])

            feature_order = base_features + cluster_columns + opponent_columns
            logger.info(f"Reconstructed {len(feature_order)} features (Base: {len(base_features)}, "
                       f"Clusters: {len(cluster_columns)}, Opponents: {len(opponent_columns)})")

            model_input_features = todays_player_features[feature_order]

        # Check if we have data to predict
        if model_input_features.empty:
            logger.warning("No data available for prediction. Skipping model prediction.")
            return None

        # Generate predictions
        logger.info(f"Making predictions for {len(model_input_features)} players with {len(model_input_features.columns)} features")
        predictions = model.predict(model_input_features.to_numpy())

        # Format results
        todays_player_features['My Model Predicted FP'] = predictions
        final_columns = ['PLAYER', 'PPG Projection', 'My Model Predicted FP', 'GAME_DATE', 'SALARY', 'POSITION', 'MIN', 'FP', 'PREV_MIN_HIST', 'Season_FP_Avg', 'Season_MIN_Avg']
        prediction_results = todays_player_features[final_columns].copy()
        prediction_results = prediction_results.rename(columns={
            'MIN': 'PROJECTED_MIN',
            'FP': 'PREV_FP',
            'PREV_MIN_HIST': 'PREV_MIN',
            'Season_FP_Avg': 'SEASON_AVG_FP',
            'Season_MIN_Avg': 'SEASON_AVG_MIN'
        })

        # Set today's date in ET timezone (Lambda runs in UTC)
        et_tz = pytz.timezone('America/New_York')
        current_date = datetime.datetime.now(et_tz).date()
        prediction_results['GAME_DATE'] = current_date

        # Load existing daily predictions from S3
        try:
            df_fp = load_dataframe_from_s3('data/daily_predictions/current.parquet')
        except:
            logger.info("No existing daily predictions found, creating new DataFrame")
            df_fp = pd.DataFrame(columns=['PLAYER', 'GAME_DATE', 'PPG_PROJECTION', 'MY_MODEL_PREDICTED_FP', 'PROJECTED_MIN', 'ACTUAL_MIN', 'ACTUAL_FP', 'MY_MODEL_CLOSER_PREDICTION', 'SALARY', 'POSITION'])

        if df_fp.empty:
            df_fp = pd.DataFrame(columns=['PLAYER', 'GAME_DATE', 'PPG_PROJECTION', 'MY_MODEL_PREDICTED_FP', 'PROJECTED_MIN', 'ACTUAL_MIN', 'ACTUAL_FP', 'MY_MODEL_CLOSER_PREDICTION', 'SALARY', 'POSITION'])

        # Update existing predictions or add new ones
        logger.info("Updating daily predictions")
        new_records = []

        for _, row in prediction_results.iterrows():
            mask = (df_fp['PLAYER'] == row['PLAYER']) & (df_fp['GAME_DATE'] == row['GAME_DATE'])
            if mask.any():
                df_fp.loc[mask, 'PPG_PROJECTION'] = safe_float(row['PPG Projection'])
                df_fp.loc[mask, 'MY_MODEL_PREDICTED_FP'] = safe_float(row['My Model Predicted FP'])
                df_fp.loc[mask, 'PROJECTED_MIN'] = safe_float(row['PROJECTED_MIN'])
                df_fp.loc[mask, 'SALARY'] = safe_float(row['SALARY'])
                df_fp.loc[mask, 'POSITION'] = row['POSITION']
                df_fp.loc[mask, 'PREV_FP'] = safe_float(row.get('PREV_FP', 0))
                df_fp.loc[mask, 'PREV_MIN'] = safe_float(row.get('PREV_MIN', 0))
                df_fp.loc[mask, 'SEASON_AVG_FP'] = safe_float(row.get('SEASON_AVG_FP', 0))
                df_fp.loc[mask, 'SEASON_AVG_MIN'] = safe_float(row.get('SEASON_AVG_MIN', 0))
                if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                    df_fp.loc[mask, 'ACTUAL_FP'] = safe_float(row['ACTUAL_FP'])
                if 'ACTUAL_MIN' in row and pd.notna(row['ACTUAL_MIN']):
                    df_fp.loc[mask, 'ACTUAL_MIN'] = safe_float(row['ACTUAL_MIN'])
                if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['MY_MODEL_CLOSER_PREDICTION']):
                    df_fp.loc[mask, 'MY_MODEL_CLOSER_PREDICTION'] = row['MY_MODEL_CLOSER_PREDICTION']
            else:
                new_record_data = {
                    "PLAYER": row['PLAYER'],
                    "GAME_DATE": row['GAME_DATE'],
                    "PPG_PROJECTION": safe_float(row['PPG Projection']),
                    "MY_MODEL_PREDICTED_FP": safe_float(row['My Model Predicted FP']),
                    "PROJECTED_MIN": safe_float(row['PROJECTED_MIN']),
                    "SALARY": safe_float(row['SALARY']),
                    "POSITION": row['POSITION'],
                    "PREV_FP": safe_float(row.get('PREV_FP', 0)),
                    "PREV_MIN": safe_float(row.get('PREV_MIN', 0)),
                    "SEASON_AVG_FP": safe_float(row.get('SEASON_AVG_FP', 0)),
                    "SEASON_AVG_MIN": safe_float(row.get('SEASON_AVG_MIN', 0)),
                }
                if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                    new_record_data["ACTUAL_FP"] = safe_float(row['ACTUAL_FP'])
                if 'ACTUAL_MIN' in row and pd.notna(row['ACTUAL_MIN']):
                    new_record_data["ACTUAL_MIN"] = safe_float(row['ACTUAL_MIN'])
                if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['MY_MODEL_CLOSER_PREDICTION']):
                    new_record_data["MY_MODEL_CLOSER_PREDICTION"] = row['MY_MODEL_CLOSER_PREDICTION']
                new_records.append(new_record_data)

        # Add all new records at once if there are any
        if new_records:
            df_fp = pd.concat([df_fp, pd.DataFrame(new_records)], ignore_index=True)
        save_dataframe_to_s3(df_fp, 'data/daily_predictions/current.parquet')
        logger.info("Model predictions completed and saved")

        return len(prediction_results)

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
        required_box_cols = {'PLAYER', 'GAME_DATE', 'FP', 'MIN'}
        missing_box_cols = required_box_cols - set(df_box.columns)
        if missing_box_cols:
            logger.error(f"Missing required columns {missing_box_cols} in box scores")
            return

        if df_fp.empty:
            logger.warning("No daily predictions found")
            return

        # Merge actual scores and actual minutes
        df_box = df_box[['PLAYER', 'GAME_DATE', 'FP', 'MIN']].rename(columns={'FP': 'ACTUAL_FP', 'MIN': 'ACTUAL_MIN'})

        # Ensure GAME_DATE is the same type in both dataframes
        df_box['GAME_DATE'] = pd.to_datetime(df_box['GAME_DATE']).dt.date
        df_fp['GAME_DATE'] = pd.to_datetime(df_fp['GAME_DATE']).dt.date

        # Drop and merge 'ACTUAL_FP' and 'ACTUAL_MIN' to avoid conflict
        df_fp = df_fp.drop(columns=['ACTUAL_FP', 'ACTUAL_MIN'], errors='ignore')
        df_fp = df_fp.merge(df_box, on=['PLAYER', 'GAME_DATE'], how='left')
        
        if 'MY_MODEL_CLOSER_PREDICTION' not in df_fp.columns:
            df_fp['MY_MODEL_CLOSER_PREDICTION'] = pd.Series(dtype=bool)

        # Calculate accuracy only for games after Oct 30, 2025 (when we fixed the stale data issue)
        import datetime as dt
        accuracy_cutoff_date = dt.date(2025, 10, 30)

        false_count = 0
        true_count = 0

        # Lists to track errors for mean error calculations
        dff_errors = []
        model_errors = []

        for index, row in df_fp.iterrows():
            actual_fp = safe_float(row['ACTUAL_FP'])
            ppg_projection = safe_float(row['PPG_PROJECTION'])
            model_predicted_fp = safe_float(row['MY_MODEL_PREDICTED_FP'])
            game_date = row['GAME_DATE']

            # Only count accuracy for games after cutoff date
            if not pd.isnull(actual_fp) and game_date >= accuracy_cutoff_date:
                ppg_diff = abs(ppg_projection - actual_fp) if not pd.isnull(ppg_projection) else float('inf')
                model_diff = abs(model_predicted_fp - actual_fp) if not pd.isnull(model_predicted_fp) else float('inf')

                # Track errors for mean calculations
                if not pd.isnull(ppg_projection):
                    dff_errors.append(ppg_diff)
                if not pd.isnull(model_predicted_fp):
                    model_errors.append(model_diff)

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
        total_with_actual = df_fp['ACTUAL_FP'].notna().sum()
        filtered_out = total_with_actual - total_predictions
        accuracy_ratio = true_count / total_predictions if total_predictions > 0 else 0

        # Calculate mean absolute errors
        dff_mae = sum(dff_errors) / len(dff_errors) if dff_errors else 0
        model_mae = sum(model_errors) / len(model_errors) if model_errors else 0

        logger.info(f"Accuracy check: {total_with_actual} predictions with actual results")
        logger.info(f"Filtered out {filtered_out} predictions before {accuracy_cutoff_date} (stale data)")
        logger.info(f"Counted accuracy for {total_predictions} predictions after {accuracy_cutoff_date}")
        logger.info(f"Accuracy: {true_count}/{total_predictions} = {accuracy_ratio:.3f}")
        logger.info(f"DFF Mean Absolute Error: {dff_mae:.2f} FP ({len(dff_errors)} predictions)")
        logger.info(f"My Model Mean Absolute Error: {model_mae:.2f} FP ({len(model_errors)} predictions)")
        logger.info(f"Mean Error Improvement: {dff_mae - model_mae:.2f} FP (negative = model is worse)")

        return {
            'accuracy_ratio': accuracy_ratio,
            'dff_mae': dff_mae,
            'model_mae': model_mae,
            'total_predictions': total_predictions
        }

    except Exception as e:
        logger.error(f"Error checking accuracy: {e}")
        return None

def run_daily_predictions_scraper():
    """Main function to run daily predictions pipeline"""
    logger.info("Starting daily predictions scraper")
    
    try:
        # Step 1: Scrape projections and minutes for today
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
        accuracy_results = check_scores_for_accuracy()

        logger.info("Daily predictions scraper completed successfully")

        # Extract results or use defaults if None
        if accuracy_results:
            return {
                'success': True,
                'scraped_players': len(df),
                'predictions_generated': predictions_count,
                'accuracy_ratio': accuracy_results.get('accuracy_ratio'),
                'dff_mae': accuracy_results.get('dff_mae'),
                'model_mae': accuracy_results.get('model_mae'),
                'total_accuracy_predictions': accuracy_results.get('total_predictions')
            }
        else:
            return {
                'success': True,
                'scraped_players': len(df),
                'predictions_generated': predictions_count,
                'accuracy_ratio': None,
                'dff_mae': None,
                'model_mae': None,
                'total_accuracy_predictions': 0
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
                    'scraped_players': result.get('scraped_players'),
                    'predictions_generated': result.get('predictions_generated'),
                    'accuracy_ratio': result.get('accuracy_ratio'),
                    'dff_mae': result.get('dff_mae'),
                    'model_mae': result.get('model_mae'),
                    'total_accuracy_predictions': result.get('total_accuracy_predictions')
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