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
from aws.s3_utils import load_dataframe_from_s3, save_dataframe_to_s3, load_model_from_s3
from unidecode import unidecode

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def normalize_name(name):
    return unidecode(name.strip().lower())

# Run within scrapeData()
def scrape_minutes_projection():
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
   
    driver = webdriver.Chrome()
    url = r"https://www.fanduel.com/research/nba/fantasy/dfs-projections"
    driver.get(url)

    try:
        # Wait for the table to load
        scrollable_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="virtuoso-scroller"]'))
        )
    except Exception as e:
        print(f"[Fallback] Could not find scrollable container: {e}")
        driver.quit()
        return {}

    player_minutes = {}
    seen_rows = set()
    scroll_pause_time = 1  # Adjust this if needed

    while True:
        # time.sleep(1)
        # Get the currently visible HTML
        html = driver.execute_script("return document.documentElement.outerHTML;")
        parser = BeautifulSoup(html, 'lxml')
        tbody = parser.find('tbody', class_='tableStyles_vtbody__Tj_Pq')

        if not tbody:
            print("No table body found.")
            break

        rows = tbody.find_all('tr')
        new_rows_found = False

        for row in rows:
            # Skip spacer rows
            if 'height' in row.get('style', ''):
                continue

            row_id = str(row)  # Unique identifier for each row based on HTML
            if row_id in seen_rows:
                continue  # Skip already processed rows

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
                    print(f"Found player: {player_name} - Minutes: {minutes}")  # Debug print
                except (AttributeError, IndexError) as e:
                    print(f"Error processing row: {e}")
                    continue

        # Scroll down to load more rows
        driver.execute_script("arguments[0].scrollBy(0, 500);", scrollable_container)
        time.sleep(scroll_pause_time)

        # Break the loop if no new rows are found
        if not new_rows_found:
            break
    

    driver.quit()
    return player_minutes



def scrapeData():
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
    driver = webdriver.Chrome()

    url_advanced = r"https://www.dailyfantasyfuel.com/nba/projections/draftkings"

    driver.get(url_advanced)

    try:
        span_element = driver.find_element(By.XPATH, r"/html/body/div[2]/div[1]/div[2]/div/div/div[3]/div/div/span")
        ActionChains(driver).move_to_element(span_element).click().perform()
    except Exception as e:
        print(f"[Fallback] Could not find or click span element: {e}")
        # Continue without clicking the span element

    src = driver.page_source 
    parser = BeautifulSoup(src, 'lxml')

    table = parser.find('table', class_="col-pad-lg-left-5 col-pad-lg-right-5 col-pad-md-left-3 col-pad-md-right-3 text-black row-pad-lg-top-2 row-pad-md-top-2 row-pad-sm-top-2 col-12 row-pad-5 row-pad-xs-1")

    if table:
        tbody = table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            # Initialize a dictionary to store results
            player_data = {}

            for row in rows:
                data_name = row.get('data-name')
                data_name = normalize_name(data_name)
                data_ppg_proj = row.get('data-ppg_proj')

                if data_name and data_ppg_proj and data_ppg_proj != "0.0":
                    player_data[data_name] = data_ppg_proj
            # Player Name and Projected FP based upon DraftKings
            df = pd.DataFrame(list(player_data.items()), columns=['Player', 'PPG Projection'])
            minutes_dict = scrape_minutes_projection()
            df['MIN'] = df['Player'].map(minutes_dict)
            df = df.dropna(subset=['MIN'])
        else:
            print("[Fallback] No tbody found in table. Creating empty DataFrame.")
            df = pd.DataFrame(columns=['Player', 'PPG Projection', 'MIN'])
    else:
        print("[Fallback] No table found. Creating empty DataFrame.")
        df = pd.DataFrame(columns=['Player', 'PPG Projection', 'MIN'])
    
    driver.quit()
    return df

def runModel(df):
    # Load the saved Random Forest model from S3
    try:
        rf = load_model_from_s3('models/RFCluster.sav')
    except Exception as e:
        print(f"[Fallback] Could not load model from S3: {e}")
        return

    # Load BoxScores from S3
    dataset = load_dataframe_from_s3('data/box_scores/current.parquet')
    
    # Defensive checks for required columns
    required_cols = {'MIN', 'W/L', 'FP', 'PLAYER', 'GAME_DATE'}
    missing_cols = required_cols - set(dataset.columns)
    if missing_cols:
        print(f"[Fallback] Missing required columns {missing_cols} in data/box_scores/current.parquet. Skipping runModel.")
        return
    
    dataset = dataset[dataset['MIN'] != 0]
    dataset = dataset.dropna(subset=['W/L'])
    dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE'])
    # Sort by PLAYER and GAME_DATE in ascending order
    dataset_sorted = dataset.sort_values(by=['PLAYER', 'GAME_DATE'], ascending=[True, True])
    windows = [3, 5, 7]
    for window in windows:
        dataset_sorted[f'Last{window}_FP_Avg'] = (
            dataset_sorted.groupby('PLAYER')['FP']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    # Recalculate the season average FP for each player
    dataset_sorted['Season_FP_Avg'] = (
        dataset_sorted.groupby('PLAYER')['FP']
        .transform(lambda x: x.expanding(min_periods=1).mean())
    )
    # One-hot encode the CLUSTER column
    dataset_sorted = pd.get_dummies(dataset_sorted, columns=['CLUSTER'], drop_first=False)
    all_cluster_columns = [col for col in dataset_sorted.columns if col.startswith('CLUSTER_')]
    # Group by PLAYER and take the first row for each group (most recent game)
    most_recent_games = dataset_sorted.groupby('PLAYER').tail(1)
    # Merge with PPG Projection from the scraped dataset; only the players in the scraped dataset will remain
    refined_most_recent_games = most_recent_games[most_recent_games['PLAYER'].isin(df['Player'].unique())]
    refined_most_recent_games = refined_most_recent_games.merge(
        df.rename(columns={'Player': 'PLAYER', 'MIN': 'PRED_MIN'}), on='PLAYER', how='left'
    )
    # If no players from a cluster are in the most recent games, add the column with all False values
    for col in all_cluster_columns:
        if col not in refined_most_recent_games.columns:
            refined_most_recent_games[col] = False
    feature_columns = ['Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg', 'PRED_MIN'] + all_cluster_columns
    # Reorder columns to match the training data
    refined_most_recent_games_model = refined_most_recent_games[feature_columns]
    
    # Defensive check: if no data to predict, return early
    if refined_most_recent_games_model.empty:
        print("[Fallback] No data available for prediction. Skipping model prediction.")
        return
    
    # Predict
    features = refined_most_recent_games_model.to_numpy()
    predictions = rf.predict(features)
    # Add predictions to the DataFrame
    refined_most_recent_games['My Model Predicted FP'] = predictions
    # Retain the Player and PPG Projection columns
    final_columns = ['PLAYER', 'PPG Projection', 'My Model Predicted FP', 'GAME_DATE']
    refined_most_recent_games = refined_most_recent_games[final_columns]
    current_date = datetime.date.today()
    refined_most_recent_games['GAME_DATE'] = current_date
    # Load DailyPlayerPredictions from S3
    df_fp = load_dataframe_from_s3('data/daily_predictions/current.parquet')
    
    # Defensive check: if df_fp is empty, create it with proper columns
    if df_fp.empty:
        df_fp = pd.DataFrame(columns=['PLAYER', 'GAME_DATE', 'PPG_PROJECTION', 'MY_MODEL_PREDICTED_FP', 'ACTUAL_FP', 'MY_MODEL_CLOSER_PREDICTION'])
        print("[Fallback] No daily player predictions found. Creating empty DataFrame.")
    # Merge or update predictions
    for _, row in refined_most_recent_games.iterrows():
        mask = (df_fp['PLAYER'] == row['PLAYER']) & (df_fp['GAME_DATE'] == row['GAME_DATE'])
        if mask.any():
            df_fp.loc[mask, 'PPG_PROJECTION'] = safe_float(row['PPG Projection'])
            df_fp.loc[mask, 'MY_MODEL_PREDICTED_FP'] = safe_float(row['My Model Predicted FP'])
            if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                df_fp.loc[mask, 'ACTUAL_FP'] = safe_float(row['ACTUAL_FP'])
            if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['MY_MODEL_CLOSER_PREDICTION']):
                df_fp.loc[mask, 'MY_MODEL_CLOSER_PREDICTION'] = row['MY_MODEL_CLOSER_PREDICTION']
        else:
            new_record_data = {
                "PLAYER": row['PLAYER'],
                "GAME_DATE": row['GAME_DATE'],
                "PPG_PROJECTION": safe_float(row['PPG Projection']),
                "MY_MODEL_PREDICTED_FP": safe_float(row['My Model Predicted FP']),
            }
            if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                new_record_data["ACTUAL_FP"] = safe_float(row['ACTUAL_FP'])
            if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['MY_MODEL_CLOSER_PREDICTION']):
                new_record_data["MY_MODEL_CLOSER_PREDICTION"] = row['MY_MODEL_CLOSER_PREDICTION']
            df_fp = pd.concat([df_fp, pd.DataFrame([new_record_data])], ignore_index=True)
    save_dataframe_to_s3(df_fp, 'data/daily_predictions/current.parquet')

def checkScoresForFP():
    df_box = load_dataframe_from_s3('data/box_scores/current.parquet')
    df_fp = load_dataframe_from_s3('data/daily_predictions/current.parquet')
    
    # Defensive checks for required columns
    required_box_cols = {'PLAYER', 'GAME_DATE', 'FP'}
    missing_box_cols = required_box_cols - set(df_box.columns)
    if missing_box_cols:
        print(f"[Fallback] Missing required columns {missing_box_cols} in data/box_scores/current.parquet. Skipping checkScoresForFP.")
        return
    
    # Defensive check: if df_fp is empty, return early
    if df_fp.empty:
        print("[Fallback] No daily player predictions found. Skipping checkScoresForFP.")
        return
    
    df_box = df_box[['PLAYER', 'GAME_DATE', 'FP']].rename(columns={'FP': 'ACTUAL_FP'})
    # Drop and merge 'ACTUAL_FP' to avoid conflict
    df_fp = df_fp.drop(columns=['ACTUAL_FP'], errors='ignore')
    df_fp = df_fp.merge(df_box, on=['PLAYER', 'GAME_DATE'], how='left')
    if 'MY_MODEL_CLOSER_PREDICTION' not in df_fp.columns:
        df_fp['MY_MODEL_CLOSER_PREDICTION'] = pd.Series(dtype=bool)
    false_count = 0
    true_count = 0
    for index, row in df_fp.iterrows():
        actual_fp = safe_float(row['ACTUAL_FP'])
        ppg_projection = safe_float(row['PPG_PROJECTION'])
        model_predicted_fp = safe_float(row['MY_MODEL_PREDICTED_FP'])
        my_model_closer_prediction = None
        if not pd.isnull(actual_fp):
            ppg_diff = abs(ppg_projection - actual_fp) if not pd.isnull(ppg_projection) else float('inf')
            model_diff = abs(model_predicted_fp - actual_fp) if not pd.isnull(model_predicted_fp) else float('inf')
            if ppg_diff < model_diff:
                my_model_closer_prediction = False
                false_count += 1
            else:
                my_model_closer_prediction = True
                true_count += 1
        df_fp.at[index, 'PPG_PROJECTION'] = ppg_projection
        df_fp.at[index, 'MY_MODEL_PREDICTED_FP'] = model_predicted_fp
        if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
            df_fp.at[index, 'ACTUAL_FP'] = actual_fp
        if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['ACTUAL_FP']):
            df_fp.at[index, 'MY_MODEL_CLOSER_PREDICTION'] = my_model_closer_prediction
    save_dataframe_to_s3(df_fp, 'data/daily_predictions/current.parquet')
    ratio = true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0
    print(f"% of more accurate predictions: {ratio}")

def run_daily_predictions_scraper():
    df = scrapeData()
    runModel(df)
    checkScoresForFP()

if __name__ == "__main__":
    run_daily_predictions_scraper()