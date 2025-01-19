from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import pickle
import time
import datetime
from postgres.config import SessionLocal
from postgres.models import BoxScores, DailyPlayerPredictions
from unidecode import unidecode

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def normalize_name(name):
    return unidecode(name.strip().lower())

def scrape_minutes_projection():
    driver = webdriver.Chrome()
    url = r"https://www.fanduel.com/research/nba/fantasy/dfs-projections"
    driver.get(url)

    # Wait for the table to load
    scrollable_container = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="virtuoso-scroller"]'))
    )

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
    driver = webdriver.Chrome()

    url_advanced = r"https://www.dailyfantasyfuel.com/nba/projections/draftkings"

    driver.get(url_advanced)

    span_element = driver.find_element(By.XPATH, r"/html/body/div[2]/div[1]/div[2]/div/div/div[3]/div/div/span")
    ActionChains(driver).move_to_element(span_element).click().perform()

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
    
    driver.quit()
    return df

def runModel(df):
    # Load the saved Random Forest model
    with open("models/RFCluster.sav", 'rb') as f:
        rf = pickle.load(f)


    session = SessionLocal()

    games = session.query(BoxScores).all()
    data = []
    for game in games:
        game_dict = {column.name: getattr(game, column.name) for column in game.__table__.columns}
        data.append(game_dict)
    dataset = pd.DataFrame(data)
    dataset = dataset[dataset['MIN'] != 0]
    dataset = dataset.dropna(subset=['WL'])


    dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE'])

    # Sort by PLAYER and GAME DATE in ascending order
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

    # Iterate over the DataFrame rows
    for _, row in refined_most_recent_games.iterrows():
        # Check if a record already exists for the given PLAYER and GAME_DATE
        existing_record = (
            session.query(DailyPlayerPredictions)
            .filter_by(PLAYER=row['PLAYER'], GAME_DATE=row['GAME_DATE'])
            .one_or_none()
        )
        if existing_record:
            # Update the existing record with new data
            existing_record.PPG_PROJECTION = safe_float(row['PPG Projection'])
            existing_record.MY_MODEL_PREDICTED_FP = safe_float(row['My Model Predicted FP'])
            if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                existing_record.ACTUAL_FP = safe_float(row['ACTUAL_FP'])
            if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['MY_MODEL_CLOSER_PREDICTION']):
                existing_record.MY_MODEL_CLOSER_PREDICTION = row['MY_MODEL_CLOSER_PREDICTION']
        else:
            new_record_data = {
                "PLAYER": row['PLAYER'],
                "GAME_DATE": row['GAME_DATE'],
                "PPG_PROJECTION": safe_float(row['PPG Projection']),
                "MY_MODEL_PREDICTED_FP": safe_float(row['My Model Predicted FP']),
            }
            # Add ACTUAL_FP and MY_MODEL_CLOSER_PREDICTION if they exist in the row
            if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                new_record_data["ACTUAL_FP"] = safe_float(row['ACTUAL_FP'])
            if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['MY_MODEL_CLOSER_PREDICTION']):
                new_record_data["MY_MODEL_CLOSER_PREDICTION"] = row['MY_MODEL_CLOSER_PREDICTION']
            
            # Create a new DailyPlayerPredictions object
            new_record = DailyPlayerPredictions(**new_record_data)
            session.add(new_record)
    session.commit()
    session.close()


def checkScoresForFP():
    session = SessionLocal()
    games = session.query(BoxScores).all()
    data = []
    for game in games:
        game_dict = {column.name: getattr(game, column.name) for column in game.__table__.columns}
        data.append(game_dict)
    df_box = pd.DataFrame(data)

    games = session.query(DailyPlayerPredictions).all()
    data = []
    for game in games:
        game_dict = {column.name: getattr(game, column.name) for column in game.__table__.columns}
        data.append(game_dict)
    df_fp = pd.DataFrame(data)


    # Add an 'ACTUAL FP' column to df_fp if it doesn't exist
    if 'ACTUAL_FP' not in df_fp.columns:
        df_fp['ACTUAL_FP'] = None

    if 'MY_MODEL_CLOSER_PREDICTION' not in df_fp.columns:
        df_fp['MY_MODEL_CLOSER_PREDICTION'] = pd.Series(dtype=bool)

    df_box = df_box[['PLAYER', 'GAME_DATE', 'FP']].rename(columns={'FP': 'ACTUAL_FP'})

    # Ensure no conflict by dropping 'ACTUAL_FP' in df_fp (if it exists)
    if 'ACTUAL_FP' in df_fp.columns:
        df_fp.drop(columns=['ACTUAL_FP'], inplace=True)

    # Now merge without suffixes
    df_fp = df_fp.merge(df_box, on=['PLAYER', 'GAME_DATE'], how='left') 

    false_count = 0
    true_count = 0

    # Iterate through rows in dailyPredictions.xlsx
    for index, row in df_fp.iterrows():
        existing_record = (
            session.query(DailyPlayerPredictions)
            .filter_by(PLAYER=row['PLAYER'], GAME_DATE=row['GAME_DATE'])
            .one_or_none()
        )
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

        if existing_record:
            existing_record.PPG_PROJECTION = ppg_projection
            existing_record.MY_MODEL_PREDICTED_FP = model_predicted_fp

            if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                existing_record.ACTUAL_FP = actual_fp
            if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['ACTUAL_FP']):
                existing_record.MY_MODEL_CLOSER_PREDICTION = my_model_closer_prediction
        else:
            new_record_data = {
                "PLAYER": row['PLAYER'],
                "GAME_DATE": row['GAME_DATE'],
                "PPG_PROJECTION": safe_float(row['PPG_PROJECTION']),
                "MY_MODEL_PREDICTED_FP": safe_float(row['MY_MODEL_PREDICTED_FP']),
            }
            if 'ACTUAL_FP' in row and pd.notna(row['ACTUAL_FP']):
                new_record_data["ACTUAL_FP"] = actual_fp
            if 'MY_MODEL_CLOSER_PREDICTION' in row and pd.notna(row['ACTUAL_FP']):
                new_record_data["MY_MODEL_CLOSER_PREDICTION"] = my_model_closer_prediction
            new_record = DailyPlayerPredictions(**new_record_data)
            session.add(new_record)
        session.commit()


    ratio = true_count / (true_count + false_count)
    print(f"% of more accurate predictions: {ratio}")
    session.close()

def run_daily_predictions_scraper():
    df = scrapeData()
    runModel(df)
    checkScoresForFP()

if __name__ == "__main__":
    # scrape_minutes_projection()
    run_daily_predictions_scraper()