from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import time
import datetime

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
                data_ppg_proj = row.get('data-ppg_proj')

                if data_name and data_ppg_proj and data_ppg_proj != "0.0":
                    player_data[data_name] = data_ppg_proj
            # Player Name and Projected FP based upon DraftKings
            df = pd.DataFrame(list(player_data.items()), columns=['Player', 'PPG Projection'])
    return df

def runModel(df):
    # Load the saved Random Forest model
    with open("models/RFCluster.sav", 'rb') as f:
        rf = pickle.load(f)

    # Import the boxScores scores
    dataset = pd.read_excel('data/boxScores.xlsx')
    dataset['GAME DATE'] = pd.to_datetime(dataset['GAME DATE'])

    # Sort by PLAYER and GAME DATE in ascending order
    dataset_sorted = dataset.sort_values(by=['PLAYER', 'GAME DATE'], ascending=[True, True])
    
    windows = [3, 5, 7]
    for window in windows:
        dataset_sorted[f'Last{window}_FP_Avg'] = (
            dataset_sorted.groupby('PLAYER')['FP']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # Calculate the season average FP for each player
    dataset_sorted['Season_FP_Avg'] = (
        dataset_sorted.groupby('PLAYER')['FP']
        .transform(lambda x: x.expanding(min_periods=1).mean())
    )

    # One-hot encode the CLUSTER column
    dataset_sorted = pd.get_dummies(dataset_sorted, columns=['CLUSTER'], drop_first=False)
    all_cluster_columns = [col for col in dataset_sorted.columns if col.startswith('CLUSTER_')]
    
    # Group by PLAYER and take the first row for each group (most recent game)
    most_recent_games = dataset_sorted.groupby('PLAYER').tail(1)


    # This removes players without clusters
    refined_most_recent_games = most_recent_games[most_recent_games['PLAYER'].isin(df['Player'].unique())]

        # Merge with PPG Projection from the scraped dataset; only the players in the scraped dataset will remain
    refined_most_recent_games = refined_most_recent_games.merge(
        df.rename(columns={'Player': 'PLAYER'}), on='PLAYER', how='left'
    )

    # If no players from a cluster are in the most recent games, add the column with all False values
    for col in all_cluster_columns:
        if col not in refined_most_recent_games.columns:
            refined_most_recent_games[col] = False


    feature_columns = ['Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg'] + all_cluster_columns

    # Reorder columns to match the training data
    refined_most_recent_games_model = refined_most_recent_games[feature_columns]

    # Predict
    features = refined_most_recent_games_model.to_numpy()
    predictions = rf.predict(features)
    # Add predictions to the DataFrame
    refined_most_recent_games['My Model Predicted FP'] = predictions
    # Retain the Player and PPG Projection columns
    final_columns = ['PLAYER', 'PPG Projection', 'My Model Predicted FP', 'GAME DATE']
    refined_most_recent_games = refined_most_recent_games[final_columns]
    current_date = datetime.date.today()
    refined_most_recent_games['GAME DATE'] = current_date

    # Export to Excel, ideally not overwriting the previous day's predictions
    output_file = 'data/dailyPredictions.xlsx'

    try:
        # Try to read the existing file
        existing_data = pd.read_excel(output_file)
        # Ensure GAME DATE columns are datetime objects for comparison
        existing_data['GAME DATE'] = pd.to_datetime(existing_data['GAME DATE'])
        refined_most_recent_games['GAME DATE'] = pd.to_datetime(refined_most_recent_games['GAME DATE'])

        # Append the new data to the existing data
        combined_data = pd.concat([existing_data, refined_most_recent_games], ignore_index=True)
        # Drop duplicates based on 'PLAYER' and 'GAME DATE'
        combined_data = combined_data.drop_duplicates(subset=['PLAYER', 'GAME DATE'], keep='first')

    except FileNotFoundError:
        # If the file doesn't exist, use the new data directly
        combined_data = refined_most_recent_games

    # Write the combined data back to the file
    combined_data.to_excel(output_file, index=False)


def checkScoresForFP():
    # TODO Add a check to boxScores.xlsx to see if the game has been played and the actual FP can be updated. This can be done by finding PLAYER AND GAME DATE there and comparing
    box_scores_file = 'data/boxScores.xlsx'
    fp_file = 'data/dailyPredictions.xlsx'
    df_box = pd.read_excel(box_scores_file)
    # Ensure 'GAME DATE' is in datetime format
    df_box['GAME DATE'] = pd.to_datetime(df_box['GAME DATE'])
    df_fp = pd.read_excel(fp_file)

    # Add an 'ACTUAL FP' column to df_fp if it doesn't exist
    if 'ACTUAL FP' not in df_fp.columns:
        df_fp['ACTUAL FP'] = None

    if 'My Model Closer Prediction' not in df_fp.columns:
        df_fp['My Model Closer Prediction'] = pd.Series(dtype=bool)

    # Iterate through rows in dailyPredictions.xlsx
    for index, row in df_fp.iterrows():
        player = row['PLAYER']
        game_date = row['GAME DATE']
        fp = row['ACTUAL FP']  # Assuming this is the column for FP
        if not pd.isnull(fp) and fp != None:
            continue
        # Check if this player's game exists in boxScores.xlsx
        matching_game = df_box[
            (df_box['PLAYER'] == player) & (df_box['GAME DATE'] == game_date)
        ]
        # If a matching game is found, update FP in df_fp
        if not matching_game.empty:
            actual_fp = matching_game.iloc[0]['FP']  # Assuming 'FP' column contains the actual FP
            if not pd.isnull(actual_fp):
                df_fp.at[index, 'ACTUAL FP'] = actual_fp
                print(f"Updated FP for {player} on {game_date.date()}: {actual_fp}")

    false_count = 0
    true_count = 0
    for index, row in df_fp.iterrows():
        actual_fp = row['ACTUAL FP']
        ppg_projection = row['PPG Projection']
        model_predicted_fp = row['My Model Predicted FP']
        if not pd.isnull(actual_fp):
            ppg_diff = abs(ppg_projection - actual_fp) if not pd.isnull(ppg_projection) else float('inf')
            model_diff = abs(model_predicted_fp - actual_fp) if not pd.isnull(model_predicted_fp) else float('inf')

            if ppg_diff < model_diff:
                df_fp.at[index, 'My Model Closer Prediction'] = False
                false_count += 1
            else:
                df_fp.at[index, 'My Model Closer Prediction'] = True
                true_count += 1



    ratio = true_count / (true_count + false_count)
    print(f"Ratio of True to False: {ratio}")
    # Save the updated dailyPredictions.xlsx
    df_fp.to_excel(fp_file, index=False)

df = scrapeData()
runModel(df)
checkScoresForFP()