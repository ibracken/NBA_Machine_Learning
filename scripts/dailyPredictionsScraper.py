from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import time

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

    # Import the pastPlayerPredictions scores
    dataset = pd.read_excel('data/playerPredictions.xlsx')
    dataset = dataset.drop('Unnamed: 0', axis=1)
    dataset['GAME DATE'] = pd.to_datetime(dataset['GAME DATE'])

    # Sort by PLAYER and GAME DATE in descending order
    dataset_sorted = dataset.sort_values(by=['PLAYER', 'GAME DATE'], ascending=[True, False])

    # Group by PLAYER and take the first row for each group (most recent game)
    most_recent_games = dataset_sorted.groupby('PLAYER').first().reset_index()

    # This removes players without clusters
    refined_most_recent_games = most_recent_games[most_recent_games['PLAYER'].isin(df['Player'].unique())]



    refined_most_recent_games = pd.get_dummies(
        refined_most_recent_games, columns=['CLUSTER'], drop_first=False
    )

    # TODO I have to manually adjust each column because the most recent game isn't being considered
    feature_columns = ['Last3_FP_Avg', 'Last5_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg']
    feature_columns += [col for col in refined_most_recent_games.columns if col.startswith('CLUSTER_')]

    # Reorder columns to match the training data
    refined_most_recent_games = refined_most_recent_games[feature_columns]

    # Predict
    features = refined_most_recent_games.to_numpy()
    predictions = rf.predict(features)
    refined_most_recent_games['Predicted_FP'] = predictions
    print(refined_most_recent_games)

df = scrapeData()
runModel(df)