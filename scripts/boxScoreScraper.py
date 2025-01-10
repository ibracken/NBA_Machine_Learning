from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time
from postgres.config import SessionLocal
from postgres.models import BoxScores, ClusteredPlayers
from unidecode import unidecode

def normalize_name(name):
    return unidecode(name.strip().lower())

def safe_float(value):
    try:
        return float(value)
    except ValueError:
        return None

def scrapeBoxScores():
    # Advanced Stats
    driver = webdriver.Chrome()

    url_advanced = r"https://www.nba.com/stats/players/boxscores"

    driver.get(url_advanced)


    # Found using copy full xml path on the dropdown menu(inspect element)s
    select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))


    select.select_by_index(0)

    # Lots of data so it helps to give selenium time to load the page
    time.sleep(10)

    src = driver.page_source 
    parser = BeautifulSoup(src, 'lxml')
    table = parser.find("div", attrs = {"class": "Crom_container__C45Ti crom-container"})
    headers = table.findAll('th')
    headerlist = [h.text.strip() for h in headers]
    headerlist1 = [a for a in headerlist if not 'RANK' in a]

    rows = table.findAll('tr')[1:]
    player_box_scores = [[td.getText().strip() for td in rows[i].findAll('td')] for i in range(len(rows))]
    df = pd.DataFrame(player_box_scores, columns=headerlist1)

    df['PLAYER'] = df['PLAYER'].apply(normalize_name)

    df['FP'] = pd.to_numeric(df['FP'], errors='coerce')

    # Convert 'GAME DATE' to datetime if it's not already
    df['GAME DATE'] = pd.to_datetime(df['GAME DATE'], format='%m/%d/%Y')

    # Sort the DataFrame by PLAYER and GAME DATE
    df = df.sort_values(by=['PLAYER', 'GAME DATE'], ascending = [True, True])

    # Takes from postgres session
    session = SessionLocal()

    players = session.query(ClusteredPlayers).all()
    # Convert ORM objects to a list of dictionaries
    data = []
    for player in players:
        player_dict = {column.name: getattr(player, column.name) for column in player.__table__.columns}
        data.append(player_dict)

    dataset_clusters = pd.DataFrame(data)

    dataset_clusters = dataset_clusters.drop('id', axis=1)

    clusterDict = {}

    # Map the 'Cluster' column from clusterdf to the dataset based on 'Player'
    clusterDict = dataset_clusters.set_index('PLAYER')['CLUSTER'].to_dict()

    # Map the 'CLUSTER' column in the dataset
    df['CLUSTER'] = df['PLAYER'].map(clusterDict)


    # Define the window sizes
    windows = [3, 5, 7]

    for window in windows:
        # Calculate the rolling average for each player
        df[f'Last{window}_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        # May add this later
        # df[f'Last{window}_Min'] = df.groupby('PLAYER')['MIN'].transform(lambda x: x.rolling(window, min_periods=1).mean())

    # Calculate Season Average FP for each player
    df['Season_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )


    for _, row in df.iterrows():
        existing_record = (
            session.query(BoxScores)
            .filter_by(PLAYER=row['PLAYER'], GAME_DATE=row['GAME DATE'])
            .one_or_none()
        )
        if existing_record:
            # Update the existing record with new data
            existing_record.TEAM = row['TEAM']
            existing_record.MATCH_UP = row['MATCH UP']
            existing_record.WL = row['W/L']
            existing_record.MIN = safe_float(row['MIN'])
            existing_record.PTS = safe_float(row['PTS'])
            existing_record.FGM = safe_float(row['FGM'])
            existing_record.FGA = safe_float(row['FGA'])
            existing_record.FG_PERCENT = safe_float(row['FG%'])
            existing_record.THREE_PM = safe_float(row['3PM'])
            existing_record.THREE_PA = safe_float(row['3PA'])
            existing_record.THREE_PERCENT = safe_float(row['3P%'])
            existing_record.FTM = safe_float(row['FTM'])
            existing_record.FTA = safe_float(row['FTA'])
            existing_record.FT_PERCENT = safe_float(row['FT%'])
            existing_record.OREB = safe_float(row['OREB'])
            existing_record.DREB = safe_float(row['DREB'])
            existing_record.REB = safe_float(row['REB'])
            existing_record.AST = safe_float(row['AST'])
            existing_record.STL = safe_float(row['STL'])
            existing_record.BLK = safe_float(row['BLK'])
            existing_record.TOV = safe_float(row['TOV'])
            existing_record.PF = safe_float(row['PF'])
            existing_record.PLUS_MINUS = safe_float(row['+/-'])
            existing_record.FP = safe_float(row['FP'])
            existing_record.CLUSTER = row['CLUSTER']
            existing_record.Last3_FP_Avg = safe_float(row['Last3_FP_Avg'])
            existing_record.Last5_FP_Avg = safe_float(row['Last5_FP_Avg'])
            existing_record.Last7_FP_Avg = safe_float(row['Last7_FP_Avg'])
            existing_record.Season_FP_Avg = safe_float(row['Season_FP_Avg'])
        else:
            # Create a new BoxScores object if no existing record is found
            new_record = BoxScores(
                PLAYER=row['PLAYER'],
                TEAM=row['TEAM'],
                MATCH_UP=row['MATCH UP'],
                GAME_DATE=row['GAME DATE'],
                WL=row['W/L'],
                MIN=safe_float(row['MIN']),
                PTS=safe_float(row['PTS']),
                FGM=safe_float(row['FGM']),
                FGA=safe_float(row['FGA']),
                FG_PERCENT=safe_float(row['FG%']),
                THREE_PM=safe_float(row['3PM']),
                THREE_PA=safe_float(row['3PA']),
                THREE_PERCENT=safe_float(row['3P%']),
                FTM=safe_float(row['FTM']),
                FTA=safe_float(row['FTA']),
                FT_PERCENT=safe_float(row['FT%']),
                OREB=safe_float(row['OREB']),
                DREB=safe_float(row['DREB']),
                REB=safe_float(row['REB']),
                AST=safe_float(row['AST']),
                STL=safe_float(row['STL']),
                BLK=safe_float(row['BLK']),
                TOV=safe_float(row['TOV']),
                PF=safe_float(row['PF']),
                PLUS_MINUS=safe_float(row['+/-']),
                FP=safe_float(row['FP']),
                CLUSTER=row['CLUSTER'],
                Last3_FP_Avg=safe_float(row['Last3_FP_Avg']),
                Last5_FP_Avg=safe_float(row['Last5_FP_Avg']),
                Last7_FP_Avg=safe_float(row['Last7_FP_Avg']),
                Season_FP_Avg=safe_float(row['Season_FP_Avg']),
            )
            session.add(new_record)
    session.commit()

    session.close()

scrapeBoxScores()