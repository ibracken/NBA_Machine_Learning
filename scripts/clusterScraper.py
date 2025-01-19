from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time
from postgres.config import SessionLocal
from postgres.models import AdvancedPlayerStats
from unidecode import unidecode

numeric_columns = {
    "AGE": "AGE",
    "GP": "GP",
    "W": "W",
    "L": "L",
    "MIN": "MIN",
    "OFFRTG": "OFFRTG",
    "DEFRTG": "DEFRTG",
    "NETRTG": "NETRTG",
    "AST%": "AST_PERCENT",
    "AST/TO": "AST_TO",
    "AST RATIO": "AST_RATIO",
    "OREB%": "OREB_PERCENT",
    "DREB%": "DREB_PERCENT",
    "REB%": "REB_PERCENT",
    "TO RATIO": "TO_RATIO",
    "EFG%": "EFG_PERCENT",
    "TS%": "TS_PERCENT",
    "USG%": "USG_PERCENT",
    "PACE": "PACE",
    "PIE": "PIE",
    "POSS": "POSS",
    "%FGA2PT": "FGA2P_PERCENT",
    "%FGA3PT": "FGA3P_PERCENT",
    "%PTS2PT": "PTS2P_PERCENT",
    "%PTS2PT MR": "PTS2P_MR_PERCENT",
    "%PTS3PT": "PTS3P_PERCENT",
    "%PTSFBPS": "PTSFBPS_PERCENT",
    "%PTSFT": "PTSFT_PERCENT",
    "%PTSOFFTO": "PTS_OFFTO_PERCENT",
    "%PTSPITP": "PTSPITP_PERCENT",
    "2FGM%AST": "FG2M_AST_PERCENT",
    "2FGM%UAST": "FG2M_UAST_PERCENT",
    "3FGM%AST": "FG3M_AST_PERCENT",
    "3FGM%UAST": "FG3M_UAST_PERCENT",
    "FGM%AST": "FGM_AST_PERCENT",
    "FGM%UAST": "FGM_UAST_PERCENT",
    "DEF RTG": "DEF_RTG",
    "DREB": "DREB",
    "DREB%TEAM": "DREB_PERCENT_TEAM",
    "STL": "STL",
    "STL%": "STL_PERCENT",
    "BLK": "BLK",
    "%BLK": "BLK_PERCENT",
    "OPP PTSOFF TOV": "OPP_PTS_OFFTO",
    "OPP PTS2ND CHANCE": "OPP_PTS_2ND_CHANCE",
    "OPP PTSFB": "OPP_PTS_FB",
    "OPP PTSPAINT": "OPP_PTS_PAINT",
    "DEFWS": "DEFWS",
}

string_columns = {
    'TEAM': 'TEAM',
}



def safe_float(value):
    try:
        return float(value)
    except ValueError:
        return None

def normalize_name(name):
    return unidecode(name.strip().lower())



def scrape_and_insert_data(url, session, table_model):
    driver = webdriver.Chrome()
    driver.get(url)
    # Found using copy full xml path on the dropdown menu(inspect element)s
    select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))
    select.select_by_index(0)

    src = driver.page_source 
    parser = BeautifulSoup(src, 'lxml')
    table = parser.find("div", attrs = {"class": "Crom_container__C45Ti crom-container"})
    
    headers = table.findAll('th')
    headerlist = [h.text.strip().upper() for h in headers[1:]]
    headerlist = [a for a in headerlist if not 'RANK' in a]
    for i, header in enumerate(headerlist):
        if header == "DREB%" and url == "https://www.nba.com/stats/players/defense":
            headerlist[i] = "DREB%TEAM"
            break
    rows = table.findAll('tr')[1:]
    player_stats = [[td.getText().strip() for td in rows[i].findAll('td')[1:]] for i in range(len(rows))]
    if url == r"https://www.nba.com/stats/players/advanced":
        headerlist = headerlist[:-5]
    stats = pd.DataFrame(player_stats, columns = headerlist)


    for _, row in stats.iterrows():
        normalized_name = normalize_name(row["PLAYER"])
        existing_player = session.query(table_model).filter_by(PLAYER=normalized_name).first()

        # Update or Insert Logic
        if existing_player:
            for scraped_col, db_col in numeric_columns.items():
                if scraped_col in row:
                    setattr(existing_player, db_col, safe_float(row[scraped_col]))
            for scraped_col, db_col in string_columns.items():
                if scraped_col in row:
                    setattr(existing_player, db_col, row[scraped_col])
        else:
            player_data = {}

            player_data['PLAYER'] = normalized_name
            
            # Add string columns
            for scraped_col, db_col in string_columns.items():
                if scraped_col in row:
                    player_data[db_col] = row[scraped_col]
            
            # Add numeric columns
            for scraped_col, db_col in numeric_columns.items():
                if scraped_col in row:
                    player_data[db_col] = safe_float(row[scraped_col])
            
            # Create and add the new player
            player_stat = AdvancedPlayerStats(**player_data)
            session.add(player_stat)
    driver.quit()

def run_cluster_scraper():
    session = SessionLocal()
    
    # Scrape Advanced Stats
    scrape_and_insert_data(
        url="https://www.nba.com/stats/players/advanced",
        session=session,
        table_model=AdvancedPlayerStats
    )
    session.commit()
    
    # Scrape Scoring Stats
    scrape_and_insert_data(
        url="https://www.nba.com/stats/players/scoring",
        session=session,
        table_model=AdvancedPlayerStats
    )
    session.commit()
    
    # Scrape Defense Stats
    scrape_and_insert_data(
        url="https://www.nba.com/stats/players/defense",
        session=session,
        table_model=AdvancedPlayerStats
    )
    session.commit()

    session.close()

if __name__ == "__main__":
    run_cluster_scraper()