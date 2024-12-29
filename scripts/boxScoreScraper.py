from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time

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

df['FP'] = pd.to_numeric(df['FP'], errors='coerce')

# Convert 'GAME DATE' to datetime if it's not already
df['GAME DATE'] = pd.to_datetime(df['GAME DATE'], format='%m/%d/%Y')

# Sort the DataFrame by PLAYER and GAME DATE
df = df.sort_values(by=['PLAYER', 'GAME DATE'], ascending = [True, True])

# Define the window sizes
windows = [3, 5, 7]

for window in windows:
    # Calculate the rolling average for each player
    df[f'Last{window}_FP_Avg'] = df.groupby('PLAYER')['FP'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df[f'Last{window}_Min'] = df.groupby('PLAYER')['MIN'].transform(lambda x: x.rolling(window, min_periods=1).mean())

# Calculate Season Average FP for each player
df['Season_FP_Avg'] = df.groupby('PLAYER')['FP'].transform('mean')

df.to_excel('data/boxScores.xlsx', index=False)
print(df)