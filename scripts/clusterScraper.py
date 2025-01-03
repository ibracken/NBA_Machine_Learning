from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time

# Advanced Stats
driver = webdriver.Chrome()

url_advanced = r"https://www.nba.com/stats/players/advanced"

driver.get(url_advanced)

# Found using copy full xml path on the dropdown menu(inspect element)s
select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))

select.select_by_index(0)

src = driver.page_source 
parser = BeautifulSoup(src, 'lxml')
table = parser.find("div", attrs = {"class": "Crom_container__C45Ti crom-container"})
headers = table.findAll('th')
headerlist = [h.text.strip() for h in headers[1:]]
# Filter out the ranking fields(no relevant stats here)
# If rank is not in headerlist, add it to headerist1
headerlist1 = [a for a in headerlist if not 'RANK' in a]
# same as this:
# for a in headerlist:
#     if not 'RANK' in a:
#         headerist1.append(a)

rows = table.findAll('tr')[1:]
player_stats = [[td.getText().strip() for td in rows[i].findAll('td')[1:]] for i in range(len(rows))]
# Cuts out some hidden columns
headerlist1 = headerlist1[:-5]

advanced_stats = pd.DataFrame(player_stats, columns = headerlist1)
driver.quit()


# Scoring Stats
driver = webdriver.Chrome()

url_scoring = r"https://www.nba.com/stats/players/scoring"

driver.get(url_scoring)

select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))

select.select_by_index(0)

src = driver.page_source 
parser = BeautifulSoup(src, 'lxml')
table = parser.find("div", attrs = {"class": "Crom_container__C45Ti crom-container"})
headers = table.findAll('th')
headerlist = [h.text.strip() for h in headers[1:]]
headerlist2 = [a for a in headerlist if not 'RANK' in a]

rows = table.findAll('tr')[1:]
player_stats_scoring = [[td.getText().strip() for td in rows[i].findAll('td')[1:]] for i in range(len(rows))]

# Test that the number of columns in the first row of player stats is equal to the number of headers
print(f"Number of headers: {len(headerlist2)}")
print(f"Number of columns in first row of player stats: {len(player_stats_scoring[0])}")


# Create DataFrame for Scoring Stats
scoring_stats = pd.DataFrame(player_stats_scoring, columns=headerlist2)
driver.quit()


# Defense Stats
driver = webdriver.Chrome()
url_defense = r"https://www.nba.com/stats/players/defense"
driver.get(url_defense)

# Handle cookie consent overlay if it exists
try:
    cookie_button = driver.find_element(By.XPATH, r"/html/body/div[3]/div[2]/div/div[1]/div/div[2]/div/button")
    cookie_button.click()
except:
    print("No cookie consent overlay found.")

# Getting the page set up right
data_sorting = driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table/thead/tr/th[5]")
data_sorting.click()
select = Select(driver.find_element(By.XPATH, r"/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"))
select.select_by_index(0)
src = driver.page_source 

# Extracting data  
parser = BeautifulSoup(src, 'lxml')
table = parser.find("div", attrs = {"class": "Crom_container__C45Ti crom-container"})
headers = table.findAll('th')
headerlist = [h.text.strip() for h in headers[1:]]
headerlist3 = [a for a in headerlist if not 'RANK' in a]
for i, header in enumerate(headerlist3):
    if header == "DREB%":
        headerlist3[i] = "DREB%TEAM"
        break
rows = table.findAll('tr')[1:]
player_stats_defense = [[td.getText().strip() for td in rows[i].findAll('td')[1:]] for i in range(len(rows))]
print(f"Number of headers: {len(headerlist3)}")
print(f"Number of columns in first row of defense stats: {len(player_stats_defense[0])}")

defense_stats = pd.DataFrame(player_stats_defense, columns=headerlist3)

# Normalize column names in both DataFrames
advanced_stats.columns = advanced_stats.columns.str.upper()
scoring_stats.columns = scoring_stats.columns.str.upper()
defense_stats.columns = defense_stats.columns.str.upper()
# Ensure No Duplicate Columns
merged_stats = advanced_stats.copy()
for col in scoring_stats.columns:
    if col not in merged_stats.columns:
        merged_stats[col] = scoring_stats[col]
for col in defense_stats.columns:
    if col not in merged_stats.columns:
        merged_stats[col] = defense_stats[col]

# Save to Excel
merged_stats.to_excel('data/NBAStats.xlsx', index=False)

print(merged_stats)
driver.quit()