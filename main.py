# Call all this shit at once
# postgers port: 5433
# username: postgres
# password: postgres

# clusterScraper.py

# nbaClustering.ipynb

# boxScoreScraper.py

# nbaSupervisedLearningClusters.ipynb

# dailyPredictionsScraper.py

from postgres.config import SessionLocal
from postgres.models import BoxScores, DailyPlayerPredictions # Import your model

# Start a session
session = SessionLocal()

# Define the player name to search for
player_name = "alec burks"

# Query all rows where PLAYER matches the specified name
matching_records = (
    session.query(BoxScores)
    .filter(BoxScores.PLAYER == player_name)
    .all()
)

# Process or print the results
for record in matching_records:
    print(f"PLAYER: {record.PLAYER}, GAME_DATE: {record.GAME_DATE}")


matching_records = (
    session.query(DailyPlayerPredictions)
    .filter(DailyPlayerPredictions.PLAYER == player_name)
    .all()
)

# Process or print the results
for record in matching_records:
    print(f"DPS: PLAYER: {record.PLAYER}, GAME_DATE: {record.GAME_DATE}")

# Close the session
session.close()
