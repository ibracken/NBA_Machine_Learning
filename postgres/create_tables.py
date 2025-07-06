import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from postgres.config import engine, Base
from postgres.models import AdvancedPlayerStats, BoxScores, ClusteredPlayers, TestPlayerPredictions, DailyPlayerPredictions

Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
