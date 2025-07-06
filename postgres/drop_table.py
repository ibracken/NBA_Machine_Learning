import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import engine
from models import BoxScores

# Drop the table
BoxScores.__table__.drop(engine, checkfirst=True)
print("BoxScores table dropped successfully!") 