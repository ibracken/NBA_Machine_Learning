from config import engine, Base
from models import AdvancedPlayerStats

Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
