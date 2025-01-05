from postgres.config import engine, Base
from postgres.models import AdvancedPlayerStats

Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
