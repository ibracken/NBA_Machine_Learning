from sqlalchemy import create_engine

DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/nba_predictions"

engine = create_engine(DATABASE_URL)

# Test the connection
with engine.connect() as connection:
    print("Connection successful!")
