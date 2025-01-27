from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:postgronner34@localhost:5432/Sonance"
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        print("Database connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
