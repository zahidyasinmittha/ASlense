# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file

DATABASE_URL = os.getenv('DATABASE_URL')      # update to Postgres later if needed

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}   # SQLite-only switch
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def get_db():
    """Yield a SQLAlchemy session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
