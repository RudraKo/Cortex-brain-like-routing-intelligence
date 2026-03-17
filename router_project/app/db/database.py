from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

import os

# Use /tmp on Vercel (where typical filesystem is read-only)
DB_PATH = "/tmp/router.db" if os.getenv("VERCEL") else "./router.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
