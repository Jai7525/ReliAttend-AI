"""
ReliAttend AI — Database Engine, Session, and Base Model Configuration
Configured for SQLite prototype with seamless PostgreSQL readiness.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import get_settings

settings = get_settings()

# For SQLite, check_same_thread=False is needed for multi-threaded FastAPI workers.
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session per request.
    Automatically closes session after request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables registered with Base metadata."""
    # Import all models to ensure they are registered with Base.metadata
    import app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
