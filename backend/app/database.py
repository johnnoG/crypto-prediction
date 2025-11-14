"""
Database Connection and Session Management

Provides SQLAlchemy engine, session factory, and base model class.
Implements connection pooling and proper resource management.
"""

from typing import Generator
from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import Pool

from app.config import settings

# Naming convention for constraints (helps with migrations)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

# Metadata with naming convention
metadata = MetaData(naming_convention=NAMING_CONVENTION)

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    settings.database_url_str,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_timeout=settings.database_pool_timeout,
    pool_recycle=settings.database_pool_recycle,
    pool_pre_ping=True,  # Verify connections before using
    echo=settings.debug,  # Log SQL statements in debug mode
)


@event.listens_for(Pool, "connect")
def set_postgres_pragma(dbapi_connection, connection_record):
    """
    Set PostgreSQL connection parameters on new connections.

    This ensures optimal settings for each connection in the pool.
    """
    cursor = dbapi_connection.cursor()
    # Set statement timeout to prevent long-running queries
    cursor.execute("SET statement_timeout = '30s'")
    cursor.close()


# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Prevent lazy loading errors after commit
)

# Base class for models
Base = declarative_base(metadata=metadata)


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.

    Yields a SQLAlchemy session and ensures proper cleanup.
    Use this in FastAPI dependency injection.

    Example:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db() -> None:
    """
    Initialize database connection.

    Creates all tables defined in models.
    In production, use Alembic migrations instead.
    """
    # Import all models here to ensure they're registered with Base
    # from app.models import cryptocurrency, market_data, predictions

    # Create tables (use Alembic in production)
    Base.metadata.create_all(bind=engine)


async def close_db() -> None:
    """
    Close database connections.

    Disposes of the connection pool on application shutdown.
    """
    engine.dispose()
