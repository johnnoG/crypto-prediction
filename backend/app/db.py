from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

try:
    from config import get_settings
except ImportError:
    from config import get_settings


_settings = get_settings()

def _build_database_url() -> str:
    if _settings.database_url:
        return _settings.database_url
    # Dev fallback to SQLite file
    return "sqlite:///./dev.db"


_engine = create_engine(_build_database_url(), echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_ping() -> bool:
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


