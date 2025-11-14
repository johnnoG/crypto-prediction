"""
Pytest Configuration and Fixtures

Provides shared fixtures for testing including:
- Test database setup/teardown
- Test Redis cache
- FastAPI test client
- Sample data fixtures
"""

import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import fakeredis

from app.main import app
from app.database import Base, get_db
from app.cache import cache_manager, CacheManager
from app.models import Cryptocurrency, MarketData, Prediction
from datetime import datetime, timedelta


# Test Database Setup
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """
    Create a fresh database for each test.

    Yields:
        Session: SQLAlchemy database session
    """
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_cache() -> Generator[CacheManager, None, None]:
    """
    Create a fake Redis cache for testing.

    Yields:
        CacheManager: Cache manager with fake Redis backend
    """
    # Create fake Redis
    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    # Replace cache manager's Redis instance
    original_redis = cache_manager._redis
    cache_manager._redis = fake_redis

    try:
        yield cache_manager
    finally:
        # Cleanup
        fake_redis.flushall()
        cache_manager._redis = original_redis


@pytest.fixture(scope="function")
def client(db: Session, test_cache: CacheManager) -> Generator[TestClient, None, None]:
    """
    Create a FastAPI test client with test database.

    Args:
        db: Test database session
        test_cache: Test cache manager

    Yields:
        TestClient: FastAPI test client
    """

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# Sample Data Fixtures


@pytest.fixture
def sample_cryptocurrency(db: Session) -> Cryptocurrency:
    """Create a sample cryptocurrency for testing."""
    crypto = Cryptocurrency(
        symbol="BTC", name="Bitcoin", coin_gecko_id="bitcoin", is_active=True
    )
    db.add(crypto)
    db.commit()
    db.refresh(crypto)
    return crypto


@pytest.fixture
def sample_cryptocurrencies(db: Session) -> list[Cryptocurrency]:
    """Create multiple sample cryptocurrencies."""
    cryptos = [
        Cryptocurrency(
            symbol="BTC", name="Bitcoin", coin_gecko_id="bitcoin", is_active=True
        ),
        Cryptocurrency(
            symbol="ETH", name="Ethereum", coin_gecko_id="ethereum", is_active=True
        ),
        Cryptocurrency(
            symbol="SOL", name="Solana", coin_gecko_id="solana", is_active=True
        ),
    ]
    db.add_all(cryptos)
    db.commit()
    for crypto in cryptos:
        db.refresh(crypto)
    return cryptos


@pytest.fixture
def sample_market_data(
    db: Session, sample_cryptocurrency: Cryptocurrency
) -> MarketData:
    """Create sample market data."""
    market_data = MarketData(
        cryptocurrency_id=sample_cryptocurrency.id,
        timestamp=datetime.utcnow(),
        open_price=45000.0,
        high_price=46000.0,
        low_price=44000.0,
        close_price=45500.0,
        volume=1000000000.0,
        market_cap=900000000000.0,
        data_source="test",
    )
    db.add(market_data)
    db.commit()
    db.refresh(market_data)
    return market_data


@pytest.fixture
def sample_market_data_series(
    db: Session, sample_cryptocurrency: Cryptocurrency
) -> list[MarketData]:
    """Create a time series of market data."""
    base_time = datetime.utcnow() - timedelta(days=7)
    data_points = []

    for i in range(7):
        market_data = MarketData(
            cryptocurrency_id=sample_cryptocurrency.id,
            timestamp=base_time + timedelta(days=i),
            open_price=45000.0 + i * 100,
            high_price=46000.0 + i * 100,
            low_price=44000.0 + i * 100,
            close_price=45500.0 + i * 100,
            volume=1000000000.0,
            market_cap=900000000000.0,
            data_source="test",
        )
        data_points.append(market_data)

    db.add_all(data_points)
    db.commit()
    for data in data_points:
        db.refresh(data)
    return data_points


@pytest.fixture
def sample_prediction(db: Session, sample_cryptocurrency: Cryptocurrency) -> Prediction:
    """Create sample prediction."""
    now = datetime.utcnow()
    prediction = Prediction(
        cryptocurrency_id=sample_cryptocurrency.id,
        prediction_timestamp=now,
        target_timestamp=now + timedelta(hours=24),
        predicted_price=46000.0,
        confidence_score=0.85,
        model_name="LSTM-v1",
        model_version="1.0.0",
        features_used={"feature1": 1.0, "feature2": 2.0},
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


# Cleanup fixtures


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before each test."""
    yield
    # Cleanup happens in db fixture


@pytest.fixture(autouse=True)
def reset_cache(test_cache: CacheManager):
    """Reset cache before each test."""
    if test_cache._redis:
        test_cache._redis.flushall()
    yield
    if test_cache._redis:
        test_cache._redis.flushall()
