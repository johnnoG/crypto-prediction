"""
Database Seed Script

Populates the database with initial cryptocurrency data for development/testing.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.database import SessionLocal, engine, Base
from app.models import Cryptocurrency, MarketData


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")


def seed_cryptocurrencies(db: Session) -> dict:
    """
    Seed cryptocurrency data.

    Args:
        db: Database session

    Returns:
        Dictionary mapping symbols to cryptocurrency objects
    """
    print("Seeding cryptocurrencies...")

    cryptos = [
        {
            "symbol": "BTC",
            "name": "Bitcoin",
            "coin_gecko_id": "bitcoin",
            "is_active": True,
        },
        {
            "symbol": "ETH",
            "name": "Ethereum",
            "coin_gecko_id": "ethereum",
            "is_active": True,
        },
        {
            "symbol": "BNB",
            "name": "Binance Coin",
            "coin_gecko_id": "binancecoin",
            "is_active": True,
        },
        {
            "symbol": "SOL",
            "name": "Solana",
            "coin_gecko_id": "solana",
            "is_active": True,
        },
        {
            "symbol": "ADA",
            "name": "Cardano",
            "coin_gecko_id": "cardano",
            "is_active": True,
        },
    ]

    crypto_objects = {}
    for crypto_data in cryptos:
        # Check if already exists
        existing = db.query(Cryptocurrency).filter_by(
            symbol=crypto_data["symbol"]
        ).first()

        if existing:
            print(f"  - {crypto_data['symbol']} already exists, skipping...")
            crypto_objects[crypto_data["symbol"]] = existing
            continue

        crypto = Cryptocurrency(**crypto_data)
        db.add(crypto)
        crypto_objects[crypto_data["symbol"]] = crypto
        print(f"  - Added {crypto_data['symbol']} ({crypto_data['name']})")

    db.commit()
    print(f"Seeded {len(crypto_objects)} cryptocurrencies.")
    return crypto_objects


def seed_market_data(db: Session, cryptos: dict):
    """
    Seed sample market data.

    Creates sample OHLCV data for the last 30 days.

    Args:
        db: Database session
        cryptos: Dictionary of cryptocurrency objects
    """
    print("Seeding market data...")

    # Sample prices (simplified for demo)
    base_prices = {
        "BTC": 45000,
        "ETH": 2500,
        "BNB": 300,
        "SOL": 100,
        "ADA": 0.50,
    }

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    current_date = start_date

    records_added = 0

    while current_date <= end_date:
        for symbol, crypto in cryptos.items():
            # Check if data already exists
            existing = db.query(MarketData).filter_by(
                cryptocurrency_id=crypto.id,
                timestamp=current_date
            ).first()

            if existing:
                continue

            base_price = base_prices.get(symbol, 100)

            # Simulate price fluctuation (±5%)
            import random
            price_variation = random.uniform(-0.05, 0.05)
            close_price = base_price * (1 + price_variation)
            open_price = base_price * (1 + random.uniform(-0.03, 0.03))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
            volume = random.uniform(1000000, 10000000)
            market_cap = close_price * 1000000  # Simplified

            market_data = MarketData(
                cryptocurrency_id=crypto.id,
                timestamp=current_date,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
                market_cap=market_cap,
                data_source="seed_script",
            )
            db.add(market_data)
            records_added += 1

        current_date += timedelta(days=1)

    db.commit()
    print(f"Seeded {records_added} market data records.")


def main():
    """Main seeding function."""
    print("\n" + "=" * 60)
    print("Database Seeding Script")
    print("=" * 60 + "\n")

    # Create tables
    create_tables()

    # Create database session
    db = SessionLocal()

    try:
        # Seed data
        cryptos = seed_cryptocurrencies(db)
        seed_market_data(db, cryptos)

        print("\n" + "=" * 60)
        print("Seeding completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nError during seeding: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
