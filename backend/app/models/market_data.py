"""
Market Data Model

Represents historical and real-time cryptocurrency market data.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Index,
    BigInteger,
    CheckConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class MarketData(Base):
    """
    Market data time series model.

    Stores OHLCV (Open, High, Low, Close, Volume) data and additional
    market metrics for cryptocurrencies.

    Attributes:
        id: Primary key
        cryptocurrency_id: Foreign key to Cryptocurrency
        timestamp: Data point timestamp (UTC)
        open_price: Opening price in USD
        high_price: Highest price in USD
        low_price: Lowest price in USD
        close_price: Closing price in USD
        volume: Trading volume
        market_cap: Market capitalization in USD
        data_source: Source of the data (e.g., 'coingecko', 'binance')
        created_at: Record creation timestamp
    """

    __tablename__ = "market_data"

    id = Column(BigInteger, primary_key=True, index=True)
    cryptocurrency_id = Column(
        Integer,
        ForeignKey("cryptocurrencies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    market_cap = Column(Float, nullable=True)
    data_source = Column(String(50), nullable=False, default="unknown")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    cryptocurrency = relationship("Cryptocurrency", back_populates="market_data")

    # Constraints
    __table_args__ = (
        # Ensure unique timestamp per cryptocurrency
        Index(
            "idx_market_data_crypto_timestamp",
            "cryptocurrency_id",
            "timestamp",
            unique=True,
        ),
        # Performance indexes
        Index("idx_market_data_timestamp", "timestamp"),
        Index("idx_market_data_crypto_created", "cryptocurrency_id", "created_at"),
        # Data validation constraints
        CheckConstraint("open_price >= 0", name="ck_market_data_open_price_positive"),
        CheckConstraint("high_price >= 0", name="ck_market_data_high_price_positive"),
        CheckConstraint("low_price >= 0", name="ck_market_data_low_price_positive"),
        CheckConstraint("close_price >= 0", name="ck_market_data_close_price_positive"),
        CheckConstraint("volume >= 0", name="ck_market_data_volume_positive"),
        CheckConstraint("high_price >= low_price", name="ck_market_data_high_gte_low"),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketData(id={self.id}, crypto_id={self.cryptocurrency_id}, "
            f"timestamp={self.timestamp}, close={self.close_price})>"
        )
