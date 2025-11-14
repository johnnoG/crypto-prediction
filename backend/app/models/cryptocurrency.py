"""
Cryptocurrency Model

Represents cryptocurrency metadata and information.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Index
from sqlalchemy.orm import relationship

from app.database import Base


class Cryptocurrency(Base):
    """
    Cryptocurrency entity model.

    Stores metadata about cryptocurrencies including symbol, name,
    and tracking status.

    Attributes:
        id: Primary key
        symbol: Trading symbol (e.g., BTC, ETH)
        name: Full name (e.g., Bitcoin, Ethereum)
        coin_gecko_id: CoinGecko API identifier
        is_active: Whether this cryptocurrency is actively tracked
        created_at: Record creation timestamp
        updated_at: Record last update timestamp
    """

    __tablename__ = "cryptocurrencies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    coin_gecko_id = Column(String(100), unique=True, nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    market_data = relationship(
        "MarketData", back_populates="cryptocurrency", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="cryptocurrency", cascade="all, delete-orphan"
    )

    # Indexes for performance
    __table_args__ = (Index("idx_crypto_symbol_active", "symbol", "is_active"),)

    def __repr__(self) -> str:
        return f"<Cryptocurrency(id={self.id}, symbol='{self.symbol}', name='{self.name}')>"
