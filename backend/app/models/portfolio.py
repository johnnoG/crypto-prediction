from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class PortfolioHolding(Base):
    """Model for user portfolio holdings."""

    id: Mapped[int] = mapped_column(primary_key=True)

    # Foreign key to user
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)

    # Crypto details
    crypto_id: Mapped[str] = mapped_column(String(50), nullable=False)       # CoinGecko ID
    crypto_symbol: Mapped[str] = mapped_column(String(20), nullable=False)   # e.g. "BTC"
    crypto_name: Mapped[str] = mapped_column(String(100), nullable=False)    # e.g. "Bitcoin"

    # Position
    amount: Mapped[float] = mapped_column(Float, nullable=False)             # quantity held
    avg_buy_price: Mapped[float] = mapped_column(Float, nullable=False)      # USD cost basis per unit

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # One entry per coin per user
    __table_args__ = (
        UniqueConstraint("user_id", "crypto_id", name="_user_crypto_portfolio"),
    )

    def __repr__(self) -> str:
        return (
            f"<PortfolioHolding(id={self.id}, user_id={self.user_id}, "
            f"crypto_symbol={self.crypto_symbol}, amount={self.amount})>"
        )
