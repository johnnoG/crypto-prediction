from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Boolean, Text, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class UserWatchlist(Base):
    """Model for user's cryptocurrency watchlist."""

    id: Mapped[int] = mapped_column(primary_key=True)

    # Foreign key to user
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)

    # Crypto details
    crypto_symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    crypto_name: Mapped[str] = mapped_column(String(100), nullable=False)
    crypto_id: Mapped[str] = mapped_column(String(50), nullable=False)  # CoinGecko ID

    # User notes and preferences
    notes: Mapped[Optional[str]] = mapped_column(Text)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    notification_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship
    # user: Mapped["User"] = relationship(back_populates="watchlist")

    # Unique constraint - each user can only have one entry per crypto
    __table_args__ = (
        UniqueConstraint('user_id', 'crypto_symbol', name='_user_crypto_watchlist'),
    )

    def __repr__(self) -> str:
        return f"<UserWatchlist(id={self.id}, user_id={self.user_id}, crypto_symbol={self.crypto_symbol})>"
