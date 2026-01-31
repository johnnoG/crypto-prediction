from __future__ import annotations

from datetime import datetime
from typing import Optional
from enum import Enum as PyEnum

from sqlalchemy import String, DateTime, Boolean, Text, Integer, Float, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class AlertType(PyEnum):
    """Types of alerts that can be set."""
    PRICE_TARGET = "price_target"
    FORECAST_CHANGE = "forecast_change"
    VOLATILITY = "volatility"


class AlertStatus(PyEnum):
    """Status of an alert."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class UserAlert(Base):
    """Model for user price alerts and forecast notifications."""

    id: Mapped[int] = mapped_column(primary_key=True)

    # Foreign key to user
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)

    # Alert details
    crypto_symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    crypto_name: Mapped[str] = mapped_column(String(100), nullable=False)
    alert_type: Mapped[AlertType] = mapped_column(Enum(AlertType), nullable=False)

    # Alert conditions
    target_price: Mapped[Optional[float]] = mapped_column(Float)
    condition: Mapped[Optional[str]] = mapped_column(String(20))  # "above", "below", "reaches"

    # Alert metadata
    status: Mapped[AlertStatus] = mapped_column(Enum(AlertStatus), default=AlertStatus.ACTIVE)
    message: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationship
    # user: Mapped["User"] = relationship(back_populates="alerts")

    def __repr__(self) -> str:
        return f"<UserAlert(id={self.id}, user_id={self.user_id}, crypto_symbol={self.crypto_symbol}, alert_type={self.alert_type})>"
