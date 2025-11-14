"""
Prediction Model

Stores model predictions for cryptocurrency prices.
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
    CheckConstraint,
    JSON,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Prediction(Base):
    """
    Prediction model for storing cryptocurrency price predictions.

    Stores predictions made by ML models including predicted values,
    confidence scores, and model metadata.

    Attributes:
        id: Primary key
        cryptocurrency_id: Foreign key to Cryptocurrency
        prediction_timestamp: When the prediction was made (UTC)
        target_timestamp: The future timestamp being predicted
        predicted_price: Predicted price in USD
        confidence_score: Model confidence (0-1)
        model_name: Name/version of the model used
        model_version: Version identifier of the model
        features_used: JSON object of features used for prediction
        actual_price: Actual price at target_timestamp (filled later)
        error: Prediction error once actual_price is known
        created_at: Record creation timestamp
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    cryptocurrency_id = Column(
        Integer,
        ForeignKey("cryptocurrencies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    target_timestamp = Column(DateTime, nullable=False, index=True)
    predicted_price = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    features_used = Column(JSON, nullable=True)
    actual_price = Column(Float, nullable=True)
    error = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    cryptocurrency = relationship("Cryptocurrency", back_populates="predictions")

    # Constraints
    __table_args__ = (
        # Performance indexes
        Index("idx_prediction_crypto_target", "cryptocurrency_id", "target_timestamp"),
        Index(
            "idx_prediction_crypto_prediction_time",
            "cryptocurrency_id",
            "prediction_timestamp",
        ),
        Index("idx_prediction_model", "model_name", "model_version"),
        # Data validation constraints
        CheckConstraint(
            "predicted_price >= 0", name="ck_prediction_predicted_price_positive"
        ),
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_prediction_confidence_range",
        ),
        CheckConstraint(
            "actual_price IS NULL OR actual_price >= 0",
            name="ck_prediction_actual_price_positive",
        ),
        CheckConstraint(
            "target_timestamp > prediction_timestamp",
            name="ck_prediction_target_after_prediction",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, crypto_id={self.cryptocurrency_id}, "
            f"target={self.target_timestamp}, predicted={self.predicted_price})>"
        )

    def calculate_error(self) -> None:
        """
        Calculate prediction error once actual price is known.

        Updates the error field with the absolute percentage error.
        """
        if self.actual_price is not None and self.predicted_price > 0:
            self.error = abs(
                (self.actual_price - self.predicted_price) / self.predicted_price * 100
            )
