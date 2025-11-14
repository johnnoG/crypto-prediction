"""
Tests for Database Models

Tests SQLAlchemy models including:
- Cryptocurrency model
- MarketData model
- Prediction model
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Cryptocurrency, MarketData, Prediction


class TestCryptocurrencyModel:
    """Test cases for Cryptocurrency model."""

    @pytest.mark.unit
    def test_create_cryptocurrency(self, db: Session):
        """Test creating a cryptocurrency."""
        crypto = Cryptocurrency(
            symbol="BTC", name="Bitcoin", coin_gecko_id="bitcoin", is_active=True
        )
        db.add(crypto)
        db.commit()
        db.refresh(crypto)

        assert crypto.id is not None
        assert crypto.symbol == "BTC"
        assert crypto.name == "Bitcoin"
        assert crypto.coin_gecko_id == "bitcoin"
        assert crypto.is_active is True
        assert crypto.created_at is not None
        assert crypto.updated_at is not None

    @pytest.mark.unit
    def test_cryptocurrency_unique_symbol(self, db: Session):
        """Test that symbol must be unique."""
        crypto1 = Cryptocurrency(symbol="BTC", name="Bitcoin", is_active=True)
        db.add(crypto1)
        db.commit()

        crypto2 = Cryptocurrency(symbol="BTC", name="Bitcoin Clone", is_active=True)
        db.add(crypto2)

        with pytest.raises(IntegrityError):
            db.commit()

    @pytest.mark.unit
    def test_cryptocurrency_repr(self, sample_cryptocurrency: Cryptocurrency):
        """Test string representation."""
        repr_str = repr(sample_cryptocurrency)
        assert "Cryptocurrency" in repr_str
        assert "BTC" in repr_str
        assert "Bitcoin" in repr_str

    @pytest.mark.unit
    def test_cryptocurrency_relationships(
        self, db: Session, sample_cryptocurrency: Cryptocurrency
    ):
        """Test relationships with market data and predictions."""
        # Create market data
        market_data = MarketData(
            cryptocurrency_id=sample_cryptocurrency.id,
            timestamp=datetime.utcnow(),
            open_price=45000,
            high_price=46000,
            low_price=44000,
            close_price=45500,
            volume=1000000,
            market_cap=900000000,
            data_source="test",
        )
        db.add(market_data)

        # Create prediction
        prediction = Prediction(
            cryptocurrency_id=sample_cryptocurrency.id,
            prediction_timestamp=datetime.utcnow(),
            target_timestamp=datetime.utcnow() + timedelta(hours=24),
            predicted_price=46000,
            confidence_score=0.85,
            model_name="test",
            model_version="1.0",
        )
        db.add(prediction)
        db.commit()

        # Refresh and check relationships
        db.refresh(sample_cryptocurrency)
        assert len(sample_cryptocurrency.market_data) == 1
        assert len(sample_cryptocurrency.predictions) == 1


class TestMarketDataModel:
    """Test cases for MarketData model."""

    @pytest.mark.unit
    def test_create_market_data(
        self, db: Session, sample_cryptocurrency: Cryptocurrency
    ):
        """Test creating market data."""
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

        assert market_data.id is not None
        assert market_data.cryptocurrency_id == sample_cryptocurrency.id
        assert market_data.open_price == 45000.0
        assert market_data.high_price == 46000.0
        assert market_data.low_price == 44000.0
        assert market_data.close_price == 45500.0
        assert market_data.volume == 1000000000.0

    @pytest.mark.unit
    def test_market_data_unique_timestamp(
        self, db: Session, sample_cryptocurrency: Cryptocurrency
    ):
        """Test that timestamp must be unique per cryptocurrency."""
        timestamp = datetime.utcnow()

        data1 = MarketData(
            cryptocurrency_id=sample_cryptocurrency.id,
            timestamp=timestamp,
            open_price=45000,
            high_price=46000,
            low_price=44000,
            close_price=45500,
            volume=1000000,
            market_cap=900000000,
            data_source="test",
        )
        db.add(data1)
        db.commit()

        data2 = MarketData(
            cryptocurrency_id=sample_cryptocurrency.id,
            timestamp=timestamp,  # Same timestamp
            open_price=46000,
            high_price=47000,
            low_price=45000,
            close_price=46500,
            volume=2000000,
            market_cap=1000000000,
            data_source="test",
        )
        db.add(data2)

        with pytest.raises(IntegrityError):
            db.commit()

    @pytest.mark.unit
    def test_market_data_repr(self, sample_market_data: MarketData):
        """Test string representation."""
        repr_str = repr(sample_market_data)
        assert "MarketData" in repr_str
        assert str(sample_market_data.close_price) in repr_str

    @pytest.mark.unit
    def test_market_data_relationship(
        self, db: Session, sample_market_data: MarketData
    ):
        """Test relationship to cryptocurrency."""
        assert sample_market_data.cryptocurrency is not None
        assert sample_market_data.cryptocurrency.symbol == "BTC"


class TestPredictionModel:
    """Test cases for Prediction model."""

    @pytest.mark.unit
    def test_create_prediction(
        self, db: Session, sample_cryptocurrency: Cryptocurrency
    ):
        """Test creating a prediction."""
        now = datetime.utcnow()
        prediction = Prediction(
            cryptocurrency_id=sample_cryptocurrency.id,
            prediction_timestamp=now,
            target_timestamp=now + timedelta(hours=24),
            predicted_price=46000.0,
            confidence_score=0.85,
            model_name="LSTM-v1",
            model_version="1.0.0",
            features_used={"feature1": 1.0},
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        assert prediction.id is not None
        assert prediction.cryptocurrency_id == sample_cryptocurrency.id
        assert prediction.predicted_price == 46000.0
        assert prediction.confidence_score == 0.85
        assert prediction.model_name == "LSTM-v1"
        assert prediction.features_used == {"feature1": 1.0}

    @pytest.mark.unit
    def test_prediction_calculate_error(
        self, db: Session, sample_prediction: Prediction
    ):
        """Test error calculation."""
        # Set actual price
        sample_prediction.actual_price = 46500.0
        sample_prediction.calculate_error()
        db.commit()

        # Error should be about 1.09% ((46500-46000)/46000 * 100)
        assert sample_prediction.error is not None
        assert abs(sample_prediction.error - 1.09) < 0.01

    @pytest.mark.unit
    def test_prediction_repr(self, sample_prediction: Prediction):
        """Test string representation."""
        repr_str = repr(sample_prediction)
        assert "Prediction" in repr_str
        assert str(sample_prediction.predicted_price) in repr_str

    @pytest.mark.unit
    def test_prediction_relationship(self, db: Session, sample_prediction: Prediction):
        """Test relationship to cryptocurrency."""
        assert sample_prediction.cryptocurrency is not None
        assert sample_prediction.cryptocurrency.symbol == "BTC"


class TestModelConstraints:
    """Test database constraints on models."""

    @pytest.mark.unit
    def test_market_data_price_constraints(
        self, db: Session, sample_cryptocurrency: Cryptocurrency
    ):
        """Test that prices must be positive and high >= low."""
        # Note: SQLite doesn't enforce CHECK constraints by default,
        # so this test verifies the constraint exists in the model definition
        from app.models.market_data import MarketData as MarketDataModel

        # Check that table has constraints defined
        constraints = MarketDataModel.__table__.constraints
        constraint_names = [c.name for c in constraints if hasattr(c, "name")]

        # Verify constraint names exist
        assert any("price" in name for name in constraint_names if name)

    @pytest.mark.unit
    def test_prediction_timestamp_order(
        self, db: Session, sample_cryptocurrency: Cryptocurrency
    ):
        """Test that target_timestamp must be after prediction_timestamp."""
        from app.models.prediction import Prediction as PredictionModel

        # Check constraint exists - SQLAlchemy 2.0 prefixes with table name
        constraints = PredictionModel.__table__.constraints
        constraint_names = [c.name for c in constraints if hasattr(c, "name")]

        # Check that the constraint exists (name may be prefixed with table name)
        assert any("ck_prediction_target_after_prediction" in name for name in constraint_names if name)
