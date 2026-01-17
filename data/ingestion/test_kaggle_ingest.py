#!/usr/bin/env python3
"""
Unit tests for Kaggle ingestion script.

Tests data parsing, normalization, and deduplication logic.
"""

import pytest
import polars as pl
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.ingestion.kaggle_ingest import KaggleIngestor


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    return pl.DataFrame({
        "Date": ["2023-01-01 00:00:00", "2023-01-01 01:00:00", "2023-01-01 02:00:00"],
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 106.0],
        "Volume": [1000.0, 1100.0, 1200.0],
    })


@pytest.fixture
def ingestor(tmp_path):
    """Create KaggleIngestor instance for testing."""
    return KaggleIngestor(tmp_path)


class TestKaggleIngestor:
    """Test cases for KaggleIngestor class."""

    def test_extract_symbol_from_filename(self, ingestor):
        """Test symbol extraction from filenames."""
        assert ingestor.extract_symbol_from_filename("BTC_data.csv") == "BTC"
        assert ingestor.extract_symbol_from_filename("bitcoin_prices.csv") == "BTC"
        assert ingestor.extract_symbol_from_filename("ETH_historical.csv") == "ETH"
        assert ingestor.extract_symbol_from_filename("unknown.csv") is None

    def test_normalize_ohlcv_data(self, ingestor, sample_ohlcv_data):
        """Test OHLCV data normalization."""
        normalized = ingestor.normalize_ohlcv_data(sample_ohlcv_data, "BTC")
        
        # Check required columns exist
        required_cols = ["ts", "open", "high", "low", "close", "volume", "symbol", "timeframe"]
        for col in required_cols:
            assert col in normalized.columns
        
        # Check data types
        assert normalized["open"].dtype == pl.Float64
        assert normalized["high"].dtype == pl.Float64
        assert normalized["low"].dtype == pl.Float64
        assert normalized["close"].dtype == pl.Float64
        assert normalized["volume"].dtype == pl.Float64
        
        # Check symbol and timeframe
        assert normalized["symbol"].unique().item() == "BTC"
        assert normalized["timeframe"].unique().item() == "1h"
        
        # Check row count
        assert len(normalized) == 3

    def test_normalize_ohlcv_data_with_missing_columns(self, ingestor):
        """Test normalization with missing columns."""
        incomplete_data = pl.DataFrame({
            "Date": ["2023-01-01 00:00:00"],
            "Open": [100.0],
            "Close": [104.0],
        })
        
        normalized = ingestor.normalize_ohlcv_data(incomplete_data, "ETH")
        
        # Should have all required columns with defaults
        assert "high" in normalized.columns
        assert "low" in normalized.columns
        assert "volume" in normalized.columns
        assert normalized["volume"].item() == 0.0

    def test_normalize_ohlcv_data_deduplication(self, ingestor):
        """Test duplicate removal."""
        duplicate_data = pl.DataFrame({
            "Date": ["2023-01-01 00:00:00", "2023-01-01 00:00:00", "2023-01-01 01:00:00"],
            "Open": [100.0, 100.0, 101.0],
            "High": [105.0, 105.0, 106.0],
            "Low": [99.0, 99.0, 100.0],
            "Close": [104.0, 104.0, 105.0],
            "Volume": [1000.0, 1000.0, 1100.0],
        })
        
        normalized = ingestor.normalize_ohlcv_data(duplicate_data, "SOL")
        
        # Should remove duplicates
        assert len(normalized) == 2
        assert normalized["symbol"].unique().item() == "SOL"

    def test_column_mapping(self, ingestor):
        """Test various column name mappings."""
        # Test different column name formats
        data_variants = [
            pl.DataFrame({
                "timestamp": ["2023-01-01 00:00:00"],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "vol": [1000.0],
            }),
            pl.DataFrame({
                "datetime": ["2023-01-01 00:00:00"],
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [104.0],
                "Volume": [1000.0],
            }),
        ]
        
        for data in data_variants:
            normalized = ingestor.normalize_ohlcv_data(data, "ADA")
            assert "ts" in normalized.columns
            assert "open" in normalized.columns
            assert "high" in normalized.columns
            assert "low" in normalized.columns
            assert "close" in normalized.columns
            assert "volume" in normalized.columns

    @patch('data.ingestion.kaggle_ingest.KaggleApi')
    def test_download_dataset(self, mock_kaggle_api, ingestor, tmp_path):
        """Test dataset download functionality."""
        mock_api = Mock()
        mock_kaggle_api.return_value = mock_api
        
        dataset_name = "test/dataset"
        result = ingestor.download_dataset(dataset_name)
        
        # Check that API was called
        mock_api.dataset_download_files.assert_called_once()
        
        # Check output directory was created
        expected_dir = tmp_path / "test_dataset"
        assert result == expected_dir

    def test_asset_mapping(self, ingestor):
        """Test asset mapping configuration."""
        # Check that common assets are mapped
        assert "BTC" in ingestor.asset_mapping
        assert "ETH" in ingestor.asset_mapping
        assert "SOL" in ingestor.asset_mapping
        
        # Check mapping structure
        btc_info = ingestor.asset_mapping["BTC"]
        assert "name" in btc_info
        assert "coingecko_id" in btc_info
        assert btc_info["name"] == "Bitcoin"
        assert btc_info["coingecko_id"] == "bitcoin"


if __name__ == "__main__":
    pytest.main([__file__])

