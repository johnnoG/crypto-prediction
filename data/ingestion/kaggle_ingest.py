#!/usr/bin/env python3
"""
Kaggle Crypto Dataset Ingestion Script

Downloads and processes Kaggle crypto datasets for historical price data.
Uses Polars for high-performance data processing and deduplication.

Usage:
    python data/ingestion/kaggle_ingest.py --dataset bitcoin-price-data --output data/raw/
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from kaggle import KaggleApi

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from backend.app.config import get_settings  # type: ignore
    from backend.app.models.market import Asset, OHLCV  # type: ignore
    from backend.app.db import get_db  # type: ignore
except ImportError:
    from app.config import get_settings  # type: ignore
    from app.models.market import Asset, OHLCV  # type: ignore
    from app.db import get_db  # type: ignore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class KaggleIngestor:
    """High-performance Kaggle dataset ingestion with Polars."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings = get_settings()
        
        # Defer Kaggle API authentication until needed (makes tests/offline usage easier)
        self._api: Optional[KaggleApi] = None
        
        # Asset mapping for common symbols
        self.asset_mapping = {
            "BTC": {"name": "Bitcoin", "coingecko_id": "bitcoin"},
            "ETH": {"name": "Ethereum", "coingecko_id": "ethereum"},
            "SOL": {"name": "Solana", "coingecko_id": "solana"},
            "ADA": {"name": "Cardano", "coingecko_id": "cardano"},
            "DOT": {"name": "Polkadot", "coingecko_id": "polkadot"},
            "LINK": {"name": "Chainlink", "coingecko_id": "chainlink"},
            "UNI": {"name": "Uniswap", "coingecko_id": "uniswap"},
            "AVAX": {"name": "Avalanche", "coingecko_id": "avalanche-2"},
            "MATIC": {"name": "Polygon", "coingecko_id": "matic-network"},
            "ATOM": {"name": "Cosmos", "coingecko_id": "cosmos"},
        }

    def _get_api(self) -> KaggleApi:
        """Lazily initialize and authenticate the Kaggle API client."""
        if self._api is None:
            api = KaggleApi()
            try:
                api.authenticate()
            except Exception as exc:  # pragma: no cover - network/credential specific
                raise RuntimeError("Kaggle API authentication failed") from exc
            self._api = api
        return self._api

    def download_dataset(self, dataset_name: str) -> Path:
        """Download Kaggle dataset to output directory."""
        logger.info(f"Downloading dataset: {dataset_name}")
        
        # Create dataset-specific directory
        dataset_dir = self.output_dir / dataset_name.replace("/", "_")
        dataset_dir.mkdir(exist_ok=True)
        
        # Download dataset
        api = self._get_api()
        api.dataset_download_files(
            dataset_name,
            path=str(dataset_dir),
            unzip=True,
        )
        
        logger.info(f"Dataset downloaded to: {dataset_dir}")
        return dataset_dir

    def parse_csv_file(self, file_path: Path) -> pl.DataFrame:
        """Parse CSV file with Polars, handling various formats."""
        logger.info(f"Parsing CSV file: {file_path}")
        
        try:
            # Try to infer schema automatically
            df = pl.read_csv(file_path)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            # Try with different options
            df = pl.read_csv(file_path, try_parse_dates=True, null_values=["", "null", "NULL"])
        
        # Log basic info
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Columns: {df.columns}")
        
        return df

    def normalize_ohlcv_data(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """Normalize OHLCV data to standard format."""
        logger.info(f"Normalizing OHLCV data for {symbol}")
        
        # Common column name mappings
        column_mappings = {
            # Date/Time columns
            "Date": "ts",
            "date": "ts", 
            "timestamp": "ts",
            "time": "ts",
            "datetime": "ts",
            
            # OHLC columns
            "Open": "open",
            "open": "open",
            "High": "high", 
            "high": "high",
            "Low": "low",
            "low": "low", 
            "Close": "close",
            "close": "close",
            "Price": "close",
            "price": "close",
            
            # Volume columns
            "Volume": "volume",
            "volume": "volume",
            "Vol": "volume",
            "vol": "volume",
        }
        
        # Rename columns
        renamed_cols = {}
        for old_col in df.columns:
            if old_col in column_mappings:
                renamed_cols[old_col] = column_mappings[old_col]
        
        if renamed_cols:
            df = df.rename(renamed_cols)
        
        # Ensure required columns exist
        required_cols = ["ts", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            # Add missing columns with defaults
            for col in missing_cols:
                if col == "volume":
                    df = df.with_columns(pl.lit(0.0).alias("volume"))
                else:
                    df = df.with_columns(pl.lit(0.0).alias(col))
        
        # Parse timestamp
        if "ts" in df.columns:
            ts_expr = None
            try:
                ts_expr = pl.col("ts").str.strptime(
                    pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
                )
            except TypeError:
                # Older Polars versions use `fmt=` instead of `format=`
                ts_expr = pl.col("ts").str.strptime(
                    pl.Datetime, fmt="%Y-%m-%d %H:%M:%S", strict=False
                )
            df = df.with_columns(ts_expr.alias("ts"))
        
        # Add symbol and timeframe
        df = df.with_columns([
            pl.lit(symbol).alias("symbol"),
            pl.lit("1h").alias("timeframe"),  # Default to 1h
        ])
        
        # Ensure numeric types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
        
        # Remove duplicates and sort by timestamp
        df = df.unique(subset=["ts"], maintain_order=False).sort("ts")
        
        logger.info(f"Normalized {len(df)} rows for {symbol}")
        return df

    def save_to_parquet(self, df: pl.DataFrame, symbol: str, dataset_name: str) -> Path:
        """Save normalized data to Parquet format."""
        output_file = self.output_dir / f"{symbol}_{dataset_name}.parquet"
        
        logger.info(f"Saving {len(df)} rows to {output_file}")
        df.write_parquet(output_file)
        
        return output_file

    def ingest_dataset(self, dataset_name: str, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main ingestion pipeline for a Kaggle dataset."""
        logger.info(f"Starting ingestion for dataset: {dataset_name}")
        
        # Download dataset
        dataset_dir = self.download_dataset(dataset_name)
        
        # Find CSV files
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {dataset_dir}")
            return {"error": "No CSV files found"}
        
        results = {
            "dataset": dataset_name,
            "files_processed": 0,
            "total_rows": 0,
            "output_files": [],
            "errors": [],
        }
        
        for csv_file in csv_files:
            try:
                # Parse CSV
                df = self.parse_csv_file(csv_file)
                
                # Determine symbol from filename or data
                symbol = self.extract_symbol_from_filename(csv_file.name)
                if not symbol and symbols:
                    symbol = symbols[0]  # Use first provided symbol
                
                if not symbol:
                    logger.warning(f"Could not determine symbol for {csv_file.name}")
                    continue
                
                # Normalize data
                normalized_df = self.normalize_ohlcv_data(df, symbol)
                
                # Save to Parquet
                output_file = self.save_to_parquet(normalized_df, symbol, dataset_name)
                
                results["files_processed"] += 1
                results["total_rows"] += len(normalized_df)
                results["output_files"].append(str(output_file))
                
                logger.info(f"Successfully processed {csv_file.name} -> {output_file}")
                
            except Exception as e:
                error_msg = f"Error processing {csv_file.name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        logger.info(f"Ingestion complete: {results}")
        return results

    def extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """Extract cryptocurrency symbol from filename."""
        # Common patterns
        filename_upper = filename.upper()
        
        for symbol in self.asset_mapping.keys():
            if symbol in filename_upper:
                return symbol
        
        # Try to extract from common patterns
        if "BTC" in filename_upper or "BITCOIN" in filename_upper:
            return "BTC"
        elif "ETH" in filename_upper or "ETHEREUM" in filename_upper:
            return "ETH"
        elif "SOL" in filename_upper or "SOLANA" in filename_upper:
            return "SOL"
        
        return None

    async def load_to_database(self, parquet_file: Path) -> Dict[str, Any]:
        """Load Parquet data into PostgreSQL database."""
        logger.info(f"Loading {parquet_file} to database")
        
        try:
            # Read Parquet file
            df = pl.read_parquet(parquet_file)
            
            # Get database session
            db = next(get_db())
            
            # Extract symbol
            symbol = df.select("symbol").unique().item()
            
            # Create or get asset
            asset = db.query(Asset).filter(Asset.symbol == symbol).first()
            if not asset:
                asset_info = self.asset_mapping.get(symbol, {})
                asset = Asset(
                    symbol=symbol,
                    name=asset_info.get("name", symbol),
                    coingecko_id=asset_info.get("coingecko_id"),
                )
                db.add(asset)
                db.commit()
                db.refresh(asset)
            
            # Convert to OHLCV records
            ohlcv_records = []
            for row in df.iter_rows(named=True):
                ohlcv_records.append(OHLCV(
                    asset_id=asset.id,
                    ts=row["ts"],
                    timeframe=row["timeframe"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                ))
            
            # Bulk insert
            db.bulk_save_objects(ohlcv_records)
            db.commit()
            
            logger.info(f"Loaded {len(ohlcv_records)} OHLCV records for {symbol}")
            return {"success": True, "records": len(ohlcv_records), "symbol": symbol}
            
        except Exception as e:
            logger.error(f"Database load error: {e}")
            return {"error": str(e)}


async def main():
    """Main ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest Kaggle crypto datasets")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset name (e.g., 'borismarjanovic/price-volume-data-for-all-us-stocks-etfs')")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to process")
    parser.add_argument("--load-db", action="store_true", help="Load data to database after processing")
    
    args = parser.parse_args()
    
    # Initialize ingestor
    ingestor = KaggleIngestor(Path(args.output))
    
    # Ingest dataset
    results = ingestor.ingest_dataset(args.dataset, args.symbols)
    
    if results.get("errors"):
        logger.error(f"Ingestion completed with errors: {results['errors']}")
    
    # Load to database if requested
    if args.load_db and results.get("output_files"):
        logger.info("Loading data to database...")
        for output_file in results["output_files"]:
            await ingestor.load_to_database(Path(output_file))
    
    logger.info("Ingestion complete!")


if __name__ == "__main__":
    asyncio.run(main())
