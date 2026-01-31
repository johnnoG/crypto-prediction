#!/usr/bin/env python3
"""
Kaggle API Setup and Dataset Testing Script

Helps set up Kaggle API credentials and test dataset access.
"""

import os
import json
from pathlib import Path
import sys

def setup_kaggle_credentials():
    """Guide user through setting up Kaggle API credentials."""
    print("🔧 Kaggle API Setup")
    print("=" * 50)
    
    # Check if credentials already exist
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials found!")
        return True
    
    print("❌ Kaggle credentials not found.")
    print("\nTo set up Kaggle API access:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print("5. Place it in your home directory: ~/.kaggle/kaggle.json")
    
    # Create directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    
    # Ask user if they want to create a template
    response = input("\nWould you like to create a template kaggle.json file? (y/n): ")
    if response.lower() == 'y':
        template = {
            "username": "your_kaggle_username",
            "key": "your_kaggle_api_key"
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"📝 Template created at: {kaggle_json}")
        print("⚠️  Please edit the file with your actual credentials!")
    
    return False


def test_kaggle_api():
    """Test Kaggle API access."""
    print("\n🧪 Testing Kaggle API...")
    
    try:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Kaggle API authentication failed: {e}")
        return False


def list_recommended_datasets():
    """List recommended crypto datasets."""
    print("\n📊 Recommended Crypto Datasets")
    print("=" * 50)
    
    datasets = [
        {
            "name": "Cryptocurrency Historical Prices",
            "ref": "sudalairajkumar/cryptocurrencypricehistory",
            "description": "BTC, ETH, LTC, XRP, BCH, ADA, XLM, NEO, EOS, IOTA (2013-2021)",
            "size": "~500MB",
            "quality": "⭐⭐⭐⭐⭐"
        },
        {
            "name": "Bitcoin Historical Data",
            "ref": "mczielinski/bitcoin-historical-data",
            "description": "BTC minute-level data (2012-2023)",
            "size": "~200MB",
            "quality": "⭐⭐⭐⭐⭐"
        },
        {
            "name": "Crypto Fear & Greed Index",
            "ref": "andrewmvd/crypto-fear-and-greed-index",
            "description": "Market sentiment index (2018-2023)",
            "size": "~1MB",
            "quality": "⭐⭐⭐⭐"
        },
        {
            "name": "Crypto News Sentiment",
            "ref": "ankurzing/sentiment-analysis-for-financial-news",
            "description": "News headlines with sentiment scores (2008-2018)",
            "size": "~50MB",
            "quality": "⭐⭐⭐"
        }
    ]
    
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Reference: {dataset['ref']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Quality: {dataset['quality']}")
        print()


def download_sample_dataset():
    """Download a small sample dataset for testing."""
    print("\n📥 Downloading Sample Dataset")
    print("=" * 50)
    
    if not test_kaggle_api():
        print("❌ Cannot download without valid Kaggle credentials")
        return False
    
    try:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download the Fear & Greed dataset (smallest)
        dataset_ref = "andrewmvd/crypto-fear-and-greed-index"
        output_dir = Path("data/raw/kaggle_test")
        
        print(f"Downloading: {dataset_ref}")
        print(f"Output directory: {output_dir}")
        
        api.dataset_download_files(
            dataset_ref,
            path=str(output_dir),
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        
        # List downloaded files
        print("\n📁 Downloaded files:")
        for file in output_dir.glob("*"):
            print(f"  - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def create_data_source_config():
    """Create a configuration file for data sources."""
    print("\n⚙️  Creating Data Source Configuration")
    print("=" * 50)
    
    config = {
        "kaggle_datasets": {
            "primary": "sudalairajkumar/cryptocurrencypricehistory",
            "secondary": "mczielinski/bitcoin-historical-data",
            "sentiment": "ankurzing/sentiment-analysis-for-financial-news",
            "fear_greed": "andrewmvd/crypto-fear-and-greed-index"
        },
        "apis": {
            "coingecko": {
                "enabled": True,
                "base_url": "https://api.coingecko.com/api/v3",
                "rate_limit": "50 calls/minute"
            },
            "binance": {
                "enabled": False,
                "base_url": "https://api.binance.com/api/v3",
                "rate_limit": "1200 requests/minute"
            },
            "cryptocompare": {
                "enabled": False,
                "base_url": "https://min-api.cryptocompare.com",
                "rate_limit": "100,000 calls/month"
            }
        },
        "news_sources": {
            "firecrawl": {
                "enabled": True,
                "base_url": "https://api.firecrawl.dev"
            },
            "newsapi": {
                "enabled": False,
                "base_url": "https://newsapi.org/v2",
                "rate_limit": "1000 requests/day"
            }
        }
    }
    
    config_file = Path("data/sources/config.json")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to: {config_file}")
    return config_file


def main():
    """Main setup function."""
    print("🚀 Crypto Data Sources Setup")
    print("=" * 60)
    
    # Setup Kaggle credentials
    has_credentials = setup_kaggle_credentials()
    
    if has_credentials:
        # Test API access
        api_working = test_kaggle_api()
        
        if api_working:
            # List recommended datasets
            list_recommended_datasets()
            
            # Ask if user wants to download sample
            response = input("Would you like to download a sample dataset? (y/n): ")
            if response.lower() == 'y':
                download_sample_dataset()
    
    # Create configuration
    create_data_source_config()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit ~/.kaggle/kaggle.json with your credentials")
    print("2. Run this script again to test API access")
    print("3. Use the ingestion script to download datasets")
    print("4. Proceed to SCRUM-4 (Feature ETL)")


if __name__ == "__main__":
    main()

