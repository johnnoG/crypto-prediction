"""
Quick Training Script

Train ML models for major cryptocurrencies.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.train_pipeline import train_model_for_crypto
from models.model_registry import model_registry


async def main():
    """Train models for major cryptocurrencies"""
    
    # Major cryptos to train
    cryptos = ['bitcoin', 'ethereum', 'solana', 'cardano', 'binancecoin']
    
    print("=" * 80)
    print("CRYPTOCURRENCY ML MODEL TRAINING")
    print("=" * 80)
    print(f"\nTraining models for {len(cryptos)} cryptocurrencies")
    print(f"Models: LightGBM (fast, recommended to start)")
    print(f"\nThis will take approximately 5-10 minutes...\n")
    
    results = {}
    
    for i, crypto_id in enumerate(cryptos, 1):
        print(f"\n[{i}/{len(cryptos)}] Training {crypto_id.upper()}")
        print("-" * 80)
        
        try:
            # Train LightGBM model (fastest)
            model = await train_model_for_crypto(
                crypto_id=crypto_id,
                model_type='lightgbm',
                days=365,
                save_model=True
            )
            
            results[crypto_id] = {
                'status': 'success',
                'metrics': model.metadata.get('metrics', {})
            }
            
            print(f"✓ {crypto_id} training complete!")
            
        except Exception as e:
            print(f"✗ {crypto_id} training failed: {e}")
            results[crypto_id] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = sum(1 for r in results.values() if r['status'] == 'failed')
    
    print(f"\nSuccessful: {successful}/{len(cryptos)}")
    print(f"Failed: {failed}/{len(cryptos)}")
    
    if successful > 0:
        print("\n✅ Models trained and saved to models/artifacts/lightgbm/")
        print("\nYou can now use these models in the API:")
        print("  http://127.0.0.1:8000/forecasts?ids=bitcoin&model=lightgbm")
    
    print("\nModel Registry Status:")
    registry_report = model_registry.generate_registry_report()
    print(f"  Total models: {registry_report['total_models']}")
    print(f"  By type: {registry_report.get('by_type', {})}")
    
    return results


if __name__ == "__main__":
    print("\nStarting model training...")
    print("Make sure you have installed: pip install lightgbm scikit-learn pandas numpy\n")
    
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

