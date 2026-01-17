"""
Quick Training Script for Bitcoin and Ethereum

Trains LightGBM models for BTC and ETH only.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.train_pipeline import train_model_for_crypto


async def main():
    """Train models for Bitcoin and Ethereum"""
    
    cryptos = ['bitcoin', 'ethereum']
    
    print("=" * 70)
    print("TRAINING ML MODELS FOR BITCOIN & ETHEREUM")
    print("=" * 70)
    print(f"\nThis will:")
    print("  1. Fetch 365 days of historical price data from CoinGecko")
    print("  2. Engineer 50+ technical features")
    print("  3. Train LightGBM models")
    print("  4. Save models to models/artifacts/lightgbm/")
    print(f"\nEstimated time: 5-8 minutes\n")
    
    results = {}
    
    for i, crypto_id in enumerate(cryptos, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(cryptos)}] TRAINING {crypto_id.upper()}")
        print(f"{'='*70}\n")
        
        try:
            # Train LightGBM model
            print(f"Starting training for {crypto_id}...")
            
            model = await train_model_for_crypto(
                crypto_id=crypto_id,
                model_type='lightgbm',
                days=365,  # 1 year of data
                save_model=True
            )
            
            # Get metrics
            metrics = model.metadata.get('metrics', {})
            
            results[crypto_id] = {
                'status': 'success',
                'train_rmse': metrics.get('train_rmse', 'N/A'),
                'val_rmse': metrics.get('val_rmse', 'N/A'),
                'model_path': str(model.model_dir)
            }
            
            print(f"\n✅ {crypto_id.upper()} training complete!")
            print(f"   Train RMSE: {metrics.get('train_rmse', 'N/A')}")
            print(f"   Val RMSE: {metrics.get('val_rmse', 'N/A')}")
            
        except Exception as e:
            print(f"\n❌ {crypto_id.upper()} training failed: {e}")
            results[crypto_id] = {
                'status': 'failed',
                'error': str(e)
            }
            import traceback
            traceback.print_exc()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"\n✅ Successful: {successful}/{len(cryptos)}")
    
    if successful > 0:
        print("\n📦 Models saved to: models/artifacts/lightgbm/")
        print("\n🚀 Next steps:")
        print("   1. Restart your backend: python main.py")
        print("   2. Go to Forecasts page in browser")
        print("   3. Select 'Machine Learning' from model dropdown")
        print("   4. See improved predictions!")
        
        print("\n📊 To use in API:")
        print("   curl \"http://127.0.0.1:8000/forecasts?ids=bitcoin&model=lightgbm\"")
    
    return results


if __name__ == "__main__":
    print("\n🚀 Starting ML model training for Bitcoin & Ethereum...\n")
    
    try:
        results = asyncio.run(main())
        
        # Check results
        if all(r['status'] == 'success' for r in results.values()):
            print("\n🎉 All models trained successfully!")
        else:
            print("\n⚠️  Some models failed. Check errors above.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

