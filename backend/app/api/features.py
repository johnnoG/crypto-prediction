from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

try:
    from services.data_aggregator import get_aggregator
    from services.feature_engineering import get_feature_pipeline
    from services.etl_scheduler import get_etl_scheduler
except ImportError:
    from services.data_aggregator import get_aggregator
    from services.feature_engineering import get_feature_pipeline
    from services.etl_scheduler import get_etl_scheduler

router = APIRouter(prefix="/features", tags=["features"])


@router.get("/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    limit: int = Query(100, ge=1, le=1000, description="Number of data points"),
    use_cache: bool = Query(True, description="Use cached data if available")
) -> Dict[str, Any]:
    """Get OHLCV data for a symbol."""
    try:
        aggregator = await get_aggregator()
        data = await aggregator.get_ohlcv(symbol, timeframe, limit, use_cache)
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_points": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching OHLCV data: {str(e)}")


@router.get("/price/{symbol}")
async def get_current_price(
    symbol: str,
    use_cache: bool = Query(True, description="Use cached data if available")
) -> Dict[str, Any]:
    """Get current price for a symbol."""
    try:
        aggregator = await get_aggregator()
        price = await aggregator.get_price(symbol, use_cache)
        
        if price is None:
            raise HTTPException(status_code=404, detail=f"No price available for {symbol}")
        
        return {
            "symbol": symbol,
            "price": price,
            "timestamp": "now"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price: {str(e)}")


@router.get("/prices")
async def get_multiple_prices(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    use_cache: bool = Query(True, description="Use cached data if available")
) -> Dict[str, Any]:
    """Get current prices for multiple symbols."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        aggregator = await get_aggregator()
        prices = await aggregator.get_multiple_prices(symbol_list, use_cache)
        
        return {
            "prices": prices,
            "timestamp": "now"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")


@router.get("/technical/{symbol}")
async def get_technical_features(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    feature_set: str = Query("basic", description="Feature set: basic, technical, advanced, all"),
    limit: int = Query(100, ge=1, le=1000, description="Number of data points"),
    use_cache: bool = Query(True, description="Use cached data if available")
) -> Dict[str, Any]:
    """Get technical features for a symbol."""
    try:
        # Get OHLCV data
        aggregator = await get_aggregator()
        ohlcv_data = await aggregator.get_ohlcv(symbol, timeframe, limit, use_cache)
        
        if not ohlcv_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Calculate features
        feature_pipeline = get_feature_pipeline()
        features_df = feature_pipeline.process_ohlcv_data(ohlcv_data, feature_set)
        
        if features_df.is_empty():
            raise HTTPException(status_code=500, detail="Error calculating features")
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "feature_set": feature_set,
            "data_points": len(features_df),
            "features": features_df.to_dicts()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating features: {str(e)}")


@router.get("/feature-sets")
async def get_available_feature_sets() -> Dict[str, Any]:
    """Get available feature sets."""
    try:
        feature_pipeline = get_feature_pipeline()
        feature_sets = feature_pipeline.get_feature_sets()
        
        return {
            "feature_sets": feature_sets,
            "available_features": feature_pipeline.engineer.get_feature_list()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature sets: {str(e)}")


@router.get("/etl/jobs")
async def get_etl_jobs() -> Dict[str, Any]:
    """Get all ETL jobs status."""
    try:
        scheduler = await get_etl_scheduler()
        return scheduler.get_all_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETL jobs: {str(e)}")


@router.get("/etl/jobs/{job_id}")
async def get_etl_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of a specific ETL job."""
    try:
        scheduler = await get_etl_scheduler()
        job_status = scheduler.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return job_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")


@router.post("/etl/jobs/{job_id}/run")
async def run_etl_job_now(job_id: str) -> Dict[str, Any]:
    """Run an ETL job immediately."""
    try:
        scheduler = await get_etl_scheduler()
        await scheduler.run_job_now(job_id)
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Job {job_id} started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running job: {str(e)}")


@router.post("/etl/jobs/{job_id}/enable")
async def enable_etl_job(job_id: str) -> Dict[str, Any]:
    """Enable an ETL job."""
    try:
        scheduler = await get_etl_scheduler()
        scheduler.enable_job(job_id)
        
        return {
            "job_id": job_id,
            "status": "enabled",
            "message": f"Job {job_id} enabled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enabling job: {str(e)}")


@router.post("/etl/jobs/{job_id}/disable")
async def disable_etl_job(job_id: str) -> Dict[str, Any]:
    """Disable an ETL job."""
    try:
        scheduler = await get_etl_scheduler()
        scheduler.disable_job(job_id)
        
        return {
            "job_id": job_id,
            "status": "disabled",
            "message": f"Job {job_id} disabled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error disabling job: {str(e)}")


@router.get("/etl/status")
async def get_etl_scheduler_status() -> Dict[str, Any]:
    """Get ETL scheduler status."""
    try:
        scheduler = await get_etl_scheduler()
        return scheduler.get_scheduler_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scheduler status: {str(e)}")


@router.get("/data-sources/health")
async def get_data_sources_health() -> Dict[str, Any]:
    """Get health status of all data sources."""
    try:
        aggregator = await get_aggregator()
        return {
            "data_sources": aggregator.get_source_health(),
            "timestamp": "now"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data sources health: {str(e)}")


@router.post("/etl/setup-default")
async def setup_default_etl_jobs() -> Dict[str, Any]:
    """Setup default ETL jobs."""
    try:
        scheduler = await get_etl_scheduler()
        await scheduler.setup_default_jobs()
        
        return {
            "status": "success",
            "message": "Default ETL jobs setup completed",
            "jobs": scheduler.get_all_jobs()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up default jobs: {str(e)}")
