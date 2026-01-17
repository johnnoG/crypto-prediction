from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from services.data_aggregator import get_aggregator
from services.feature_engineering import get_feature_pipeline
from cache import AsyncCache
from config import get_settings
from db import get_db
from models.market import Asset, OHLCV


class JobStatus(Enum):
    """ETL job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


class ETLJob:
    """ETL job definition."""
    
    def __init__(
        self,
        job_id: str,
        name: str,
        func: callable,
        trigger: Any,
        enabled: bool = True,
        max_instances: int = 1,
        **kwargs
    ):
        self.job_id = job_id
        self.name = name
        self.func = func
        self.trigger = trigger
        self.enabled = enabled
        self.max_instances = max_instances
        self.kwargs = kwargs
        self.last_run = None
        self.last_success = None
        self.last_error = None
        self.run_count = 0
        self.success_count = 0
        self.failure_count = 0


class ETLScheduler:
    """ETL job scheduler for crypto data processing."""
    
    def __init__(self):
        self.settings = get_settings()
        self.scheduler = None
        self.jobs: Dict[str, ETLJob] = {}
        self.cache = AsyncCache()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the ETL scheduler."""
        if self._initialized:
            return
            
        await self.cache.initialize()
        
        # Configure job stores and executors
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': AsyncIOExecutor()
        }
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 300  # 5 minutes
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        
        self._initialized = True
    
    async def start(self) -> None:
        """Start the ETL scheduler."""
        await self.initialize()
        
        if not self.scheduler.running:
            self.scheduler.start()
            print("ETL Scheduler started")
    
    async def stop(self) -> None:
        """Stop the ETL scheduler."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            print("ETL Scheduler stopped")
    
    async def add_job(self, job: ETLJob) -> None:
        """Add an ETL job to the scheduler."""
        await self.initialize()
        
        if job.enabled:
            self.scheduler.add_job(
                func=job.func,
                trigger=job.trigger,
                id=job.job_id,
                name=job.name,
                max_instances=job.max_instances,
                **job.kwargs
            )
        
        self.jobs[job.job_id] = job
        print(f"Added ETL job: {job.name} ({job.job_id})")
    
    async def remove_job(self, job_id: str) -> None:
        """Remove an ETL job from the scheduler."""
        if self.scheduler and job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            print(f"Removed ETL job: {job_id}")
    
    async def enable_job(self, job_id: str) -> None:
        """Enable an ETL job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.enabled = True
            
            if self.scheduler:
                self.scheduler.add_job(
                    func=job.func,
                    trigger=job.trigger,
                    id=job.job_id,
                    name=job.name,
                    max_instances=job.max_instances,
                    **job.kwargs
                )
            
            print(f"Enabled ETL job: {job_id}")
    
    async def disable_job(self, job_id: str) -> None:
        """Disable an ETL job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.enabled = False
            
            if self.scheduler:
                self.scheduler.remove_job(job_id)
            
            print(f"Disabled ETL job: {job_id}")
    
    async def run_job_now(self, job_id: str) -> None:
        """Run an ETL job immediately."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            try:
                job.last_run = datetime.now()
                job.run_count += 1
                
                if asyncio.iscoroutinefunction(job.func):
                    await job.func()
                else:
                    job.func()
                
                job.last_success = datetime.now()
                job.success_count += 1
                job.last_error = None
                
                print(f"Successfully ran ETL job: {job_id}")
                
            except Exception as e:
                job.last_error = str(e)
                job.failure_count += 1
                print(f"Failed to run ETL job {job_id}: {e}")
                raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific ETL job."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        return {
            "job_id": job.job_id,
            "name": job.name,
            "enabled": job.enabled,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "last_success": job.last_success.isoformat() if job.last_success else None,
            "last_error": job.last_error,
            "run_count": job.run_count,
            "success_count": job.success_count,
            "failure_count": job.failure_count,
            "success_rate": job.success_count / job.run_count if job.run_count > 0 else 0
        }
    
    def get_all_jobs(self) -> Dict[str, Any]:
        """Get status of all ETL jobs."""
        return {
            "jobs": [self.get_job_status(job_id) for job_id in self.jobs.keys()],
            "total_jobs": len(self.jobs),
            "enabled_jobs": sum(1 for job in self.jobs.values() if job.enabled),
            "disabled_jobs": sum(1 for job in self.jobs.values() if not job.enabled)
        }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get ETL scheduler status."""
        return {
            "running": self.scheduler.running if self.scheduler else False,
            "initialized": self._initialized,
            "total_jobs": len(self.jobs),
            "enabled_jobs": sum(1 for job in self.jobs.values() if job.enabled),
            "last_check": datetime.now().isoformat()
        }
    
    async def setup_default_jobs(self) -> None:
        """Setup default ETL jobs."""
        await self.initialize()
        
        # Price data refresh job (every 5 minutes)
        price_job = ETLJob(
            job_id="price_refresh",
            name="Price Data Refresh",
            func=self._refresh_price_data,
            trigger=IntervalTrigger(minutes=5),
            enabled=True
        )
        await self.add_job(price_job)
        
        # OHLCV data refresh job (every hour)
        ohlcv_job = ETLJob(
            job_id="ohlcv_refresh",
            name="OHLCV Data Refresh",
            func=self._refresh_ohlcv_data,
            trigger=IntervalTrigger(hours=1),
            enabled=True
        )
        await self.add_job(ohlcv_job)
        
        # Feature calculation job (every 4 hours)
        feature_job = ETLJob(
            job_id="feature_calculation",
            name="Feature Calculation",
            func=self._calculate_features,
            trigger=IntervalTrigger(hours=4),
            enabled=True
        )
        await self.add_job(feature_job)
        
        # Cache cleanup job (daily at 2 AM)
        cleanup_job = ETLJob(
            job_id="cache_cleanup",
            name="Cache Cleanup",
            func=self._cleanup_cache,
            trigger=CronTrigger(hour=2, minute=0),
            enabled=True
        )
        await self.add_job(cleanup_job)
        
        print("Default ETL jobs setup completed")
    
    async def _refresh_price_data(self) -> None:
        """Refresh price data for major cryptocurrencies."""
        try:
            aggregator = await get_aggregator()
            
            # Major crypto symbols
            symbols = ["BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "UNI", "AVAX", "MATIC", "ATOM"]
            
            # Get current prices
            prices = await aggregator.get_multiple_prices(symbols, use_cache=False)
            
            # Store in cache
            for symbol, price in prices.items():
                if price is not None:
                    cache_key = f"price:{symbol}"
                    await self.cache.set(
                        cache_key,
                        {"price": price, "timestamp": datetime.now().isoformat()},
                        ttl_seconds=300  # 5 minutes
                    )
            
            print(f"Refreshed price data for {len([p for p in prices.values() if p is not None])} symbols")
            
        except Exception as e:
            print(f"Error refreshing price data: {e}")
            raise
    
    async def _refresh_ohlcv_data(self) -> None:
        """Refresh OHLCV data for major cryptocurrencies."""
        try:
            aggregator = await get_aggregator()
            
            # Major crypto symbols
            symbols = ["BTC", "ETH", "SOL", "ADA", "DOT"]
            
            for symbol in symbols:
                # Get OHLCV data
                ohlcv_data = await aggregator.get_ohlcv(symbol, "1h", 100, use_cache=False)
                
                if ohlcv_data:
                    # Store in cache
                    cache_key = f"ohlcv:{symbol}:1h:100"
                    await self.cache.set(cache_key, ohlcv_data, ttl_seconds=3600)  # 1 hour
            
            print(f"Refreshed OHLCV data for {len(symbols)} symbols")
            
        except Exception as e:
            print(f"Error refreshing OHLCV data: {e}")
            raise
    
    async def _calculate_features(self) -> None:
        """Calculate technical features for major cryptocurrencies."""
        try:
            aggregator = await get_aggregator()
            feature_pipeline = get_feature_pipeline()
            
            # Major crypto symbols
            symbols = ["BTC", "ETH", "SOL"]
            
            for symbol in symbols:
                # Get OHLCV data
                ohlcv_data = await aggregator.get_ohlcv(symbol, "1h", 200, use_cache=True)
                
                if ohlcv_data:
                    # Calculate features
                    features_df = feature_pipeline.process_ohlcv_data(ohlcv_data, "technical")
                    
                    if not features_df.is_empty():
                        # Store features in cache
                        cache_key = f"features:{symbol}:1h:technical"
                        await self.cache.set(
                            cache_key,
                            features_df.to_dicts(),
                            ttl_seconds=14400  # 4 hours
                        )
            
            print(f"Calculated features for {len(symbols)} symbols")
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            raise
    
    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        try:
            # This is a placeholder - actual cache cleanup depends on the cache backend
            print("Cache cleanup completed")
            
        except Exception as e:
            print(f"Error during cache cleanup: {e}")
            raise


# Global instance
_etl_scheduler_instance: Optional[ETLScheduler] = None


async def get_etl_scheduler() -> ETLScheduler:
    """Get or create the global ETL scheduler instance."""
    global _etl_scheduler_instance
    
    if _etl_scheduler_instance is None:
        _etl_scheduler_instance = ETLScheduler()
        await _etl_scheduler_instance.initialize()
    
    return _etl_scheduler_instance