from __future__ import annotations

from typing import Dict, Any

from fastapi import APIRouter

try:
    from db import db_ping, get_db
    from models.market import Asset, OHLCV
    from models.news import NewsSource, NewsArticle
except ImportError:
    from db import db_ping, get_db
    from models.market import Asset, OHLCV
    from models.news import NewsSource, NewsArticle


router = APIRouter(prefix="/db", tags=["db"])


@router.get("/status")
def db_status() -> Dict[str, Any]:
    """Get database status and table information."""
    healthy = db_ping()
    
    if not healthy:
        return {
            "healthy": False,
            "status": "unhealthy",
            "tables": {},
            "error": "Database connection failed"
        }
    
    try:
        db = next(get_db())
        
        # Get table counts
        asset_count = db.query(Asset).count()
        ohlcv_count = db.query(OHLCV).count()
        news_source_count = db.query(NewsSource).count()
        news_article_count = db.query(NewsArticle).count()
        
        # Get recent data info
        recent_ohlcv = db.query(OHLCV).order_by(OHLCV.ts.desc()).first()
        recent_article = db.query(NewsArticle).order_by(NewsArticle.fetched_at.desc()).first()
        
        return {
            "healthy": True,
            "status": "healthy",
            "tables": {
                "assets": {
                    "count": asset_count,
                    "description": "Cryptocurrency assets"
                },
                "ohlcv": {
                    "count": ohlcv_count,
                    "description": "OHLCV price data",
                    "latest_timestamp": recent_ohlcv.ts.isoformat() if recent_ohlcv else None
                },
                "news_sources": {
                    "count": news_source_count,
                    "description": "News sources configuration"
                },
                "news_articles": {
                    "count": news_article_count,
                    "description": "News articles",
                    "latest_fetch": recent_article.fetched_at.isoformat() if recent_article else None
                }
            },
            "timestamp": "now"
        }
        
    except Exception as e:
        return {
            "healthy": False,
            "status": "error",
            "tables": {},
            "error": str(e)
        }


