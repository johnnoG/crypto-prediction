from .prices import router as prices_router
from .cache_admin import router as cache_router
from .db_admin import router as db_router
from .news import router as news_router
from .features import router as features_router
from .crypto_data import router as crypto_data_router
from .forecasts import router as forecasts_router
from .stream import router as stream_router
from .quick_prices import router as quick_prices_router

__all__ = [
    "prices_router",
    "cache_router",
    "db_router",
    "news_router", 
    "features_router",
    "crypto_data_router",
    "forecasts_router",
    "stream_router",
    "quick_prices_router"
]


