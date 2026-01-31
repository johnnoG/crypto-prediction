from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Uses pydantic-settings for robust env parsing with sensible defaults.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Core
    app_name: str = Field(default="Crypto Forecast & Real‑Time Dashboard API")

    # CORS - Simple string field to avoid parsing issues
    allowed_origins: str = Field(default="*")

    # External APIs / Data sources
    coingecko_base_url: str = Field(
        default="https://api.coingecko.com/api/v3",
        alias="COINGECKO_BASE_URL",
    )

    coingecko_api_key: Optional[str] = Field(default=None, alias="COINGECKO_API_KEY")

    # Firecrawl MCP + REST
    firecrawl_base_url: str = Field(
        default="https://api.firecrawl.dev",
        alias="FIRECRAWL_BASE_URL",
    )
    firecrawl_api_key: Optional[str] = Field(default=None, alias="FIRECRAWL_API_KEY")
    mcp_firecrawl_port: int = Field(default=7355, alias="MCP_FIRECRAWL_PORT")
    news_max_content_bytes: int = Field(default=1048576, alias="NEWS_MAX_CONTENT_BYTES")

    # CryptoPanic API
    cryptopanic_api_key: Optional[str] = Field(default=None, alias="CRYPTOPANIC_API_KEY")

    # Caching
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    cache_sqlite_path: Optional[str] = Field(default=None, alias="CACHE_SQLITE_PATH")

    # Database
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    
    # Authentication
    secret_key: str = Field(default="your-secret-key-change-in-production", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Google OAuth
    google_client_id: Optional[str] = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(default=None, alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(default="http://localhost:3000/oauth/callback/google", alias="GOOGLE_REDIRECT_URI")

    # API rate limiting (slowapi)
    rate_limit_default: str = Field(default="120/minute", alias="RATE_LIMIT_DEFAULT")
    rate_limit_prices: str = Field(default="60/minute", alias="RATE_LIMIT_PRICES")
    rate_limit_market: str = Field(default="45/minute", alias="RATE_LIMIT_PRICES_MARKET")
    rate_limit_history: str = Field(default="30/minute", alias="RATE_LIMIT_PRICES_HISTORY")
    rate_limit_forecasts: str = Field(default="20/minute", alias="RATE_LIMIT_FORECASTS")
    rate_limit_news_list: str = Field(default="30/minute", alias="RATE_LIMIT_NEWS_LIST")
    rate_limit_news_refresh: str = Field(default="5/minute", alias="RATE_LIMIT_NEWS_REFRESH")
    rate_limit_cache_status: str = Field(default="30/minute", alias="RATE_LIMIT_CACHE_STATUS")

    def get_allowed_origins_list(self) -> List[str]:
        """Get allowed origins as a list."""
        if not self.allowed_origins or self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
