"""
Application Configuration

Centralized configuration management using pydantic-settings.
Loads environment variables and provides typed configuration objects.
"""

from typing import List
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Main application settings.

    Configuration is loaded from environment variables with fallback to .env file.
    All sensitive data should be provided through environment variables in production.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application Metadata
    app_name: str = Field(
        default="Crypto Prediction API", description="Application name"
    )
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Logging level")

    # API Configuration
    api_v1_prefix: str = Field(default="/api/v1", description="API version 1 prefix")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], description="Allowed CORS origins"
    )

    # Database Configuration
    database_url: PostgresDsn = Field(
        default="postgresql://crypto_user:crypto_pass@localhost:5432/crypto_prediction_db",
        description="PostgreSQL connection URL",
    )
    database_pool_size: int = Field(
        default=20, ge=1, le=100, description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum overflow connections beyond pool_size",
    )
    database_pool_timeout: int = Field(
        default=30, ge=5, le=120, description="Connection pool timeout in seconds"
    )
    database_pool_recycle: int = Field(
        default=3600, ge=300, description="Connection recycle time in seconds"
    )

    # Redis Configuration
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    redis_max_connections: int = Field(
        default=50, ge=1, le=200, description="Maximum Redis connections in pool"
    )
    redis_socket_timeout: int = Field(
        default=5, ge=1, le=30, description="Redis socket timeout in seconds"
    )
    redis_socket_connect_timeout: int = Field(
        default=5, ge=1, le=30, description="Redis socket connect timeout in seconds"
    )

    # Cache TTL Settings (in seconds)
    cache_ttl_short: int = Field(
        default=300,
        ge=60,
        description="Short cache TTL (5 minutes) for frequently updated data",
    )
    cache_ttl_medium: int = Field(
        default=1800,
        ge=300,
        description="Medium cache TTL (30 minutes) for moderately stable data",
    )
    cache_ttl_long: int = Field(
        default=3600,
        ge=600,
        description="Long cache TTL (1 hour) for relatively static data",
    )

    # Security
    secret_key: str = Field(
        default="change-this-to-a-secure-random-string-in-production",
        min_length=32,
        description="Secret key for signing tokens",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def database_url_str(self) -> str:
        """Get database URL as string."""
        return str(self.database_url)

    @property
    def redis_url_str(self) -> str:
        """Get Redis URL as string."""
        return str(self.redis_url)


# Global settings instance
settings = Settings()
