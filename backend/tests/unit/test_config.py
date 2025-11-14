"""
Tests for Application Configuration

Tests configuration management including:
- Settings loading
- Environment variables
- Validation
- Default values
"""

import pytest
import os
from pydantic import ValidationError
from app.config import Settings


class TestSettings:
    """Test cases for Settings configuration."""

    @pytest.mark.unit
    def test_default_settings(self):
        """Test that settings load with defaults."""
        settings = Settings()

        assert settings.app_name == "Crypto Prediction API"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.api_v1_prefix == "/api/v1"

    @pytest.mark.unit
    def test_database_settings(self):
        """Test database configuration settings."""
        settings = Settings()

        assert settings.database_pool_size >= 1
        assert settings.database_pool_size <= 100
        assert settings.database_max_overflow >= 0
        assert settings.database_max_overflow <= 50
        assert settings.database_pool_timeout >= 5
        assert settings.database_pool_recycle >= 300

    @pytest.mark.unit
    def test_redis_settings(self):
        """Test Redis configuration settings."""
        settings = Settings()

        assert settings.redis_max_connections >= 1
        assert settings.redis_max_connections <= 200
        assert settings.redis_socket_timeout >= 1
        assert settings.redis_socket_connect_timeout >= 1

    @pytest.mark.unit
    def test_cache_ttl_settings(self):
        """Test cache TTL configuration."""
        settings = Settings()

        assert settings.cache_ttl_short >= 60
        assert settings.cache_ttl_medium >= 300
        assert settings.cache_ttl_long >= 600

        # Verify TTL hierarchy
        assert settings.cache_ttl_short < settings.cache_ttl_medium
        assert settings.cache_ttl_medium < settings.cache_ttl_long

    @pytest.mark.unit
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from string."""
        # Test string input
        settings = Settings(cors_origins="http://localhost:3000,http://localhost:8000")
        assert len(settings.cors_origins) == 2
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:8000" in settings.cors_origins

        # Test list input
        settings = Settings(cors_origins=["http://example.com"])
        assert len(settings.cors_origins) == 1
        assert "http://example.com" in settings.cors_origins

    @pytest.mark.unit
    def test_database_url_str_property(self):
        """Test database URL string property."""
        settings = Settings()
        url_str = settings.database_url_str

        assert isinstance(url_str, str)
        assert "postgresql" in url_str

    @pytest.mark.unit
    def test_redis_url_str_property(self):
        """Test Redis URL string property."""
        settings = Settings()
        url_str = settings.redis_url_str

        assert isinstance(url_str, str)
        assert "redis" in url_str


class TestSettingsValidation:
    """Test settings validation."""

    @pytest.mark.unit
    def test_pool_size_validation(self):
        """Test that pool size is validated."""
        # Too small
        with pytest.raises(ValidationError):
            Settings(database_pool_size=0)

        # Too large
        with pytest.raises(ValidationError):
            Settings(database_pool_size=101)

        # Valid
        settings = Settings(database_pool_size=20)
        assert settings.database_pool_size == 20

    @pytest.mark.unit
    def test_ttl_validation(self):
        """Test that TTL values are validated."""
        # Too small
        with pytest.raises(ValidationError):
            Settings(cache_ttl_short=30)

        # Valid
        settings = Settings(cache_ttl_short=300)
        assert settings.cache_ttl_short == 300

    @pytest.mark.unit
    def test_secret_key_length_validation(self):
        """Test that secret key must meet minimum length."""
        # Too short
        with pytest.raises(ValidationError):
            Settings(secret_key="short")

        # Valid
        settings = Settings(secret_key="a" * 32)
        assert len(settings.secret_key) >= 32


class TestEnvironmentVariables:
    """Test loading from environment variables."""

    @pytest.mark.unit
    def test_load_from_env_var(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Set environment variable
        monkeypatch.setenv("APP_NAME", "Test API")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = Settings()

        assert settings.app_name == "Test API"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    @pytest.mark.unit
    def test_database_url_from_env(self, monkeypatch):
        """Test loading database URL from environment."""
        db_url = "postgresql://testuser:testpass@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", db_url)

        settings = Settings()
        assert "testuser" in settings.database_url_str
        assert "testdb" in settings.database_url_str

    @pytest.mark.unit
    def test_redis_url_from_env(self, monkeypatch):
        """Test loading Redis URL from environment."""
        redis_url = "redis://localhost:6380/1"
        monkeypatch.setenv("REDIS_URL", redis_url)

        settings = Settings()
        assert "6380" in settings.redis_url_str

    @pytest.mark.unit
    def test_cors_origins_from_env(self, monkeypatch):
        """Test loading CORS origins from environment."""
        # pydantic-settings 2.6+ expects JSON format for List fields
        monkeypatch.setenv(
            "CORS_ORIGINS", '["http://example.com", "https://api.example.com"]'
        )

        settings = Settings()
        assert len(settings.cors_origins) == 2
        assert "http://example.com" in settings.cors_origins
        assert "https://api.example.com" in settings.cors_origins


class TestConfigurationProfiles:
    """Test different configuration profiles."""

    @pytest.mark.unit
    def test_development_profile(self):
        """Test development configuration."""
        settings = Settings(debug=True, log_level="DEBUG")

        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    @pytest.mark.unit
    def test_production_profile(self):
        """Test production configuration."""
        settings = Settings(
            debug=False, log_level="INFO", secret_key="a" * 64  # Strong secret
        )

        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert len(settings.secret_key) >= 32

    @pytest.mark.unit
    def test_production_checklist(self):
        """Test that production settings are secure."""
        settings = Settings(
            debug=False, secret_key="a" * 64, cors_origins=["https://production.com"]
        )

        # Debug should be off
        assert settings.debug is False

        # Secret key should be strong
        assert len(settings.secret_key) >= 32

        # CORS should be specific
        assert "localhost" not in settings.cors_origins[0]
