"""
Sentry Configuration for Error Tracking

Captures and reports errors, performance issues, and traces.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    print("Warning: sentry-sdk not installed")


def init_sentry(
    dsn: Optional[str] = None,
    environment: str = "development",
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1
) -> None:
    """
    Initialize Sentry error tracking.
    
    Args:
        dsn: Sentry DSN (if None, tries to load from env)
        environment: Deployment environment
        traces_sample_rate: Percentage of transactions to trace (0.0-1.0)
        profiles_sample_rate: Percentage of transactions to profile
    """
    if not SENTRY_AVAILABLE:
        print("Sentry not available, skipping initialization")
        return
    
    # Get DSN from environment if not provided
    sentry_dsn = dsn or os.getenv("SENTRY_DSN")
    
    if not sentry_dsn:
        print("Sentry DSN not configured, skipping initialization")
        return
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        integrations=[
            FastApiIntegration(
                transaction_style="endpoint",
                failed_request_status_codes=[500, 502, 503, 504]
            ),
            SqlalchemyIntegration()
        ],
        # Capture breadcrumbs
        max_breadcrumbs=50,
        # Send PII (disable in production if handling sensitive data)
        send_default_pii=False,
        # Performance monitoring
        enable_tracing=True,
        # Release tracking
        release=os.getenv("APP_VERSION", "1.0.0")
    )
    
    print(f"Sentry initialized for environment: {environment}")


def capture_exception(error: Exception, context: Optional[dict] = None) -> None:
    """
    Manually capture an exception in Sentry.
    
    Args:
        error: Exception to capture
        context: Additional context information
    """
    if not SENTRY_AVAILABLE:
        return
    
    if context:
        sentry_sdk.set_context("additional_info", context)
    
    sentry_sdk.capture_exception(error)


def capture_message(message: str, level: str = "info", context: Optional[dict] = None) -> None:
    """
    Capture a message in Sentry.
    
    Args:
        message: Message to capture
        level: Severity level (info, warning, error, fatal)
        context: Additional context
    """
    if not SENTRY_AVAILABLE:
        return
    
    if context:
        sentry_sdk.set_context("additional_info", context)
    
    sentry_sdk.capture_message(message, level=level)


def add_breadcrumb(message: str, category: str = "default", level: str = "info", data: Optional[dict] = None) -> None:
    """
    Add a breadcrumb for debugging context.
    
    Args:
        message: Breadcrumb message
        category: Category (e.g., 'api_call', 'cache', 'model')
        level: Severity level
        data: Additional data
    """
    if not SENTRY_AVAILABLE:
        return
    
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )

