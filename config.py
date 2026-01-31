"""Compatibility shim so `from config import ...` works in all runtimes.

Tests and utility scripts executed from the repository root import `config`
directly, while backend modules (running via `backend.app.main`) expect the
same API inside `backend.app.config`.  Re-export the canonical objects so both
execution paths stay in sync.
"""

from backend.app.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]

