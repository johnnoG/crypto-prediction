"""
Integration Test Configuration

Automatically applies 'integration' marker to all tests in this directory.
"""

import pytest


def pytest_collection_modifyitems(items):
    """
    Automatically mark all tests in tests/integration/ with the 'integration' marker.
    """
    for item in items:
        if "tests/integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
