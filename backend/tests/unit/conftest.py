"""
Unit Test Configuration

Automatically applies 'unit' marker to all tests in this directory.
"""

import pytest


def pytest_collection_modifyitems(items):
    """
    Automatically mark all tests in tests/unit/ with the 'unit' marker.
    """
    for item in items:
        if "tests/unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
