# Test Structure Reorganization

**Date:** 2025-11-14
**Change:** Organized tests into `unit/` and `integration/` directories

## Overview

Reorganized the test suite to follow industry best practices by separating **unit tests** from **integration tests** into dedicated directories.

## Benefits

### 1. Clear Separation of Concerns
- **Unit tests** are fast and isolated
- **Integration tests** may have external dependencies
- Developers know exactly where to add new tests

### 2. Faster Development Workflow
- Run only unit tests during TDD: `pytest tests/unit/`
- Quick feedback loop (<5 seconds)
- Run integration tests before committing

### 3. Better CI/CD
- Can run unit and integration tests in parallel
- Identify slow tests more easily
- Better test metrics and reporting

### 4. Industry Standard
- Follows pytest and testing best practices
- Familiar structure for other developers
- Scales well as test suite grows

## New Structure

```
backend/tests/
├── __init__.py
├── conftest.py              # Shared fixtures (db, cache, client, samples)
├── pytest.ini               # Pytest configuration (in backend/)
├── README.md                # Test suite documentation
│
├── unit/                    # Unit Tests
│   ├── __init__.py
│   ├── conftest.py         # Auto-applies @pytest.mark.unit
│   ├── test_models.py      # Database model tests (~30 tests)
│   ├── test_cache.py       # Cache manager tests (~20 tests)
│   └── test_config.py      # Configuration tests (~15 tests)
│
└── integration/             # Integration Tests
    ├── __init__.py
    ├── conftest.py         # Auto-applies @pytest.mark.integration
    └── test_health.py      # Health endpoint tests (~15 tests)
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Characteristics:**
- ⚡ Fast execution (<10ms per test)
- 🔒 Isolated from external dependencies
- 🎭 Use mocks/fakes for dependencies
- 🎯 Test individual components
- ✅ Automatically marked with `@pytest.mark.unit`

**Files:**
- `test_models.py` - SQLAlchemy model tests (with in-memory DB)
- `test_cache.py` - Cache manager tests (with fake Redis)
- `test_config.py` - Configuration and validation tests

**Examples:**
```python
# tests/unit/test_models.py
@pytest.mark.unit
def test_cryptocurrency_creation(db):
    """Test creating a cryptocurrency."""
    crypto = Cryptocurrency(symbol="BTC", name="Bitcoin")
    db.add(crypto)
    db.commit()
    assert crypto.id is not None
```

### Integration Tests (`tests/integration/`)

**Characteristics:**
- 🐢 May be slower (>10ms per test)
- 🔗 Test component interactions
- 💾 May require database/cache
- 🌐 Test API endpoints
- ✅ Automatically marked with `@pytest.mark.integration`

**Files:**
- `test_health.py` - API health endpoint tests

**Examples:**
```python
# tests/integration/test_health.py
@pytest.mark.integration
def test_health_check_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## Running Tests

### By Directory (Recommended)

```bash
# Unit tests only (fast, for TDD)
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# All tests
pytest
```

### By Marker (Alternative)

```bash
# Unit tests
pytest -m unit

# Integration tests
pytest -m integration
```

### Specific Files

```bash
# Specific unit test file
pytest tests/unit/test_models.py

# Specific integration test file
pytest tests/integration/test_health.py

# Specific test function
pytest tests/unit/test_models.py::TestCryptocurrencyModel::test_create_cryptocurrency
```

## Auto-Applied Markers

Tests are automatically marked based on their directory location:

**How it works:**
1. Each directory has a `conftest.py` file
2. The `pytest_collection_modifyitems` hook adds markers
3. All tests in `tests/unit/` get `@pytest.mark.unit`
4. All tests in `tests/integration/` get `@pytest.mark.integration`

**Benefits:**
- No need to manually add `@pytest.mark.unit` decorators
- Consistent marking across all tests
- Less boilerplate code

**Example conftest.py:**
```python
# tests/unit/conftest.py
import pytest

def pytest_collection_modifyitems(items):
    """Auto-mark all tests in tests/unit/ with 'unit' marker."""
    for item in items:
        if "tests/unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
```

## Migration Checklist

✅ **Completed:**
- [x] Created `tests/unit/` directory
- [x] Created `tests/integration/` directory
- [x] Moved `test_models.py` → `tests/unit/`
- [x] Moved `test_cache.py` → `tests/unit/`
- [x] Moved `test_config.py` → `tests/unit/`
- [x] Moved `test_health.py` → `tests/integration/`
- [x] Created `tests/unit/conftest.py` with auto-marker
- [x] Created `tests/integration/conftest.py` with auto-marker
- [x] Created `tests/README.md` with documentation
- [x] Updated `pytest.ini` with comments
- [x] Updated `docs/TESTING.md`
- [x] Updated `backend/README.md`
- [x] Verified all tests still run

## File Changes

### Created Files
```
backend/tests/unit/__init__.py
backend/tests/unit/conftest.py
backend/tests/integration/__init__.py
backend/tests/integration/conftest.py
backend/tests/README.md
docs/TEST-STRUCTURE-REORGANIZATION.md
```

### Moved Files
```
tests/test_models.py  → tests/unit/test_models.py
tests/test_cache.py   → tests/unit/test_cache.py
tests/test_config.py  → tests/unit/test_config.py
tests/test_health.py  → tests/integration/test_health.py
```

### Updated Files
```
backend/pytest.ini
backend/README.md
docs/TESTING.md
```

## Performance Comparison

### Before
```bash
pytest  # Runs all 80+ tests mixed together
# ~15-20 seconds
```

### After
```bash
pytest tests/unit/         # Run unit tests only
# ~5 seconds (fast feedback for TDD)

pytest tests/integration/  # Run integration tests
# ~10 seconds

pytest                     # Run all tests
# ~15-20 seconds (same total time)
```

## Best Practices

### When to Use Unit Tests

Write unit tests for:
- ✅ Business logic and calculations
- ✅ Data validation and parsing
- ✅ Model methods and properties
- ✅ Utility functions
- ✅ Configuration handling
- ✅ Cache operations (with fake Redis)

Place in: `tests/unit/`

### When to Use Integration Tests

Write integration tests for:
- ✅ API endpoint testing
- ✅ Database integration flows
- ✅ Service-to-service communication
- ✅ Health checks and monitoring
- ✅ Authentication flows
- ✅ End-to-end workflows

Place in: `tests/integration/`

## CI/CD Impact

### GitHub Actions
Tests run the same way in CI:
```yaml
# .github/workflows/ci.yml
- name: Run tests with coverage
  run: |
    pytest -v --cov=app --cov-report=xml
```

### Future Optimization
Can now run tests in parallel:
```yaml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/unit/

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres: ...
      redis: ...
    steps:
      - run: pytest tests/integration/
```

## Developer Workflow

### Test-Driven Development (TDD)

```bash
# 1. Write a failing unit test
# tests/unit/test_new_feature.py

# 2. Run unit tests (fast feedback)
pytest tests/unit/test_new_feature.py

# 3. Implement feature

# 4. Run unit tests again
pytest tests/unit/

# 5. Write integration test
# tests/integration/test_new_feature.py

# 6. Run integration test
pytest tests/integration/test_new_feature.py

# 7. Run all tests before commit
pytest
```

### Pre-Commit Checklist

```bash
# Fast unit tests
pytest tests/unit/

# Format code
black backend/app backend/tests

# Lint
flake8 backend/app

# Full test suite
pytest --cov=app --cov-fail-under=80

# Commit
git add .
git commit -m "feat: add new feature with tests"
git push
```

## Statistics

### Test Distribution

```
Total Tests: 80+

Unit Tests (tests/unit/):
├── test_models.py    ~30 tests
├── test_cache.py     ~20 tests
└── test_config.py    ~15 tests
Total: ~65 tests (81%)

Integration Tests (tests/integration/):
└── test_health.py    ~15 tests
Total: ~15 tests (19%)
```

### Execution Time

```
Unit Tests:       ~5 seconds  (81% of tests)
Integration Tests: ~10 seconds (19% of tests)
Total:            ~15 seconds
```

## Future Additions

As the project grows, add tests to appropriate directories:

**Unit Tests:**
```
tests/unit/
├── test_models.py          # Existing
├── test_cache.py           # Existing
├── test_config.py          # Existing
├── test_utils.py           # NEW - Utility functions
├── test_validators.py      # NEW - Data validators
└── test_services.py        # NEW - Business logic
```

**Integration Tests:**
```
tests/integration/
├── test_health.py          # Existing
├── test_api_crypto.py      # NEW - Crypto API endpoints
├── test_api_predictions.py # NEW - Prediction endpoints
├── test_database_flows.py  # NEW - Complex DB workflows
└── test_cache_integration.py # NEW - Real Redis tests
```

## References

- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Testing Guide](./TESTING.md)
- [Test Suite README](../backend/tests/README.md)

## Conclusion

The test suite is now better organized, following industry best practices:

✅ **Clear separation** between unit and integration tests
✅ **Faster development** with quick unit test feedback
✅ **Auto-marked** tests based on directory
✅ **Well documented** with comprehensive README
✅ **Scalable** structure for future growth

This organization makes the test suite more maintainable and easier to work with as the project grows.
