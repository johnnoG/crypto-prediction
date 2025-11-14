# Test Suite

Comprehensive test suite for the Crypto Prediction backend.

## Structure

```
tests/
├── conftest.py              # Shared fixtures for all tests
├── pytest.ini               # Pytest configuration (in parent dir)
│
├── unit/                    # Unit tests - fast, isolated
│   ├── __init__.py
│   ├── conftest.py         # Auto-applies @unit marker
│   ├── test_models.py      # Database model tests
│   ├── test_cache.py       # Cache manager tests
│   └── test_config.py      # Configuration tests
│
└── integration/             # Integration tests - with dependencies
    ├── __init__.py
    ├── conftest.py         # Auto-applies @integration marker
    └── test_health.py      # Health endpoint tests
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Characteristics:**
- Fast execution (<10ms per test)
- Isolated from external dependencies
- Use mocks/fakes for dependencies
- Test individual components
- No database or cache required
- Automatically marked with `@pytest.mark.unit`

**Examples:**
- Model validation logic
- Configuration parsing
- Cache manager operations (with fake Redis)
- Business logic functions

**Running unit tests only:**
```bash
# All unit tests
pytest tests/unit/

# Or use marker
pytest -m unit

# Specific file
pytest tests/unit/test_models.py
```

### Integration Tests (`tests/integration/`)

**Characteristics:**
- May be slower (>10ms per test)
- Test component interactions
- May require database/cache
- Test API endpoints
- Test data flow between components
- Automatically marked with `@pytest.mark.integration`

**Examples:**
- API endpoint tests
- Database connectivity
- Cache connectivity
- Health checks
- End-to-end workflows

**Running integration tests only:**
```bash
# All integration tests
pytest tests/integration/

# Or use marker
pytest -m integration

# Specific file
pytest tests/integration/test_health.py
```

## Running Tests

### Quick Commands

```bash
# All tests
pytest

# Only unit tests (fast)
pytest tests/unit/
pytest -m unit

# Only integration tests
pytest tests/integration/
pytest -m integration

# With coverage
pytest --cov=app --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### By Directory

```bash
# Unit tests
cd tests/unit && pytest

# Integration tests
cd tests/integration && pytest

# Specific test file
pytest tests/unit/test_models.py

# Specific test class
pytest tests/unit/test_models.py::TestCryptocurrencyModel

# Specific test function
pytest tests/unit/test_models.py::TestCryptocurrencyModel::test_create_cryptocurrency
```

### By Markers

```bash
# Unit tests (fast, isolated)
pytest -m unit

# Integration tests
pytest -m integration

# Database tests
pytest -m db

# Cache tests
pytest -m cache

# Exclude slow tests
pytest -m "not slow"

# Combined markers
pytest -m "unit and cache"
pytest -m "integration and not slow"
```

## Test Organization Guidelines

### When to Write Unit Tests

Write unit tests for:
- ✅ Business logic
- ✅ Data validation
- ✅ Model methods
- ✅ Utility functions
- ✅ Configuration parsing
- ✅ Cache operations (with fake)

Place in: `tests/unit/`

### When to Write Integration Tests

Write integration tests for:
- ✅ API endpoints
- ✅ Database operations
- ✅ Service interactions
- ✅ Health checks
- ✅ Authentication flows
- ✅ Data pipelines

Place in: `tests/integration/`

## Fixtures

### Shared Fixtures (`tests/conftest.py`)

Available to all tests:
- `db` - SQLAlchemy session (in-memory SQLite)
- `test_cache` - Fake Redis instance
- `client` - FastAPI TestClient
- `sample_cryptocurrency` - Sample crypto data
- `sample_market_data` - Sample OHLCV data
- `sample_prediction` - Sample prediction

### Using Fixtures

```python
# Unit test with fake cache
def test_cache_operation(test_cache):
    test_cache.set("key", "value")
    assert test_cache.get("key") == "value"

# Integration test with database
def test_api_endpoint(client, sample_cryptocurrency):
    response = client.get(f"/crypto/{sample_cryptocurrency.symbol}")
    assert response.status_code == 200
```

## Best Practices

### Unit Tests

```python
# tests/unit/test_example.py

def test_pure_function():
    """Unit test - no dependencies."""
    result = calculate_profit(100, 110)
    assert result == 10

def test_with_fake(test_cache):
    """Unit test - fake dependency."""
    test_cache.set("price", 45000)
    assert test_cache.exists("price")
```

### Integration Tests

```python
# tests/integration/test_example.py

def test_api_endpoint(client):
    """Integration test - real API."""
    response = client.get("/health")
    assert response.status_code == 200

def test_database_flow(db, sample_cryptocurrency):
    """Integration test - database interaction."""
    crypto = db.query(Cryptocurrency).filter_by(
        symbol=sample_cryptocurrency.symbol
    ).first()
    assert crypto is not None
```

## Test Metrics

### Current Coverage

- **Overall:** 80%+
- **Unit Tests:** ~50 tests
- **Integration Tests:** ~30 tests
- **Total:** 80+ tests

### Performance Targets

- **Unit tests:** <5 seconds total
- **Integration tests:** <15 seconds total
- **Full suite:** <20 seconds total

## CI/CD

Tests run automatically on every push:

**GitHub Actions:**
- ✅ All unit tests
- ✅ All integration tests
- ✅ Coverage report
- ✅ Code quality checks

**Services in CI:**
- PostgreSQL 16
- Redis 7

## Writing New Tests

### 1. Decide Test Type

Is it testing a single component in isolation?
→ **Unit test** (`tests/unit/`)

Is it testing component interaction or needs real dependencies?
→ **Integration test** (`tests/integration/`)

### 2. Choose Appropriate File

- Models → `tests/unit/test_models.py`
- Cache → `tests/unit/test_cache.py`
- Config → `tests/unit/test_config.py`
- API → `tests/integration/test_health.py` or new file
- Database → New file in `tests/integration/`

### 3. Follow AAA Pattern

```python
def test_example():
    # Arrange - setup test data
    data = {"key": "value"}

    # Act - execute the code
    result = process_data(data)

    # Assert - verify result
    assert result["status"] == "success"
```

### 4. Use Descriptive Names

```python
# Good
def test_cryptocurrency_creation_sets_default_active_status()
def test_market_data_requires_unique_timestamp_per_crypto()

# Avoid
def test_crypto()
def test_data()
```

## Troubleshooting

### Tests Not Found

```bash
# Rebuild pytest cache
pytest --cache-clear

# Verify test discovery
pytest --collect-only
```

### Import Errors

```bash
# Verify PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from backend directory
cd backend && pytest
```

### Database Issues

```bash
# Tests use in-memory SQLite by default
# No database setup required

# If issues persist, check conftest.py fixtures
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest tests/unit/` | Run unit tests only |
| `pytest tests/integration/` | Run integration tests only |
| `pytest -m unit` | Run tests marked as unit |
| `pytest -m integration` | Run tests marked as integration |
| `pytest --cov=app` | Run with coverage |
| `pytest -v` | Verbose output |
| `pytest -x` | Stop on first failure |
| `pytest -k "test_name"` | Run tests matching name |
| `pytest --lf` | Run last failed tests |

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](../../docs/TESTING.md)
- [CI/CD Setup](../../docs/CI-CD-COMPLETION.md)
