# Testing Guide

Comprehensive guide for running tests and understanding the test suite.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Writing Tests](#writing-tests)
- [Best Practices](#best-practices)

## Overview

The project uses **pytest** as the testing framework with comprehensive coverage of:
- Database models and relationships
- Cache operations
- API endpoints
- Configuration management
- Integration scenarios

**Current Test Coverage:** 80%+ target

## Test Structure

Tests are organized into **unit** and **integration** directories:

```
backend/tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── README.md                # Test suite documentation
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

### Test Categories

#### Unit Tests (`tests/unit/`)
- **Fast** (<10ms per test)
- **Isolated** from external dependencies
- Use mocks/fakes for dependencies
- Test individual components
- Automatically marked with `@pytest.mark.unit`

#### Integration Tests (`tests/integration/`)
- May be **slower** (>10ms per test)
- Test **component interactions**
- May require database/cache
- Test API endpoints
- Automatically marked with `@pytest.mark.integration`

### Additional Markers

- **`@pytest.mark.db`** - Tests requiring database
- **`@pytest.mark.cache`** - Tests requiring Redis cache
- **`@pytest.mark.slow`** - Slow-running tests

## Running Tests

### Quick Start

```bash
# Run all tests
docker-compose exec backend pytest

# Run only unit tests (fast)
docker-compose exec backend pytest tests/unit/

# Run only integration tests
docker-compose exec backend pytest tests/integration/

# Run with verbose output
docker-compose exec backend pytest -v

# Run specific test file
docker-compose exec backend pytest tests/unit/test_models.py

# Run specific test
docker-compose exec backend pytest tests/unit/test_models.py::TestCryptocurrencyModel::test_create_cryptocurrency
```

### Local Development

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run unit tests only (fast, good for TDD)
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=app --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Running by Directory

```bash
# Unit tests (fast, isolated)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Specific test file
pytest tests/unit/test_models.py
pytest tests/integration/test_health.py
```

### Running by Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run database tests
pytest -m db

# Run cache tests
pytest -m cache

# Exclude slow tests
pytest -m "not slow"
```

### Advanced Options

```bash
# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run with specific verbosity
pytest -v -s  # -s shows print statements

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Generate multiple report formats
pytest --cov=app --cov-report=html --cov-report=xml --cov-report=term
```

## Test Coverage

### Viewing Coverage

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# View in browser
open htmlcov/index.html

# Terminal coverage report
pytest --cov=app --cov-report=term-missing
```

### Coverage Configuration

Coverage settings are in `pytest.ini`:

```ini
[coverage:run]
source = app
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

### Coverage Requirements

- **Minimum coverage:** 80%
- **Critical modules:** 90%+ (models, cache, database)
- **New code:** Must have tests
- **Bug fixes:** Must include regression tests

### Coverage Metrics

View coverage for specific modules:

```bash
# Coverage by module
pytest --cov=app --cov-report=term

# Coverage for specific module
pytest --cov=app.models --cov-report=term

# Missing lines report
pytest --cov=app --cov-report=term-missing
```

## CI/CD Pipeline

### GitHub Actions Workflows

The project includes two GitHub Actions workflows:

#### 1. CI Pipeline (`.github/workflows/ci.yml`)

Runs on every push and pull request:

**Jobs:**
- **Backend Tests** - Run full test suite with coverage
- **Code Quality** - Black formatting check
- **Database Migrations** - Verify migrations work
- **Security Scan** - Check for vulnerabilities
- **Docker Build** - Verify Docker image builds

**Services:**
- PostgreSQL 16
- Redis 7

**Triggers:**
- Push to `main`, `develop`, `SCRUM-*`, `feature/*` branches
- Pull requests to `main` and `develop`

#### 2. Coverage Badge (`.github/workflows/coverage-badge.yml`)

Runs on push to main:
- Generates coverage badge
- Updates `backend/coverage.svg`
- Commits badge to repository

### Viewing CI Results

1. Go to GitHub repository
2. Click "Actions" tab
3. View workflow runs
4. Click on a run to see detailed logs

### CI Environment Variables

CI uses these test database credentials:

```yaml
POSTGRES_USER: test_user
POSTGRES_PASSWORD: test_password
POSTGRES_DB: test_db
REDIS_URL: redis://localhost:6379/0
```

## Writing Tests

### Test Structure Template

```python
"""
Test module description.
"""

import pytest
from app.models import YourModel


class TestYourFeature:
    """Test cases for your feature."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = "test"

        # Act
        result = your_function(input_data)

        # Assert
        assert result == expected_value

    @pytest.mark.integration
    @pytest.mark.db
    def test_database_interaction(self, db):
        """Test database interaction."""
        # Use db fixture
        model = YourModel(field="value")
        db.add(model)
        db.commit()

        assert model.id is not None
```

### Using Fixtures

#### Available Fixtures

**Database Fixtures:**
```python
def test_with_database(db):
    """db provides SQLAlchemy session."""
    pass

def test_with_sample_data(sample_cryptocurrency):
    """Use pre-created test data."""
    assert sample_cryptocurrency.symbol == "BTC"
```

**Cache Fixtures:**
```python
def test_with_cache(test_cache):
    """test_cache provides fake Redis."""
    test_cache.set("key", "value")
    assert test_cache.get("key") == "value"
```

**API Fixtures:**
```python
def test_api_endpoint(client):
    """client provides FastAPI TestClient."""
    response = client.get("/health")
    assert response.status_code == 200
```

### Creating Custom Fixtures

Add to `conftest.py`:

```python
@pytest.fixture
def your_fixture():
    """Your fixture description."""
    # Setup
    resource = create_resource()

    yield resource

    # Teardown
    cleanup_resource(resource)
```

### Testing Best Practices

#### 1. Arrange-Act-Assert Pattern

```python
def test_example():
    # Arrange - Set up test data
    crypto = Cryptocurrency(symbol="BTC")

    # Act - Execute the code
    result = crypto.do_something()

    # Assert - Verify results
    assert result is not None
```

#### 2. Test One Thing

```python
# Good - Tests one specific behavior
def test_cryptocurrency_has_default_active_status():
    crypto = Cryptocurrency(symbol="BTC")
    assert crypto.is_active is True

# Avoid - Tests multiple behaviors
def test_cryptocurrency():  # Too broad
    crypto = Cryptocurrency(symbol="BTC")
    assert crypto.is_active is True
    assert crypto.symbol == "BTC"
    assert crypto.name is not None
```

#### 3. Descriptive Test Names

```python
# Good
def test_market_data_unique_timestamp_per_cryptocurrency()

# Bad
def test_market_data()
```

#### 4. Use Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("symbol,expected_name", [
    ("BTC", "Bitcoin"),
    ("ETH", "Ethereum"),
    ("SOL", "Solana"),
])
def test_cryptocurrency_names(symbol, expected_name):
    crypto = get_crypto(symbol)
    assert crypto.name == expected_name
```

#### 5. Test Edge Cases

```python
def test_with_empty_string():
    """Test with empty string."""
    pass

def test_with_none_value():
    """Test with None value."""
    pass

def test_with_max_length():
    """Test with maximum length input."""
    pass
```

#### 6. Test Error Handling

```python
def test_raises_error_on_invalid_input():
    with pytest.raises(ValueError):
        create_invalid_model()
```

### Testing Database Models

```python
@pytest.mark.unit
def test_model_creation(db):
    """Test creating a model."""
    model = YourModel(field="value")
    db.add(model)
    db.commit()
    db.refresh(model)

    assert model.id is not None
    assert model.field == "value"

@pytest.mark.unit
def test_model_constraint(db):
    """Test database constraints."""
    model1 = YourModel(unique_field="value")
    db.add(model1)
    db.commit()

    model2 = YourModel(unique_field="value")
    db.add(model2)

    with pytest.raises(IntegrityError):
        db.commit()
```

### Testing API Endpoints

```python
@pytest.mark.integration
def test_endpoint_success(client):
    """Test successful API call."""
    response = client.get("/api/v1/endpoint")

    assert response.status_code == 200
    data = response.json()
    assert "expected_field" in data

@pytest.mark.integration
def test_endpoint_validation(client):
    """Test input validation."""
    response = client.post("/api/v1/endpoint", json={
        "invalid": "data"
    })

    assert response.status_code == 422  # Validation error
```

### Testing Cache Operations

```python
@pytest.mark.cache
def test_cache_set_get(test_cache):
    """Test basic cache operations."""
    test_cache.set("key", {"data": "value"}, ttl=300)

    result = test_cache.get("key")

    assert result == {"data": "value"}
    assert test_cache.get_ttl("key") <= 300
```

## Best Practices

### General Guidelines

1. **Test behavior, not implementation**
   - Focus on what the code does, not how it does it

2. **Keep tests independent**
   - Each test should run in isolation
   - Use fixtures for shared setup

3. **Fast tests**
   - Unit tests should be very fast (<10ms)
   - Use mocks for external dependencies

4. **Readable tests**
   - Clear test names
   - Simple assertions
   - Good documentation

5. **Maintainable tests**
   - DRY principle with fixtures
   - Update tests when code changes
   - Remove obsolete tests

### Code Coverage Guidelines

- Aim for 80%+ overall coverage
- Critical paths should have 100% coverage
- Don't write tests just for coverage
- Focus on meaningful tests

### Performance Testing

```python
@pytest.mark.slow
def test_performance():
    """Test performance characteristics."""
    import time

    start = time.time()
    result = expensive_operation()
    duration = time.time() - start

    assert result is not None
    assert duration < 1.0  # Should complete in 1 second
```

### Debugging Tests

```bash
# Run with debugger
pytest --pdb

# Debug on failure
pytest --pdb -x

# Print output
pytest -s

# Verbose output
pytest -vv
```

### Common Pitfalls

1. **Don't use production database**
   - Always use test database
   - Clean up after tests

2. **Don't rely on test order**
   - Tests should run in any order
   - Use fixtures, not global state

3. **Don't skip cleanup**
   - Use fixtures with teardown
   - Clean up test data

4. **Don't test framework code**
   - Test your code, not FastAPI/SQLAlchemy

5. **Don't ignore warnings**
   - Fix deprecation warnings
   - Update dependencies

## Continuous Integration

### Pre-commit Checks

Before pushing code:

```bash
# Format code
black backend/app backend/tests

# Run linting
flake8 backend/app

# Run type checking
mypy backend/app --ignore-missing-imports

# Run tests
pytest

# Check coverage
pytest --cov=app --cov-fail-under=80
```

### Pull Request Requirements

- ✅ All tests pass
- ✅ Coverage ≥ 80%
- ✅ No linting errors
- ✅ Code formatted with Black
- ✅ New features have tests
- ✅ Bug fixes have regression tests

## Troubleshooting

### Tests Failing Locally

```bash
# Clean pytest cache
rm -rf .pytest_cache

# Clean coverage data
rm -rf .coverage htmlcov/

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.11+
```

### Database Issues

```bash
# Reset test database
docker-compose down -v
docker-compose up -d postgres

# Check database connection
docker-compose exec postgres psql -U test_user -d test_db
```

### Cache Issues

```bash
# Clear Redis
docker-compose exec redis redis-cli FLUSHALL

# Restart Redis
docker-compose restart redis
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites)

## Getting Help

If tests are failing:
1. Check the error message carefully
2. Run with `-vv` for verbose output
3. Check test isolation (run single test)
4. Review recent code changes
5. Check CI logs on GitHub
