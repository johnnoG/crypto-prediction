# Quick Fixes & Immediate Action Items

## 🚨 CRITICAL - Fix Immediately

### 1. Install Missing Dependency ⚠️
```powershell
# Run this now:
.\.venv\Scripts\Activate.ps1
pip install aiofiles>=23.2.1
```
**Why:** Smart cache service will crash without this.

### 2. Create News Database Migration ⚠️
```powershell
cd backend
alembic revision --autogenerate -m "add_news_schema"
alembic upgrade head
```
**Why:** News endpoints may fail without proper database schema.

### 3. Verify Environment Variables ⚠️
Check your `.env` file has these critical settings:
```bash
# REQUIRED for production:
SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">

# IMPORTANT:
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/crypto
REDIS_URL=redis://localhost:6379

# OPTIONAL (for enhanced features):
COINGECKO_API_KEY=<your-key>
FIRECRAWL_API_KEY=<your-key>
```

---

## 🎨 FRONTEND - Add Missing UI Components

### Install shadcn/ui
```powershell
cd frontend
npx shadcn@latest init
# Follow prompts, choose defaults

# Add essential components:
npx shadcn@latest add button
npx shadcn@latest add card
npx shadcn@latest add table
npx shadcn@latest add toast
npx shadcn@latest add skeleton
```

### Add Chart Library
```powershell
cd frontend
npm install recharts
npm install --save-dev @types/recharts
```

---

## 🔒 SECURITY - Production Hardening

### 1. Update CORS Settings
Edit `backend/app/main.py` around line 92:
```python
# BEFORE (insecure):
response.headers["Access-Control-Allow-Origin"] = "*"

# AFTER (secure):
origin = request.headers.get("origin")
if origin in settings.get_allowed_origins_list():
    response.headers["Access-Control-Allow-Origin"] = origin
```

### 2. Add Rate Limiting
```powershell
pip install slowapi
```

Then in `backend/app/main.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

---

## 🧪 TESTING - Add Basic Tests

### Backend Tests
Create `backend/tests/test_prices.py`:
```python
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_get_prices():
    response = client.get("/prices?ids=bitcoin,ethereum")
    assert response.status_code == 200
    data = response.json()
    assert "bitcoin" in data
    assert "ethereum" in data

def test_health():
    response = client.get("/health/quick")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

Run with:
```powershell
pytest backend/tests/
```

### Frontend Tests
Create `frontend/src/lib/__tests__/api.test.ts`:
```typescript
import { describe, it, expect } from 'vitest';
import { apiClient } from '../api';

describe('API Client', () => {
  it('should have correct base URL', () => {
    expect(apiClient['baseUrl']).toBeDefined();
  });
});
```

Run with:
```powershell
cd frontend
npm test
```

---

## 📊 FORECASTS - Use Real Data

Update `backend/app/api/forecasts.py` around line 319:
```python
# BEFORE: Synthetic data
historical_prices = []
base_price = current_price
for j in range(50):
    volatility = 0.02
    change = random.gauss(0, volatility)
    base_price *= (1 + change)
    historical_prices.append(base_price)

# AFTER: Real OHLC data
from clients.coingecko_client import CoinGeckoClient
client = CoinGeckoClient()
try:
    ohlc_data = await client.get_coin_ohlc_by_id(
        coin_id=crypto_id,
        vs_currency="usd",
        days=90  # Get 90 days of history
    )
    # Extract closing prices
    historical_prices = [candle[4] for candle in ohlc_data]  # index 4 is close price
finally:
    await client.close()
```

---

## 🎯 NEXT DEVELOPMENT PRIORITIES

### Week 1:
- [ ] Fix critical issues above
- [ ] Add basic testing
- [ ] Install frontend components

### Week 2:
- [ ] Implement rate limiting
- [ ] Use real OHLC data for forecasts
- [ ] Add authentication guards to admin endpoints

### Week 3:
- [ ] Complete news database integration
- [ ] Add monitoring/logging
- [ ] Implement Prophet baseline model

---

## 🐛 Known Issues Reference

See `PROJECT_ANALYSIS.md` for detailed analysis of:
- Backend architecture issues
- Frontend missing dependencies
- Security concerns
- Database migration needs
- Testing coverage gaps

---

## 📞 Quick Commands Reference

### Start Everything:
```powershell
python main.py
```

### Backend Only (for debugging):
```powershell
cd backend
python app/main.py
```

### Frontend Only (for UI work):
```powershell
cd frontend
npm run dev
```

### Run Tests:
```powershell
# Backend
pytest

# Frontend
cd frontend
npm test
```

### Database Migrations:
```powershell
cd backend
alembic revision --autogenerate -m "description"
alembic upgrade head
```

### Check Dependencies:
```powershell
pip list | grep -i <package>
npm list <package>
```

---

## ✅ Verification Checklist

After applying fixes, verify:

- [ ] `python main.py` starts without errors
- [ ] Can access http://127.0.0.1:8000/docs
- [ ] Can access http://127.0.0.1:5173
- [ ] `/prices` endpoint returns data
- [ ] `/forecasts` endpoint returns data
- [ ] `/news` endpoint returns data
- [ ] Dark mode toggle works
- [ ] No console errors in browser

---

## 🆘 Troubleshooting

### "Module 'aiofiles' not found"
```powershell
pip install aiofiles
```

### "Connection to database failed"
```powershell
# Check PostgreSQL is running
# Or use SQLite fallback:
# DATABASE_URL=sqlite:///dev.db
```

### "Redis connection failed"
```powershell
# App will fallback to in-memory cache automatically
# To use Redis: install Redis for Windows or Memurai
```

### Frontend build errors
```powershell
cd frontend
rm -rf node_modules package-lock.json
npm install
```

---

**Need More Help?** 

See `PROJECT_ANALYSIS.md` for comprehensive documentation.

