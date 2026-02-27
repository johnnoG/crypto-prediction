# Frontend README

Crypto Forecast & Real‑Time Dashboard frontend built with React + Vite. This document covers what’s implemented, how it talks to the backend, and how to run it locally.

## What’s Implemented

- Single‑page dashboard with in-app page switching (no react-router).
- **Public landing page** — hero section ("The Future of Crypto Intelligence") is visible without signing in. Forecasts, News, Watchlist, and Portfolio are auth-gated: clicking them while signed out opens the sign-in modal.
- Real‑time crypto grid with streaming + polling fallback.
- Forecasting UI with real trained ML models (LightGBM, LSTM, Transformer, TCN, DLinear) replacing the previous ARIMA/ETS/SARIMA placeholders.
- News feed with sentiment highlights and pagination.
- User auth (JWT) + OAuth callback handling.
- Watchlist + alerts management UI.
- **Portfolio tracker** — full CRUD holdings manager backed by real API. Displays live prices via CoinGecko, calculates total portfolio value, total cost basis, and overall P&L per holding. Supports adding (15-coin dropdown), inline editing amount/avg buy price, and removing holdings. Loading, error, and empty states included.
- Global error boundaries, loading/skeleton states, and toasts.

## Tech Stack

- React 19 + TypeScript
- Vite
- Tailwind CSS + tailwind-merge + tailwindcss-animate
- TanStack Query (data fetching/cache)
- Recharts + Lightweight Charts
- Radix UI (toast)

## Project Structure

- `frontend/src/App.tsx` — app shell, providers, OAuth callback routing
- `frontend/src/components` — feature UI (dashboard, charts, news, forecasts)
- `frontend/src/components/pages` — page content for each section
- `frontend/src/contexts/AuthContext.tsx` — auth state + token handling
- `frontend/src/hooks` — data hooks + streaming
- `frontend/src/lib/api.ts` — API client + endpoint wrappers
- `frontend/src/index.css` — global styles + theme

## Pages & Key Components

- Home (`HomePage`): **public** — hero section only when signed out; full content (markets grid, forecasts, news, features) when signed in
- Forecasts (`ForecastsPage` + `ForecastPanel`): **requires sign-in** — live ML model selection and forecast display
- News (`NewsPage` + `NewsPanel`): **requires sign-in**
- Watchlist (`WatchlistPage`): **requires sign-in**
- Portfolio (`PortfolioPage`): **requires sign-in** — live holdings table with P&L, add/edit/delete per row, summary cards (total value, total cost, total gain/loss). Uses `usePortfolio`, `useAddHolding`, `useUpdateHolding`, `useDeleteHolding` hooks backed by real REST API.
- Alerts: `AlertsPage`
- Settings: `SettingsPage`
- Markets (`MarketsPage`): accessible via app routing but **removed from the nav bar** (charting widgets had no data source).

### Auth-gated Navigation (`Header.tsx`)

`PROTECTED_PAGES = ['forecasts', 'news', 'watchlist', 'portfolio']`

Nav bar items: Home → Forecasts → News → Watchlist → Portfolio

When an unauthenticated user clicks a protected nav item:
- The sign-in modal opens immediately
- Navigation does not occur
- A lock icon is shown next to each protected nav item label
- Lock icons disappear after signing in

Portfolio is also reachable from the **user dropdown menu** (avatar → "My Portfolio"), alongside My Watchlist and My Alerts.

## Backend Integration

### Base URL

The API base URL comes from `VITE_API_BASE_URL` in `frontend/env.example`.

Note: `frontend/src/contexts/AuthContext.tsx` currently uses a hard-coded base URL (`http://127.0.0.1:8000/api`) for auth calls, while `frontend/src/lib/api.ts` uses `VITE_API_BASE_URL`. If you change the backend host, update both.

### Main Endpoints Used

From `frontend/src/lib/api.ts` and hooks:

- Prices + market data:
  - `GET /api/prices` — cache-backed simple prices; returns `{ coin_id: { usd: price } }`
  - `GET /api/market/data`
- Forecasts:
  - `GET /api/forecasts`
  - `GET /api/forecasts/models`
- News:
  - `GET /api/news`
- Health + cache:
  - `GET /health`
  - `GET /api/cache/status`
- Rate limits:
  - `GET /api/rate-limit/status`
- Auth:
  - `POST /api/auth/signup`
  - `POST /api/auth/signin`
  - `POST /api/auth/refresh`
  - `GET /api/auth/me`
  - `PUT /api/auth/me`
  - `PUT /api/auth/me/password`
  - `DELETE /api/auth/me`
  - `GET /api/auth/verify-token`
- Alerts:
  - `POST /api/alerts`
  - `GET /api/alerts`
  - `DELETE /api/alerts/{id}`
- Watchlist:
  - `POST /api/watchlist`
  - `GET /api/watchlist`
  - `PUT /api/watchlist/{id}`
  - `DELETE /api/watchlist/{id}`
- Portfolio:
  - `GET /api/portfolio/holdings`
  - `POST /api/portfolio/holdings`
  - `PUT /api/portfolio/holdings/{id}`
  - `DELETE /api/portfolio/holdings/{id}`
- Streaming:
  - `GET /stream/snapshot` (polling fallback)
  - `GET /health/quick` (streaming preflight)

## Streaming Behavior

The `useWebSocketStream` hook currently defaults to polling `GET /stream/snapshot` and uses `GET /health/quick` as a readiness check. WebSocket support is scaffolded but intentionally bypassed for reliability.

## Environment Variables

See `frontend/env.example`:

- `VITE_API_BASE_URL` — backend base URL (default: `http://localhost:8000`)
- `VITE_DEV` — dev flag
- `VITE_ENABLE_DEVTOOLS` — React Query Devtools toggle

## Run Locally

From `frontend/`:

```bash
npm install
npm run dev
```

Build:

```bash
npm run build
```

Preview:

```bash
npm run preview
```

## Notes

- The UI forces dark mode in `frontend/src/App.tsx`.
- Auth tokens are stored in `localStorage` (`auth_tokens`, `auth_user`).
- Errors on protected endpoints trigger a client-side logout event.
- `ForecastPanel` fetches `/api/forecasts/models` on mount to determine which ML models are available from the backend. Model options shown in the UI (LightGBM, LSTM, Transformer, TCN, DLinear, Ensemble) reflect actual trained artifacts rather than static placeholders.
- `apiClient.getPrices()` calls `GET /api/prices` (cache-backed, returns CoinGecko simple-price format). Do **not** use `GET /api/crypto/prices` for live price data — that endpoint returns a different aggregated format and ignores the `ids` parameter.
