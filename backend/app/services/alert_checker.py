"""Background job: check active price alerts against the prices cache and trigger.

Runs on a schedule (default every 5 minutes) via APScheduler.
Only PRICE_TARGET alerts are evaluated; FORECAST_CHANGE / VOLATILITY are skipped.

Trigger logic:
  condition="above" or "reaches"  → fires when current_price >= target_price
  condition="below"               → fires when current_price <= target_price

On trigger:
  - Sets alert.status = TRIGGERED, alert.triggered_at = now(), alert.is_active = False
  - Sends an email via notification_service (no-ops if SMTP not configured)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# Uppercase ticker → CoinGecko ID (must match keys in prices_major_cryptos.json)
SYMBOL_TO_COINGECKO: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "LTC": "litecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "BNB": "binancecoin",
    "SOL": "solana",
    "ADA": "cardano",
    "LINK": "chainlink",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "NEAR": "near",
}

_CACHE_FILE = Path(__file__).parent.parent / "data" / "cache" / "prices_major_cryptos.json"


def _load_prices() -> Dict[str, float]:
    """Return {SYMBOL: usd_price} from the major-cryptos cache file."""
    if not _CACHE_FILE.exists():
        return {}
    try:
        raw: dict = json.loads(_CACHE_FILE.read_text())
        prices: Dict[str, float] = {}
        for symbol, coingecko_id in SYMBOL_TO_COINGECKO.items():
            entry = raw.get(coingecko_id)
            if isinstance(entry, dict) and "usd" in entry:
                prices[symbol] = float(entry["usd"])
        return prices
    except Exception as e:
        print(f"[ALERT CHECKER] Failed to load price cache: {e}")
        return {}


def _condition_met(condition: str | None, current: float, target: float) -> bool:
    if condition in ("above", "reaches"):
        return current >= target
    if condition == "below":
        return current <= target
    return False


async def check_and_trigger_alerts() -> None:
    """Main entry-point called by the scheduler every N seconds."""
    print("[ALERT CHECKER] Running price-alert check…")

    try:
        from db import SessionLocal
        from models.alert import UserAlert, AlertType, AlertStatus
        from models.user import User
        from services.notification_service import send_alert_email
    except ImportError:
        from app.db import SessionLocal
        from app.models.alert import UserAlert, AlertType, AlertStatus
        from app.models.user import User
        from app.services.notification_service import send_alert_email

    prices = _load_prices()
    if not prices:
        print("[ALERT CHECKER] No price data available — skipping")
        return

    db = SessionLocal()
    try:
        active_alerts = (
            db.query(UserAlert)
            .filter(
                UserAlert.is_active == True,          # noqa: E712
                UserAlert.status == AlertStatus.ACTIVE,
                UserAlert.alert_type == AlertType.PRICE_TARGET,
                UserAlert.target_price.isnot(None),
                UserAlert.condition.isnot(None),
            )
            .all()
        )

        if not active_alerts:
            print("[ALERT CHECKER] No active price alerts")
            return

        print(f"[ALERT CHECKER] Evaluating {len(active_alerts)} alert(s)")
        triggered = 0
        now = datetime.now(timezone.utc)

        for alert in active_alerts:
            # Expire stale alerts
            if alert.expires_at and alert.expires_at.replace(tzinfo=timezone.utc) < now:
                alert.status = AlertStatus.EXPIRED
                alert.is_active = False
                db.commit()
                print(f"[ALERT CHECKER] Alert {alert.id} expired ({alert.crypto_symbol})")
                continue

            symbol = alert.crypto_symbol.upper()
            current_price = prices.get(symbol)

            if current_price is None:
                print(f"[ALERT CHECKER] No price for {symbol} — skipping alert {alert.id}")
                continue

            if not _condition_met(alert.condition, current_price, alert.target_price):
                continue

            # --- TRIGGER ---
            alert.status = AlertStatus.TRIGGERED
            alert.triggered_at = now
            alert.is_active = False
            db.commit()
            triggered += 1

            print(
                f"[ALERT CHECKER] Triggered alert {alert.id}: "
                f"{symbol} @ ${current_price:,.2f} "
                f"(condition: {alert.condition} ${alert.target_price:,.2f})"
            )

            user = db.get(User, alert.user_id)
            if user:
                await send_alert_email(
                    to_email=user.email,
                    user_name=user.full_name or user.username,
                    alert=alert,
                    current_price=current_price,
                )

        print(f"[ALERT CHECKER] Done — {triggered}/{len(active_alerts)} triggered")

    except Exception as e:
        print(f"[ALERT CHECKER] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
