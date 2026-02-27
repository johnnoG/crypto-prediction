"""Email notification service for triggered price alerts.

Uses Python's stdlib smtplib — no extra dependencies required.
Call send_alert_email(); it silently no-ops when SMTP is not configured.
"""
from __future__ import annotations

import asyncio
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.alert import UserAlert


def _build_html(user_name: str, alert: "UserAlert", current_price: float) -> str:
    direction = "above" if alert.condition in ("above", "reaches") else "below"
    arrow = "↑" if direction == "above" else "↓"
    price_color = "#16a34a" if direction == "above" else "#dc2626"
    return f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:Arial,sans-serif;">
  <div style="max-width:560px;margin:40px auto;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
    <div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);padding:32px;text-align:center;">
      <p style="color:#93c5fd;margin:0 0 8px;font-size:13px;letter-spacing:0.1em;text-transform:uppercase;">CryptoForecast Pro</p>
      <h1 style="color:#ffffff;margin:0;font-size:22px;font-weight:700;">🔔 Price Alert Triggered</h1>
    </div>
    <div style="padding:32px;">
      <p style="color:#374151;font-size:15px;margin:0 0 20px;">Hi <strong>{user_name}</strong>,</p>
      <p style="color:#374151;font-size:15px;margin:0 0 24px;">
        Your price alert for <strong>{alert.crypto_name} ({alert.crypto_symbol})</strong> has fired.
      </p>
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:24px;text-align:center;margin-bottom:24px;">
        <p style="color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 6px;">Current Price</p>
        <p style="color:{price_color};font-size:40px;font-weight:800;margin:0;line-height:1;">{arrow} ${current_price:,.2f}</p>
        <p style="color:#94a3b8;font-size:13px;margin:10px 0 0;">
          Alert condition: price <strong>{direction}</strong> ${alert.target_price:,.2f}
        </p>
      </div>
      {f'<p style="color:#6b7280;font-size:13px;font-style:italic;margin-bottom:20px;">"{alert.message}"</p>' if alert.message else ""}
      <p style="color:#94a3b8;font-size:12px;border-top:1px solid #f1f5f9;padding-top:20px;margin:0;">
        This alert has been marked as triggered and will no longer fire.
        You can create a new alert in the app at any time.
      </p>
    </div>
  </div>
</body>
</html>
"""


def _send_smtp(to_email: str, subject: str, html: str, smtp_host: str, smtp_port: int,
               smtp_user: str, smtp_password: str, smtp_from: str, smtp_tls: bool) -> None:
    """Blocking SMTP send — runs in a thread via asyncio.to_thread."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_from or smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    if smtp_tls:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_user, smtp_password)
            server.sendmail(msg["From"], to_email, msg.as_string())
    else:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(msg["From"], to_email, msg.as_string())


async def send_alert_email(
    to_email: str,
    user_name: str,
    alert: "UserAlert",
    current_price: float,
) -> None:
    """Send a triggered-alert email.

    Silently no-ops when SMTP is not configured (smtp_host / smtp_user empty).
    Email delivery failure is logged but never raises — the alert is already
    marked triggered in the DB regardless.
    """
    try:
        from config import get_settings
    except ImportError:
        from app.config import get_settings

    settings = get_settings()

    if not settings.smtp_host or not settings.smtp_user:
        print(f"[ALERTS] SMTP not configured — skipping email for alert {alert.id} ({alert.crypto_symbol})")
        return

    subject = (
        f"🔔 {alert.crypto_symbol} is {alert.condition} ${alert.target_price:,.2f} "
        f"— alert triggered"
    )
    html = _build_html(user_name, alert, current_price)

    try:
        await asyncio.to_thread(
            _send_smtp,
            to_email,
            subject,
            html,
            settings.smtp_host,
            settings.smtp_port,
            settings.smtp_user,
            settings.smtp_password,
            settings.smtp_from,
            settings.smtp_tls,
        )
        print(f"[ALERTS] Email sent → {to_email} | alert {alert.id} ({alert.crypto_symbol})")
    except Exception as e:
        print(f"[ALERTS] Email failed for alert {alert.id}: {e}")
