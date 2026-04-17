# FILE: core/telegram_alerts.py
"""
TITAN v2.0 — Telegram Alert Integration
Send trade signals, performance reports, and alerts to Telegram
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests
from datetime import datetime, timezone
from typing import Dict, Optional

from core.logger import get_logger

log = get_logger("TelegramAlerts")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.debug("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set)")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": parse_mode,
        }
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            log.info("Telegram alert sent ✓")
            return True
        else:
            log.warning(f"Telegram send failed: {r.status_code} {r.text[:100]}")
            return False
    except Exception as e:
        log.warning(f"Telegram error: {e}")
        return False


def format_trade_alert(verdict: Dict) -> str:
    direction  = verdict.get("direction", "?")
    confidence = verdict.get("confidence", 0)
    entry      = verdict.get("entry")
    sl         = verdict.get("sl")
    tp         = verdict.get("tp")
    regime     = verdict.get("regime", "?")
    session    = verdict.get("session", "?")

    emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}.get(direction, "⚪")
    ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")

    lines = [
        f"⚡ <b>TITAN v2.0 — Gold Signal</b>",
        f"",
        f"{emoji} <b>{direction}</b> @ {ts}",
        f"📊 Confidence: <b>{confidence:.0%}</b>",
        f"🌍 Session: {session} | Regime: {regime}",
    ]

    if entry:
        lines.append(f"")
        lines.append(f"💰 Entry:  <code>${entry:.2f}</code>")
    if sl:
        lines.append(f"🛑 Stop:   <code>${sl:.2f}</code>")
    if tp:
        lines.append(f"🎯 Target: <code>${tp:.2f}</code>")
    if entry and sl and tp:
        rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0
        lines.append(f"📐 R:R: {rr:.1f}:1")

    warnings = verdict.get("soft_warnings", [])
    if warnings:
        lines.append(f"")
        lines.append(f"⚠️ <i>{warnings[0]}</i>")

    return "\n".join(lines)


def format_performance_report(perf: Dict) -> str:
    capital    = perf.get("capital", 0)
    pnl        = perf.get("total_pnl", 0)
    pnl_pct    = perf.get("total_return_pct", 0)
    trades     = perf.get("total_trades", 0)
    wr         = perf.get("win_rate", 0)
    pf         = perf.get("profit_factor", 0)

    trend = "📈" if pnl >= 0 else "📉"

    return (
        f"📋 <b>TITAN Paper Trading Report</b>\n\n"
        f"{trend} Capital: <b>${capital:,.2f}</b> ({pnl_pct:+.1f}%)\n"
        f"💵 Total P&L: <b>${pnl:+.2f}</b>\n"
        f"📊 Trades: {trades} | WR: {wr:.0%} | PF: {pf:.2f}\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


def format_macro_alert(macro: Dict) -> str:
    snapshot = macro.get("snapshot", {})
    div      = macro.get("divergences", [])

    lines = ["🌐 <b>TITAN Macro Alert</b>\n"]

    dxy = snapshot.get("dxy", {})
    if dxy.get("chg_1d"):
        lines.append(f"💵 DXY: {dxy['chg_1d']:+.2f}% today")

    vix = snapshot.get("vix", {})
    if vix.get("latest"):
        lines.append(f"😨 VIX: {vix['latest']:.1f} ({vix.get('chg_1d', 0):+.1f}%)")

    tnx = snapshot.get("tnx", {})
    if tnx.get("latest"):
        lines.append(f"📈 10Y Yield: {tnx['latest']:.2f}% ({tnx.get('chg_1d', 0):+.2f}%)")

    if div:
        lines.append(f"\n⚠️ Divergence: {div[0].get('msg', '')}")

    return "\n".join(lines)


def send_trade_alert(verdict: Dict) -> bool:
    msg = format_trade_alert(verdict)
    return send_message(msg)


def send_performance_report(perf: Dict) -> bool:
    msg = format_performance_report(perf)
    return send_message(msg)


def send_macro_alert(macro: Dict) -> bool:
    msg = format_macro_alert(macro)
    return send_message(msg)


def send_startup_message() -> bool:
    return send_message(
        "⚡ <b>TITAN v2.0 Online</b>\n"
        "9-Engine Gold Intelligence System started.\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
