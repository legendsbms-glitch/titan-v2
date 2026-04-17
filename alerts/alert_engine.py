# FILE: alerts/alert_engine.py
"""
TITAN v2.0 — Alert Engine
Trade alerts, macro alerts, formatting, log management
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.config import ALERT_MIN_CONFIDENCE, ALERT_LOG_PATH, ALERT_JSON_PATH
from core.logger import get_logger

log = get_logger("AlertEngine")

os.makedirs("alerts", exist_ok=True)
os.makedirs("data",   exist_ok=True)


def format_alert_message(verdict: Dict, engine_results: Dict = None) -> str:
    engine_results = engine_results or {}
    direction  = verdict.get("direction", "?")
    confidence = verdict.get("confidence", 0)
    entry      = verdict.get("entry")
    sl         = verdict.get("sl")
    tp         = verdict.get("tp")
    regime     = verdict.get("regime", "?")
    session    = verdict.get("session", "?")

    emoji = {"BUY": "🟢 LONG", "SELL": "🔴 SHORT", "NEUTRAL": "⚪ NEUTRAL"}.get(direction, direction)

    lines = [
        "=" * 50,
        f"⚡ TITAN v2.0 ALERT — GOLD (XAUUSD)",
        f"{'=' * 50}",
        f"Signal:     {emoji}",
        f"Confidence: {confidence:.1%}",
        f"Regime:     {regime}",
        f"Session:    {session}",
    ]

    if entry:
        lines += [f"Entry:      ${entry:.2f}"]
    if sl:
        lines += [f"Stop Loss:  ${sl:.2f}"]
    if tp:
        lines += [f"Take Profit:${tp:.2f}"]
    if entry and sl and tp:
        rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0
        lines += [f"R:R Ratio:  {rr:.2f}:1"]

    # Key confluences from E1
    e1 = engine_results.get("engine1_price_matrix", {})
    confluences = e1.get("confluences", [])
    if confluences:
        lines += ["", "Key Confluences:"]
        for c in confluences[:3]:
            lines += [f"  • {c}"]

    # Warnings
    warnings = verdict.get("soft_warnings", [])
    if warnings:
        lines += ["", "⚠️  Warnings:"]
        for w in warnings[:2]:
            lines += [f"  • {w}"]

    lines += [
        "",
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 50,
    ]

    return "\n".join(lines)


def write_alert(message: str, alert_type: str = "TRADE", severity: str = "HIGH"):
    """Write alert to log and JSON"""
    ts    = datetime.now(timezone.utc).isoformat()
    entry = {"timestamp": ts, "type": alert_type, "severity": severity, "message": message}

    # Append to log
    try:
        with open(ALERT_LOG_PATH, "a") as f:
            f.write(f"\n[{ts}] [{alert_type}] [{severity}]\n{message}\n")
    except Exception as e:
        log.warning(f"Alert log write error: {e}")

    # Write latest to JSON
    try:
        with open(ALERT_JSON_PATH, "w") as f:
            json.dump(entry, f, indent=2)
    except Exception as e:
        log.warning(f"Alert JSON write error: {e}")

    log.info(f"Alert written: {alert_type} / {severity}")
    return entry


def get_recent_alerts(n: int = 20) -> List[Dict]:
    """Read last N alerts from log"""
    alerts = []
    try:
        if not os.path.exists(ALERT_LOG_PATH):
            return []
        with open(ALERT_LOG_PATH) as f:
            content = f.read()
        # Parse basic blocks
        blocks = content.strip().split("\n\n")
        for block in blocks[-n:]:
            if block.strip():
                alerts.append({"raw": block.strip()})
    except Exception:
        pass
    return list(reversed(alerts))


def check_verdict_alert(verdict: Dict) -> Optional[str]:
    """Should we alert on this verdict?"""
    if verdict.get("direction") == "NEUTRAL":
        return None
    if verdict.get("blocked"):
        return None
    if verdict.get("confidence", 0) < ALERT_MIN_CONFIDENCE:
        return None
    return "TRADE_SIGNAL"


def check_macro_alert(macro_result: Dict) -> Optional[str]:
    """Alert on significant macro moves"""
    snapshot = macro_result.get("snapshot", {})

    dxy  = snapshot.get("dxy", {})
    vix  = snapshot.get("vix", {})
    tnx  = snapshot.get("tnx", {})

    alerts = []
    if dxy.get("chg_1d", 0) and abs(dxy["chg_1d"]) > 0.5:
        direction = "surging" if dxy["chg_1d"] > 0 else "collapsing"
        alerts.append(f"DXY {direction} ({dxy['chg_1d']:+.2f}%)")

    if vix.get("latest", 0) and vix.get("chg_1d", 0):
        if abs(vix["chg_1d"]) > 15:
            alerts.append(f"VIX spike ({vix['chg_1d']:+.1f}%) — fear event")

    if tnx.get("chg_1d", 0) and abs(tnx["chg_1d"]) > 5:  # bps equivalent
        alerts.append(f"10Y yield moving sharply ({tnx['chg_1d']:+.2f}%)")

    return "MACRO_MOVE" if alerts else None


def run(verdict: Dict = None, engine_results: Dict = None) -> Dict:
    """Run alert checks on latest verdict"""
    engine_results = engine_results or {}
    results = {"alerts_sent": [], "alert_type": None}

    if verdict:
        alert_type = check_verdict_alert(verdict)
        if alert_type:
            msg = format_alert_message(verdict, engine_results)
            entry = write_alert(msg, alert_type=alert_type, severity="HIGH")
            results["alerts_sent"].append(entry)
            results["alert_type"] = alert_type
            results["message"] = msg
            log.info(f"🚨 TRADE ALERT: {verdict.get('direction')} @ {verdict.get('confidence'):.1%}")

    macro = engine_results.get("engine4_macro_correlation", {})
    if macro:
        macro_alert = check_macro_alert(macro)
        if macro_alert:
            macro_msg = f"MACRO ALERT\n{macro.get('divergences', [])}"
            write_alert(macro_msg, alert_type=macro_alert, severity="MEDIUM")
            results["alerts_sent"].append({"type": macro_alert})

    return results


if __name__ == "__main__":
    # Test alert
    test_verdict = {
        "direction": "BUY", "confidence": 0.78, "entry": 3250.0,
        "sl": 3238.0, "tp": 3268.0, "regime": "BULLISH", "session": "LONDON",
        "soft_warnings": ["Moderate confidence — half size"],
    }
    result = run(verdict=test_verdict)
    print(result.get("message", "No alert generated"))
