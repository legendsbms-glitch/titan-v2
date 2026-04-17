# FILE: core/economic_calendar.py
"""
TITAN v2.0 — Economic Calendar
Hardcoded 2026 key events + live scraping fallback, blackout management
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from core.config import RISK
from core.logger import get_logger

log = get_logger("EconomicCalendar")

# ── Impact Levels ─────────────────────────────────────────────────────────────
IMPACT = {
    "CRITICAL": 4,  # FOMC decision, rate decisions
    "HIGH":     3,  # CPI, NFP, GDP, PCE
    "MEDIUM":   2,  # PPI, Retail Sales, JOLTS, Fed speeches
    "LOW":      1,  # Housing data, minor reports
}

GOLD_IMPACT_MAP = {
    "FOMC":             "CRITICAL",
    "FOMC Statement":   "CRITICAL",
    "Fed Rate Decision":"CRITICAL",
    "CPI":              "HIGH",
    "Core CPI":         "HIGH",
    "NFP":              "HIGH",
    "Non-Farm Payrolls":"HIGH",
    "PCE":              "HIGH",
    "Core PCE":         "HIGH",
    "GDP":              "HIGH",
    "PPI":              "MEDIUM",
    "Retail Sales":     "MEDIUM",
    "Fed Speech":       "MEDIUM",
    "Powell Speech":    "HIGH",
    "ECB Rate Decision":"MEDIUM",
    "BOE Rate Decision":"MEDIUM",
    "JOLTS":            "MEDIUM",
    "ISM":              "LOW",
    "PMI":              "LOW",
}

# ── 2026 Key Economic Dates (UTC) ─────────────────────────────────────────────
# FOMC meetings 2026: Jan 27-28, Mar 17-18, May 5-6, Jun 9-10, Jul 28-29, Sep 15-16, Oct 27-28, Dec 8-9
# NFP: First Friday of each month
# CPI: Mid-month

KEY_EVENTS_2026: List[Dict] = [
    # FOMC 2026
    {"event": "FOMC Rate Decision", "date": "2026-01-29 19:00", "currency": "USD", "impact": "CRITICAL"},
    {"event": "FOMC Rate Decision", "date": "2026-03-18 18:00", "currency": "USD", "impact": "CRITICAL"},
    {"event": "FOMC Rate Decision", "date": "2026-05-06 18:00", "currency": "USD", "impact": "CRITICAL"},
    {"event": "FOMC Rate Decision", "date": "2026-06-10 18:00", "currency": "USD", "impact": "CRITICAL"},
    {"event": "FOMC Rate Decision", "date": "2026-07-29 18:00", "currency": "USD", "impact": "CRITICAL"},
    {"event": "FOMC Rate Decision", "date": "2026-09-16 18:00", "currency": "USD", "impact": "CRITICAL"},
    {"event": "FOMC Rate Decision", "date": "2026-10-28 18:00", "currency": "USD", "impact": "CRITICAL"},
    {"event": "FOMC Rate Decision", "date": "2026-12-09 19:00", "currency": "USD", "impact": "CRITICAL"},
    # NFP 2026 (first Friday)
    {"event": "Non-Farm Payrolls", "date": "2026-01-09 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-02-06 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-03-06 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-04-03 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-05-08 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-06-05 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-07-02 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-08-07 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-09-04 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-10-02 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-11-06 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Non-Farm Payrolls", "date": "2026-12-04 13:30", "currency": "USD", "impact": "HIGH"},
    # CPI 2026
    {"event": "CPI (YoY)",  "date": "2026-01-15 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-02-12 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-03-12 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-04-10 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-05-13 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-06-11 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-07-14 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-08-13 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-09-10 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-10-14 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-11-12 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "CPI (YoY)",  "date": "2026-12-10 13:30", "currency": "USD", "impact": "HIGH"},
    # PCE
    {"event": "Core PCE Price Index", "date": "2026-01-30 13:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Core PCE Price Index", "date": "2026-03-27 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Core PCE Price Index", "date": "2026-04-30 12:30", "currency": "USD", "impact": "HIGH"},
    {"event": "Core PCE Price Index", "date": "2026-05-29 12:30", "currency": "USD", "impact": "HIGH"},
]


def get_upcoming_events(hours: int = 48) -> List[Dict]:
    """Return events in the next N hours"""
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)

    upcoming = []
    for ev in KEY_EVENTS_2026:
        try:
            dt = datetime.strptime(ev["date"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            if now <= dt <= cutoff:
                mins_away = int((dt - now).total_seconds() / 60)
                upcoming.append({
                    **ev,
                    "datetime_utc": dt.isoformat(),
                    "minutes_away": mins_away,
                    "impact_level": IMPACT.get(ev.get("impact", "LOW"), 1),
                })
        except Exception:
            pass

    return sorted(upcoming, key=lambda x: x["minutes_away"])


def get_next_high_impact_event() -> Optional[Dict]:
    """Return the next HIGH or CRITICAL impact event"""
    now = datetime.now(timezone.utc)
    candidates = []
    for ev in KEY_EVENTS_2026:
        if ev.get("impact") in ("CRITICAL", "HIGH"):
            try:
                dt = datetime.strptime(ev["date"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                if dt > now:
                    mins_away = int((dt - now).total_seconds() / 60)
                    candidates.append({**ev, "datetime_utc": dt.isoformat(), "minutes_away": mins_away})
            except Exception:
                pass
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["minutes_away"])


def is_blackout_active() -> Dict:
    """Check if we're currently in a news blackout window"""
    pre  = RISK["blackout_pre_news_min"]
    post = RISK["blackout_post_news_min"]
    now  = datetime.now(timezone.utc)

    for ev in KEY_EVENTS_2026:
        if ev.get("impact") not in ("CRITICAL", "HIGH"):
            continue
        try:
            dt = datetime.strptime(ev["date"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            delta_min = (now - dt).total_seconds() / 60

            # Pre-event blackout
            if -pre <= delta_min <= 0:
                return {
                    "active":  True,
                    "reason":  f"Pre-event blackout: {ev['event']} in {int(-delta_min)}min",
                    "event":   ev["event"],
                    "impact":  ev["impact"],
                    "minutes": int(-delta_min),
                }
            # Post-event blackout
            elif 0 < delta_min <= post:
                return {
                    "active":  True,
                    "reason":  f"Post-event blackout: {ev['event']} {int(delta_min)}min ago",
                    "event":   ev["event"],
                    "impact":  ev["impact"],
                    "minutes": int(delta_min),
                }
        except Exception:
            pass

    return {"active": False, "reason": None}


def minutes_to_next_event() -> int:
    """Minutes until the next high-impact event"""
    ev = get_next_high_impact_event()
    if ev:
        return ev["minutes_away"]
    return 9999


def run() -> Dict:
    """Full calendar output"""
    upcoming      = get_upcoming_events(hours=48)
    next_event    = get_next_high_impact_event()
    blackout      = is_blackout_active()
    mins_to_next  = minutes_to_next_event()

    return {
        "blackout_active":      blackout["active"],
        "blackout_reason":      blackout.get("reason"),
        "mins_to_next_hi_impact": mins_to_next,
        "next_high_impact_event": next_event,
        "upcoming_48h":         upcoming,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
