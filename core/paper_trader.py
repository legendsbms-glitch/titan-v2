# FILE: core/paper_trader.py
"""
TITAN v2.0 — Paper Trading Engine
Simulates trades from live verdicts, tracks P&L in real-time,
auto-manages positions, sends performance reports
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import yfinance as yf

from core.config import GOLD_SYMBOL, RISK
from core.logger import get_logger
from core.db import log_trade, get_trades_df, init_db

log = get_logger("PaperTrader")

PAPER_STATE_PATH = "data/paper_state.json"
os.makedirs("data", exist_ok=True)


def load_state() -> Dict:
    try:
        if os.path.exists(PAPER_STATE_PATH):
            with open(PAPER_STATE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "capital":       10000.0,
        "open_position": None,
        "trades":        [],
        "daily_pnl":     0.0,
        "consecutive_losses": 0,
        "trades_today":  0,
        "last_date":     None,
    }


def save_state(state: Dict):
    try:
        with open(PAPER_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"State save error: {e}")


def get_live_price() -> float:
    try:
        df = yf.download(GOLD_SYMBOL, period="1d", interval="1m",
                         progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def open_position(state: Dict, verdict: Dict) -> Dict:
    """Open a paper trade based on verdict"""
    direction  = verdict.get("direction")
    entry      = verdict.get("entry") or verdict.get("current_price")
    sl         = verdict.get("sl")
    tp         = verdict.get("tp")
    confidence = verdict.get("confidence", 0)
    session    = verdict.get("session", "UNKNOWN")
    regime     = verdict.get("regime", "UNKNOWN")

    if not all([direction, entry, sl, tp]):
        return state

    if direction not in ("BUY", "SELL"):
        return state

    # Position sizing: risk 1% of capital
    sl_dist    = abs(entry - sl)
    risk_amt   = state["capital"] * (RISK["max_position_pct"] / 100)
    units      = risk_amt / sl_dist if sl_dist > 0 else 0

    position = {
        "id":           f"PT_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "direction":    direction,
        "entry":        round(float(entry), 2),
        "sl":           round(float(sl), 2),
        "tp":           round(float(tp), 2),
        "units":        round(units, 4),
        "risk_amount":  round(risk_amt, 2),
        "opened_at":    datetime.now(timezone.utc).isoformat(),
        "session":      session,
        "regime":       regime,
        "confidence":   confidence,
        "status":       "OPEN",
    }

    state["open_position"] = position
    state["trades_today"]  = state.get("trades_today", 0) + 1
    log.info(f"📊 Paper trade opened: {direction} {units:.2f} units @ ${entry:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}")
    save_state(state)
    return state


def check_and_close(state: Dict) -> Dict:
    """Check if open position hit SL or TP"""
    pos = state.get("open_position")
    if not pos or pos.get("status") != "OPEN":
        return state

    current = get_live_price()
    if current == 0:
        return state

    direction = pos["direction"]
    entry     = pos["entry"]
    sl        = pos["sl"]
    tp        = pos["tp"]
    units     = pos["units"]

    outcome = None
    exit_price = None

    if direction == "BUY":
        if current <= sl:
            outcome, exit_price = "LOSS", sl
        elif current >= tp:
            outcome, exit_price = "WIN", tp
    elif direction == "SELL":
        if current >= sl:
            outcome, exit_price = "LOSS", sl
        elif current <= tp:
            outcome, exit_price = "WIN", tp

    if outcome:
        if direction == "BUY":
            pnl = (exit_price - entry) * units
        else:
            pnl = (entry - exit_price) * units

        sl_dist = abs(entry - sl)
        rr_actual = abs(exit_price - entry) / sl_dist if sl_dist > 0 else 0
        if outcome == "LOSS":
            rr_actual = -1.0

        # Update state
        state["capital"]    += pnl
        state["daily_pnl"]  = state.get("daily_pnl", 0) + pnl

        if outcome == "LOSS":
            state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        else:
            state["consecutive_losses"] = 0

        duration = _duration_minutes(pos.get("opened_at"))

        closed_trade = {
            **pos,
            "exit_price": round(exit_price, 2),
            "pnl":        round(pnl, 2),
            "outcome":    outcome,
            "rr_actual":  round(rr_actual, 2),
            "closed_at":  datetime.now(timezone.utc).isoformat(),
            "duration_min": duration,
            "status":     "CLOSED",
        }

        state["open_position"] = None
        state.setdefault("trades", []).append(closed_trade)

        # Log to DB
        log_trade({
            "direction":    pos["direction"],
            "entry":        pos["entry"],
            "sl":           pos["sl"],
            "tp":           pos["tp"],
            "exit_price":   exit_price,
            "pnl":          pnl,
            "outcome":      outcome,
            "rr_actual":    rr_actual,
            "duration_min": duration,
            "session":      pos.get("session"),
            "regime":       pos.get("regime"),
            "notes":        f"Paper trade | Confidence: {pos.get('confidence',.0):.0%}",
        })

        emoji = "✅" if outcome == "WIN" else "❌"
        log.info(f"{emoji} Paper trade closed: {outcome} | PnL: ${pnl:+.2f} | RR: {rr_actual:.2f} | Capital: ${state['capital']:.2f}")
        save_state(state)

    return state


def _duration_minutes(opened_at: str) -> float:
    try:
        opened = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
        now    = datetime.now(timezone.utc)
        return round((now - opened).total_seconds() / 60, 1)
    except Exception:
        return 0.0


def get_paper_performance(state: Dict) -> Dict:
    """Compute paper trading performance summary"""
    trades = state.get("trades", [])
    closed = [t for t in trades if t.get("status") == "CLOSED"]

    if not closed:
        return {
            "capital":     state.get("capital", 10000),
            "total_trades": 0,
            "message":     "No closed paper trades yet",
        }

    import numpy as np
    pnl_list = [t["pnl"] for t in closed if "pnl" in t]
    wins     = [p for p in pnl_list if p > 0]
    losses   = [p for p in pnl_list if p < 0]

    return {
        "capital":        round(state.get("capital", 10000), 2),
        "starting_capital": 10000,
        "total_pnl":      round(sum(pnl_list), 2),
        "total_return_pct": round(sum(pnl_list) / 10000 * 100, 2),
        "total_trades":   len(closed),
        "win_rate":       round(len(wins) / len(pnl_list), 3) if pnl_list else 0,
        "avg_win":        round(float(np.mean(wins)), 2)   if wins   else 0,
        "avg_loss":       round(float(np.mean(losses)), 2) if losses else 0,
        "profit_factor":  round(sum(wins) / abs(sum(losses)), 3) if sum(losses) != 0 else 999,
        "open_position":  state.get("open_position"),
        "consecutive_losses": state.get("consecutive_losses", 0),
        "daily_pnl":      round(state.get("daily_pnl", 0), 2),
    }


def run(verdict: Dict = None) -> Dict:
    """Main paper trader loop: check open positions, open new if verdict says so"""
    import pandas as pd
    state = load_state()

    # Reset daily counters if new day
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_date") != today:
        state["daily_pnl"]    = 0.0
        state["trades_today"] = 0
        state["last_date"]    = today

    # Check existing position
    state = check_and_close(state)

    # Open new trade if verdict is tradeable and no open position
    if verdict and verdict.get("tradeable") and not state.get("open_position"):
        # Risk checks
        daily_loss_pct = abs(state["daily_pnl"]) / state["capital"] * 100
        if daily_loss_pct < RISK["max_daily_loss_pct"]:
            if state.get("consecutive_losses", 0) < RISK["max_consecutive_losses"]:
                if state.get("trades_today", 0) < RISK["max_trades_per_day"]:
                    state = open_position(state, verdict)

    perf = get_paper_performance(state)
    save_state(state)
    return perf
