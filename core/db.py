# FILE: core/db.py
"""TITAN v2.0 - SQLAlchemy Core database layer"""
import json
import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Float, String, Text, DateTime,
    insert, select, desc, func
)
from sqlalchemy.engine import Engine

from core.config import DB_PATH
from core.logger import get_logger

log = get_logger("db")
_engine: Optional[Engine] = None
_meta   = MetaData()

# ── Table Definitions ─────────────────────────────────────────────────────────
t_engine_signals = Table("engine_signals", _meta,
    Column("id",         Integer, primary_key=True, autoincrement=True),
    Column("timestamp",  String(32)),
    Column("engine",     String(64)),
    Column("signal",     String(16)),
    Column("confidence", Float),
    Column("details",    Text),
)

t_verdicts = Table("verdicts", _meta,
    Column("id",         Integer, primary_key=True, autoincrement=True),
    Column("timestamp",  String(32)),
    Column("direction",  String(16)),
    Column("confidence", Float),
    Column("regime",     String(32)),
    Column("sl",         Float),
    Column("tp",         Float),
    Column("entry",      Float),
    Column("details",    Text),
)

t_trades = Table("trades", _meta,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("timestamp",    String(32)),
    Column("direction",    String(8)),
    Column("entry",        Float),
    Column("sl",           Float),
    Column("tp",           Float),
    Column("exit_price",   Float),
    Column("pnl",          Float),
    Column("outcome",      String(8)),
    Column("rr_actual",    Float),
    Column("duration_min", Float),
    Column("session",      String(16)),
    Column("regime",       String(32)),
    Column("notes",        Text),
)

t_engine_perf = Table("engine_performance", _meta,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("timestamp",    String(32)),
    Column("engine",       String(64)),
    Column("accuracy_7d",  Float),
    Column("accuracy_30d", Float),
    Column("weight",       Float),
)

t_economic_events = Table("economic_events", _meta,
    Column("id",        Integer, primary_key=True, autoincrement=True),
    Column("timestamp", String(32)),
    Column("event_name",String(128)),
    Column("currency",  String(8)),
    Column("impact",    String(16)),
    Column("forecast",  String(32)),
    Column("actual",    String(32)),
    Column("surprise",  Float),
)

t_market_snapshots = Table("market_snapshots", _meta,
    Column("id",        Integer, primary_key=True, autoincrement=True),
    Column("timestamp", String(32)),
    Column("price",     Float),
    Column("atr",       Float),
    Column("spread",    Float),
    Column("session",   String(16)),
    Column("regime",    String(32)),
    Column("details",   Text),
)


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    return _engine


def init_db():
    engine = _get_engine()
    _meta.create_all(engine)
    log.info("Database initialized.")


def _now() -> str:
    return datetime.utcnow().isoformat()


def _j(d: Any) -> str:
    try:
        return json.dumps(d, default=str)
    except Exception:
        return "{}"


# ── Write Functions ───────────────────────────────────────────────────────────

def log_signal(engine_name: str, signal: str, confidence: float, details: dict):
    try:
        with _get_engine().begin() as conn:
            conn.execute(insert(t_engine_signals).values(
                timestamp=_now(), engine=engine_name,
                signal=signal, confidence=confidence, details=_j(details)
            ))
    except Exception as e:
        log.warning(f"log_signal error: {e}")


def log_verdict(direction: str, confidence: float, regime: str,
                sl: float = None, tp: float = None, entry: float = None, details: dict = None):
    try:
        with _get_engine().begin() as conn:
            conn.execute(insert(t_verdicts).values(
                timestamp=_now(), direction=direction, confidence=confidence,
                regime=regime, sl=sl, tp=tp, entry=entry, details=_j(details or {})
            ))
    except Exception as e:
        log.warning(f"log_verdict error: {e}")


def log_trade(trade: dict):
    try:
        with _get_engine().begin() as conn:
            conn.execute(insert(t_trades).values(
                timestamp  = _now(),
                direction  = trade.get("direction"),
                entry      = trade.get("entry"),
                sl         = trade.get("sl"),
                tp         = trade.get("tp"),
                exit_price = trade.get("exit_price"),
                pnl        = trade.get("pnl"),
                outcome    = trade.get("outcome"),
                rr_actual  = trade.get("rr_actual"),
                duration_min = trade.get("duration_min"),
                session    = trade.get("session"),
                regime     = trade.get("regime"),
                notes      = trade.get("notes", ""),
            ))
    except Exception as e:
        log.warning(f"log_trade error: {e}")


def log_event(event: dict):
    try:
        with _get_engine().begin() as conn:
            conn.execute(insert(t_economic_events).values(
                timestamp  = _now(),
                event_name = event.get("event_name"),
                currency   = event.get("currency", "USD"),
                impact     = event.get("impact", "MEDIUM"),
                forecast   = str(event.get("forecast", "")),
                actual     = str(event.get("actual", "")),
                surprise   = event.get("surprise"),
            ))
    except Exception as e:
        log.warning(f"log_event error: {e}")


def log_snapshot(price: float, atr: float, session: str, regime: str,
                 spread: float = 0.0, details: dict = None):
    try:
        with _get_engine().begin() as conn:
            conn.execute(insert(t_market_snapshots).values(
                timestamp=_now(), price=price, atr=atr,
                spread=spread, session=session, regime=regime,
                details=_j(details or {})
            ))
    except Exception as e:
        log.warning(f"log_snapshot error: {e}")


# ── Read Functions ────────────────────────────────────────────────────────────

def get_trades_df() -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            result = conn.execute(select(t_trades).order_by(desc(t_trades.c.timestamp)))
            rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=result.keys())
    except Exception as e:
        log.warning(f"get_trades_df error: {e}")
        return pd.DataFrame()


def get_signals_df(limit: int = 1000) -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            result = conn.execute(
                select(t_engine_signals)
                .order_by(desc(t_engine_signals.c.timestamp))
                .limit(limit)
            )
            rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=result.keys())
    except Exception as e:
        log.warning(f"get_signals_df error: {e}")
        return pd.DataFrame()


def get_recent_verdicts(limit: int = 50) -> pd.DataFrame:
    try:
        with _get_engine().connect() as conn:
            result = conn.execute(
                select(t_verdicts)
                .order_by(desc(t_verdicts.c.timestamp))
                .limit(limit)
            )
            rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=result.keys())
    except Exception as e:
        log.warning(f"get_recent_verdicts error: {e}")
        return pd.DataFrame()


def get_engine_accuracy() -> Dict[str, float]:
    """Calculate per-engine accuracy vs trade outcomes (simplified match)"""
    from core.config import ENGINE_WEIGHTS
    try:
        signals = get_signals_df(limit=500)
        trades  = get_trades_df()
        if signals.empty or trades.empty:
            return {e: 0.5 for e in ENGINE_WEIGHTS}

        accuracy = {}
        for engine in ENGINE_WEIGHTS:
            eng_sigs = signals[signals["engine"] == engine]
            if eng_sigs.empty:
                accuracy[engine] = 0.5
                continue
            correct, total = 0, 0
            for _, sig in eng_sigs.iterrows():
                wins  = trades[(trades["outcome"] == "WIN")  & (trades["direction"] == sig["signal"])]
                loses = trades[(trades["outcome"] == "LOSS") & (trades["direction"] == sig["signal"])]
                if not wins.empty:
                    correct += 1; total += 1
                elif not loses.empty:
                    total += 1
            accuracy[engine] = round(correct / total, 3) if total > 0 else 0.5
        return accuracy
    except Exception as e:
        log.warning(f"get_engine_accuracy error: {e}")
        from core.config import ENGINE_WEIGHTS
        return {e: 0.5 for e in ENGINE_WEIGHTS}
