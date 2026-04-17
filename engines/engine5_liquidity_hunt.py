# FILE: engines/engine5_liquidity_hunt.py
"""
TITAN v2.0 — Engine 5: Liquidity Hunt (Full ICT)
BSL/SSL pools, stop hunts, Judas swings, retail traps, liquidity voids
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import List, Dict

from core.config import GOLD_SYMBOL
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine5.LiquidityHunt")


def _fetch(symbol: str = GOLD_SYMBOL, period: str = "10d", interval: str = "15m") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df.dropna()


# ── Swing Points ──────────────────────────────────────────────────────────────

def find_swing_highs_lows(df: pd.DataFrame, lookback: int = 4) -> Dict:
    """Find significant swing highs and lows"""
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    sh, sl = [], []
    for i in range(lookback, n - lookback):
        if all(highs[i] >= highs[i-k] for k in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+k] for k in range(1, lookback+1)):
            sh.append({"price": float(highs[i]), "index": i, "ts": str(df.index[i])})
        if all(lows[i] <= lows[i-k] for k in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+k] for k in range(1, lookback+1)):
            sl.append({"price": float(lows[i]), "index": i, "ts": str(df.index[i])})
    return {"swing_highs": sh, "swing_lows": sl}


# ── Equal Levels (Liquidity Pools) ───────────────────────────────────────────

def find_equal_levels(df: pd.DataFrame, tolerance: float = 0.0012) -> Dict:
    """Equal highs and lows = liquidity pools (clustered stops)"""
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    eq_highs, eq_lows = [], []

    for i in range(n - 1):
        for j in range(i + 2, min(i + 30, n)):
            if highs[i] > 0 and abs(highs[i] - highs[j]) / highs[i] < tolerance:
                level = round((highs[i] + highs[j]) / 2, 2)
                if not any(abs(e["price"] - level) / level < tolerance for e in eq_highs):
                    eq_highs.append({
                        "price": level,
                        "touches": 2,
                        "type": "EQUAL_HIGHS",
                        "ts_first": str(df.index[i]),
                        "ts_last":  str(df.index[j]),
                    })
            if lows[i] > 0 and abs(lows[i] - lows[j]) / lows[i] < tolerance:
                level = round((lows[i] + lows[j]) / 2, 2)
                if not any(abs(e["price"] - level) / level < tolerance for e in eq_lows):
                    eq_lows.append({
                        "price": level,
                        "touches": 2,
                        "type": "EQUAL_LOWS",
                        "ts_first": str(df.index[i]),
                        "ts_last":  str(df.index[j]),
                    })

    return {
        "bsl": eq_highs[-5:],  # Buy-side liquidity above equal highs
        "ssl": eq_lows[-5:],   # Sell-side liquidity below equal lows
    }


# ── Stop Hunt Detection ───────────────────────────────────────────────────────

def detect_stop_hunt_candles(df: pd.DataFrame) -> List[Dict]:
    """Long wicks through key levels then close back inside = stop hunt"""
    hunts = []
    if len(df) < 5:
        return hunts

    recent = df.tail(15)
    for i in range(len(recent)):
        row   = recent.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body  = abs(c - o)
        rng   = h - l
        if rng < 1e-6:
            continue

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick_ratio = (rng - body) / rng

        if wick_ratio > 0.65:
            if upper_wick > lower_wick and upper_wick > body * 1.5:
                hunts.append({
                    "type":      "BEARISH_STOP_HUNT",
                    "price":     round(h, 2),
                    "timestamp": str(recent.index[i]),
                    "wick_ratio": round(wick_ratio, 3),
                    "msg":       "Upper wick stop hunt — stops above taken, likely reversal down",
                    "bias":      "SELL",
                })
            elif lower_wick > upper_wick and lower_wick > body * 1.5:
                hunts.append({
                    "type":      "BULLISH_STOP_HUNT",
                    "price":     round(l, 2),
                    "timestamp": str(recent.index[i]),
                    "wick_ratio": round(wick_ratio, 3),
                    "msg":       "Lower wick stop hunt — stops below taken, likely reversal up",
                    "bias":      "BUY",
                })

    return hunts


# ── Judas Swing ───────────────────────────────────────────────────────────────

def detect_judas_swing(df: pd.DataFrame) -> List[Dict]:
    """
    Judas Swing: session open moves one direction strongly,
    then completely reverses — trapping retail traders
    """
    patterns = []
    if df.empty:
        return patterns

    try:
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index, utc=True)

        for session_name, (h_start, h_end) in [("LONDON", (8, 10)), ("NEW_YORK", (13, 15))]:
            session_df = df_copy[(df_copy.index.hour >= h_start) & (df_copy.index.hour < h_end)]
            if len(session_df) < 3:
                continue

            open_price  = float(session_df["open"].iloc[0])
            first_close = float(session_df["close"].iloc[0])
            last_close  = float(session_df["close"].iloc[-1])

            first_move = first_close - open_price
            total_move = last_close - open_price

            # Reversal if total move is more than 50% opposite to first move
            if abs(first_move) > 0 and abs(total_move) > 0:
                if first_move > 0 and total_move < -first_move * 0.5:
                    patterns.append({
                        "type":    "JUDAS_SWING_BEARISH",
                        "session": session_name,
                        "first_move": round(first_move, 2),
                        "total_move": round(total_move, 2),
                        "msg": f"{session_name}: initial pump reversed — bearish trap",
                        "bias": "SELL",
                    })
                elif first_move < 0 and total_move > -first_move * 0.5:
                    patterns.append({
                        "type":    "JUDAS_SWING_BULLISH",
                        "session": session_name,
                        "first_move": round(first_move, 2),
                        "total_move": round(total_move, 2),
                        "msg": f"{session_name}: initial dump reversed — bullish trap",
                        "bias": "BUY",
                    })
    except Exception as e:
        log.debug(f"Judas swing error: {e}")

    return patterns


# ── Liquidity Void ────────────────────────────────────────────────────────────

def detect_liquidity_void(df: pd.DataFrame, min_gap_pct: float = 0.002) -> List[Dict]:
    """Price areas with no candle bodies — fast move zones (will be revisited)"""
    voids = []
    if len(df) < 3:
        return voids

    for i in range(1, len(df)):
        prev_c  = float(df["close"].iloc[i-1])
        curr_o  = float(df["open"].iloc[i])
        curr_c  = float(df["close"].iloc[i])
        gap     = abs(curr_o - prev_c)

        if gap / prev_c >= min_gap_pct:
            direction = "UP" if curr_o > prev_c else "DOWN"
            voids.append({
                "type":      f"VOID_{direction}",
                "top":       round(max(curr_o, prev_c), 2),
                "bottom":    round(min(curr_o, prev_c), 2),
                "gap_pct":   round(gap / prev_c * 100, 4),
                "timestamp": str(df.index[i]),
                "filled":    False,
            })

    # Check if voids are filled by current price
    current = float(df["close"].iloc[-1])
    for v in voids:
        if v["bottom"] <= current <= v["top"]:
            v["filled"] = True

    return voids[-6:]


# ── Retail Trap ───────────────────────────────────────────────────────────────

def detect_retail_traps(df: pd.DataFrame) -> List[Dict]:
    """
    Obvious S/R levels that get swept and reversed — retail traders get trapped
    Turtle Soup pattern: breakout of 20-bar H/L that immediately fails
    """
    traps = []
    if len(df) < 25:
        return traps

    period = 20
    recent = df.tail(10)

    for i in range(len(recent)):
        row = recent.iloc[i]
        idx = len(df) - len(recent) + i

        if idx < period:
            continue

        lookback   = df.iloc[idx-period:idx]
        period_high = float(lookback["high"].max())
        period_low  = float(lookback["low"].min())

        bar_high = float(row["high"])
        bar_low  = float(row["low"])
        bar_close = float(row["close"])

        # Price broke above 20-bar high but closed below (turtle soup short)
        if bar_high > period_high and bar_close < period_high:
            traps.append({
                "type":        "TURTLE_SOUP_BEARISH",
                "broken_level": round(period_high, 2),
                "close":        round(bar_close, 2),
                "timestamp":    str(recent.index[i]),
                "msg":          f"Breakout of {period}-bar high failed — retail longs trapped",
                "bias":         "SELL",
            })

        # Price broke below 20-bar low but closed above (turtle soup long)
        elif bar_low < period_low and bar_close > period_low:
            traps.append({
                "type":        "TURTLE_SOUP_BULLISH",
                "broken_level": round(period_low, 2),
                "close":        round(bar_close, 2),
                "timestamp":    str(recent.index[i]),
                "msg":          f"Breakdown of {period}-bar low failed — retail shorts trapped",
                "bias":         "BUY",
            })

    return traps


# ── Proximity Score ───────────────────────────────────────────────────────────

def proximity_score(price: float, pools: List[Dict], threshold_pct: float = 0.003) -> Dict:
    """How close is price to the nearest liquidity pool?"""
    if not pools:
        return {"nearest": None, "distance_pct": None, "in_range": False}

    nearest = min(pools, key=lambda p: abs(p["price"] - price))
    dist_pct = abs(nearest["price"] - price) / price

    return {
        "nearest":      nearest,
        "distance_pct": round(dist_pct * 100, 4),
        "in_range":     dist_pct < threshold_pct,
        "side":         "ABOVE" if nearest["price"] > price else "BELOW",
    }


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("Running Engine 5: Liquidity Hunt")
    try:
        df           = _fetch(period="10d", interval="15m")
        df_1h        = _fetch(period="30d", interval="1h")

        if df.empty:
            return {"signal": "ERROR", "confidence": 0.0, "error": "No data"}

        current_price = float(df["close"].iloc[-1])

        swings      = find_swing_highs_lows(df)
        eq_levels   = find_equal_levels(df)
        stop_hunts  = detect_stop_hunt_candles(df)
        judas       = detect_judas_swing(df)
        voids       = detect_liquidity_void(df_1h if not df_1h.empty else df)
        traps       = detect_retail_traps(df)

        all_bsl = eq_levels["bsl"]
        all_ssl = eq_levels["ssl"]

        bsl_prox = proximity_score(current_price, all_bsl)
        ssl_prox = proximity_score(current_price, all_ssl)

        score = 0.0
        bias_votes = {"BUY": 0, "SELL": 0}
        notes = []

        # Stop hunts
        for sh in stop_hunts[-2:]:
            bias = sh.get("bias")
            if bias == "BUY":
                score += 0.30
                bias_votes["BUY"] += 1
                notes.append(sh["msg"])
            elif bias == "SELL":
                score -= 0.30
                bias_votes["SELL"] += 1
                notes.append(sh["msg"])

        # Judas swings
        for j in judas:
            if j["bias"] == "BUY":
                score += 0.25
                bias_votes["BUY"] += 1
                notes.append(j["msg"])
            elif j["bias"] == "SELL":
                score -= 0.25
                bias_votes["SELL"] += 1
                notes.append(j["msg"])

        # Retail traps
        for t in traps[-2:]:
            if t["bias"] == "BUY":
                score += 0.20
                bias_votes["BUY"] += 1
                notes.append(t["msg"])
            elif t["bias"] == "SELL":
                score -= 0.20
                bias_votes["SELL"] += 1
                notes.append(t["msg"])

        # Proximity to pools
        if bsl_prox["in_range"]:
            score += 0.10
            notes.append(f"Near BSL pool at {bsl_prox['nearest']['price']}")
        if ssl_prox["in_range"]:
            score -= 0.10
            notes.append(f"Near SSL pool at {ssl_prox['nearest']['price']}")

        score = max(-1.0, min(1.0, score))

        if score > 0.15:
            signal     = "BUY"
            confidence = min(0.5 + score * 0.42, 0.92)
        elif score < -0.15:
            signal     = "SELL"
            confidence = min(0.5 + abs(score) * 0.42, 0.92)
        else:
            signal     = "NEUTRAL"
            confidence = 0.40

        all_patterns = stop_hunts + judas + traps
        result = {
            "signal":           signal,
            "confidence":       round(confidence, 3),
            "score":            round(score, 3),
            "current_price":    current_price,
            "notes":            notes,
            "stop_hunts":       stop_hunts,
            "judas_swings":     judas,
            "retail_traps":     traps,
            "bsl_pools":        all_bsl,
            "ssl_pools":        all_ssl,
            "bsl_proximity":    bsl_prox,
            "ssl_proximity":    ssl_prox,
            "liquidity_voids":  voids,
            "pattern_count":    len(all_patterns),
            "bias_votes":       bias_votes,
        }

        log_signal("engine5_liquidity_hunt", signal, confidence, result)
        log.info(f"Engine 5 → {signal} @ {confidence:.1%} | {len(all_patterns)} patterns | BSL near: {bsl_prox['in_range']}")
        return result

    except Exception as e:
        log.error(f"Engine 5 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
