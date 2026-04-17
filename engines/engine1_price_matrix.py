# FILE: engines/engine1_price_matrix.py
"""
TITAN v2.0 — Engine 1: Price Matrix (ICT Edition)
Multi-TF structure, FVG, Order Blocks, BOS/CHoCH, Kill Zones, Premium/Discount
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Optional
import ta

from core.config import GOLD_SYMBOL, TIMEFRAMES, KILL_ZONES, SESSION_TIMES
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine1.PriceMatrix")


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str = GOLD_SYMBOL, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.dropna(inplace=True)
    return df


# ── Indicator Suite ───────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    df = df.copy()
    df["ema20"]  = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"]  = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi14"]  = ta.momentum.rsi(df["close"], window=14)
    df["atr14"]  = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    macd = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"]    = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    return df


# ── ICT: Fair Value Gaps ──────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame, min_gap_pct: float = 0.001) -> list:
    """Detect Fair Value Gaps (3-candle imbalance)"""
    fvgs = []
    for i in range(2, len(df)):
        c1_h = float(df["high"].iloc[i-2])
        c1_l = float(df["low"].iloc[i-2])
        c3_h = float(df["high"].iloc[i])
        c3_l = float(df["low"].iloc[i])
        price_ref = float(df["close"].iloc[i])

        # Bullish FVG: gap up — candle3 low > candle1 high
        if c3_l > c1_h:
            gap_size = c3_l - c1_h
            if gap_size / price_ref >= min_gap_pct:
                fvgs.append({
                    "type":      "BULLISH_FVG",
                    "top":       round(c3_l, 2),
                    "bottom":    round(c1_h, 2),
                    "midpoint":  round((c3_l + c1_h) / 2, 2),
                    "size_pct":  round(gap_size / price_ref * 100, 4),
                    "timestamp": str(df.index[i]),
                    "filled":    False,
                })

        # Bearish FVG: gap down — candle3 high < candle1 low
        elif c3_h < c1_l:
            gap_size = c1_l - c3_h
            if gap_size / price_ref >= min_gap_pct:
                fvgs.append({
                    "type":      "BEARISH_FVG",
                    "top":       round(c1_l, 2),
                    "bottom":    round(c3_h, 2),
                    "midpoint":  round((c1_l + c3_h) / 2, 2),
                    "size_pct":  round(gap_size / price_ref * 100, 4),
                    "timestamp": str(df.index[i]),
                    "filled":    False,
                })

    # Check if recent FVGs have been filled by current price
    current = float(df["close"].iloc[-1])
    for fvg in fvgs:
        if fvg["type"] == "BULLISH_FVG" and current < fvg["top"]:
            fvg["filled"] = True
        elif fvg["type"] == "BEARISH_FVG" and current > fvg["bottom"]:
            fvg["filled"] = True

    return fvgs[-8:]  # Last 8 FVGs


# ── ICT: Order Blocks ─────────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, impulse_threshold: float = 0.003) -> list:
    """
    Bullish OB = last bearish candle before strong impulsive bullish move
    Bearish OB = last bullish candle before strong impulsive bearish move
    """
    obs = []
    if len(df) < 5:
        return obs

    for i in range(1, len(df) - 3):
        candle     = df.iloc[i]
        body       = abs(float(candle["close"]) - float(candle["open"]))
        price_ref  = float(candle["close"])

        # Look at next 3 candles for impulse
        next_move  = float(df["close"].iloc[i+3]) - float(df["close"].iloc[i])
        move_pct   = abs(next_move) / price_ref

        if move_pct < impulse_threshold:
            continue

        # Bullish OB: bearish candle followed by bullish impulse
        if float(candle["close"]) < float(candle["open"]) and next_move > 0:
            obs.append({
                "type":      "BULLISH_OB",
                "top":       round(float(candle["open"]), 2),
                "bottom":    round(float(candle["low"]), 2),
                "midpoint":  round((float(candle["open"]) + float(candle["low"])) / 2, 2),
                "timestamp": str(df.index[i]),
                "impulse_pct": round(move_pct * 100, 3),
            })

        # Bearish OB: bullish candle followed by bearish impulse
        elif float(candle["close"]) > float(candle["open"]) and next_move < 0:
            obs.append({
                "type":      "BEARISH_OB",
                "top":       round(float(candle["high"]), 2),
                "bottom":    round(float(candle["close"]), 2),
                "midpoint":  round((float(candle["high"]) + float(candle["close"])) / 2, 2),
                "timestamp": str(df.index[i]),
                "impulse_pct": round(move_pct * 100, 3),
            })

    return obs[-6:]


# ── Market Structure ──────────────────────────────────────────────────────────

def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> dict:
    """Find significant swing highs and lows"""
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs = []
    swing_lows  = []

    for i in range(lookback, n - lookback):
        if all(highs[i] >= highs[i-k] for k in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+k] for k in range(1, lookback+1)):
            swing_highs.append({"price": float(highs[i]), "index": i, "ts": str(df.index[i])})
        if all(lows[i] <= lows[i-k] for k in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+k] for k in range(1, lookback+1)):
            swing_lows.append({"price": float(lows[i]), "index": i, "ts": str(df.index[i])})

    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def detect_structure(df: pd.DataFrame) -> dict:
    """Detect HH/HL (bullish) or LH/LL (bearish) market structure"""
    swings = find_swing_points(df)
    sh = [s["price"] for s in swings["swing_highs"]]
    sl = [s["price"] for s in swings["swing_lows"]]

    bias = "NEUTRAL"
    bos  = False
    choch = False

    if len(sh) >= 2 and len(sl) >= 2:
        hh = sh[-1] > sh[-2]
        hl = sl[-1] > sl[-2]
        lh = sh[-1] < sh[-2]
        ll = sl[-1] < sl[-2]

        if hh and hl:
            bias = "BULLISH"
        elif lh and ll:
            bias = "BEARISH"
        elif hh and ll:
            bias = "EXPANSION_UP"
        elif lh and hl:
            bias = "RANGING"

        # BOS: price breaks above last swing high (bearish structure) = BOS bullish
        current = float(df["close"].iloc[-1])
        if sh and current > sh[-1]:
            bos = True
        if sl and current < sl[-1]:
            bos = True  # bearish BOS

    return {
        "bias":         bias,
        "bos":          bos,
        "choch":        choch,
        "last_sh":      sh[-1] if sh else None,
        "last_sl":      sl[-1] if sl else None,
        "swing_highs":  sh[-3:],
        "swing_lows":   sl[-3:],
    }


# ── Premium / Discount Zones ──────────────────────────────────────────────────

def get_premium_discount(df: pd.DataFrame) -> dict:
    """
    ICT Premium/Discount:
    - Range = last significant swing high to swing low
    - Equilibrium = 50% of range
    - Premium = above 50% (sell zone)
    - Discount = below 50% (buy zone)
    """
    swings = find_swing_points(df)
    sh_list = [s["price"] for s in swings["swing_highs"]]
    sl_list = [s["price"] for s in swings["swing_lows"]]

    if not sh_list or not sl_list:
        return {"zone": "UNKNOWN", "equilibrium": None, "pct_of_range": None}

    range_high = max(sh_list[-3:]) if len(sh_list) >= 3 else max(sh_list)
    range_low  = min(sl_list[-3:]) if len(sl_list) >= 3 else min(sl_list)
    equilibrium = (range_high + range_low) / 2
    current     = float(df["close"].iloc[-1])

    if range_high == range_low:
        return {"zone": "UNKNOWN", "equilibrium": None, "pct_of_range": None}

    pct = (current - range_low) / (range_high - range_low) * 100

    if pct > 62.5:
        zone = "PREMIUM"     # Optimal sell zone (OTE sell: 62.5-79%)
    elif pct > 50:
        zone = "PREMIUM_WEAK"
    elif pct > 37.5:
        zone = "DISCOUNT_WEAK"
    else:
        zone = "DISCOUNT"    # Optimal buy zone (OTE buy: 21-37.5%)

    return {
        "zone":         zone,
        "equilibrium":  round(equilibrium, 2),
        "range_high":   round(range_high, 2),
        "range_low":    round(range_low, 2),
        "pct_of_range": round(pct, 1),
        "current":      round(current, 2),
    }


# ── Session Ranges ────────────────────────────────────────────────────────────

def get_session_ranges(df: pd.DataFrame) -> dict:
    """Get Asian/London/NY session range highs/lows"""
    if df.empty:
        return {}

    try:
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index, utc=True)

        def session_range(h_start, h_end, n_days=1):
            mask = (df_copy.index.hour >= h_start) & (df_copy.index.hour < h_end)
            session_df = df_copy[mask].tail(n_days * 12)
            if session_df.empty:
                return {"high": None, "low": None}
            return {
                "high": round(float(session_df["high"].max()), 2),
                "low":  round(float(session_df["low"].min()), 2),
            }

        return {
            "asian":    session_range(0, 8),
            "london":   session_range(8, 13),
            "new_york": session_range(13, 22),
        }
    except Exception as e:
        log.warning(f"Session ranges error: {e}")
        return {}


# ── Key Levels ────────────────────────────────────────────────────────────────

def get_key_levels(df: pd.DataFrame) -> dict:
    """Previous day/week H/L, round numbers ($50 increments for gold)"""
    if df.empty:
        return {}

    try:
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index, utc=True)

        current = float(df_copy["close"].iloc[-1])

        # Round numbers: every $50
        base = (current // 50) * 50
        round_numbers = [base - 100, base - 50, base, base + 50, base + 100]

        # Pivot
        h  = float(df_copy["high"].max())
        l  = float(df_copy["low"].min())
        c  = float(df_copy["close"].iloc[-1])
        p  = (h + l + c) / 3
        r1 = 2 * p - l
        r2 = p + (h - l)
        s1 = 2 * p - h
        s2 = p - (h - l)

        return {
            "pivot": round(p, 2),
            "r1": round(r1, 2), "r2": round(r2, 2),
            "s1": round(s1, 2), "s2": round(s2, 2),
            "period_high": round(h, 2),
            "period_low":  round(l, 2),
            "round_numbers": [round(r, 0) for r in round_numbers],
            "current": round(current, 2),
        }
    except Exception as e:
        log.warning(f"Key levels error: {e}")
        return {}


# ── Kill Zone Detection ───────────────────────────────────────────────────────

def detect_killzone() -> dict:
    """Is current UTC time inside an ICT kill zone?"""
    hour = datetime.now(timezone.utc).hour
    active = []
    for name, (start, end) in KILL_ZONES.items():
        if start <= end:
            in_zone = start <= hour < end
        else:
            in_zone = hour >= start or hour < end
        if in_zone:
            active.append(name)

    return {
        "active_killzones": active,
        "in_killzone":      len(active) > 0,
        "utc_hour":         hour,
    }


def get_current_session() -> str:
    hour = datetime.now(timezone.utc).hour
    if 8 <= hour < 17:
        return "LONDON"
    elif 13 <= hour < 22:
        return "NEW_YORK"
    elif 0 <= hour < 8:
        return "ASIA"
    else:
        return "DEAD_ZONE"


# ── Multi-TF Confluence ───────────────────────────────────────────────────────

def score_confluence(structures: dict) -> dict:
    """Score multi-timeframe confluence"""
    scores = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    weights = {"1h": 0.3, "4h": 0.4, "1d": 0.3}

    for tf, s in structures.items():
        bias = s.get("bias", "NEUTRAL")
        w    = weights.get(tf, 0.1)
        if "BULLISH" in bias:
            scores["BULLISH"] += w
        elif "BEARISH" in bias:
            scores["BEARISH"] += w
        else:
            scores["NEUTRAL"] += w

    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 3) for k, v in scores.items()}

    direction = max(scores, key=scores.get)
    strength  = scores[direction]

    return {
        "direction": direction,
        "strength":  strength,
        "scores":    scores,
        "aligned_tfs": [tf for tf, s in structures.items()
                         if direction.lower() in s.get("bias", "").lower()],
    }


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("Running Engine 1: Price Matrix (ICT)")
    try:
        # Fetch multiple timeframes
        tf_data = {}
        for tf, cfg in [("1h", {"period": "30d", "interval": "1h"}),
                         ("4h", {"period": "90d", "interval": "4h"}),
                         ("1d", {"period": "365d","interval": "1d"})]:
            df = fetch_ohlcv(period=cfg["period"], interval=cfg["interval"])
            if not df.empty:
                tf_data[tf] = add_indicators(df)

        if not tf_data:
            return {"signal": "ERROR", "confidence": 0.0, "error": "No data fetched"}

        primary = tf_data.get("1h", list(tf_data.values())[0])
        current_price = float(primary["close"].iloc[-1])
        atr = float(primary["atr14"].iloc[-1]) if "atr14" in primary.columns and not primary["atr14"].isna().all() else None

        # Run all analyses
        structures   = {tf: detect_structure(df) for tf, df in tf_data.items()}
        confluence   = score_confluence(structures)
        fvgs         = detect_fvg(primary)
        obs          = detect_order_blocks(primary)
        pd_zone      = get_premium_discount(primary)
        session_rng  = get_session_ranges(primary)
        key_levels   = get_key_levels(tf_data.get("4h", primary))
        killzone     = detect_killzone()
        session      = get_current_session()

        # Compute signal
        direction   = confluence["direction"]
        strength    = confluence["strength"]

        # Bonuses
        bonus = 0.0
        confluences_list = []

        # In kill zone = higher probability
        if killzone["in_killzone"]:
            bonus += 0.05
            confluences_list.append(f"Kill zone active: {killzone['active_killzones']}")

        # Premium/Discount alignment
        if direction == "BUY" and pd_zone.get("zone") == "DISCOUNT":
            bonus += 0.07
            confluences_list.append("Price in discount zone — ICT buy area")
        elif direction == "SELL" and pd_zone.get("zone") == "PREMIUM":
            bonus += 0.07
            confluences_list.append("Price in premium zone — ICT sell area")

        # Unfilled FVGs nearby (within 0.5% of price)
        nearby_fvgs = [f for f in fvgs if not f["filled"] and
                       abs(f["midpoint"] - current_price) / current_price < 0.005]
        if nearby_fvgs:
            bonus += 0.04
            confluences_list.append(f"{len(nearby_fvgs)} nearby unfilled FVG(s)")

        # OBs nearby
        nearby_obs = [o for o in obs if
                      abs(o["midpoint"] - current_price) / current_price < 0.005]
        if nearby_obs:
            bonus += 0.04
            confluences_list.append(f"{len(nearby_obs)} Order Block(s) near price")

        # EMA stack check
        if "ema20" in primary.columns and "ema50" in primary.columns and "ema200" in primary.columns:
            e20  = float(primary["ema20"].iloc[-1])
            e50  = float(primary["ema50"].iloc[-1])
            e200 = float(primary["ema200"].iloc[-1])
            if direction == "BUY" and current_price > e20 > e50 > e200:
                bonus += 0.06
                confluences_list.append("Bullish EMA stack (price > 20 > 50 > 200)")
            elif direction == "SELL" and current_price < e20 < e50 < e200:
                bonus += 0.06
                confluences_list.append("Bearish EMA stack (price < 20 < 50 < 200)")

        base_confidence = 0.45 + strength * 0.35
        confidence      = min(base_confidence + bonus, 0.94)

        if direction == "NEUTRAL" or strength < 0.4:
            signal = "NEUTRAL"
            confidence = min(confidence, 0.48)
        elif direction == "BULLISH":
            signal = "BUY"
        else:
            signal = "SELL"

        # Entry zone suggestion
        entry_zone = {"near": round(current_price * 0.9995, 2),
                      "far":  round(current_price * 1.001, 2)}
        if atr:
            if signal == "BUY":
                entry_zone = {"near": round(current_price, 2),
                              "far":  round(current_price - atr * 0.5, 2)}
            elif signal == "SELL":
                entry_zone = {"near": round(current_price, 2),
                              "far":  round(current_price + atr * 0.5, 2)}

        result = {
            "signal":        signal,
            "confidence":    round(confidence, 3),
            "current_price": current_price,
            "atr":           round(atr, 4) if atr else None,
            "session":       session,
            "structures":    structures,
            "confluence":    confluence,
            "confluences":   confluences_list,
            "fvgs":          fvgs,
            "order_blocks":  obs,
            "premium_discount": pd_zone,
            "session_ranges":   session_rng,
            "key_levels":       key_levels,
            "killzone":         killzone,
            "entry_zone":       entry_zone,
        }

        log_signal("engine1_price_matrix", signal, confidence, result)
        log.info(f"Engine 1 → {signal} @ {confidence:.1%} | {pd_zone.get('zone','?')} | {session}")
        return result

    except Exception as e:
        log.error(f"Engine 1 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
