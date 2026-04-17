# FILE: engines/engine10_smc_advanced.py
"""
TITAN v2.0 — Engine 10: Smart Money Concepts (Advanced ICT)
IPDA liquidity model, PD arrays, dealing ranges, displacement detection,
CISD (Change In State of Delivery), time-price theory, optimal trade entry
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

from core.config import GOLD_SYMBOL
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine10.SMCAdvanced")


def _fetch(period: str = "30d", interval: str = "1h") -> pd.DataFrame:
    df = yf.download(GOLD_SYMBOL, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df.dropna()


# ── IPDA (Interbank Price Delivery Algorithm) ─────────────────────────────────

def ipda_liquidity_draw(df: pd.DataFrame) -> Dict:
    """
    IPDA concept: market seeks liquidity every 20-40 candles.
    Identifies the next likely draw on liquidity.
    """
    if len(df) < 40:
        return {"draw": "UNKNOWN", "target": None}

    lookback_20 = df.tail(20)
    lookback_40 = df.tail(40)
    current     = float(df["close"].iloc[-1])

    high_20  = float(lookback_20["high"].max())
    low_20   = float(lookback_20["low"].min())
    high_40  = float(lookback_40["high"].max())
    low_40   = float(lookback_40["low"].min())

    # Distance to nearest external range liquidity
    dist_to_high = abs(current - high_40) / current
    dist_to_low  = abs(current - low_40)  / current

    if dist_to_high < dist_to_low:
        draw   = "DRAW_TO_BSL"
        target = round(high_40, 2)
        note   = f"Price likely drawing to buy-side liquidity at {high_40:.2f}"
    else:
        draw   = "DRAW_TO_SSL"
        target = round(low_40, 2)
        note   = f"Price likely drawing to sell-side liquidity at {low_40:.2f}"

    return {
        "draw":      draw,
        "target":    target,
        "note":      note,
        "high_40":   round(high_40, 2),
        "low_40":    round(low_40, 2),
        "high_20":   round(high_20, 2),
        "low_20":    round(low_20, 2),
        "distance_to_target_pct": round(min(dist_to_high, dist_to_low) * 100, 3),
    }


# ── PD Arrays (Premium/Discount Arrays) ──────────────────────────────────────

def detect_pd_arrays(df: pd.DataFrame) -> List[Dict]:
    """
    PD Arrays in order of delivery priority (ICT):
    1. Old Highs/Lows (external liquidity)
    2. Fair Value Gaps
    3. Order Blocks
    4. Breaker Blocks
    5. Mitigation Blocks
    6. Rejection Blocks
    7. Propulsion Blocks
    """
    arrays = []
    current = float(df["close"].iloc[-1])

    if len(df) < 10:
        return arrays

    # Breaker Blocks: failed Order Blocks (OB taken out → becomes breaker)
    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values
    opens  = df["open"].values

    for i in range(3, len(df) - 5):
        # Bullish breaker: bearish OB that was penetrated to upside
        if closes[i] < opens[i]:  # bearish candle
            future_high = df["high"].iloc[i+1:i+6].max()
            if future_high > highs[i]:  # price went above bearish OB → bullish breaker
                arrays.append({
                    "type":      "BULLISH_BREAKER",
                    "top":       round(float(highs[i]), 2),
                    "bottom":    round(float(lows[i]), 2),
                    "midpoint":  round((float(highs[i]) + float(lows[i])) / 2, 2),
                    "timestamp": str(df.index[i]),
                    "priority":  "HIGH",
                    "note":      "Failed bearish OB — bullish breaker block",
                })

        # Bearish breaker: bullish OB taken to downside
        if closes[i] > opens[i]:  # bullish candle
            future_low = df["low"].iloc[i+1:i+6].min()
            if future_low < lows[i]:   # price went below bullish OB → bearish breaker
                arrays.append({
                    "type":      "BEARISH_BREAKER",
                    "top":       round(float(highs[i]), 2),
                    "bottom":    round(float(lows[i]), 2),
                    "midpoint":  round((float(highs[i]) + float(lows[i])) / 2, 2),
                    "timestamp": str(df.index[i]),
                    "priority":  "HIGH",
                    "note":      "Failed bullish OB — bearish breaker block",
                })

    # Rejection Blocks: series of wicks rejecting a level
    for i in range(5, len(df)):
        window = df.iloc[i-5:i]
        upper_wicks = []
        lower_wicks = []
        for _, row in window.iterrows():
            uw = float(row["high"]) - max(float(row["open"]), float(row["close"]))
            lw = min(float(row["open"]), float(row["close"])) - float(row["low"])
            upper_wicks.append(uw)
            lower_wicks.append(lw)

        avg_uw = np.mean(upper_wicks)
        avg_lw = np.mean(lower_wicks)
        avg_body = float((abs(window["close"] - window["open"])).mean())

        if avg_body > 0:
            if avg_uw > avg_body * 1.2:
                arrays.append({
                    "type":     "REJECTION_BLOCK_BEARISH",
                    "price":    round(float(window["high"].max()), 2),
                    "note":     "Repeated upper wicks — sellers rejecting highs",
                    "priority": "MEDIUM",
                })
            if avg_lw > avg_body * 1.2:
                arrays.append({
                    "type":     "REJECTION_BLOCK_BULLISH",
                    "price":    round(float(window["low"].min()), 2),
                    "note":     "Repeated lower wicks — buyers supporting lows",
                    "priority": "MEDIUM",
                })

    # Deduplicate and return the most recent/relevant
    seen = set()
    unique = []
    for a in arrays[-10:]:
        key = (a["type"], a.get("midpoint", a.get("price", 0)))
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique[-8:]


# ── CISD (Change In State of Delivery) ───────────────────────────────────────

def detect_cisd(df: pd.DataFrame) -> Dict:
    """
    CISD: when price changes delivery from bearish to bullish (or vice versa).
    Identified by: strong displacement candle that closes above/below key structure.
    This is different from BOS — it's the internal shift.
    """
    if len(df) < 10:
        return {"cisd": None, "direction": None}

    recent = df.tail(20)
    closes = recent["close"].values
    opens  = recent["open"].values
    highs  = recent["high"].values
    lows   = recent["low"].values

    # Look for displacement: large body candle (> 1.5x avg body size)
    avg_body = float(np.mean(abs(closes - opens)))
    if avg_body == 0:
        return {"cisd": None, "direction": None}

    for i in range(5, len(recent)):
        body = abs(float(closes[i]) - float(opens[i]))
        if body > avg_body * 1.8:
            # Bullish displacement: closes above last 3 highs
            if closes[i] > opens[i]:  # bullish bar
                prev_highs = highs[max(0, i-3):i]
                if float(closes[i]) > float(max(prev_highs)):
                    return {
                        "cisd":        "BULLISH_CISD",
                        "direction":   "BUY",
                        "timestamp":   str(recent.index[i]),
                        "close":       round(float(closes[i]), 2),
                        "body_ratio":  round(body / avg_body, 2),
                        "note":        "Bullish displacement — delivery changed to bullish",
                    }
            # Bearish displacement: closes below last 3 lows
            else:
                prev_lows = lows[max(0, i-3):i]
                if float(closes[i]) < float(min(prev_lows)):
                    return {
                        "cisd":        "BEARISH_CISD",
                        "direction":   "SELL",
                        "timestamp":   str(recent.index[i]),
                        "close":       round(float(closes[i]), 2),
                        "body_ratio":  round(body / avg_body, 2),
                        "note":        "Bearish displacement — delivery changed to bearish",
                    }

    return {"cisd": None, "direction": None}


# ── Optimal Trade Entry (OTE) ─────────────────────────────────────────────────

def compute_ote(swing_high: float, swing_low: float, direction: str) -> Dict:
    """
    ICT Optimal Trade Entry (OTE):
    Fibonacci retracements 62%-79% of the last swing
    OTE buy: 62.5%-79% retracement in bullish trend
    OTE sell: 21%-37.5% retracement in bearish trend
    """
    rng = swing_high - swing_low
    if rng <= 0:
        return {}

    fib_levels = {
        "0.0":    round(swing_low, 2),
        "0.236":  round(swing_low + rng * 0.236, 2),
        "0.382":  round(swing_low + rng * 0.382, 2),
        "0.5":    round(swing_low + rng * 0.5, 2),
        "0.618":  round(swing_low + rng * 0.618, 2),
        "0.705":  round(swing_low + rng * 0.705, 2),  # OTE midpoint
        "0.786":  round(swing_low + rng * 0.786, 2),
        "1.0":    round(swing_high, 2),
    }

    if direction == "BUY":
        ote_zone = {
            "top":    round(swing_low + rng * 0.786, 2),
            "bottom": round(swing_low + rng * 0.625, 2),
            "ideal":  round(swing_low + rng * 0.705, 2),
            "note":   "OTE buy zone: 62.5%-78.6% retracement",
        }
    else:
        ote_zone = {
            "top":    round(swing_low + rng * 0.375, 2),
            "bottom": round(swing_low + rng * 0.214, 2),
            "ideal":  round(swing_low + rng * 0.295, 2),
            "note":   "OTE sell zone: 21.4%-37.5% retracement",
        }

    return {"fib_levels": fib_levels, "ote_zone": ote_zone}


# ── Time-Price Theory (Power of 3) ────────────────────────────────────────────

def power_of_three(df: pd.DataFrame) -> Dict:
    """
    ICT Power of 3: Accumulation → Manipulation → Distribution
    Session-based: Asian accumulates range, London manipulates, NY distributes
    AMD model for intraday gold.
    """
    if df.empty:
        return {"phase": "UNKNOWN", "note": ""}

    try:
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index, utc=True)
        hour = datetime.now(timezone.utc).hour

        # Asian range (accumulation): 0-8 UTC
        asian = df_copy[df_copy.index.hour.isin(range(0, 8))].tail(24)
        asian_high = float(asian["high"].max()) if not asian.empty else 0
        asian_low  = float(asian["low"].min())  if not asian.empty else 0
        asian_rng  = asian_high - asian_low

        if 0 <= hour < 8:
            phase = "ACCUMULATION"
            note  = "Asian session building range — await manipulation in London"
        elif 8 <= hour < 13:
            # London: manipulate the Asian range
            london = df_copy[df_copy.index.hour.isin(range(8, 13))].tail(10)
            if not london.empty:
                london_high = float(london["high"].max())
                london_low  = float(london["low"].min())
                if asian_rng > 0:
                    if london_high > asian_high:
                        phase = "MANIPULATION_HIGH"
                        note  = f"London swept Asian high ({asian_high:.2f}) — expect reversal and distribution downward"
                    elif london_low < asian_low:
                        phase = "MANIPULATION_LOW"
                        note  = f"London swept Asian low ({asian_low:.2f}) — expect reversal and distribution upward"
                    else:
                        phase = "MANIPULATION"
                        note  = "London in manipulation phase — watch for Asian range sweep"
                else:
                    phase = "MANIPULATION"
                    note  = "London open — manipulation phase"
            else:
                phase = "MANIPULATION"
                note  = "London opening — manipulation expected"
        else:
            phase = "DISTRIBUTION"
            note  = "NY session — distribution phase, real directional move"

        return {
            "phase":      phase,
            "note":       note,
            "asian_high": round(asian_high, 2),
            "asian_low":  round(asian_low, 2),
            "asian_range": round(asian_rng, 2),
        }
    except Exception as e:
        log.debug(f"Power of 3 error: {e}")
        return {"phase": "UNKNOWN", "note": str(e)}


# ── Dealing Range ─────────────────────────────────────────────────────────────

def compute_dealing_range(df: pd.DataFrame) -> Dict:
    """
    ICT Dealing Range: the range between the last significant swing high and low.
    Premium above 50% (sell bias), Discount below 50% (buy bias).
    Optimal buy discount: 0%-25%
    Optimal sell premium: 75%-100%
    """
    if len(df) < 20:
        return {}

    window = df.tail(60)
    high   = float(window["high"].max())
    low    = float(window["low"].min())
    close  = float(df["close"].iloc[-1])

    if high == low:
        return {}

    pct     = (close - low) / (high - low) * 100
    eq      = (high + low) / 2

    zone_labels = {
        (0,   25):  "DEEP_DISCOUNT",
        (25,  37.5):"DISCOUNT",
        (37.5,50):  "DISCOUNT_WEAK",
        (50,  62.5):"PREMIUM_WEAK",
        (62.5,75):  "PREMIUM",
        (75,  100): "DEEP_PREMIUM",
    }

    zone = "EQUILIBRIUM"
    for (lo, hi), label in zone_labels.items():
        if lo <= pct < hi:
            zone = label
            break

    bias = "BUY"  if pct < 50 else "SELL" if pct > 50 else "NEUTRAL"
    ideal_buy  = round(low  + (high - low) * 0.25, 2)
    ideal_sell = round(low  + (high - low) * 0.75, 2)

    return {
        "zone":          zone,
        "bias":          bias,
        "pct":           round(pct, 1),
        "equilibrium":   round(eq, 2),
        "range_high":    round(high, 2),
        "range_low":     round(low, 2),
        "current":       round(close, 2),
        "ideal_buy":     ideal_buy,
        "ideal_sell":    ideal_sell,
        "note":          f"Price at {pct:.1f}% of dealing range — {zone}",
    }


# ── Silver Bullet Pattern ─────────────────────────────────────────────────────

def detect_silver_bullet(df: pd.DataFrame) -> List[Dict]:
    """
    ICT Silver Bullet: specific time-based FVG entry
    - 3AM-4AM UTC silver bullet
    - 10AM-11AM UTC silver bullet  
    - 2PM-3PM UTC (14-15) silver bullet
    Requires: FVG formed during the window + killzone alignment
    """
    patterns = []
    if df.empty:
        return patterns

    try:
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index, utc=True)

        silver_bullet_windows = [(3, 4), (10, 11), (14, 15)]

        for start_h, end_h in silver_bullet_windows:
            window = df_copy[(df_copy.index.hour >= start_h) & (df_copy.index.hour < end_h)].tail(6)
            if len(window) < 3:
                continue

            for i in range(2, len(window)):
                c1_h = float(window["high"].iloc[i-2])
                c3_l = float(window["low"].iloc[i])

                if c3_l > c1_h:  # Bullish FVG in silver bullet window
                    patterns.append({
                        "type":      "SILVER_BULLET_LONG",
                        "window":    f"{start_h:02d}:00-{end_h:02d}:00 UTC",
                        "fvg_top":   round(c3_l, 2),
                        "fvg_bot":   round(c1_h, 2),
                        "timestamp": str(window.index[i]),
                        "note":      f"Silver Bullet long setup in {start_h}AM window",
                        "bias":      "BUY",
                    })

                c1_l = float(window["low"].iloc[i-2])
                c3_h = float(window["high"].iloc[i])
                if c3_h < c1_l:  # Bearish FVG
                    patterns.append({
                        "type":      "SILVER_BULLET_SHORT",
                        "window":    f"{start_h:02d}:00-{end_h:02d}:00 UTC",
                        "fvg_top":   round(c1_l, 2),
                        "fvg_bot":   round(c3_h, 2),
                        "timestamp": str(window.index[i]),
                        "note":      f"Silver Bullet short setup in {start_h}AM window",
                        "bias":      "SELL",
                    })
    except Exception as e:
        log.debug(f"Silver bullet error: {e}")

    return patterns


# ── Run ───────────────────────────────────────────────────────────────────────

def run() -> Dict:
    log.info("Running Engine 10: SMC Advanced (ICT)")
    try:
        df    = _fetch(period="30d",  interval="1h")
        df_15 = _fetch(period="10d",  interval="15m")

        if df.empty:
            return {"signal": "ERROR", "confidence": 0.0, "error": "No data"}

        current = float(df["close"].iloc[-1])

        ipda        = ipda_liquidity_draw(df)
        pd_arrays   = detect_pd_arrays(df)
        cisd        = detect_cisd(df)
        dealing_rng = compute_dealing_range(df)
        pow3        = power_of_three(df)
        sb          = detect_silver_bullet(df_15 if not df_15.empty else df)

        # OTE from recent swing
        highs = df["high"].values
        lows  = df["low"].values
        swing_high = float(np.max(highs[-40:]))
        swing_low  = float(np.min(lows[-40:]))
        ote_long  = compute_ote(swing_high, swing_low, "BUY")
        ote_short = compute_ote(swing_high, swing_low, "SELL")

        # Scoring
        score = 0.0
        notes = []

        # CISD is the strongest signal
        if cisd.get("cisd"):
            if cisd["direction"] == "BUY":
                score += 0.40
                notes.append(f"Bullish CISD detected — delivery changed bullish")
            elif cisd["direction"] == "SELL":
                score -= 0.40
                notes.append(f"Bearish CISD detected — delivery changed bearish")

        # IPDA draw
        if ipda["draw"] == "DRAW_TO_BSL":
            score += 0.20
            notes.append(f"IPDA drawing to BSL at {ipda['target']}")
        elif ipda["draw"] == "DRAW_TO_SSL":
            score -= 0.20
            notes.append(f"IPDA drawing to SSL at {ipda['target']}")

        # Dealing range bias
        dr_bias = dealing_rng.get("bias", "NEUTRAL")
        if dr_bias == "BUY":
            score += 0.15
            notes.append(f"Price in discount zone ({dealing_rng.get('zone')})")
        elif dr_bias == "SELL":
            score -= 0.15
            notes.append(f"Price in premium zone ({dealing_rng.get('zone')})")

        # Power of 3 phase
        pow3_phase = pow3.get("phase", "")
        if "MANIPULATION_LOW" in pow3_phase:
            score += 0.20
            notes.append("Power of 3: Asian low swept — bullish distribution expected")
        elif "MANIPULATION_HIGH" in pow3_phase:
            score -= 0.20
            notes.append("Power of 3: Asian high swept — bearish distribution expected")

        # Silver Bullet
        buy_sb  = [p for p in sb if p["bias"] == "BUY"]
        sell_sb = [p for p in sb if p["bias"] == "SELL"]
        if buy_sb:
            score += 0.15
            notes.append(f"Silver Bullet LONG setup: {buy_sb[0]['window']}")
        elif sell_sb:
            score -= 0.15
            notes.append(f"Silver Bullet SHORT setup: {sell_sb[0]['window']}")

        # OTE proximity
        if ote_long.get("ote_zone"):
            ote = ote_long["ote_zone"]
            if ote["bottom"] <= current <= ote["top"]:
                score += 0.12
                notes.append(f"Price in OTE buy zone ({ote['bottom']}-{ote['top']})")

        score = max(-1.0, min(1.0, score))

        if score > 0.18:
            signal, confidence = "BUY",  min(0.5 + score * 0.42, 0.93)
        elif score < -0.18:
            signal, confidence = "SELL", min(0.5 + abs(score) * 0.42, 0.93)
        else:
            signal, confidence = "NEUTRAL", 0.40

        result = {
            "signal":       signal,
            "confidence":   round(confidence, 3),
            "score":        round(score, 3),
            "notes":        notes,
            "ipda":         ipda,
            "cisd":         cisd,
            "dealing_range": dealing_rng,
            "power_of_3":   pow3,
            "pd_arrays":    pd_arrays[:5],
            "ote_long":     ote_long,
            "ote_short":    ote_short,
            "silver_bullets": sb,
        }

        log_signal("engine10_smc_advanced", signal, confidence, result)
        log.info(f"Engine 10 → {signal} @ {confidence:.1%} | CISD: {cisd.get('cisd')} | P3: {pow3.get('phase')}")
        return result

    except Exception as e:
        log.error(f"Engine 10 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
