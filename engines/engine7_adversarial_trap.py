# FILE: engines/engine7_adversarial_trap.py
"""
TITAN v2.0 — Engine 7: Adversarial Trap
Institutional playbook matching, fake breakouts, turtle soup, Asian range fade
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict

from core.config import GOLD_SYMBOL
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine7.AdversarialTrap")


def _fetch(period: str = "10d", interval: str = "15m") -> pd.DataFrame:
    df = yf.download(GOLD_SYMBOL, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df.dropna()


# ── Fake Breakout ─────────────────────────────────────────────────────────────

def detect_fake_breakout(df: pd.DataFrame, range_bars: int = 24,
                          confirm_bars: int = 4) -> List[Dict]:
    """Price breaks range convincingly, then reverses back inside"""
    patterns = []
    if len(df) < range_bars + confirm_bars + 5:
        return patterns

    for i in range(range_bars, len(df) - confirm_bars):
        lookback   = df.iloc[i - range_bars:i]
        range_high = float(lookback["high"].max())
        range_low  = float(lookback["low"].min())
        range_size = range_high - range_low
        if range_size <= 0:
            continue

        breakout_bar = df.iloc[i]
        b_high = float(breakout_bar["high"])
        b_low  = float(breakout_bar["low"])
        b_close = float(breakout_bar["close"])

        # After-bar confirmation
        confirm_df  = df.iloc[i+1:i+confirm_bars+1]
        if confirm_df.empty:
            continue
        final_close = float(confirm_df["close"].iloc[-1])

        # Bullish fake: broke above, confirmed back inside
        if b_high > range_high * 1.001 and final_close < range_high:
            depth = round((range_high - final_close) / range_size * 100, 1)
            patterns.append({
                "type":         "FAKE_BREAKOUT_BEARISH",
                "broken_level": round(range_high, 2),
                "current":      round(final_close, 2),
                "depth_pct":    depth,
                "timestamp":    str(df.index[i]),
                "msg":          f"Fake breakout above {round(range_high,2)} — longs trapped",
                "bias":         "SELL",
                "strength":     min(depth / 50, 1.0),
            })

        # Bearish fake: broke below, confirmed back inside
        elif b_low < range_low * 0.999 and final_close > range_low:
            depth = round((final_close - range_low) / range_size * 100, 1)
            patterns.append({
                "type":         "FAKE_BREAKOUT_BULLISH",
                "broken_level": round(range_low, 2),
                "current":      round(final_close, 2),
                "depth_pct":    depth,
                "timestamp":    str(df.index[i]),
                "msg":          f"Fake breakdown below {round(range_low,2)} — shorts trapped",
                "bias":         "BUY",
                "strength":     min(depth / 50, 1.0),
            })

    return patterns[-4:]


# ── Double Top/Bottom Liquidity ───────────────────────────────────────────────

def detect_double_levels(df: pd.DataFrame, lookback: int = 50,
                          tolerance: float = 0.0015) -> List[Dict]:
    """Double top/bottom = liquidity grab before reversal"""
    patterns = []
    if len(df) < lookback:
        return patterns

    window = df.tail(lookback)
    highs  = window["high"].values
    lows   = window["low"].values

    for i in range(5, len(highs) - 3):
        for j in range(i + 3, len(highs)):
            # Double top
            if highs[i] > 0 and abs(highs[i] - highs[j]) / highs[i] < tolerance:
                between_min = float(window["low"].iloc[i:j].min())
                depth_pct   = (highs[i] - between_min) / highs[i]
                if depth_pct > 0.003:  # Meaningful pullback between tops
                    current = float(window["close"].iloc[-1])
                    swept   = current < highs[j] * 0.999  # Price has come back down
                    patterns.append({
                        "type":      "DOUBLE_TOP_LIQUIDITY",
                        "level":     round(float(highs[i]), 2),
                        "first_ts":  str(window.index[i]),
                        "second_ts": str(window.index[j]),
                        "depth_pct": round(depth_pct * 100, 2),
                        "swept":     swept,
                        "msg":       f"Double top at {round(float(highs[i]),2)} — BSL grab, watch for reversal",
                        "bias":      "SELL",
                    })
                    break

            # Double bottom
            if lows[i] > 0 and abs(lows[i] - lows[j]) / lows[i] < tolerance:
                between_max = float(window["high"].iloc[i:j].max())
                depth_pct   = (between_max - lows[i]) / lows[i]
                if depth_pct > 0.003:
                    current = float(window["close"].iloc[-1])
                    swept   = current > lows[j] * 1.001
                    patterns.append({
                        "type":      "DOUBLE_BOTTOM_LIQUIDITY",
                        "level":     round(float(lows[i]), 2),
                        "first_ts":  str(window.index[i]),
                        "second_ts": str(window.index[j]),
                        "depth_pct": round(depth_pct * 100, 2),
                        "swept":     swept,
                        "msg":       f"Double bottom at {round(float(lows[i]),2)} — SSL grab, watch for reversal up",
                        "bias":      "BUY",
                    })
                    break

    return patterns[-4:]


# ── Turtle Soup ───────────────────────────────────────────────────────────────

def detect_turtle_soup(df: pd.DataFrame, period: int = 20) -> List[Dict]:
    """Breakout of N-bar H/L immediately fails"""
    patterns = []
    if len(df) < period + 3:
        return patterns

    recent = df.tail(8)
    for i in range(len(recent)):
        idx = len(df) - len(recent) + i
        if idx < period:
            continue
        lb  = df.iloc[idx - period:idx]
        bar = recent.iloc[i]
        b_h, b_l, b_c = float(bar["high"]), float(bar["low"]), float(bar["close"])
        ph, pl = float(lb["high"].max()), float(lb["low"].min())

        if b_h > ph and b_c < ph:
            patterns.append({
                "type":      "TURTLE_SOUP_SHORT",
                "level":     round(ph, 2),
                "close":     round(b_c, 2),
                "timestamp": str(recent.index[i]),
                "msg":       f"Turtle soup: {period}-bar high broken & failed — short setup",
                "bias":      "SELL",
            })
        elif b_l < pl and b_c > pl:
            patterns.append({
                "type":      "TURTLE_SOUP_LONG",
                "level":     round(pl, 2),
                "close":     round(b_c, 2),
                "timestamp": str(recent.index[i]),
                "msg":       f"Turtle soup: {period}-bar low broken & failed — long setup",
                "bias":      "BUY",
            })

    return patterns


# ── Asian Range Fade ──────────────────────────────────────────────────────────

def detect_asian_range_fade(df: pd.DataFrame) -> List[Dict]:
    """
    NY often fades London's breakout of the Asian range.
    Asian range breaks up in London → NY sells it back down.
    """
    patterns = []
    if df.empty:
        return patterns

    try:
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index, utc=True)

        # Asian range
        asian = df_copy[(df_copy.index.hour >= 0) & (df_copy.index.hour < 8)].tail(20)
        if asian.empty:
            return patterns
        asian_high = float(asian["high"].max())
        asian_low  = float(asian["low"].min())

        # London break
        london = df_copy[(df_copy.index.hour >= 8) & (df_copy.index.hour < 13)].tail(20)
        if london.empty:
            return patterns
        london_close = float(london["close"].iloc[-1])
        broke_up   = london_close > asian_high
        broke_down = london_close < asian_low

        # NY reaction
        ny = df_copy[(df_copy.index.hour >= 13) & (df_copy.index.hour < 18)].tail(10)
        if ny.empty:
            return patterns
        ny_current = float(ny["close"].iloc[-1])

        if broke_up and ny_current < asian_high:
            patterns.append({
                "type":    "ASIAN_RANGE_FADE_BEARISH",
                "asian_high": round(asian_high, 2),
                "london_close": round(london_close, 2),
                "ny_current":   round(ny_current, 2),
                "msg":  "London broke Asian high, NY fading back — bearish",
                "bias": "SELL",
            })
        elif broke_down and ny_current > asian_low:
            patterns.append({
                "type":    "ASIAN_RANGE_FADE_BULLISH",
                "asian_low":    round(asian_low, 2),
                "london_close": round(london_close, 2),
                "ny_current":   round(ny_current, 2),
                "msg":  "London broke Asian low, NY reversing — bullish",
                "bias": "BUY",
            })
    except Exception as e:
        log.debug(f"Asian range fade error: {e}")

    return patterns


# ── Inducement ────────────────────────────────────────────────────────────────

def detect_inducement(df: pd.DataFrame) -> List[Dict]:
    """
    Inducement: small obvious consolidation with clear breakout point
    Retail trades the breakout → price goes the other way.
    """
    patterns = []
    if len(df) < 15:
        return patterns

    recent = df.tail(15)
    rng    = float(recent["high"].max() - recent["low"].min())
    body_avg = float((abs(recent["close"] - recent["open"])).mean())
    vol_avg  = float(recent["volume"].mean()) if "volume" in recent.columns else 1

    # Tight consolidation + small bodies = inducement building
    is_tight = rng / float(recent["close"].mean()) < 0.008

    if is_tight:
        last_bar = recent.iloc[-1]
        lb_high = float(recent["high"].max())
        lb_low  = float(recent["low"].min())
        price   = float(last_bar["close"])

        # Price near top of range = inducement for shorts (will go up)
        if (price - lb_low) / rng > 0.75 if rng > 0 else False:
            patterns.append({
                "type": "INDUCEMENT_LONG",
                "range_high": round(lb_high, 2),
                "range_low":  round(lb_low, 2),
                "msg": "Tight range, price near top — inducement may break above to trap shorts",
                "bias": "BUY",
            })
        elif rng > 0 and (price - lb_low) / rng < 0.25:
            patterns.append({
                "type": "INDUCEMENT_SHORT",
                "range_high": round(lb_high, 2),
                "range_low":  round(lb_low, 2),
                "msg": "Tight range, price near bottom — inducement may break below to trap longs",
                "bias": "SELL",
            })

    return patterns


# ── Counter Strategies ────────────────────────────────────────────────────────

COUNTER_STRATEGIES = {
    "FAKE_BREAKOUT_BEARISH": "Wait for close back below breakout level, enter short with SL above wick high",
    "FAKE_BREAKOUT_BULLISH": "Wait for close back above breakdown level, enter long with SL below wick low",
    "DOUBLE_TOP_LIQUIDITY":  "Enter short after second top with SL above, target previous lows",
    "DOUBLE_BOTTOM_LIQUIDITY": "Enter long after second bottom with SL below, target previous highs",
    "TURTLE_SOUP_SHORT":     "Short below the failed breakout bar, SL above bar high",
    "TURTLE_SOUP_LONG":      "Long above the failed breakdown bar, SL below bar low",
    "ASIAN_RANGE_FADE_BEARISH": "Short when price re-enters Asian range from above",
    "ASIAN_RANGE_FADE_BULLISH": "Long when price re-enters Asian range from below",
    "INDUCEMENT_LONG":       "Buy when price breaks above inducement range after fakeout low",
    "INDUCEMENT_SHORT":      "Sell when price breaks below inducement range after fakeout high",
}


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("Running Engine 7: Adversarial Trap")
    try:
        df = _fetch(period="10d", interval="15m")
        if df.empty:
            return {"signal": "ERROR", "confidence": 0.0, "error": "No data"}

        fake_breaks    = detect_fake_breakout(df)
        double_levels  = detect_double_levels(df)
        turtle_soup    = detect_turtle_soup(df)
        asian_fade     = detect_asian_range_fade(df)
        inducement     = detect_inducement(df)

        all_patterns = fake_breaks + double_levels + turtle_soup + asian_fade + inducement

        score = 0.0
        buy_weight  = 0.0
        sell_weight = 0.0

        pattern_weights = {
            "FAKE_BREAKOUT_BULLISH":     0.35,
            "FAKE_BREAKOUT_BEARISH":     0.35,
            "DOUBLE_BOTTOM_LIQUIDITY":   0.25,
            "DOUBLE_TOP_LIQUIDITY":      0.25,
            "TURTLE_SOUP_LONG":          0.30,
            "TURTLE_SOUP_SHORT":         0.30,
            "ASIAN_RANGE_FADE_BULLISH":  0.20,
            "ASIAN_RANGE_FADE_BEARISH":  0.20,
            "INDUCEMENT_LONG":           0.15,
            "INDUCEMENT_SHORT":          0.15,
        }

        pattern_details = []
        for p in all_patterns:
            w    = pattern_weights.get(p["type"], 0.15)
            bias = p.get("bias", "NEUTRAL")
            counter = COUNTER_STRATEGIES.get(p["type"], "No strategy")
            pattern_details.append({**p, "counter_strategy": counter, "weight": w})
            if bias == "BUY":
                buy_weight += w
                score += w
            elif bias == "SELL":
                sell_weight += w
                score -= w

        score = max(-1.0, min(1.0, score))
        trap_probability = min(abs(score), 1.0)

        if score > 0.18:
            signal     = "BUY"
            confidence = min(0.50 + score * 0.42, 0.92)
        elif score < -0.18:
            signal     = "SELL"
            confidence = min(0.50 + abs(score) * 0.42, 0.92)
        else:
            signal     = "NEUTRAL"
            confidence = 0.38

        result = {
            "signal":           signal,
            "confidence":       round(confidence, 3),
            "trap_score":       round(score, 3),
            "trap_probability": round(trap_probability, 3),
            "buy_weight":       round(buy_weight, 3),
            "sell_weight":      round(sell_weight, 3),
            "patterns":         pattern_details,
            "pattern_count":    len(all_patterns),
            "fake_breakouts":   fake_breaks,
            "double_levels":    double_levels,
            "turtle_soup":      turtle_soup,
            "asian_fade":       asian_fade,
            "inducement":       inducement,
        }

        log_signal("engine7_adversarial_trap", signal, confidence, result)
        log.info(f"Engine 7 → {signal} @ {confidence:.1%} | {len(all_patterns)} trap patterns detected")
        return result

    except Exception as e:
        log.error(f"Engine 7 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
