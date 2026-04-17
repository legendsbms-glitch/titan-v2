# FILE: quant/technical_model.py
"""
TITAN v2.0 — Quantitative Technical Model
Multi-timeframe trend following + mean reversion + volatility regime
Based on academic research: momentum (Jegadeesh & Titman), 
mean reversion (Poterba & Summers), regime-switching (Hamilton 1989)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from typing import Dict, Tuple

from core.config import GOLD_SYMBOL
from core.logger import get_logger

log = get_logger("QuantTechnicalModel")


def _fetch_all_tfs() -> Dict[str, pd.DataFrame]:
    """Fetch gold data on multiple timeframes"""
    configs = {
        "daily":  {"period": "2y",  "interval": "1d"},
        "weekly": {"period": "5y",  "interval": "1wk"},
        "hourly": {"period": "30d", "interval": "1h"},
    }
    data = {}
    for name, cfg in configs.items():
        df = yf.download(GOLD_SYMBOL, progress=False, auto_adjust=True, **cfg)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            data[name] = df.dropna()
    return data


# ── Trend Strength System ─────────────────────────────────────────────────────

def trend_system(df: pd.DataFrame) -> Dict:
    """
    Multi-indicator trend assessment.
    Uses: EMA alignment, ADX, MACD, price vs moving averages.
    """
    if len(df) < 200:
        return {"trend": "INSUFFICIENT_DATA", "score": 0.0}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # EMA stack
    e8   = close.ewm(span=8).mean()
    e21  = close.ewm(span=21).mean()
    e50  = close.ewm(span=50).mean()
    e200 = close.ewm(span=200).mean()

    price = float(close.iloc[-1])
    e8v   = float(e8.iloc[-1])
    e21v  = float(e21.iloc[-1])
    e50v  = float(e50.iloc[-1])
    e200v = float(e200.iloc[-1])

    # EMA alignment score (0-4 for bullish, 0 to -4 for bearish)
    alignment = 0
    if price > e8v:   alignment += 1
    else:             alignment -= 1
    if e8v > e21v:    alignment += 1
    else:             alignment -= 1
    if e21v > e50v:   alignment += 1
    else:             alignment -= 1
    if e50v > e200v:  alignment += 1
    else:             alignment -= 1

    # ADX
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    adx_val = float(adx_ind.adx().iloc[-1])
    dmp     = float(adx_ind.adx_pos().iloc[-1])
    dmn     = float(adx_ind.adx_neg().iloc[-1])

    # Trend direction from ADX
    adx_signal = 0
    if adx_val > 25:
        adx_signal = 1 if dmp > dmn else -1
    elif adx_val > 20:
        adx_signal = 0.5 if dmp > dmn else -0.5

    # MACD
    macd_ind = ta.trend.MACD(close)
    macd_v   = float(macd_ind.macd().iloc[-1])
    macd_sig = float(macd_ind.macd_signal().iloc[-1])
    macd_hist = float(macd_ind.macd_diff().iloc[-1])
    macd_signal_v = 1 if macd_v > macd_sig and macd_hist > 0 else -1 if macd_v < macd_sig and macd_hist < 0 else 0

    # Composite trend score
    trend_score = (alignment / 4 * 0.40) + (adx_signal / 1 * 0.35) + (macd_signal_v * 0.25)

    if trend_score > 0.35:
        trend = "STRONG_UPTREND"
    elif trend_score > 0.15:
        trend = "UPTREND"
    elif trend_score > -0.15:
        trend = "RANGING"
    elif trend_score > -0.35:
        trend = "DOWNTREND"
    else:
        trend = "STRONG_DOWNTREND"

    # Distance from 200 EMA (mean reversion indicator)
    dist_200 = (price - e200v) / e200v * 100

    return {
        "trend":        trend,
        "score":        round(trend_score, 4),
        "ema_alignment": alignment,
        "adx":          round(adx_val, 2),
        "macd_hist":    round(macd_hist, 4),
        "price":        round(price, 2),
        "e200":         round(e200v, 2),
        "dist_200ema":  round(dist_200, 2),
        "note":         f"{trend} (ADX={adx_val:.0f}, align={alignment}/4)",
    }


# ── Momentum System ───────────────────────────────────────────────────────────

def momentum_system(df: pd.DataFrame) -> Dict:
    """
    Multi-period momentum (academic: Jegadeesh & Titman 1993).
    Gold has strong momentum properties — 6-12 month momentum is predictive.
    """
    if len(df) < 250:
        return {"momentum": "INSUFFICIENT", "score": 0.0}

    close = df["close"]
    price = float(close.iloc[-1])

    # Calculate returns over multiple periods
    periods = {
        "1m":   21,
        "3m":   63,
        "6m":   126,
        "9m":   189,
        "12m":  252,
    }

    returns = {}
    for label, days in periods.items():
        if len(close) > days:
            ret = float((close.iloc[-1] / close.iloc[-days] - 1) * 100)
            returns[label] = round(ret, 3)

    # Momentum score: weight longer-term more (skip last month for reversal)
    # Academic: use 12-2 month momentum (skip most recent month)
    score = 0.0
    if "12m" in returns and "1m" in returns:
        mom_12_2 = returns.get("12m", 0) - returns.get("1m", 0)  # 12-2 month
        if mom_12_2 > 10:
            score += 0.40
        elif mom_12_2 > 5:
            score += 0.20
        elif mom_12_2 < -10:
            score -= 0.40
        elif mom_12_2 < -5:
            score -= 0.20

    # 3-month momentum (intermediate)
    mom_3m = returns.get("3m", 0)
    if mom_3m > 5:
        score += 0.20
    elif mom_3m < -5:
        score -= 0.20

    # RSI momentum
    rsi = ta.momentum.rsi(close, window=14)
    rsi_now = float(rsi.iloc[-1])
    if rsi_now > 60:
        score += 0.10
    elif rsi_now < 40:
        score -= 0.10

    # Rate of change
    roc_20 = float(ta.momentum.ROCIndicator(close, window=20).roc().iloc[-1])
    if abs(roc_20) > 0:
        score += np.sign(roc_20) * min(abs(roc_20) / 20, 0.15)

    # Percentile rank in 52-week range
    high_52  = float(close.rolling(252).max().iloc[-1])
    low_52   = float(close.rolling(252).min().iloc[-1])
    pct_52   = (price - low_52) / (high_52 - low_52) * 100 if high_52 != low_52 else 50

    return {
        "momentum":    "STRONG_UP" if score > 0.35 else "UP" if score > 0.10 else
                       "STRONG_DOWN" if score < -0.35 else "DOWN" if score < -0.10 else "NEUTRAL",
        "score":       round(max(-1.0, min(1.0, score)), 4),
        "returns":     returns,
        "rsi_14":      round(rsi_now, 2),
        "roc_20d":     round(roc_20, 3),
        "pct_52w":     round(pct_52, 1),
        "note":        f"12-2M mom: {returns.get('12m',0)-returns.get('1m',0):.1f}% | 52W pct: {pct_52:.0f}%",
    }


# ── Mean Reversion System ─────────────────────────────────────────────────────

def mean_reversion_system(df: pd.DataFrame) -> Dict:
    """
    Mean reversion when price deviates significantly from equilibrium.
    Academic: Poterba & Summers (1988) — evidence of mean reversion in equity prices.
    For gold: strong mean reversion when 2+ standard deviations from 200-day SMA.
    """
    if len(df) < 200:
        return {"signal": 0.0, "note": "Insufficient data"}

    close = df["close"]
    price = float(close.iloc[-1])

    # Z-scores at multiple lookback windows
    zscores = {}
    for window in [20, 50, 200]:
        if len(close) > window:
            roll_mean = close.rolling(window).mean()
            roll_std  = close.rolling(window).std()
            z = float((close.iloc[-1] - roll_mean.iloc[-1]) / roll_std.iloc[-1]) if float(roll_std.iloc[-1]) > 0 else 0.0
            zscores[f"z_{window}d"] = round(z, 3)

    # Bollinger Band position
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_pct  = float(bb.bollinger_pband().iloc[-1])
    bb_w    = float(bb.bollinger_wband().iloc[-1])

    # RSI extremes
    rsi = float(ta.momentum.rsi(close, window=14).iloc[-1])

    z_200 = zscores.get("z_200d", 0)
    z_50  = zscores.get("z_50d", 0)

    # Mean reversion signal (only strong when overextended)
    if z_200 > 2.5 or (bb_pct > 1.0 and rsi > 75):
        signal = -0.60   # Extremely overbought → mean reversion expected
        note   = "EXTREMELY OVERBOUGHT — high mean reversion probability"
    elif z_200 > 1.8:
        signal = -0.30
        note   = "Overbought vs 200-day — mild mean reversion signal"
    elif z_200 < -2.5 or (bb_pct < 0 and rsi < 25):
        signal = +0.60   # Extremely oversold → bounce expected
        note   = "EXTREMELY OVERSOLD — high bounce probability"
    elif z_200 < -1.8:
        signal = +0.30
        note   = "Oversold vs 200-day — mild bounce signal"
    else:
        signal = 0.0
        note   = "No significant mean reversion setup"

    return {
        "signal":     round(signal, 3),
        "zscores":    zscores,
        "bb_pct":     round(bb_pct, 4),
        "bb_width":   round(bb_w, 4),
        "rsi_14":     round(rsi, 2),
        "overbought": signal < -0.3,
        "oversold":   signal > 0.3,
        "note":       note,
    }


# ── Volatility Regime ─────────────────────────────────────────────────────────

def volatility_regime(df: pd.DataFrame) -> Dict:
    """
    Volatility regime: low vol → breakout likely, high vol → range likely.
    Uses: ATR ratio, Bollinger Band width, historical vol ratio.
    """
    if len(df) < 30:
        return {"regime": "UNKNOWN"}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    atr50 = ta.volatility.AverageTrueRange(high, low, close, window=50).average_true_range()

    atr_ratio = float(atr14.iloc[-1] / atr50.iloc[-1]) if float(atr50.iloc[-1]) > 0 else 1.0

    hv10 = float(close.pct_change().rolling(10).std().iloc[-1] * np.sqrt(252) * 100)
    hv30 = float(close.pct_change().rolling(30).std().iloc[-1] * np.sqrt(252) * 100)
    hv_ratio = hv10 / hv30 if hv30 > 0 else 1.0

    bb = ta.volatility.BollingerBands(close, window=20)
    bb_width = float(bb.bollinger_wband().iloc[-1])
    bb_width_pct = float(close.pct_change().rolling(252).apply(
        lambda x: np.percentile(x, 50)).iloc[-1] if len(close) > 252 else 0.5)

    # Regime
    if atr_ratio > 1.6 or hv_ratio > 1.5:
        regime = "HIGH_VOLATILITY"
        bias   = "CAUTION"
    elif atr_ratio < 0.7 and bb_width < 2.0:
        regime = "LOW_VOLATILITY"
        bias   = "BREAKOUT_WATCH"
    else:
        regime = "NORMAL_VOLATILITY"
        bias   = "NORMAL_TRADING"

    # Volatility of volatility (VoV) — rising VoV = instability
    if len(close) > 60:
        daily_rv = close.pct_change().rolling(5).std()
        vov      = float(daily_rv.rolling(20).std().iloc[-1])
    else:
        vov = 0.0

    return {
        "regime":    regime,
        "bias":      bias,
        "atr_ratio": round(atr_ratio, 3),
        "hv_10d":    round(hv10, 2),
        "hv_30d":    round(hv30, 2),
        "hv_ratio":  round(hv_ratio, 3),
        "bb_width":  round(bb_width, 3),
        "note":      f"{regime} (ATR ratio={atr_ratio:.2f})",
    }


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> Dict:
    log.info("Running Quant Technical Model")

    tf_data = _fetch_all_tfs()
    daily   = tf_data.get("daily", pd.DataFrame())
    weekly  = tf_data.get("weekly", pd.DataFrame())

    if daily.empty:
        return {"error": "No data"}

    trend   = trend_system(daily)
    mom     = momentum_system(daily)
    mr      = mean_reversion_system(daily)
    vol     = volatility_regime(daily)
    trend_w = trend_system(weekly) if not weekly.empty else {}

    # Weekly trend confirmation bonus
    trend_score  = trend.get("score", 0)
    mom_score    = mom.get("score", 0)
    mr_score     = mr.get("signal", 0)

    # Weight: trend + momentum dominate, mean reversion as override
    composite = trend_score * 0.45 + mom_score * 0.35 + mr_score * 0.20
    composite = round(max(-1.0, min(1.0, composite)), 4)

    # Weekly alignment
    weekly_trend = trend_w.get("trend", "")
    if composite > 0 and "UP" in weekly_trend:
        composite = min(composite + 0.05, 1.0)
    elif composite < 0 and "DOWN" in weekly_trend:
        composite = max(composite - 0.05, -1.0)

    signal = "BUY" if composite > 0.20 else "SELL" if composite < -0.20 else "NEUTRAL"
    confidence = min(0.50 + abs(composite) * 0.40, 0.88)

    return {
        "signal":         signal,
        "confidence":     round(confidence, 3),
        "composite":      composite,
        "trend":          trend,
        "momentum":       mom,
        "mean_reversion": mr,
        "volatility":     vol,
        "weekly_trend":   trend_w,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
