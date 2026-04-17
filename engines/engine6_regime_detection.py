# FILE: engines/engine6_regime_detection.py
"""
TITAN v2.0 — Engine 6: Regime Detection
GARCH volatility, HMM regime, Wyckoff cycle, ADX, Hurst exponent, MER
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
import ta

from core.config import GOLD_SYMBOL
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine6.RegimeDetection")


def _fetch(period: str = "365d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(GOLD_SYMBOL, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df.dropna()


# ── GARCH Volatility ──────────────────────────────────────────────────────────

def detect_volatility_regime(df: pd.DataFrame) -> dict:
    returns = df["close"].pct_change().dropna() * 100
    if len(returns) < 30:
        return {"regime": "UNKNOWN", "method": "INSUFFICIENT_DATA"}

    try:
        from arch import arch_model
        model = arch_model(returns, vol="Garch", p=1, q=1, rescale=False)
        res   = model.fit(disp="off", show_warning=False)
        cond_vol   = float(res.conditional_volatility.iloc[-1])
        hist_vol20 = float(returns.rolling(20).std().iloc[-1])
        hist_vol60 = float(returns.rolling(60).std().iloc[-1])
        ratio = cond_vol / hist_vol60 if hist_vol60 > 0 else 1.0
        if ratio > 1.6:
            regime = "HIGH_VOLATILITY"
        elif ratio < 0.65:
            regime = "LOW_VOLATILITY"
        else:
            regime = "NORMAL_VOLATILITY"
        return {
            "regime":      regime,
            "cond_vol":    round(cond_vol, 4),
            "hist_vol20d": round(hist_vol20, 4),
            "hist_vol60d": round(hist_vol60, 4),
            "ratio":       round(ratio, 3),
            "method":      "GARCH",
        }
    except Exception:
        hist_vol10 = float(returns.rolling(10).std().iloc[-1])
        hist_vol60 = float(returns.rolling(60).std().iloc[-1])
        ratio = hist_vol10 / hist_vol60 if hist_vol60 > 0 else 1.0
        regime = "HIGH_VOLATILITY" if ratio > 1.5 else "LOW_VOLATILITY" if ratio < 0.7 else "NORMAL_VOLATILITY"
        return {"regime": regime, "ratio": round(ratio, 3), "method": "ROLLING_STD"}


# ── HMM Regime ────────────────────────────────────────────────────────────────

def detect_hmm_regime(df: pd.DataFrame, n_states: int = 3) -> dict:
    try:
        from hmmlearn import hmm
        returns = df["close"].pct_change().dropna()
        vol_10  = returns.rolling(10).std().fillna(method="bfill")
        rng_pct = ((df["high"] - df["low"]) / df["close"]).reindex(returns.index).fillna(0)

        # Combine features
        features = np.column_stack([
            returns.values,
            vol_10.values,
            rng_pct.values,
        ])
        features = np.nan_to_num(features)

        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                                 n_iter=200, random_state=42)
        model.fit(features)
        states        = model.predict(features)
        current_state = int(states[-1])

        # Label states by mean return
        state_means = {}
        for s in range(n_states):
            mask = states == s
            if mask.sum() > 0:
                state_means[s] = float(np.mean(returns.values[mask]))
            else:
                state_means[s] = 0.0

        sorted_states = sorted(state_means.items(), key=lambda x: x[1])
        labels = {}
        if len(sorted_states) == 3:
            labels[sorted_states[0][0]] = "BEARISH"
            labels[sorted_states[1][0]] = "RANGING"
            labels[sorted_states[2][0]] = "BULLISH"
        elif len(sorted_states) == 2:
            labels[sorted_states[0][0]] = "BEARISH"
            labels[sorted_states[1][0]] = "BULLISH"

        current_label = labels.get(current_state, "RANGING")
        persistence   = float(model.transmat_[current_state][current_state])

        return {
            "current_state":   current_label,
            "state_id":        current_state,
            "persistence":     round(persistence, 3),
            "state_means":     {labels.get(k, str(k)): round(v*100, 4) for k, v in state_means.items()},
            "method":          "HMM",
        }
    except Exception as e:
        log.debug(f"HMM failed: {e}, using fallback")
        return _ema_regime_fallback(df)


def _ema_regime_fallback(df: pd.DataFrame) -> dict:
    close  = df["close"]
    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    price  = float(close.iloc[-1])
    e20    = float(ema20.iloc[-1])
    e50    = float(ema50.iloc[-1])
    e200   = float(ema200.iloc[-1])

    if price > e20 > e50 > e200:
        state = "BULLISH"
    elif price < e20 < e50 < e200:
        state = "BEARISH"
    else:
        state = "RANGING"
    return {"current_state": state, "method": "EMA_STACK"}


# ── Wyckoff Phase ─────────────────────────────────────────────────────────────

def detect_wyckoff_phase(df: pd.DataFrame) -> dict:
    if len(df) < 60:
        return {"phase": "UNKNOWN", "confidence": 0.0}

    w   = 60
    rec = df.tail(w)
    rng = float(rec["high"].max() - rec["low"].min())
    mid = float(rec["close"].mean())
    if mid == 0 or rng == 0:
        return {"phase": "UNKNOWN", "confidence": 0.0}

    rng_pct    = rng / mid
    vol_trend  = float(rec["volume"].tail(20).mean() / rec["volume"].head(20).mean()) if float(rec["volume"].head(20).mean()) > 0 else 1.0
    price_chg  = float((rec["close"].iloc[-1] - rec["close"].iloc[0]) / rec["close"].iloc[0])
    pos        = float((rec["close"].iloc[-1] - rec["low"].min()) / rng)

    if rng_pct < 0.06 and pos < 0.40 and vol_trend > 1.15:
        phase, conf = "ACCUMULATION", 0.68
    elif price_chg > 0.04 and vol_trend > 0.95 and pos > 0.50:
        phase, conf = "MARKUP", 0.72
    elif rng_pct < 0.06 and pos > 0.60 and vol_trend > 1.15:
        phase, conf = "DISTRIBUTION", 0.65
    elif price_chg < -0.04 and vol_trend > 0.95 and pos < 0.50:
        phase, conf = "MARKDOWN", 0.70
    else:
        phase, conf = "TRANSITION", 0.42

    return {
        "phase":       phase,
        "confidence":  conf,
        "range_pct":   round(rng_pct, 4),
        "vol_trend":   round(vol_trend, 3),
        "price_chg":   round(price_chg, 4),
        "position":    round(pos, 3),
    }


# ── Hurst Exponent (R/S Analysis) ────────────────────────────────────────────

def hurst_exponent(price_series: pd.Series, min_lag: int = 10) -> float:
    """R/S Hurst exponent: H>0.6=trending, H<0.4=mean-reverting"""
    try:
        ts = price_series.values
        lags = range(min_lag, len(ts) // 4)
        if len(list(lags)) < 3:
            return 0.5

        rs_vals = []
        for lag in lags:
            sub = ts[-lag:]
            mean = np.mean(sub)
            deviations = np.cumsum(sub - mean)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(sub, ddof=1)
            if S > 0:
                rs_vals.append((lag, R / S))

        if len(rs_vals) < 3:
            return 0.5

        lags_arr = np.log([x[0] for x in rs_vals])
        rs_arr   = np.log([x[1] for x in rs_vals])
        hurst    = float(np.polyfit(lags_arr, rs_arr, 1)[0])
        return round(max(0.0, min(1.0, hurst)), 3)
    except Exception:
        return 0.5


# ── Market Efficiency Ratio ───────────────────────────────────────────────────

def market_efficiency_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """
    MER = abs(net displacement) / sum(abs(moves))
    1.0 = perfectly trending, 0.0 = perfectly choppy
    """
    if len(df) < period + 1:
        return 0.5
    prices = df["close"].values[-period-1:]
    net_move = abs(prices[-1] - prices[0])
    total_path = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
    if total_path == 0:
        return 0.0
    return round(net_move / total_path, 3)


# ── ADX Trend Strength ────────────────────────────────────────────────────────

def get_adx(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {"adx": None, "trend_strength": "UNKNOWN"}
    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    adx_val = float(adx_ind.adx().iloc[-1])
    dmp     = float(adx_ind.adx_pos().iloc[-1])
    dmn     = float(adx_ind.adx_neg().iloc[-1])

    if adx_val > 40:
        strength = "STRONG_TREND"
    elif adx_val > 25:
        strength = "TRENDING"
    elif adx_val > 15:
        strength = "WEAK_TREND"
    else:
        strength = "CHOPPY"

    return {
        "adx":           round(adx_val, 2),
        "dm_plus":       round(dmp, 2),
        "dm_minus":      round(dmn, 2),
        "trend_strength": strength,
        "direction":     "BULLISH" if dmp > dmn else "BEARISH",
    }


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("Running Engine 6: Regime Detection")
    try:
        df = _fetch(period="365d", interval="1d")
        if df.empty:
            return {"signal": "ERROR", "confidence": 0.0, "error": "No data"}

        vol_regime = detect_volatility_regime(df)
        hmm_regime = detect_hmm_regime(df)
        wyckoff    = detect_wyckoff_phase(df)
        adx_data   = get_adx(df)
        hurst      = hurst_exponent(df["close"])
        mer        = market_efficiency_ratio(df, period=20)
        is_trending = hurst > 0.58 or mer > 0.55

        current_state  = hmm_regime.get("current_state", "RANGING")
        wyckoff_phase  = wyckoff.get("phase", "TRANSITION")
        vol_reg        = vol_regime.get("regime", "NORMAL_VOLATILITY")

        # Signal logic
        signal_map = {
            "BULLISH":      "BUY",
            "BEARISH":      "SELL",
            "RANGING":      "NEUTRAL",
            "MARKUP":       "BUY",
            "MARKDOWN":     "SELL",
            "ACCUMULATION": "BUY",
            "DISTRIBUTION": "SELL",
            "TRANSITION":   "NEUTRAL",
        }

        signal = signal_map.get(current_state, "NEUTRAL")
        base_conf = {"BULLISH": 0.65, "BEARISH": 0.65, "RANGING": 0.45,
                     "MARKUP": 0.68, "MARKDOWN": 0.68, "ACCUMULATION": 0.58,
                     "DISTRIBUTION": 0.58, "TRANSITION": 0.42}
        confidence = base_conf.get(current_state, 0.45)

        # Wyckoff alignment bonus
        wy_bonus = {("BUY","ACCUMULATION"): 0.06, ("BUY","MARKUP"): 0.04,
                    ("SELL","DISTRIBUTION"): 0.06, ("SELL","MARKDOWN"): 0.04}
        confidence += wy_bonus.get((signal, wyckoff_phase), 0.0)

        # ADX trend confirmation
        if is_trending and adx_data.get("trend_strength") in ("STRONG_TREND", "TRENDING"):
            confidence += 0.04

        # High volatility reduces confidence
        if vol_reg == "HIGH_VOLATILITY":
            confidence -= 0.06

        confidence = round(min(confidence, 0.92), 3)

        result = {
            "signal":          signal,
            "confidence":      confidence,
            "current_regime":  current_state,
            "volatility":      vol_regime,
            "hmm":             hmm_regime,
            "wyckoff":         wyckoff,
            "adx":             adx_data,
            "hurst":           hurst,
            "mer":             mer,
            "is_trending":     is_trending,
        }

        log_signal("engine6_regime_detection", signal, confidence, result)
        log.info(f"Engine 6 → {signal} | HMM: {current_state} | Wyckoff: {wyckoff_phase} | H={hurst} MER={mer}")
        return result

    except Exception as e:
        log.error(f"Engine 6 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
