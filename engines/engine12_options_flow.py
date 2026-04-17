# FILE: engines/engine12_options_flow.py
"""
TITAN v2.0 — Engine 12: Options Flow & Derivatives Intelligence
GLD/GC options proxy, put/call ratio, implied volatility term structure,
gamma exposure, max pain calculation, volatility skew analysis
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional

from core.config import GOLD_ETF
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine12.OptionsFlow")


# ── Options Chain Fetching ────────────────────────────────────────────────────

def fetch_options_chain(symbol: str = GOLD_ETF) -> Dict:
    """Fetch options chain from yfinance (GLD has liquid options)"""
    try:
        ticker = yf.Ticker(symbol)
        exps   = ticker.options

        if not exps:
            return {"error": "No options data available"}

        # Get the nearest two expiries
        chains = {}
        for exp in exps[:3]:
            try:
                chain  = ticker.option_chain(exp)
                calls  = chain.calls
                puts   = chain.puts
                chains[exp] = {"calls": calls, "puts": puts}
            except Exception as e:
                log.debug(f"Options chain error for {exp}: {e}")

        return chains

    except Exception as e:
        log.debug(f"Options fetch error: {e}")
        return {}


def compute_put_call_ratio(chains: Dict) -> Dict:
    """Put/Call ratio — above 1 = bearish sentiment, below 0.7 = bullish"""
    if not chains:
        return {"pcr": None, "sentiment": "UNKNOWN"}

    total_call_vol = 0
    total_put_vol  = 0
    total_call_oi  = 0
    total_put_oi   = 0

    for exp, chain in chains.items():
        calls = chain.get("calls", pd.DataFrame())
        puts  = chain.get("puts",  pd.DataFrame())

        if not calls.empty and "volume" in calls.columns:
            total_call_vol += float(calls["volume"].fillna(0).sum())
        if not puts.empty and "volume" in puts.columns:
            total_put_vol  += float(puts["volume"].fillna(0).sum())
        if not calls.empty and "openInterest" in calls.columns:
            total_call_oi  += float(calls["openInterest"].fillna(0).sum())
        if not puts.empty and "openInterest" in puts.columns:
            total_put_oi   += float(puts["openInterest"].fillna(0).sum())

    pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else None
    pcr_oi  = total_put_oi  / total_call_oi  if total_call_oi  > 0 else None

    if pcr_vol is not None:
        if pcr_vol > 1.2:
            sentiment = "BEARISH"
            score     = -min(pcr_vol - 1.0, 0.5) * 2
        elif pcr_vol < 0.7:
            sentiment = "BULLISH"
            score     = min(1.0 - pcr_vol, 0.5) * 2
        else:
            sentiment = "NEUTRAL"
            score     = 0.0
    else:
        sentiment = "UNKNOWN"
        score     = 0.0

    return {
        "pcr_volume":  round(pcr_vol, 3) if pcr_vol else None,
        "pcr_oi":      round(pcr_oi, 3)  if pcr_oi  else None,
        "sentiment":   sentiment,
        "score":       round(score, 3),
        "call_volume": int(total_call_vol),
        "put_volume":  int(total_put_vol),
    }


def compute_iv_surface(chains: Dict, current_price: float) -> Dict:
    """Implied volatility: term structure and skew"""
    iv_data = {}

    for exp, chain in chains.items():
        calls = chain.get("calls", pd.DataFrame())
        puts  = chain.get("puts",  pd.DataFrame())

        atm_calls, atm_puts = [], []

        if not calls.empty and "impliedVolatility" in calls.columns and "strike" in calls.columns:
            # ATM = within 2% of current price
            atm = calls[abs(calls["strike"] - current_price) / current_price < 0.02]
            if not atm.empty:
                atm_calls = atm["impliedVolatility"].dropna().tolist()

        if not puts.empty and "impliedVolatility" in puts.columns and "strike" in puts.columns:
            atm = puts[abs(puts["strike"] - current_price) / current_price < 0.02]
            if not atm.empty:
                atm_puts = atm["impliedVolatility"].dropna().tolist()

        atm_iv = np.mean(atm_calls + atm_puts) if (atm_calls or atm_puts) else None

        # Skew: OTM put IV vs OTM call IV
        skew = None
        if not puts.empty and not calls.empty:
            try:
                otm_puts_iv  = puts[puts["strike"] < current_price * 0.97]["impliedVolatility"].mean()
                otm_calls_iv = calls[calls["strike"] > current_price * 1.03]["impliedVolatility"].mean()
                if pd.notna(otm_puts_iv) and pd.notna(otm_calls_iv) and otm_calls_iv > 0:
                    skew = float(otm_puts_iv / otm_calls_iv)
            except Exception:
                pass

        iv_data[exp] = {
            "atm_iv":  round(float(atm_iv), 4) if atm_iv else None,
            "skew":    round(skew, 3) if skew else None,
        }

    # Term structure
    exps = list(iv_data.keys())
    ivs  = [iv_data[e]["atm_iv"] for e in exps if iv_data[e]["atm_iv"]]

    if len(ivs) >= 2:
        ts_slope = ivs[1] - ivs[0]  # Positive = contango IV, negative = backwardation
        ts_signal = "ELEVATED_NEAR_TERM" if ts_slope < -0.01 else "NORMAL"
    else:
        ts_slope  = None
        ts_signal = "UNKNOWN"

    return {
        "by_expiry":  iv_data,
        "ts_slope":   round(ts_slope, 4) if ts_slope is not None else None,
        "ts_signal":  ts_signal,
    }


def compute_max_pain(chains: Dict, current_price: float) -> Dict:
    """
    Max Pain: the price at which option sellers (MMs) lose the least money.
    Price tends to gravitate toward max pain near expiry.
    """
    if not chains:
        return {"max_pain": None}

    # Use nearest expiry
    exp     = list(chains.keys())[0]
    chain   = chains[exp]
    calls   = chain.get("calls", pd.DataFrame())
    puts    = chain.get("puts",  pd.DataFrame())

    if calls.empty or puts.empty:
        return {"max_pain": None}

    try:
        # Get all strikes
        all_strikes = set()
        if "strike" in calls.columns:
            all_strikes.update(calls["strike"].tolist())
        if "strike" in puts.columns:
            all_strikes.update(puts["strike"].tolist())

        all_strikes = sorted(all_strikes)
        if not all_strikes:
            return {"max_pain": None}

        pain_by_strike = {}
        for test_price in all_strikes:
            total_pain = 0

            # Call pain: sum of in-the-money call OI × (strike - test_price) for strike < test_price
            if "strike" in calls.columns and "openInterest" in calls.columns:
                itm_calls = calls[calls["strike"] < test_price]
                if not itm_calls.empty:
                    call_pain = ((test_price - itm_calls["strike"]) * itm_calls["openInterest"].fillna(0)).sum()
                    total_pain += float(call_pain)

            # Put pain: sum of in-the-money put OI × (test_price - strike) for strike > test_price
            if "strike" in puts.columns and "openInterest" in puts.columns:
                itm_puts = puts[puts["strike"] > test_price]
                if not itm_puts.empty:
                    put_pain = ((itm_puts["strike"] - test_price) * itm_puts["openInterest"].fillna(0)).sum()
                    total_pain += float(put_pain)

            pain_by_strike[test_price] = total_pain

        if pain_by_strike:
            max_pain_price = min(pain_by_strike, key=pain_by_strike.get)
            bias = "PULL_UP" if max_pain_price > current_price else "PULL_DOWN"

            return {
                "max_pain":    round(float(max_pain_price), 2),
                "current":     round(float(current_price), 2),
                "distance_pct": round(abs(max_pain_price - current_price) / current_price * 100, 2),
                "expiry":      exp,
                "bias":        bias,
                "note":        f"Max pain at ${max_pain_price:.2f} — MMs benefit from price moving {'up' if bias == 'PULL_UP' else 'down'}",
            }

    except Exception as e:
        log.debug(f"Max pain calc error: {e}")

    return {"max_pain": None}


def estimate_gex(chains: Dict, current_price: float) -> Dict:
    """
    Gamma Exposure (GEX) estimation.
    Positive GEX → market makers dampen moves (range bound)
    Negative GEX → market makers amplify moves (trending/volatile)
    """
    if not chains:
        return {"gex": None, "regime": "UNKNOWN"}

    total_gex = 0.0
    exp = list(chains.keys())[0]
    chain = chains[exp]
    calls = chain.get("calls", pd.DataFrame())
    puts  = chain.get("puts",  pd.DataFrame())

    try:
        contract_size = 100  # GLD standard

        for df_opt, sign in [(calls, 1), (puts, -1)]:
            if df_opt.empty or "impliedVolatility" not in df_opt.columns:
                continue
            if "strike" not in df_opt.columns or "openInterest" not in df_opt.columns:
                continue

            for _, row in df_opt.iterrows():
                strike = float(row.get("strike", 0))
                oi     = float(row.get("openInterest", 0) or 0)
                iv     = float(row.get("impliedVolatility", 0) or 0)
                if iv == 0 or strike == 0:
                    continue

                # Simplified gamma approximation (Black-Scholes)
                moneyness = current_price / strike
                gamma_approx = np.exp(-0.5 * ((moneyness - 1) / iv) ** 2) / (current_price * iv * np.sqrt(2 * np.pi))
                gex_contrib  = sign * gamma_approx * oi * contract_size * (current_price ** 2) * 0.01
                total_gex   += gex_contrib

    except Exception as e:
        log.debug(f"GEX error: {e}")

    if total_gex > 0:
        regime = "POSITIVE_GEX"
        note   = "Positive GEX — MMs dampen volatility, likely range-bound"
        score  = 0.0
    elif total_gex < 0:
        regime = "NEGATIVE_GEX"
        note   = "Negative GEX — MMs amplify moves, trending/volatile environment"
        score  = 0.05  # Slight bias for trending moves
    else:
        regime = "NEUTRAL_GEX"
        note   = "Neutral gamma environment"
        score  = 0.0

    return {
        "gex":    round(total_gex, 2),
        "regime": regime,
        "note":   note,
        "score":  score,
    }


# ── VIX Proxy for Gold ────────────────────────────────────────────────────────

def gold_vol_index() -> Dict:
    """Use GLD options IV as a Gold VIX proxy (GVIX)"""
    try:
        gld    = yf.Ticker(GOLD_ETF)
        exps   = gld.options
        if not exps:
            return {"gvix": None}

        chain  = gld.option_chain(exps[0])
        calls  = chain.calls
        price  = float(gld.fast_info.get("lastPrice", 200))

        if "impliedVolatility" in calls.columns:
            atm = calls[abs(calls["strike"] - price) / price < 0.03]
            if not atm.empty:
                gvix = float(atm["impliedVolatility"].mean()) * 100
                return {
                    "gvix":  round(gvix, 2),
                    "note":  f"Gold IV Index (GVIX): {gvix:.1f}%",
                    "high":  gvix > 20,
                }
    except Exception as e:
        log.debug(f"GVIX error: {e}")

    return {"gvix": None}


# ── Run ───────────────────────────────────────────────────────────────────────

def run() -> Dict:
    log.info("Running Engine 12: Options Flow")
    try:
        # Get current price
        gld_data = yf.download(GOLD_ETF, period="1d", interval="1d",
                               progress=False, auto_adjust=True)
        if gld_data.empty:
            return {"signal": "NEUTRAL", "confidence": 0.35, "note": "No GLD data"}

        if isinstance(gld_data.columns, pd.MultiIndex):
            gld_data.columns = [c[0].lower() for c in gld_data.columns]
        else:
            gld_data.columns = [c.lower() for c in gld_data.columns]

        current_price = float(gld_data["close"].iloc[-1])

        chains    = fetch_options_chain()
        pcr       = compute_put_call_ratio(chains)
        iv_surf   = compute_iv_surface(chains, current_price)
        max_pain  = compute_max_pain(chains, current_price)
        gex       = estimate_gex(chains, current_price)
        gvix      = gold_vol_index()

        score = 0.0
        notes = []

        # PCR signal
        if pcr.get("sentiment") == "BULLISH":
            score += 0.30
            notes.append(f"PCR {pcr.get('pcr_volume'):.2f} — bullish options flow")
        elif pcr.get("sentiment") == "BEARISH":
            score -= 0.30
            notes.append(f"PCR {pcr.get('pcr_volume'):.2f} — bearish options flow")

        # Max pain draw
        mp = max_pain.get("max_pain")
        if mp:
            if max_pain.get("bias") == "PULL_UP":
                score += 0.15
                notes.append(f"Max pain at ${mp} above current — magnetic pull up")
            elif max_pain.get("bias") == "PULL_DOWN":
                score -= 0.15
                notes.append(f"Max pain at ${mp} below current — magnetic pull down")

        # IV term structure
        if iv_surf.get("ts_signal") == "ELEVATED_NEAR_TERM":
            notes.append("Near-term IV elevated — expect volatile move")

        # GEX
        if gex.get("regime") == "NEGATIVE_GEX":
            notes.append("Negative GEX — trending environment")
        elif gex.get("regime") == "POSITIVE_GEX":
            notes.append("Positive GEX — range-bound expected")

        score = max(-1.0, min(1.0, score))

        if score > 0.15:
            signal, confidence = "BUY",  min(0.5 + score * 0.38, 0.85)
        elif score < -0.15:
            signal, confidence = "SELL", min(0.5 + abs(score) * 0.38, 0.85)
        else:
            signal, confidence = "NEUTRAL", 0.38

        result = {
            "signal":     signal,
            "confidence": round(confidence, 3),
            "score":      round(score, 3),
            "notes":      notes,
            "pcr":        pcr,
            "iv_surface": iv_surf,
            "max_pain":   max_pain,
            "gex":        gex,
            "gvix":       gvix,
            "data_available": bool(chains),
        }

        log_signal("engine12_options_flow", signal, confidence, result)
        log.info(f"Engine 12 → {signal} @ {confidence:.1%} | PCR: {pcr.get('pcr_volume')} | MaxPain: {max_pain.get('max_pain')}")
        return result

    except Exception as e:
        log.error(f"Engine 12 error: {e}", exc_info=True)
        return {"signal": "NEUTRAL", "confidence": 0.35,
                "note": f"Options data limited: {str(e)}", "data_available": False}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
