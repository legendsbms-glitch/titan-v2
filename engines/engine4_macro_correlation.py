# FILE: engines/engine4_macro_correlation.py
"""
TITAN v2.0 — Engine 4: Macro Correlation
DXY, real yields, VIX, cross-asset, divergence, safe-haven flow
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List

from core.config import GOLD_SYMBOL, MACRO_SYMBOLS
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine4.MacroCorrelation")


def _fetch(symbol: str, period: str = "90d", interval: str = "1d") -> pd.Series:
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df["close"].dropna()
    except Exception as e:
        log.debug(f"Fetch error {symbol}: {e}")
        return pd.Series(dtype=float)


def fetch_all_macro(period: str = "90d") -> pd.DataFrame:
    """Download gold + all macro assets, return combined DataFrame"""
    all_data = {}
    symbols = {"gold": GOLD_SYMBOL, **MACRO_SYMBOLS}
    for label, sym in symbols.items():
        s = _fetch(sym, period=period)
        if not s.empty:
            all_data[label] = s
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all").ffill()


def compute_correlations(df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> Dict:
    """Rolling correlation of gold vs each macro asset"""
    if "gold" not in df.columns or df.empty:
        return {}

    returns = df.pct_change().dropna()
    result  = {}

    for col in returns.columns:
        if col == "gold":
            continue
        entry = {}
        for w in windows:
            if len(returns) >= w:
                rc = returns["gold"].rolling(w).corr(returns[col])
                val = float(rc.iloc[-1]) if not pd.isna(rc.iloc[-1]) else 0.0
                entry[f"corr_{w}d"] = round(val, 3)
        result[col] = entry

    return result


def detect_divergences(df: pd.DataFrame) -> List[Dict]:
    """Detect key cross-asset divergences"""
    if df.empty or len(df) < 5:
        return []

    divergences = []
    ret5  = df.pct_change(5).iloc[-1]
    ret1  = df.pct_change(1).iloc[-1]

    g5  = ret5.get("gold",    0)
    g1  = ret1.get("gold",    0)
    d5  = ret5.get("dxy",     0)
    v5  = ret5.get("vix",     0)
    y5  = ret5.get("tnx",     0)
    s5  = ret5.get("spx",     0)
    si5 = ret5.get("silver",  0)
    t5  = ret5.get("tlt",     0)

    # Gold + DXY moving same direction (unusual — normally inverse)
    if g5 > 0.01 and d5 > 0.005:
        divergences.append({"type": "GOLD_DXY_SAME", "severity": "MEDIUM",
            "msg": "Gold & Dollar both rising — inflation fear, not typical safe haven"})
    if g5 < -0.01 and d5 < -0.005:
        divergences.append({"type": "GOLD_DXY_INVERSE_MISS", "severity": "MEDIUM",
            "msg": "Both gold and dollar falling — risk off breakdown"})

    # Gold rising with yields (unusual)
    if g5 > 0.01 and y5 > 0.02:
        divergences.append({"type": "GOLD_YIELD_SAME", "severity": "HIGH",
            "msg": "Gold & yields both rising — inflation panic, very bullish gold signal"})

    # Silver not confirming gold move
    if abs(g5) > 0.01 and abs(si5) < 0.003:
        divergences.append({"type": "GOLD_SILVER_DIVERGE", "severity": "LOW",
            "msg": f"Gold moving {'+' if g5>0 else '-'}{abs(g5):.1%} but silver flat — weak conviction"})

    # Safe haven flows: TLT + Gold + VIX all up = strong risk-off
    safe_haven_score = (1 if g5 > 0 else -1) + (1 if t5 > 0 else -1) + (1 if v5 > 0 else -1)
    if safe_haven_score >= 3:
        divergences.append({"type": "SAFE_HAVEN_FLOW", "severity": "HIGH",
            "msg": "Gold + TLT + VIX all rising — strong risk-off, very bullish gold"})
    elif safe_haven_score <= -3:
        divergences.append({"type": "RISK_ON_FLOW", "severity": "HIGH",
            "msg": "Gold + TLT + VIX all falling — strong risk-on, bearish gold"})

    return divergences


def compute_real_yield(df: pd.DataFrame) -> Dict:
    """Estimate real yield from nominal 10Y and gold relationship"""
    if "tnx" not in df.columns:
        return {"real_yield_proxy": None, "note": "No yield data"}

    tnx_latest = float(df["tnx"].iloc[-1]) if not df["tnx"].isna().all() else 0
    # Without TIPS data, estimate real yield from gold inverse relationship
    gold_ret_60d = float(df["gold"].pct_change(60).iloc[-1]) * 100 if "gold" in df.columns else 0

    # Real yield proxy: if gold up 5%+ over 60d while nominal yield steady, real yields probably low
    real_yield_proxy = tnx_latest - (gold_ret_60d * 0.1)  # very rough
    return {
        "nominal_10y":     round(tnx_latest, 3),
        "real_yield_proxy": round(real_yield_proxy, 3),
        "interpretation": "BULLISH" if real_yield_proxy < 1.0 else
                          "NEUTRAL" if real_yield_proxy < 2.5 else "BEARISH",
    }


def dollar_regime(df: pd.DataFrame) -> Dict:
    """DXY momentum regime: rising/falling/ranging"""
    if "dxy" not in df.columns or df.empty:
        return {"regime": "UNKNOWN", "momentum": 0}
    dxy = df["dxy"].dropna()
    if len(dxy) < 5:
        return {"regime": "UNKNOWN", "momentum": 0}
    ret5  = float(dxy.pct_change(5).iloc[-1]) * 100
    ret20 = float(dxy.pct_change(20).iloc[-1]) * 100
    if ret5 > 0.5 and ret20 > 1.0:
        regime = "STRONG_RISING"   # bearish gold
    elif ret5 < -0.5 and ret20 < -1.0:
        regime = "STRONG_FALLING"  # bullish gold
    elif ret5 > 0.2:
        regime = "RISING"
    elif ret5 < -0.2:
        regime = "FALLING"
    else:
        regime = "RANGING"
    return {"regime": regime, "ret_5d": round(ret5, 3), "ret_20d": round(ret20, 3)}


def safe_haven_detector(df: pd.DataFrame) -> Dict:
    """Are safe-haven assets flowing together?"""
    if df.empty:
        return {"flow": "NEUTRAL", "strength": 0}
    ret5 = df.pct_change(5).iloc[-1]
    sh_assets = ["gold", "tlt", "vix"]
    available = [a for a in sh_assets if a in ret5.index]
    if not available:
        return {"flow": "NEUTRAL", "strength": 0}

    votes = [1 if ret5[a] > 0 else -1 for a in available]
    avg   = sum(votes) / len(votes)
    if avg > 0.5:
        return {"flow": "RISK_OFF", "strength": round(abs(avg), 2),
                "assets": available, "note": "Safe havens all bid"}
    elif avg < -0.5:
        return {"flow": "RISK_ON",  "strength": round(abs(avg), 2),
                "assets": available, "note": "Safe havens all sold"}
    return {"flow": "MIXED", "strength": round(abs(avg), 2)}


def compute_macro_score(df: pd.DataFrame, correlations: Dict,
                        divergences: List, dxy_regime: Dict, sh: Dict) -> float:
    """Synthesize all macro signals into gold score"""
    score = 0.0

    if df.empty:
        return 0.0

    ret1 = df.pct_change(1).iloc[-1]

    # DXY impact (inverse)
    if "dxy" in ret1:
        score -= float(ret1["dxy"]) * 4.0

    # Real yield proxy (inverse)
    if "tnx" in ret1:
        score -= float(ret1["tnx"]) * 3.0

    # VIX spike = safe haven demand
    if "vix" in df.columns:
        vix_now = float(df["vix"].iloc[-1])
        vix_avg = float(df["vix"].mean())
        if vix_now > vix_avg * 1.3:
            score += 0.20

    # SPX falling = safe haven demand
    if "spx" in ret1 and float(ret1["spx"]) < -0.01:
        score += 0.15

    # TLT rising = rates falling = gold bullish
    if "tlt" in ret1 and float(ret1["tlt"]) > 0.002:
        score += 0.12

    # DXY regime adjustments
    dreg = dxy_regime.get("regime", "RANGING")
    if dreg == "STRONG_FALLING":
        score += 0.20
    elif dreg == "STRONG_RISING":
        score -= 0.20

    # Safe haven flow
    sh_flow = sh.get("flow", "NEUTRAL")
    sh_str  = sh.get("strength", 0)
    if sh_flow == "RISK_OFF":
        score += sh_str * 0.25
    elif sh_flow == "RISK_ON":
        score -= sh_str * 0.25

    # Divergence boosts
    for div in divergences:
        if div["type"] in ("SAFE_HAVEN_FLOW",):
            score += 0.10
        elif div["type"] in ("RISK_ON_FLOW",):
            score -= 0.10
        elif div["type"] == "GOLD_YIELD_SAME":
            score += 0.08

    return round(max(-1.0, min(1.0, score)), 3)


def run() -> dict:
    log.info("Running Engine 4: Macro Correlation")
    try:
        df           = fetch_all_macro(period="90d")
        correlations = compute_correlations(df)
        divergences  = detect_divergences(df)
        real_yield   = compute_real_yield(df)
        dxy_regime   = dollar_regime(df)
        sh_flow      = safe_haven_detector(df)
        macro_score  = compute_macro_score(df, correlations, divergences, dxy_regime, sh_flow)

        # Snapshot of current macro state
        snapshot = {}
        if not df.empty:
            for col in df.columns:
                try:
                    snapshot[col] = {
                        "latest":    round(float(df[col].iloc[-1]), 4),
                        "chg_1d":   round(float(df[col].pct_change(1).iloc[-1]) * 100, 3),
                        "chg_5d":   round(float(df[col].pct_change(5).iloc[-1]) * 100, 3),
                        "chg_20d":  round(float(df[col].pct_change(20).iloc[-1]) * 100, 3),
                    }
                except Exception:
                    pass

        if macro_score > 0.15:
            signal     = "BUY"
            confidence = min(0.5 + macro_score * 0.38, 0.88)
        elif macro_score < -0.15:
            signal     = "SELL"
            confidence = min(0.5 + abs(macro_score) * 0.38, 0.88)
        else:
            signal     = "NEUTRAL"
            confidence = 0.40

        result = {
            "signal":       signal,
            "confidence":   round(confidence, 3),
            "macro_score":  macro_score,
            "correlations": correlations,
            "divergences":  divergences,
            "real_yield":   real_yield,
            "dxy_regime":   dxy_regime,
            "safe_haven":   sh_flow,
            "snapshot":     snapshot,
        }

        log_signal("engine4_macro_correlation", signal, confidence, result)
        log.info(f"Engine 4 → {signal} @ {confidence:.1%} | DXY: {dxy_regime.get('regime')} | Safe haven: {sh_flow.get('flow')}")
        return result

    except Exception as e:
        log.error(f"Engine 4 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
