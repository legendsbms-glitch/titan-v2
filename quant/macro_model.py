# FILE: quant/macro_model.py
"""
TITAN v2.0 — Quantitative Macro Model for Gold
Deep macro scoring: real yields, inflation regime, dollar cycle,
central bank flows, risk appetite index, geopolitical risk proxy.

This is the core quant engine — NOT a simple crossover.
Every signal is grounded in the academic/institutional literature on gold pricing.

Gold Price Drivers (in order of importance):
1. Real interest rates (TIPS yields) — inverse relationship
2. USD strength — inverse relationship  
3. Inflation expectations — positive relationship
4. Risk appetite (VIX, SPX) — safe haven demand
5. Central bank demand — positive
6. Geopolitical risk — positive
7. Opportunity cost (S&P earnings yield vs gold)
8. Positioning (COT) — contrarian indicator at extremes
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

from core.config import FRED_API_KEY, GOLD_SYMBOL, FRED_CACHE_PATH
from core.logger import get_logger

log = get_logger("QuantMacroModel")
os.makedirs("data", exist_ok=True)


# ── FRED Data Layer ───────────────────────────────────────────────────────────

REQUIRED_SERIES = {
    # Real yields (most important for gold)
    "DFII10":    "real_yield_10y",       # 10Y TIPS yield
    "DFII5":     "real_yield_5y",        # 5Y TIPS yield
    "DFII20":    "real_yield_20y",       # 20Y TIPS yield
    # Nominal yields
    "DGS10":     "nominal_yield_10y",
    "DGS2":      "nominal_yield_2y",
    "DGS30":     "nominal_yield_30y",
    # Inflation expectations
    "T10YIE":    "breakeven_10y",        # 10Y breakeven inflation
    "T5YIFR":    "forward_5y5y",         # 5Y5Y forward inflation
    "T5YIE":     "breakeven_5y",
    # Dollar
    "DTWEXBGS":  "dollar_broad",         # Trade-weighted dollar
    "DTWEXM":    "dollar_major",         # Major currencies dollar
    # Monetary policy
    "DFF":       "fed_funds_rate",
    "IORB":      "interest_on_reserves",
    # Macro
    "CPIAUCSL":  "cpi",
    "CPILFESL":  "core_cpi",
    "PCEPI":     "pce",
    "PCEPILFE":  "core_pce",
    "UNRATE":    "unemployment",
    "GDPC1":     "real_gdp",
    # Financial conditions
    "BAMLH0A0HYM2":  "hy_spread",       # High yield spread
    "BAMLC0A0CM":    "ig_spread",        # Investment grade spread
    "TEDRATE":       "ted_spread",       # TED spread (banking stress)
    "T10Y2Y":        "yield_curve_10_2", # 10Y-2Y spread (recession indicator)
    "T10Y3M":        "yield_curve_10_3m",
    # Gold specific
    "GOLDAMGBD228NLBM": "gold_lbma",    # LBMA gold price
}


def fetch_fred_series(series_id: str, lookback_years: int = 3) -> pd.Series:
    """Fetch a single FRED series"""
    if not FRED_API_KEY:
        return pd.Series(dtype=float)
    try:
        start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_API_KEY}"
            f"&file_type=json&observation_start={start}&sort_order=asc"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        data = {}
        for o in obs:
            if o.get("value", ".") != ".":
                try:
                    data[o["date"]] = float(o["value"])
                except ValueError:
                    pass
        if not data:
            return pd.Series(dtype=float)
        s = pd.Series(data)
        s.index = pd.to_datetime(s.index)
        return s.ffill()
    except Exception as e:
        log.debug(f"FRED {series_id} error: {e}")
        return pd.Series(dtype=float)


def fetch_all_fred(use_cache: bool = True) -> Dict[str, pd.Series]:
    """Fetch all FRED series with caching"""
    cache_path = "data/fred_full_cache.json"

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            if time.time() - cache.get("_ts", 0) < 8 * 3600:  # 8h cache
                log.debug("FRED full cache hit")
                result = {}
                for key, data in cache.items():
                    if key == "_ts":
                        continue
                    try:
                        s = pd.Series(data)
                        s.index = pd.to_datetime(s.index)
                        result[key] = s
                    except Exception:
                        pass
                return result
        except Exception:
            pass

    log.info("Fetching full FRED dataset...")
    result = {}
    for series_id, label in REQUIRED_SERIES.items():
        s = fetch_fred_series(series_id, lookback_years=5)
        if not s.empty:
            result[label] = s
            log.debug(f"  ✓ {label}: {len(s)} observations")
        time.sleep(0.1)  # Rate limit

    # Cache
    try:
        cache = {"_ts": time.time()}
        for key, s in result.items():
            cache[key] = {str(k): float(v) for k, v in s.items()}
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

    return result


# ── Real Yield Model ──────────────────────────────────────────────────────────

def real_yield_model(fred_data: Dict) -> Dict:
    """
    The single most important factor for gold.
    Academic basis: Gibson (1975), Barsky & Summers (1988):
    Gold is essentially the reciprocal of the real interest rate on money.

    When real yields are negative → gold is attractive (no opportunity cost)
    When real yields > 2% → gold is expensive vs bonds
    """
    ry10 = fred_data.get("real_yield_10y", pd.Series(dtype=float))
    ry5  = fred_data.get("real_yield_5y",  pd.Series(dtype=float))

    if ry10.empty:
        return {"signal": 0.0, "note": "No TIPS data"}

    current_ry  = float(ry10.iloc[-1])
    change_30d  = float(ry10.iloc[-1] - ry10.iloc[-min(22, len(ry10)-1)])  # ~1 month
    trend_90d   = float(np.polyfit(range(min(66, len(ry10))), ry10.iloc[-min(66, len(ry10)):], 1)[0])

    # z-score vs 2-year history
    lookback = ry10.iloc[-min(500, len(ry10)):]
    zscore   = float((current_ry - lookback.mean()) / lookback.std()) if lookback.std() > 0 else 0.0

    # Signal: real yields → gold (inverse)
    if current_ry < -1.0:
        signal = +0.90   # Very negative real yields = very bullish gold
    elif current_ry < 0.0:
        signal = +0.60   # Negative = bullish
    elif current_ry < 1.0:
        signal = +0.20   # Low positive = mild bullish
    elif current_ry < 2.0:
        signal = -0.10   # Moderate positive = mild bearish
    else:
        signal = -0.50   # High real yields = bearish gold

    # Trend adjustment
    if trend_90d < -0.02:   # Yields falling → gold more bullish
        signal += 0.10
    elif trend_90d > 0.02:  # Yields rising → gold less bullish
        signal -= 0.10

    return {
        "signal":       round(max(-1.0, min(1.0, signal)), 3),
        "current_ry":   round(current_ry, 3),
        "change_30d":   round(change_30d, 4),
        "trend_90d":    round(trend_90d, 6),
        "zscore":       round(zscore, 3),
        "regime":       "NEGATIVE" if current_ry < 0 else "LOW" if current_ry < 1.5 else "HIGH",
        "note":         f"Real yield: {current_ry:.2f}% (z={zscore:.1f}σ)",
    }


def inflation_regime_model(fred_data: Dict) -> Dict:
    """
    Inflation expectations model for gold.
    Key insight: gold outperforms in HIGH inflation expectations regimes.
    Gold underperforms when inflation is declining (disinflation).

    Academic basis: Jastram (1978) — gold is poor short-term inflation hedge
    but excellent long-term store of value in inflationary environments.
    """
    bei_10  = fred_data.get("breakeven_10y",   pd.Series(dtype=float))
    bei_5   = fred_data.get("breakeven_5y",    pd.Series(dtype=float))
    fwd_5y5 = fred_data.get("forward_5y5y",    pd.Series(dtype=float))
    cpi     = fred_data.get("cpi",             pd.Series(dtype=float))

    if bei_10.empty:
        return {"signal": 0.0, "note": "No breakeven data"}

    current_bei = float(bei_10.iloc[-1])
    bei_trend   = 0.0
    if len(bei_10) > 44:
        bei_trend = float(np.polyfit(range(44), bei_10.iloc[-44:], 1)[0])

    # CPI momentum
    cpi_mom = 0.0
    if not cpi.empty and len(cpi) > 2:
        cpi_mom = float((cpi.iloc[-1] - cpi.iloc[-min(13, len(cpi)-1)]) / cpi.iloc[-min(13, len(cpi)-1)] * 100)

    # Signal
    if current_bei > 2.75:
        signal = +0.70   # High inflation expectations = very bullish gold
    elif current_bei > 2.25:
        signal = +0.40
    elif current_bei > 2.0:
        signal = +0.15   # Near Fed target = neutral-bullish
    elif current_bei > 1.5:
        signal = -0.15
    else:
        signal = -0.40   # Low inflation expectations = bearish gold

    # Rising trend = additional bullish
    if bei_trend > 0.002:
        signal += 0.10
    elif bei_trend < -0.002:
        signal -= 0.10

    # CPI momentum
    if cpi_mom > 3.5:
        signal += 0.10
    elif cpi_mom < 2.0:
        signal -= 0.05

    return {
        "signal":         round(max(-1.0, min(1.0, signal)), 3),
        "breakeven_10y":  round(current_bei, 3),
        "bei_trend":      round(bei_trend, 5),
        "cpi_yoy":        round(cpi_mom, 3),
        "regime":         "HIGH_INFLATION" if current_bei > 2.75 else
                          "ELEVATED" if current_bei > 2.25 else
                          "TARGET" if current_bei > 1.75 else "LOW_INFLATION",
        "note":           f"BEI: {current_bei:.2f}%, CPI MoM trend: {cpi_mom:.1f}%",
    }


def dollar_cycle_model(fred_data: Dict) -> Dict:
    """
    Dollar cycle impact on gold.
    Long-run: inverse relationship confirmed by academic literature.
    Short-run: can break down during crisis (both gold and dollar rally).

    Key metric: trade-weighted dollar index vs gold correlation.
    """
    dollar = fred_data.get("dollar_broad", pd.Series(dtype=float))

    # Fallback to yfinance DXY
    if dollar.empty:
        try:
            dxy = yf.download("DX-Y.NYB", period="2y", progress=False, auto_adjust=True)
            if not dxy.empty:
                if isinstance(dxy.columns, pd.MultiIndex):
                    dxy.columns = [c[0].lower() for c in dxy.columns]
                else:
                    dxy.columns = [c.lower() for c in dxy.columns]
                dollar = dxy["close"]
        except Exception:
            pass

    if dollar.empty:
        return {"signal": 0.0, "note": "No dollar data"}

    current   = float(dollar.iloc[-1])
    ret_1m    = float((dollar.iloc[-1] / dollar.iloc[-min(22, len(dollar)-1)] - 1) * 100)
    ret_3m    = float((dollar.iloc[-1] / dollar.iloc[-min(66, len(dollar)-1)] - 1) * 100)
    ret_12m   = float((dollar.iloc[-1] / dollar.iloc[-min(250, len(dollar)-1)] - 1) * 100)

    # z-score vs 1-year
    lookback  = dollar.iloc[-min(250, len(dollar)):]
    zscore    = float((current - lookback.mean()) / lookback.std()) if lookback.std() > 0 else 0.0

    # Dollar regime
    if ret_3m > 3.0:
        dollar_regime = "STRONG_UPTREND"
        signal = -0.50   # Strong rising dollar = bearish gold
    elif ret_3m > 1.5:
        dollar_regime = "UPTREND"
        signal = -0.25
    elif ret_3m < -3.0:
        dollar_regime = "STRONG_DOWNTREND"
        signal = +0.50   # Falling dollar = bullish gold
    elif ret_3m < -1.5:
        dollar_regime = "DOWNTREND"
        signal = +0.25
    else:
        dollar_regime = "RANGING"
        signal = 0.0

    # Short-term momentum
    if ret_1m > 1.0:
        signal -= 0.10
    elif ret_1m < -1.0:
        signal += 0.10

    # Dollar smile: very weak dollar in crisis can flip (both gold and dollar rise)
    # but at zscore > +2 (extremely strong dollar), signal is most bearish gold
    if zscore > 2.0:
        signal -= 0.10
    elif zscore < -2.0:
        signal += 0.10

    return {
        "signal":        round(max(-1.0, min(1.0, signal)), 3),
        "dollar_level":  round(current, 3),
        "ret_1m":        round(ret_1m, 3),
        "ret_3m":        round(ret_3m, 3),
        "ret_12m":       round(ret_12m, 3),
        "zscore":        round(zscore, 3),
        "regime":        dollar_regime,
        "note":          f"Dollar 3M: {ret_3m:+.1f}% | Regime: {dollar_regime}",
    }


def financial_stress_model(fred_data: Dict) -> Dict:
    """
    Financial stress → safe haven demand for gold.
    Uses: VIX, TED spread, high yield spreads, yield curve inversion.
    """
    hy_spread = fred_data.get("hy_spread",       pd.Series(dtype=float))
    ig_spread = fred_data.get("ig_spread",       pd.Series(dtype=float))
    ted       = fred_data.get("ted_spread",      pd.Series(dtype=float))
    yc_10_2   = fred_data.get("yield_curve_10_2", pd.Series(dtype=float))

    # Also get VIX from yfinance
    vix_signal = 0.0
    try:
        vix_df = yf.download("^VIX", period="1y", progress=False, auto_adjust=True)
        if not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0].lower() for c in vix_df.columns]
            else:
                vix_df.columns = [c.lower() for c in vix_df.columns]
            vix_now = float(vix_df["close"].iloc[-1])
            vix_avg = float(vix_df["close"].mean())
            vix_pct = float(vix_df["close"].rank(pct=True).iloc[-1]) * 100

            if vix_now > 30:
                vix_signal = +0.40  # Fear spike = safe haven gold
            elif vix_now > 20:
                vix_signal = +0.15
            elif vix_now < 12:
                vix_signal = -0.20  # Complacency = less safe haven demand
            else:
                vix_signal = 0.0
        else:
            vix_now, vix_avg, vix_pct = 20, 20, 50
    except Exception:
        vix_now, vix_avg, vix_pct = 20, 20, 50

    stress_score = vix_signal

    # High yield spread: elevated = risk off = gold bullish
    if not hy_spread.empty:
        hy_now = float(hy_spread.iloc[-1])
        if hy_now > 600:
            stress_score += 0.20   # Crisis level
        elif hy_now > 450:
            stress_score += 0.10
        elif hy_now < 300:
            stress_score -= 0.05   # Risk on = mild gold bearish

    # Yield curve inversion: inverted = recession fear = gold bullish
    if not yc_10_2.empty:
        yc_now = float(yc_10_2.iloc[-1])
        if yc_now < -0.5:
            stress_score += 0.15   # Deep inversion
        elif yc_now < 0:
            stress_score += 0.08   # Inverted
        elif yc_now > 1.5:
            stress_score -= 0.05   # Steep curve = growth optimism

    # TED spread: elevated = banking stress = gold bullish
    if not ted.empty:
        ted_now = float(ted.iloc[-1])
        if ted_now > 0.8:
            stress_score += 0.10

    return {
        "signal":         round(max(-1.0, min(1.0, stress_score)), 3),
        "vix":            round(vix_now, 2),
        "vix_percentile": round(vix_pct, 1),
        "hy_spread":      round(float(hy_spread.iloc[-1]), 2) if not hy_spread.empty else None,
        "yield_curve":    round(float(yc_10_2.iloc[-1]), 3) if not yc_10_2.empty else None,
        "regime":         "CRISIS" if stress_score > 0.5 else
                          "ELEVATED" if stress_score > 0.2 else
                          "NORMAL" if stress_score > -0.1 else "RISK_ON",
        "note":           f"VIX: {vix_now:.1f} ({vix_pct:.0f}th pct) | Stress: {stress_score:+.2f}",
    }


def monetary_policy_model(fred_data: Dict) -> Dict:
    """
    Fed policy cycle and its impact on gold.
    Key principle: gold loves rate CUTS (dovish), hates rate HIKES (hawkish).
    But the EXPECTATION of cuts matters more than cuts themselves.
    """
    ffr    = fred_data.get("fed_funds_rate", pd.Series(dtype=float))
    ry10   = fred_data.get("real_yield_10y", pd.Series(dtype=float))
    m2     = pd.Series(dtype=float)  # M2 not always available

    if ffr.empty:
        return {"signal": 0.0, "note": "No Fed funds data"}

    current_ffr = float(ffr.iloc[-1])
    ffr_12m_ago = float(ffr.iloc[-min(250, len(ffr)-1)])
    ffr_change  = current_ffr - ffr_12m_ago

    # Current cycle: hiking, cutting, or on hold?
    ffr_3m_chg = float(ffr.iloc[-1] - ffr.iloc[-min(66, len(ffr)-1)])
    if ffr_3m_chg > 0.25:
        cycle = "HIKING"
        signal = -0.30  # Rate hikes = bearish gold (short term)
    elif ffr_3m_chg < -0.25:
        cycle = "CUTTING"
        signal = +0.40  # Rate cuts = bullish gold
    elif abs(ffr_3m_chg) < 0.05:
        cycle = "PAUSED"
        # On pause after hiking = neutral to slightly bullish (peak rates)
        signal = +0.10 if ffr_change > 0 else -0.05
    else:
        cycle = "UNCERTAIN"
        signal = 0.0

    # High absolute rate level = higher opportunity cost for gold
    if current_ffr > 5.0:
        signal -= 0.10
    elif current_ffr < 2.0:
        signal += 0.10

    # Real rate: if FFR is high but inflation is higher → still gold-positive
    if not ry10.empty:
        ry_now = float(ry10.iloc[-1])
        if ry_now < 0.5 and cycle == "HIKING":
            signal += 0.10  # Hiking but real rates still low → not that bad for gold

    return {
        "signal":       round(max(-1.0, min(1.0, signal)), 3),
        "ffr":          round(current_ffr, 3),
        "ffr_12m_chg":  round(ffr_change, 3),
        "cycle":        cycle,
        "note":         f"FFR: {current_ffr:.2f}% | Cycle: {cycle} | 12M change: {ffr_change:+.2f}%",
    }


# ── Composite Macro Scoring ───────────────────────────────────────────────────

MACRO_WEIGHTS = {
    "real_yield":     0.30,   # Most important
    "inflation":      0.20,
    "dollar":         0.20,
    "stress":         0.18,
    "monetary_policy": 0.12,
}


def compute_composite_macro_score(
    ry_model: Dict, inf_model: Dict, dollar_model: Dict,
    stress_model: Dict, mp_model: Dict
) -> Dict:
    """Weighted composite macro score for gold"""

    signals = {
        "real_yield":      ry_model.get("signal", 0),
        "inflation":       inf_model.get("signal", 0),
        "dollar":          dollar_model.get("signal", 0),
        "stress":          stress_model.get("signal", 0),
        "monetary_policy": mp_model.get("signal", 0),
    }

    composite = sum(signals[k] * MACRO_WEIGHTS[k] for k in signals)
    composite  = round(max(-1.0, min(1.0, composite)), 4)

    if composite > 0.35:
        macro_bias  = "STRONGLY_BULLISH"
        confidence  = min(0.55 + composite * 0.35, 0.92)
    elif composite > 0.15:
        macro_bias  = "BULLISH"
        confidence  = min(0.50 + composite * 0.30, 0.80)
    elif composite > -0.15:
        macro_bias  = "NEUTRAL"
        confidence  = 0.42
    elif composite > -0.35:
        macro_bias  = "BEARISH"
        confidence  = min(0.50 + abs(composite) * 0.30, 0.80)
    else:
        macro_bias  = "STRONGLY_BEARISH"
        confidence  = min(0.55 + abs(composite) * 0.35, 0.92)

    # Regime confluence: if 4/5 models agree → conviction bonus
    positive = sum(1 for s in signals.values() if s > 0.1)
    negative = sum(1 for s in signals.values() if s < -0.1)
    if positive >= 4 or negative >= 4:
        confidence = min(confidence + 0.05, 0.93)

    return {
        "composite_score": composite,
        "macro_bias":      macro_bias,
        "confidence":      round(confidence, 3),
        "component_signals": {k: round(v, 4) for k, v in signals.items()},
        "weights":         MACRO_WEIGHTS,
        "positive_models": positive,
        "negative_models": negative,
    }


# ── Regression Model ──────────────────────────────────────────────────────────

def fit_gold_regression(fred_data: Dict) -> Dict:
    """
    Linear regression of gold price on key macro factors.
    Used to identify fair value and deviation from fundamental value.
    """
    try:
        # Get gold price
        gold_df = yf.download(GOLD_SYMBOL, period="5y", interval="1d",
                               progress=False, auto_adjust=True)
        if gold_df.empty:
            return {"model": "failed", "error": "No gold data"}

        if isinstance(gold_df.columns, pd.MultiIndex):
            gold_df.columns = [c[0].lower() for c in gold_df.columns]
        else:
            gold_df.columns = [c.lower() for c in gold_df.columns]

        gold = gold_df["close"].resample("W").last()

        # Build factor matrix (weekly)
        factors = {}
        factor_series = {
            "real_yield_10y": fred_data.get("real_yield_10y", pd.Series(dtype=float)),
            "breakeven_10y":  fred_data.get("breakeven_10y",  pd.Series(dtype=float)),
            "dollar_broad":   fred_data.get("dollar_broad",   pd.Series(dtype=float)),
        }

        for name, series in factor_series.items():
            if not series.empty:
                weekly = series.resample("W").last()
                factors[name] = weekly

        if not factors or len(gold) < 52:
            return {"model": "insufficient_data"}

        # Align all series
        factor_df = pd.DataFrame(factors)
        combined  = pd.concat([gold.rename("gold"), factor_df], axis=1).dropna()

        if len(combined) < 52:
            return {"model": "insufficient_data"}

        X = combined.drop("gold", axis=1)
        y = np.log(combined["gold"])  # Log price

        # Normalize
        X_norm = (X - X.mean()) / X.std()
        X_norm = X_norm.fillna(0)

        # OLS regression
        from scipy import stats as sp_stats
        results = {}
        for col in X_norm.columns:
            slope, intercept, r, p, se = sp_stats.linregress(X_norm[col].values, y.values)
            results[col] = {
                "coefficient": round(slope, 4),
                "r_squared":   round(r**2, 4),
                "p_value":     round(p, 6),
                "significant": p < 0.05,
            }

        # Multi-factor: compute "fundamental" gold price estimate
        from numpy.linalg import lstsq
        X_with_const = np.column_stack([np.ones(len(X_norm)), X_norm.values])
        coeffs, _, _, _ = lstsq(X_with_const, y.values, rcond=None)

        y_pred  = X_with_const @ coeffs
        resid   = y.values - y_pred
        current_gold = float(combined["gold"].iloc[-1])
        fair_value   = float(np.exp(y_pred[-1]))
        deviation    = (current_gold - fair_value) / fair_value * 100

        return {
            "model":          "OLS",
            "r_squared":      round(float(1 - np.var(resid) / np.var(y.values)), 4),
            "factor_betas":   results,
            "fair_value":     round(fair_value, 2),
            "current_price":  round(current_gold, 2),
            "deviation_pct":  round(deviation, 2),
            "verdict":        "OVERVALUED" if deviation > 5 else "UNDERVALUED" if deviation < -5 else "FAIR_VALUE",
            "note":           f"Gold at ${current_gold:.0f} vs fundamental fair value ${fair_value:.0f} ({deviation:+.1f}%)",
        }

    except Exception as e:
        log.debug(f"Regression model error: {e}")
        return {"model": "error", "error": str(e)}


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> Dict:
    log.info("Running Quant Macro Model")
    fred = fetch_all_fred(use_cache=True)

    ry_model     = real_yield_model(fred)
    inf_model    = inflation_regime_model(fred)
    dollar_model = dollar_cycle_model(fred)
    stress_model = financial_stress_model(fred)
    mp_model     = monetary_policy_model(fred)
    composite    = compute_composite_macro_score(ry_model, inf_model, dollar_model, stress_model, mp_model)
    regression   = fit_gold_regression(fred)

    log.info(f"Macro Model: {composite['macro_bias']} (score={composite['composite_score']:.3f})")
    return {
        "macro_bias":        composite["macro_bias"],
        "composite_score":   composite["composite_score"],
        "confidence":        composite["confidence"],
        "component_signals": composite["component_signals"],
        "models": {
            "real_yield":     ry_model,
            "inflation":      inf_model,
            "dollar":         dollar_model,
            "financial_stress": stress_model,
            "monetary_policy": mp_model,
        },
        "regression": regression,
    }


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2, default=str))
