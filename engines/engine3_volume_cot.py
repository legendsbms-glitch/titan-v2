# FILE: engines/engine3_volume_cot.py
"""
TITAN v2.0 — Engine 3: Volume & COT
CFTC COT positioning, volume profile, delta, absorption detection
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import io
import json
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from core.config import (GOLD_SYMBOL, GOLD_ETF, GOLD_MINERS,
                          GOLD_COT_CODE, COT_URL, COT_CACHE_PATH)
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine3.VolumeCOT")

os.makedirs("data", exist_ok=True)


# ── OHLCV Helper ──────────────────────────────────────────────────────────────

def _fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df.dropna()


# ── CFTC COT Data ─────────────────────────────────────────────────────────────

def _load_cot_cache() -> dict:
    try:
        if os.path.exists(COT_CACHE_PATH):
            with open(COT_CACHE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cot_cache(data: dict):
    try:
        with open(COT_CACHE_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def fetch_cot_data() -> dict:
    """Download & parse CFTC COT legacy futures data for Gold"""
    cache = _load_cot_cache()
    if cache.get("_ts") and time.time() - cache["_ts"] < 24 * 3600:
        log.debug("COT cache hit")
        return cache

    try:
        r = requests.get(COT_URL, timeout=30)
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = z.namelist()
            fname = names[0]
            with z.open(fname) as f:
                raw = f.read().decode("latin-1")

        # Parse CSV lines
        lines = raw.strip().split("\n")
        header = [h.strip().strip('"').lower() for h in lines[0].split(",")]

        gold_rows = []
        for line in lines[1:]:
            if GOLD_COT_CODE in line or "GOLD" in line.upper():
                parts = line.split(",")
                if len(parts) >= len(header):
                    row = dict(zip(header, [p.strip().strip('"') for p in parts]))
                    gold_rows.append(row)

        if not gold_rows:
            log.warning("Gold not found in COT file, using fallback")
            return _cot_fallback()

        # Parse most recent row
        row = gold_rows[0]

        def safe_int(key):
            try:
                for k in header:
                    if key in k:
                        return int(row.get(k, "0").replace(",", "") or "0")
            except Exception:
                return 0
            return 0

        # Try various column name patterns
        noncomm_long  = 0
        noncomm_short = 0
        comm_long     = 0
        comm_short    = 0
        open_interest = 0

        for k, v in row.items():
            clean = v.replace(",", "").strip()
            if not clean or clean == ".":
                continue
            try:
                val = int(clean)
            except ValueError:
                continue
            kl = k.lower()
            if "noncomm" in kl and "long" in kl and "spread" not in kl:
                noncomm_long = val
            elif "noncomm" in kl and "short" in kl and "spread" not in kl:
                noncomm_short = val
            elif "comm" in kl and "long" in kl and "noncomm" not in kl:
                comm_long = val
            elif "comm" in kl and "short" in kl and "noncomm" not in kl:
                comm_short = val
            elif "open_interest" in kl or "open interest" in kl:
                open_interest = val

        net_noncomm = noncomm_long - noncomm_short
        net_comm    = comm_long - comm_short

        spec_ratio = net_noncomm / open_interest if open_interest > 0 else 0

        result = {
            "_ts":             time.time(),
            "noncomm_long":    noncomm_long,
            "noncomm_short":   noncomm_short,
            "comm_long":       comm_long,
            "comm_short":      comm_short,
            "net_noncomm":     net_noncomm,
            "net_commercial":  net_comm,
            "open_interest":   open_interest,
            "spec_ratio":      round(spec_ratio, 4),
            "source":          "CFTC_LIVE",
        }
        _save_cot_cache(result)
        return result

    except Exception as e:
        log.warning(f"COT fetch error: {e}")
        return _cot_fallback()


def _cot_fallback() -> dict:
    return {
        "_ts": time.time(),
        "net_noncomm": 0, "net_commercial": 0,
        "spec_ratio": 0, "open_interest": 0, "source": "FALLBACK"
    }


def cot_percentile(net_pos: int, history_net: list) -> float:
    """Return percentile rank of current net position in history"""
    if not history_net or len(history_net) < 2:
        return 50.0
    arr = sorted(history_net)
    count_below = sum(1 for x in arr if x < net_pos)
    return round(count_below / len(arr) * 100, 1)


# ── Volume Profile ────────────────────────────────────────────────────────────

def build_volume_profile(df: pd.DataFrame, bins: int = 30) -> dict:
    """Volume Profile: POC, VAH, VAL"""
    if df.empty or len(df) < 5:
        return {"poc": None, "vah": None, "val": None}

    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if price_max <= price_min:
        return {"poc": None, "vah": None, "val": None}

    bin_size = (price_max - price_min) / bins
    vol_map  = {}

    for _, row in df.iterrows():
        candle_bins = max(1, int((float(row["high"]) - float(row["low"])) / bin_size))
        vol_per_bin = float(row["volume"]) / candle_bins if candle_bins > 0 else 0
        low_bin = int((float(row["low"]) - price_min) / bin_size)
        for b in range(candle_bins):
            idx = low_bin + b
            if 0 <= idx < bins:
                price_lvl = round(price_min + idx * bin_size, 2)
                vol_map[price_lvl] = vol_map.get(price_lvl, 0) + vol_per_bin

    if not vol_map:
        return {"poc": None, "vah": None, "val": None}

    sorted_vol = sorted(vol_map.items(), key=lambda x: x[1], reverse=True)
    poc        = sorted_vol[0][0]
    total_vol  = sum(v for _, v in sorted_vol)
    target     = total_vol * 0.70

    cum_vol = 0.0
    value_area = []
    for price, vol in sorted_vol:
        if cum_vol < target:
            value_area.append(price)
            cum_vol += vol

    return {
        "poc": round(float(poc), 2),
        "vah": round(float(max(value_area)), 2) if value_area else None,
        "val": round(float(min(value_area)), 2) if value_area else None,
        "bins": bins,
        "total_volume": round(total_vol, 0),
    }


# ── Volume Delta ──────────────────────────────────────────────────────────────

def calculate_delta(df: pd.DataFrame) -> dict:
    """Approximate buying/selling pressure per candle"""
    if df.empty:
        return {"delta_score": 0.0, "trend": "NEUTRAL"}

    df = df.copy()
    rng = df["high"] - df["low"]
    rng = rng.replace(0, np.nan)

    df["buy_ratio"]  = (df["close"] - df["low"]) / rng
    df["sell_ratio"] = (df["high"] - df["close"]) / rng

    recent = df.tail(20)
    avg_buy  = float(recent["buy_ratio"].mean())
    avg_sell = float(recent["sell_ratio"].mean())

    delta_score = avg_buy - avg_sell  # -1 to +1
    trend = "BUYING" if delta_score > 0.1 else "SELLING" if delta_score < -0.1 else "NEUTRAL"

    return {
        "delta_score": round(delta_score, 3),
        "avg_buy_ratio":  round(avg_buy, 3),
        "avg_sell_ratio": round(avg_sell, 3),
        "trend":          trend,
    }


# ── Absorption Detection ──────────────────────────────────────────────────────

def detect_absorption(df: pd.DataFrame) -> dict:
    """High volume + small candle body = absorption (smart money accumulating/distributing)"""
    if df.empty or len(df) < 10:
        return {"absorption_detected": False, "type": None}

    df = df.copy()
    df["body"]  = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)

    avg_volume = float(df["volume"].mean())
    recent     = df.tail(5)

    absorption_candles = []
    for i, row in recent.iterrows():
        is_high_vol  = float(row["volume"]) > avg_volume * 1.5
        is_small_body = float(row.get("body_ratio", 1)) < 0.30
        if is_high_vol and is_small_body:
            candle_type = "BULLISH_ABSORB" if float(row["close"]) > float(row["open"]) else "BEARISH_ABSORB"
            absorption_candles.append({
                "type":        candle_type,
                "timestamp":   str(i),
                "vol_ratio":   round(float(row["volume"]) / avg_volume, 2),
                "body_ratio":  round(float(row.get("body_ratio", 0)), 3),
            })

    detected = len(absorption_candles) > 0
    atype    = absorption_candles[-1]["type"] if detected else None
    return {
        "absorption_detected": detected,
        "type":                atype,
        "candles":             absorption_candles,
        "signal":              "BUY" if atype == "BULLISH_ABSORB" else
                               "SELL" if atype == "BEARISH_ABSORB" else None,
    }


# ── ETF Flow Analysis ─────────────────────────────────────────────────────────

def fetch_etf_flows() -> dict:
    """GLD/GDX ETF as volume/flow proxy for gold"""
    try:
        gld = _fetch(GOLD_ETF,   "10d", "1d")
        gdx = _fetch(GOLD_MINERS,"10d", "1d")

        results = {}
        for name, df in [("GLD", gld), ("GDX", gdx)]:
            if len(df) >= 2:
                p_chg = float((df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100)
                v_chg = float((df["volume"].iloc[-1] - df["volume"].iloc[-2]) / df["volume"].iloc[-2] * 100) if df["volume"].iloc[-2] != 0 else 0
                results[name] = {
                    "price_change_pct":  round(p_chg, 3),
                    "volume_change_pct": round(v_chg, 3),
                    "accumulation": p_chg > 0 and v_chg > 10,
                    "distribution": p_chg < 0 and v_chg > 10,
                }

        # GDX leading indicator: gold miners often lead gold
        gdx_data = results.get("GDX", {})
        gld_data = results.get("GLD", {})
        divergence = None
        if gdx_data and gld_data:
            if gdx_data["price_change_pct"] > 0.5 and gld_data["price_change_pct"] < 0:
                divergence = "GDX_LEADING_BULLISH"
            elif gdx_data["price_change_pct"] < -0.5 and gld_data["price_change_pct"] > 0:
                divergence = "GDX_LEADING_BEARISH"

        results["divergence"] = divergence
        return results

    except Exception as e:
        log.warning(f"ETF flow error: {e}")
        return {}


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("Running Engine 3: Volume & COT")
    try:
        df = _fetch(GOLD_SYMBOL, "20d", "1h")

        cot_data   = fetch_cot_data()
        vol_prof   = build_volume_profile(df, bins=30) if not df.empty else {}
        delta      = calculate_delta(df) if not df.empty else {}
        absorption = detect_absorption(df) if not df.empty else {}
        etf_flows  = fetch_etf_flows()

        score = 0.0
        notes = []

        # COT signal
        spec_ratio = cot_data.get("spec_ratio", 0)
        if spec_ratio > 0.08:
            score += 0.3
            notes.append(f"Specs heavily net long ({spec_ratio:.2%}) — bullish")
        elif spec_ratio > 0.03:
            score += 0.15
        elif spec_ratio < -0.03:
            score -= 0.15
        elif spec_ratio < -0.08:
            score -= 0.3
            notes.append(f"Specs heavily net short — bearish")

        # Delta
        delta_score = delta.get("delta_score", 0)
        score += delta_score * 0.25

        # Absorption
        if absorption.get("signal") == "BUY":
            score += 0.25
            notes.append("Bullish absorption detected — smart money buying")
        elif absorption.get("signal") == "SELL":
            score -= 0.25
            notes.append("Bearish absorption detected — smart money selling")

        # ETF flows
        gld_d = etf_flows.get("GLD", {})
        gdx_d = etf_flows.get("GDX", {})
        if gld_d.get("accumulation"):
            score += 0.15
            notes.append("GLD accumulation")
        elif gld_d.get("distribution"):
            score -= 0.15
            notes.append("GLD distribution")
        if etf_flows.get("divergence") == "GDX_LEADING_BULLISH":
            score += 0.12
            notes.append("GDX leading gold higher")

        # Volume profile context
        if vol_prof.get("poc") and not df.empty:
            current = float(df["close"].iloc[-1])
            poc     = vol_prof["poc"]
            if current > poc:
                score += 0.05
                notes.append(f"Price above POC ({poc}) — bullish")
            else:
                score -= 0.05
                notes.append(f"Price below POC ({poc}) — bearish")

        score = max(-1.0, min(1.0, score))

        if score > 0.18:
            signal     = "BUY"
            confidence = min(0.5 + score * 0.40, 0.90)
        elif score < -0.18:
            signal     = "SELL"
            confidence = min(0.5 + abs(score) * 0.40, 0.90)
        else:
            signal     = "NEUTRAL"
            confidence = 0.40

        result = {
            "signal":        signal,
            "confidence":    round(confidence, 3),
            "score":         round(score, 3),
            "notes":         notes,
            "cot": {
                "net_noncomm":   cot_data.get("net_noncomm"),
                "spec_ratio":    cot_data.get("spec_ratio"),
                "open_interest": cot_data.get("open_interest"),
                "source":        cot_data.get("source"),
            },
            "volume_profile": vol_prof,
            "delta":          delta,
            "absorption":     absorption,
            "etf_flows":      etf_flows,
        }

        log_signal("engine3_volume_cot", signal, confidence, result)
        log.info(f"Engine 3 → {signal} @ {confidence:.1%} | COT spec ratio: {spec_ratio:.3f}")
        return result

    except Exception as e:
        log.error(f"Engine 3 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
