# FILE: quant/geopolitical_model.py
"""
TITAN v2.0 — Geopolitical Risk Model for Gold
Proxies for geopolitical risk, central bank demand, and safe-haven flows.
Uses: news sentiment, gold/silver ratio, gold/oil ratio, 
COMEX positioning, ETF flows, currency crises.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import feedparser
from typing import Dict, List
from datetime import datetime, timezone

from core.config import GOLD_SYMBOL, GOLD_ETF, GOLD_MINERS, NEWS_API_KEY
from core.logger import get_logger

log = get_logger("QuantGeopoliticalModel")

# Geopolitical keywords that historically move gold
GEO_RISK_KEYWORDS = [
    "war", "conflict", "invasion", "attack", "sanctions", "terrorism",
    "nuclear", "crisis", "tension", "escalation", "missile", "troops",
    "military", "nato", "iran", "russia", "china", "north korea",
    "middle east", "taiwan", "wto", "trade war", "tariff", "embargo",
]

CENTRAL_BANK_KEYWORDS = [
    "central bank", "gold reserve", "gold purchase", "gold buying",
    "china gold", "india gold", "russia gold", "de-dollarization",
    "brics gold", "gold-backed", "currency crisis", "devaluation",
]

RISK_OFF_KEYWORDS = [
    "safe haven", "flight to quality", "risk off", "panic", "selloff",
    "market crash", "recession", "bank failure", "contagion", "systemic",
]


def news_geopolitical_score() -> Dict:
    """Score geopolitical risk from news headlines"""
    risk_score = 0.0
    cb_score   = 0.0
    headlines  = []

    # RSS feeds
    feeds = [
        "https://feeds.reuters.com/reuters/worldNews",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.kitco.com/rss/",
    ]

    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "").lower()
                headlines.append(title)

                geo_hits = sum(1 for kw in GEO_RISK_KEYWORDS if kw in title)
                cb_hits  = sum(1 for kw in CENTRAL_BANK_KEYWORDS if kw in title)
                ro_hits  = sum(1 for kw in RISK_OFF_KEYWORDS if kw in title)

                risk_score += geo_hits * 0.05 + ro_hits * 0.04
                cb_score   += cb_hits * 0.08
        except Exception:
            pass

    risk_score = min(risk_score, 0.60)
    cb_score   = min(cb_score, 0.40)

    # NewsAPI for more comprehensive coverage
    if NEWS_API_KEY:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything"
                "?q=gold+war+OR+gold+crisis+OR+central+bank+gold"
                "&language=en&sortBy=publishedAt&pageSize=20"
                f"&apiKey={NEWS_API_KEY}",
                timeout=10
            )
            for article in r.json().get("articles", []):
                title = (article.get("title") or "").lower()
                geo_hits = sum(1 for kw in GEO_RISK_KEYWORDS if kw in title)
                cb_hits  = sum(1 for kw in CENTRAL_BANK_KEYWORDS if kw in title)
                risk_score += geo_hits * 0.03
                cb_score   += cb_hits * 0.05
        except Exception:
            pass

    return {
        "geo_risk_score": round(min(risk_score, 0.70), 3),
        "cb_buying_score": round(min(cb_score, 0.50), 3),
        "total_score":    round(min(risk_score + cb_score, 1.0), 3),
        "headline_count": len(headlines),
    }


def gold_ratios() -> Dict:
    """
    Key gold ratios as market signals:
    - Gold/Silver ratio: very high (>80) historically bullish gold long-term
    - Gold/Oil ratio: gold relative to commodities
    - Gold/SPX: gold as % of equity valuations
    """
    symbols = {
        "gold":   GOLD_SYMBOL,
        "silver": "SI=F",
        "oil":    "CL=F",
        "spx":    "^GSPC",
    }

    prices = {}
    for name, sym in symbols.items():
        try:
            df = yf.download(sym, period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                prices[name] = float(df["close"].iloc[-1])
        except Exception:
            pass

    results = {}
    signal  = 0.0

    if "gold" in prices and "silver" in prices and prices["silver"] > 0:
        gsr = prices["gold"] / prices["silver"]
        results["gold_silver_ratio"] = round(gsr, 2)
        # Historical avg ~65-70; >80 is historically bullish for gold
        if gsr > 90:
            signal += 0.20
            results["gsr_signal"] = "VERY HIGH — historically bullish for gold"
        elif gsr > 80:
            signal += 0.10
            results["gsr_signal"] = "HIGH — mildly bullish for gold"
        elif gsr < 60:
            signal -= 0.05
            results["gsr_signal"] = "LOW — neutral to bearish"

    if "gold" in prices and "oil" in prices and prices["oil"] > 0:
        gor = prices["gold"] / prices["oil"]
        results["gold_oil_ratio"] = round(gor, 2)
        # High ratio = gold expensive vs oil (commodity cycle)
        if gor > 30:
            results["gor_signal"] = "Gold expensive vs oil"
        elif gor < 15:
            signal += 0.08
            results["gor_signal"] = "Gold cheap vs oil — potential catchup"

    if "gold" in prices and "spx" in prices and prices["spx"] > 0:
        spx_ratio = prices["gold"] / prices["spx"] * 100
        results["gold_spx_ratio_pct"] = round(spx_ratio, 4)

    results["signal"] = round(max(-0.5, min(0.5, signal)), 3)
    results["prices"] = {k: round(v, 2) for k, v in prices.items()}
    return results


def etf_flow_analysis() -> Dict:
    """
    Gold ETF flows as a proxy for institutional demand.
    Rising GLD volume + price = accumulation = bullish
    Falling GLD volume + price = distribution = bearish
    """
    try:
        gld_df = yf.download(GOLD_ETF, period="30d", interval="1d",
                             progress=False, auto_adjust=True)
        gdx_df = yf.download(GOLD_MINERS, period="30d", interval="1d",
                             progress=False, auto_adjust=True)

        result = {}

        for name, df in [("GLD", gld_df), ("GDX", gdx_df)]:
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            vol_20    = float(df["volume"].rolling(20).mean().iloc[-1])
            vol_today = float(df["volume"].iloc[-1])
            vol_ratio = vol_today / vol_20 if vol_20 > 0 else 1.0

            price_5d  = float(df["close"].pct_change(5).iloc[-1] * 100)
            price_20d = float(df["close"].pct_change(20).iloc[-1] * 100)

            # Accumulation/Distribution classification
            if vol_ratio > 1.3 and price_5d > 0:
                flow = "ACCUMULATION"
            elif vol_ratio > 1.3 and price_5d < 0:
                flow = "DISTRIBUTION"
            elif vol_ratio < 0.7:
                flow = "LOW_ACTIVITY"
            else:
                flow = "NORMAL"

            result[name] = {
                "vol_ratio":  round(vol_ratio, 3),
                "price_5d":   round(price_5d, 3),
                "price_20d":  round(price_20d, 3),
                "flow":       flow,
            }

        # Composite flow signal
        flow_signal = 0.0
        for name, data in result.items():
            if data["flow"] == "ACCUMULATION":
                flow_signal += 0.15
            elif data["flow"] == "DISTRIBUTION":
                flow_signal -= 0.15

        # GDX leading indicator
        gdx = result.get("GDX", {})
        gld = result.get("GLD", {})
        if gdx and gld:
            if gdx.get("price_5d", 0) > gld.get("price_5d", 0) + 1:
                flow_signal += 0.10  # Miners outperforming = bullish gold
                result["miners_signal"] = "OUTPERFORMING — bullish leading indicator"
            elif gdx.get("price_5d", 0) < gld.get("price_5d", 0) - 1:
                flow_signal -= 0.08  # Miners underperforming = warning
                result["miners_signal"] = "UNDERPERFORMING — watch for gold weakness"

        result["composite_signal"] = round(max(-0.5, min(0.5, flow_signal)), 3)
        return result

    except Exception as e:
        log.debug(f"ETF flow error: {e}")
        return {"composite_signal": 0.0, "error": str(e)}


def currency_crisis_detector() -> Dict:
    """
    Currency crises drive safe-haven gold demand.
    Monitor EM currency ETFs and key pairs.
    """
    crisis_pairs = {
        "eur_usd":  "EURUSD=X",
        "jpy_usd":  "JPYUSD=X",
        "cny_usd":  "CNYUSD=X",
        "em_etf":   "EEM",
        "dev_etf":  "EFA",
    }

    crisis_score = 0.0
    results = {}

    for name, sym in crisis_pairs.items():
        try:
            df = yf.download(sym, period="30d", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            ret_30d = float(df["close"].pct_change(20).iloc[-1] * 100)
            ret_5d  = float(df["close"].pct_change(5).iloc[-1] * 100)

            results[name] = {"ret_5d": round(ret_5d, 2), "ret_30d": round(ret_30d, 2)}

            # EM assets falling = safe haven gold demand
            if "em" in name or "dev" in name.lower():
                if ret_5d < -3:
                    crisis_score += 0.15  # EM selling = flight to gold
                elif ret_5d > 3:
                    crisis_score -= 0.05
        except Exception:
            pass

    results["crisis_score"] = round(max(-0.3, min(0.5, crisis_score)), 3)
    return results


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> Dict:
    log.info("Running Quant Geopolitical Model")

    news     = news_geopolitical_score()
    ratios   = gold_ratios()
    etf      = etf_flow_analysis()
    currency = currency_crisis_detector()

    # Composite
    composite = (
        news["total_score"] * 0.35 +
        ratios.get("signal", 0) * 0.20 +
        etf.get("composite_signal", 0) * 0.25 +
        currency.get("crisis_score", 0) * 0.20
    )
    composite = round(max(-0.5, min(0.5, composite)), 4)

    signal = "BUY" if composite > 0.10 else "SELL" if composite < -0.10 else "NEUTRAL"
    confidence = 0.45 + abs(composite) * 0.35

    log.info(f"Geo Model: {signal} (score={composite:.3f})")
    return {
        "signal":     signal,
        "confidence": round(confidence, 3),
        "composite":  composite,
        "news":       news,
        "ratios":     ratios,
        "etf_flows":  etf,
        "currency":   currency,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
