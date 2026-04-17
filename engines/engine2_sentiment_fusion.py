# FILE: engines/engine2_sentiment_fusion.py
"""
TITAN v2.0 — Engine 2: Sentiment Fusion
FinBERT/keyword sentiment + FRED macro + Fed stance + seasonal bias
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict

from core.config import (FRED_API_KEY, NEWS_API_KEY, FRED_SERIES,
                          GOLD_KEYWORDS, RSS_FEEDS, HAWKISH_KEYWORDS,
                          DOVISH_KEYWORDS, SEASONAL_BIAS, FRED_CACHE_PATH)
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine2.SentimentFusion")

os.makedirs("data", exist_ok=True)

# ── Bullish / Bearish news keywords ──────────────────────────────────────────
BULLISH_WORDS = [
    "safe haven", "inflation", "rate cut", "dovish", "crisis", "geopolitical",
    "demand surge", "etf inflow", "central bank buying", "war", "conflict",
    "recession", "stagflation", "weak dollar", "dollar falls", "rate pause",
    "qe", "stimulus", "lower yields", "negative real yield", "fear",
]
BEARISH_WORDS = [
    "rate hike", "hawkish", "strong dollar", "dollar rally", "risk on",
    "tightening", "qt", "quantitative tightening", "outflows", "selloff",
    "rising yields", "high rates", "good data", "strong jobs", "fed hike",
    "50bps", "75bps", "aggressive fed", "dollar strength",
]


# ── RSS Headline Fetching ─────────────────────────────────────────────────────

def fetch_rss_headlines() -> List[Dict]:
    headlines = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:25]:
                title = entry.get("title", "").lower()
                if any(kw in title for kw in GOLD_KEYWORDS):
                    headlines.append({
                        "title":     entry.get("title", ""),
                        "source":    feed.feed.get("title", url),
                        "published": entry.get("published", ""),
                    })
        except Exception as e:
            log.debug(f"RSS error ({url}): {e}")
    return headlines[:40]


def fetch_newsapi_headlines() -> List[Dict]:
    if not NEWS_API_KEY:
        return []
    try:
        url = (
            "https://newsapi.org/v2/everything"
            "?q=gold+OR+XAUUSD+OR+bullion+OR+%22precious+metal%22"
            "&language=en&sortBy=publishedAt&pageSize=20"
            f"&apiKey={NEWS_API_KEY}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [{"title": a["title"], "source": a["source"]["name"]} for a in articles]
    except Exception as e:
        log.debug(f"NewsAPI error: {e}")
        return []


# ── Keyword Sentiment Scoring ─────────────────────────────────────────────────

def score_sentiment_keywords(texts: List[str]) -> Dict:
    """Score sentiment without ML: keyword counting with weight"""
    if not texts:
        return {"score": 0.0, "bullish_hits": 0, "bearish_hits": 0, "method": "keyword"}

    combined = " ".join(texts).lower()
    bull = sum(combined.count(kw) for kw in BULLISH_WORDS)
    bear = sum(combined.count(kw) for kw in BEARISH_WORDS)
    total = bull + bear

    if total == 0:
        score = 0.0
    else:
        score = (bull - bear) / total

    return {
        "score":        round(score, 3),
        "bullish_hits": bull,
        "bearish_hits": bear,
        "method":       "keyword",
    }


def score_sentiment_finbert(texts: List[str]) -> Dict:
    """Try FinBERT — fall back to keyword if unavailable"""
    if not texts:
        return {"score": 0.0, "method": "none"}
    try:
        from transformers import pipeline
        pipe = pipeline("text-classification", model="ProsusAI/finbert",
                        truncation=True, max_length=512)
        results = pipe(texts[:15])
        pos = sum(1 for r in results if r["label"].lower() == "positive")
        neg = sum(1 for r in results if r["label"].lower() == "negative")
        total = len(results)
        score = (pos - neg) / total if total > 0 else 0.0
        return {"score": round(score, 3), "positive": pos, "negative": neg, "method": "finbert"}
    except Exception:
        return score_sentiment_keywords(texts)


# ── FRED Data with Caching ────────────────────────────────────────────────────

def _load_fred_cache() -> Dict:
    try:
        if os.path.exists(FRED_CACHE_PATH):
            with open(FRED_CACHE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_fred_cache(data: Dict):
    try:
        with open(FRED_CACHE_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def fetch_fred_data() -> Dict:
    """Fetch FRED economic data with 6h cache"""
    cache = _load_fred_cache()
    now   = time.time()

    if cache.get("_ts") and now - cache["_ts"] < 6 * 3600:
        log.debug("FRED cache hit")
        return cache

    if not FRED_API_KEY:
        return {}

    result = {"_ts": now}
    for series_id, label in FRED_SERIES.items():
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={FRED_API_KEY}"
                f"&file_type=json&limit=3&sort_order=desc"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            obs = [o for o in r.json().get("observations", []) if o.get("value", ".") != "."]
            if len(obs) >= 1:
                result[label] = {
                    "latest": float(obs[0]["value"]),
                    "prev":   float(obs[1]["value"]) if len(obs) > 1 else None,
                    "change": float(obs[0]["value"]) - float(obs[1]["value"]) if len(obs) > 1 else 0.0,
                }
        except Exception as e:
            log.debug(f"FRED {series_id} error: {e}")

    _save_fred_cache(result)
    return result


def score_macro_from_fred(fred: Dict) -> Dict:
    """Score macro environment for gold based on FRED data"""
    score = 0.0
    notes = []

    ry = fred.get("real_yield_10y", {})
    if ry:
        real_yield = ry.get("latest", 0)
        score -= real_yield * 0.4  # Negative real yields = very bullish gold
        if real_yield < 0:
            notes.append(f"Negative real yield ({real_yield:.2f}%) — very bullish gold")
        elif real_yield > 2:
            notes.append(f"High real yield ({real_yield:.2f}%) — bearish gold")

    bei = fred.get("breakeven_inflation_10y", {})
    if bei:
        inflation_exp = bei.get("latest", 0)
        score += (inflation_exp - 2.0) * 0.3  # Above 2% = bullish gold
        if inflation_exp > 2.5:
            notes.append(f"High inflation expectations ({inflation_exp:.2f}%) — bullish gold")

    dxy = fred.get("dollar_index_broad", {})
    if dxy and dxy.get("change"):
        score -= dxy["change"] * 0.5  # Dollar rising = bearish gold
        if dxy["change"] > 1:
            notes.append("Dollar strengthening — bearish gold")
        elif dxy["change"] < -1:
            notes.append("Dollar weakening — bullish gold")

    ffr = fred.get("fed_funds_rate", {})
    if ffr and ffr.get("change"):
        if ffr["change"] > 0:
            score -= 0.2
            notes.append("Fed tightening — mild bearish gold")
        elif ffr["change"] < 0:
            score += 0.3
            notes.append("Fed easing — bullish gold")

    return {
        "score": round(max(-1.0, min(1.0, score)), 3),
        "notes": notes,
        "real_yield": ry.get("latest") if ry else None,
        "inflation_expectations": bei.get("latest") if bei else None,
    }


# ── Fed Stance Detector ───────────────────────────────────────────────────────

def detect_fed_stance(texts: List[str]) -> Dict:
    """Detect hawkish vs dovish Fed language in news"""
    combined = " ".join(texts).lower()
    hawk_count = sum(combined.count(kw) for kw in HAWKISH_KEYWORDS)
    dove_count = sum(combined.count(kw) for kw in DOVISH_KEYWORDS)
    total = hawk_count + dove_count

    if total == 0:
        return {"stance": "NEUTRAL", "strength": 0.0, "hawk": 0, "dove": 0}

    hawk_ratio = hawk_count / total
    if hawk_ratio > 0.65:
        stance = "HAWKISH"
    elif hawk_ratio < 0.35:
        stance = "DOVISH"
    else:
        stance = "MIXED"

    strength = abs(hawk_count - dove_count) / total
    return {
        "stance":   stance,
        "strength": round(strength, 3),
        "hawk":     hawk_count,
        "dove":     dove_count,
    }


# ── Seasonal Bias ─────────────────────────────────────────────────────────────

def get_seasonal_bias() -> Dict:
    month = datetime.utcnow().month
    bias  = SEASONAL_BIAS.get(month, 0.0)
    month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    return {
        "month":     month_names[month],
        "bias":      bias,
        "direction": "BULLISH" if bias > 0 else "BEARISH" if bias < 0 else "NEUTRAL",
        "note":      "Strong seasonal demand" if bias > 0.3 else
                     "Weak seasonal period" if bias < -0.2 else "Neutral seasonal",
    }


# ── Main Runner ───────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("Running Engine 2: Sentiment Fusion")
    try:
        rss       = fetch_rss_headlines()
        newsapi   = fetch_newsapi_headlines()
        all_heads = rss + newsapi
        titles    = [h["title"] for h in all_heads]

        sentiment  = score_sentiment_finbert(titles)
        fred_data  = fetch_fred_data()
        macro      = score_macro_from_fred(fred_data) if fred_data else {"score": 0.0, "notes": []}
        fed_stance = detect_fed_stance(titles)
        seasonal   = get_seasonal_bias()

        # Combine scores
        sent_score     = sentiment.get("score", 0.0)
        macro_score    = macro.get("score", 0.0)
        seasonal_score = seasonal["bias"] * 0.4  # Reduced weight

        # Fed stance adjustment
        fed_adj = 0.0
        if fed_stance["stance"] == "DOVISH":
            fed_adj = +0.15
        elif fed_stance["stance"] == "HAWKISH":
            fed_adj = -0.15

        combined = (sent_score * 0.35) + (macro_score * 0.40) + (seasonal_score * 0.15) + (fed_adj * 0.10)
        combined = max(-1.0, min(1.0, combined))

        if combined > 0.12:
            signal     = "BUY"
            confidence = min(0.5 + combined * 0.38, 0.88)
        elif combined < -0.12:
            signal     = "SELL"
            confidence = min(0.5 + abs(combined) * 0.38, 0.88)
        else:
            signal     = "NEUTRAL"
            confidence = 0.38

        result = {
            "signal":           signal,
            "confidence":       round(confidence, 3),
            "combined_score":   round(combined, 3),
            "sentiment":        sentiment,
            "macro_score":      round(macro_score, 3),
            "macro_notes":      macro.get("notes", []),
            "fed_stance":       fed_stance,
            "seasonal_bias":    seasonal,
            "headline_count":   len(all_heads),
            "top_headlines":    titles[:5],
            "real_yield":       macro.get("real_yield"),
        }

        log_signal("engine2_sentiment_fusion", signal, confidence, result)
        log.info(f"Engine 2 → {signal} @ {confidence:.1%} | Fed: {fed_stance['stance']} | Season: {seasonal['direction']}")
        return result

    except Exception as e:
        log.error(f"Engine 2 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
