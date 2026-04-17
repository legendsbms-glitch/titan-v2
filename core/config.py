# FILE: core/config.py
"""TITAN v2.0 - Master Configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
FRED_API_KEY     = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY     = os.getenv("NEWS_API_KEY", "")
TWELVE_DATA_KEY  = os.getenv("TWELVE_DATA_API_KEY", "")

# ── Asset Symbols ─────────────────────────────────────────────────────────────
GOLD_SYMBOL  = "GC=F"        # Gold Futures (yfinance)
GOLD_SPOT    = "XAUUSD=X"    # Gold Spot
GOLD_ETF     = "GLD"         # SPDR Gold ETF
GOLD_MINERS  = "GDX"         # Gold Miners ETF
SILVER_ETF   = "SLV"

MACRO_SYMBOLS = {
    "dxy":     "DX-Y.NYB",
    "vix":     "^VIX",
    "tnx":     "^TNX",     # 10Y yield
    "fvx":     "^FVX",     # 5Y yield
    "spx":     "^GSPC",
    "oil":     "CL=F",
    "silver":  "SI=F",
    "copper":  "HG=F",
    "tlt":     "TLT",      # 20Y Treasury ETF
    "uup":     "UUP",      # Dollar bull ETF
}

# ── Session Times (UTC hours) ─────────────────────────────────────────────────
SESSION_TIMES = {
    "SYDNEY":    (22, 7),   # 22:00 - 07:00 UTC
    "ASIA":      (0,  8),   # 00:00 - 08:00 UTC
    "LONDON":    (8,  17),  # 08:00 - 17:00 UTC
    "NEW_YORK":  (13, 22),  # 13:00 - 22:00 UTC
    "DEAD_ZONE": (22, 24),  # Low liquidity
}

# ── ICT Kill Zones (UTC hours) ────────────────────────────────────────────────
KILL_ZONES = {
    "ASIAN_RANGE":    (22, 3),   # Asian session range build
    "LONDON_OPEN":    (7,  9),   # London open killzone
    "LONDON_CLOSE":   (11, 13),  # London close killzone
    "NY_AM":          (13, 16),  # NY morning killzone (most active)
    "NY_PM":          (19, 21),  # NY afternoon reversal zone
    "NY_CLOSE":       (20, 22),  # NY close
}

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES = {
    "1m":  {"interval": "1m",  "period": "1d",   "weight": 0.05},
    "5m":  {"interval": "5m",  "period": "5d",   "weight": 0.10},
    "15m": {"interval": "15m", "period": "5d",   "weight": 0.15},
    "1h":  {"interval": "1h",  "period": "30d",  "weight": 0.25},
    "4h":  {"interval": "4h",  "period": "90d",  "weight": 0.25},
    "1d":  {"interval": "1d",  "period": "365d", "weight": 0.15},
    "1wk": {"interval": "1wk", "period": "3y",   "weight": 0.05},
}

# ── Engine Weights (default, overridden by Engine 9) ─────────────────────────
ENGINE_WEIGHTS = {
    "engine1_price_matrix":      0.22,
    "engine2_sentiment_fusion":  0.09,
    "engine3_volume_cot":        0.13,
    "engine4_macro_correlation": 0.14,
    "engine5_liquidity_hunt":    0.16,
    "engine6_regime_detection":  0.10,
    "engine7_adversarial_trap":  0.11,
    "engine8_memory_learning":   0.05,
}

# ── Risk Parameters ───────────────────────────────────────────────────────────
RISK = {
    "max_daily_loss_pct":        2.0,
    "max_position_pct":          1.0,
    "max_trades_per_day":        5,
    "max_consecutive_losses":    3,
    "min_rr_ratio":              1.5,
    "max_spread_pips":           3.0,
    "blackout_pre_news_min":     15,
    "blackout_post_news_min":    10,
    "min_confidence":            0.58,
    "high_conviction_threshold": 0.75,
    "half_kelly":                True,
    "atr_sl_multiplier":         1.5,
    "atr_tp_multiplier":         2.25,
    "flash_crash_atr_mult":      3.0,   # Block if ATR spike > 3x normal
}

# ── FRED Economic Series ──────────────────────────────────────────────────────
FRED_SERIES = {
    "DFF":      "fed_funds_rate",
    "T10YIE":   "breakeven_inflation_10y",
    "DFII10":   "real_yield_10y",
    "CPIAUCSL": "cpi_yoy",
    "DTWEXBGS": "dollar_index_broad",
    "UNRATE":   "unemployment_rate",
    "GDPC1":    "real_gdp",
    "T5YIE":    "breakeven_inflation_5y",
    "BAMLH0A0HYM2": "high_yield_spread",
}

# ── Gold Keywords for News Filtering ─────────────────────────────────────────
GOLD_KEYWORDS = [
    "gold", "xauusd", "bullion", "precious metal", "safe haven",
    "inflation", "fed", "fomc", "federal reserve", "rate cut", "rate hike",
    "interest rate", "real yield", "treasury yield", "dollar", "dxy",
    "geopolitical", "conflict", "war", "crisis", "recession",
    "central bank", "etf flow", "comex", "futures", "spot gold",
    "commodity", "powell", "yellen", "ecb", "boe", "cpi", "pce",
    "quantitative easing", "qe", "tightening", "pivot",
]

# ── News RSS Feeds ────────────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://www.kitco.com/rss/",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.mining.com/feed/",
    "https://goldprice.org/rss",
]

# ── Hawkish / Dovish Keywords ─────────────────────────────────────────────────
HAWKISH_KEYWORDS = [
    "rate hike", "tightening", "hawkish", "inflation fight",
    "higher for longer", "restrictive", "strong dollar", "reduce balance sheet",
    "quantitative tightening", "qt", "50bps", "75bps", "aggressive",
]
DOVISH_KEYWORDS = [
    "rate cut", "dovish", "pause", "pivot", "easing", "accommodation",
    "lower rates", "stimulus", "qe", "quantitative easing",
    "data dependent", "flexible", "support growth",
]

# ── Gold Seasonal Bias (monthly, +1 = strong, -1 = weak) ─────────────────────
SEASONAL_BIAS = {
    1:  +0.6,   # January — strong (Indian wedding, Chinese New Year)
    2:  +0.5,   # February — strong
    3:  -0.1,   # March — neutral
    4:  -0.2,   # April — slight weakness
    5:  -0.3,   # May — weak
    6:  -0.4,   # June — weakest month historically
    7:  -0.1,   # July — neutral
    8:  +0.2,   # August — recovering
    9:  +0.5,   # September — strong (Indian festival season)
    10: +0.3,   # October — positive
    11: +0.4,   # November — strong
    12: +0.2,   # December — positive (year-end flows)
}

# ── Cache Settings ────────────────────────────────────────────────────────────
CACHE_TTL_MINUTES      = 5
FRED_CACHE_TTL_HOURS   = 6
COT_CACHE_TTL_HOURS    = 24   # COT is weekly data

# ── Alert Settings ────────────────────────────────────────────────────────────
ALERT_MIN_CONFIDENCE = 0.68

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH          = os.getenv("DB_PATH", "db/titan.db")
JOURNAL_PATH     = os.getenv("JOURNAL_PATH", "journal/trades.json")
FRED_CACHE_PATH  = "data/fred_cache.json"
COT_CACHE_PATH   = "data/cot_cache.json"
VERDICT_PATH     = "data/last_verdict.json"
ALERT_LOG_PATH   = "alerts/alerts.log"
ALERT_JSON_PATH  = "data/latest_alert.json"
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")

# ── CFTC COT Gold Market Code ─────────────────────────────────────────────────
GOLD_COT_CODE = "088691"
COT_URL       = "https://www.cftc.gov/dea/newcot/deafutw.zip"
