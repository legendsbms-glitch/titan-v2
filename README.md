# ⚡ TITAN v2.0 — Gold Intelligence System

> **The most complete free-stack gold trading analysis framework ever built.**
> 12 engines. Bayesian fusion. ICT concepts. ML predictor. Paper trading. Telegram alerts. Docker. CI/CD.

[![CI](https://github.com/legendsbms-glitch/titan-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/legendsbms-glitch/titan-v2/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 Architecture

```
TITAN v2.0
├── Engine  1: Price Matrix          → Multi-TF ICT structure, FVG, OBs, BOS/CHoCH
├── Engine  2: Sentiment Fusion      → FinBERT/keyword NLP + FRED macro + seasonal bias
├── Engine  3: Volume & COT          → CFTC live COT, volume profile, delta, absorption
├── Engine  4: Macro Correlation     → DXY, yields, VIX, safe-haven flow, divergences
├── Engine  5: Liquidity Hunt        → ICT BSL/SSL, stop hunts, Judas swings, retail traps
├── Engine  6: Regime Detection      → GARCH, HMM 3-state, Wyckoff, Hurst, MER, ADX
├── Engine  7: Adversarial Trap      → Fake breakouts, turtle soup, Asian range fade
├── Engine  8: Memory & Learning     → Sharpe, Calmar, Kelly, mistake classifier
├── Engine  9: Meta-Learning         → Dynamic EWMA weights, trust scoring, pruning
├── Engine 10: SMC Advanced          → IPDA, PD arrays, CISD, OTE, Power of 3, Silver Bullet
├── Engine 11: ML Predictor          → Gradient Boosting + RF ensemble, 60+ features, walk-forward CV
├── Engine 12: Options Flow          → GLD options PCR, max pain, GEX, IV surface, GVIX
│
├── Verdict Engine    → Bayesian log-odds fusion + calibrated confidence
├── Risk Fortress     → 10 hard blocks + 5 soft warnings
├── Economic Calendar → 2026 FOMC/NFP/CPI/PCE dates + blackout management
├── Paper Trader      → Live paper trading with auto position management
├── Telegram Alerts   → Trade signals, performance reports, macro alerts
└── Backtester        → Monte Carlo simulation, walk-forward, full metrics
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/legendsbms-glitch/titan-v2.git
cd titan-v2
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env — add free API keys:
# FRED:    https://fred.stlouisfed.org/docs/api/api_key.html
# NewsAPI: https://newsapi.org/register  (optional)
# Telegram: https://t.me/BotFather (optional)
```

### 3. Run
```bash
# Full analysis (all 12 engines)
python main.py analyze

# REST API
python main.py api

# Visual dashboard
python main.py dashboard

# Auto-run every 15 minutes
python main.py schedule --interval 15

# Historical backtest
python main.py backtest --start 2023-01-01 --end 2024-01-01 --capital 10000

# Run tests (40 passing)
python main.py test
```

---

## 🐳 Docker

```bash
# Copy and configure
cp .env.example .env && nano .env

# Start all services (API + Dashboard + Scheduler)
docker-compose up -d

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Docs:      http://localhost:8000/docs
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System info |
| GET | `/analyze` | Full 12-engine analysis |
| GET | `/analyze?force=true` | Force re-run (bypass 5min cache) |
| GET | `/engine/{1-12}` | Run single engine |
| GET | `/quick` | Fast price + structure only |
| POST | `/trades` | Log a trade |
| GET | `/trades` | Trade history |
| GET | `/performance` | System performance metrics |
| GET | `/weights` | Current engine weights |
| GET | `/signals/history` | Historical signals |
| GET | `/calendar` | Upcoming economic events |
| GET | `/alerts` | Recent alerts |
| GET | `/backtest?start=&end=` | Run backtest |
| GET | `/paper` | Paper trading status |
| WS | `/ws/live` | WebSocket live updates |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## 📊 Engine Details

### Engine 1 — Price Matrix (ICT)
- Multi-timeframe analysis: 1m, 5m, 15m, 1h, 4h, 1D, 1W
- **Fair Value Gaps (FVG)**: 3-candle imbalance detection
- **Order Blocks**: bullish/bearish OB with impulse strength filter
- **BOS/CHoCH**: Break of Structure and Change of Character
- **Premium/Discount zones**: ICT OTE Fibonacci zones
- Session ranges: Asian/London/NY high/low
- Kill zone detection: London open, NY AM/PM, Asian range
- Multi-TF confluence scoring

### Engine 5 — Liquidity Hunt (ICT)
- BSL/SSL pool mapping from equal highs/lows
- Stop hunt candle detection (wick pattern analysis)
- **Judas Swing**: session open fake move detection
- **Turtle Soup**: 20-bar H/L breakout failure
- **Retail traps**: obvious S/R sweep detection
- Liquidity void mapping

### Engine 10 — SMC Advanced
- **IPDA**: Interbank Price Delivery Algorithm 20-40 bar liquidity seek
- **PD Arrays**: Breaker blocks, rejection blocks, propulsion blocks
- **CISD**: Change in State of Delivery (displacement detection)
- **OTE**: Optimal Trade Entry Fibonacci zones (62.5%-78.6%)
- **Power of 3**: Accumulation → Manipulation → Distribution
- **Silver Bullet**: Time-based FVG setups (3AM, 10AM, 2PM UTC)

### Engine 11 — ML Predictor
- **60+ features**: returns, momentum, volatility, volume, price action, time
- **Ensemble**: Gradient Boosting + Random Forest
- **Walk-forward CV**: Time-series cross-validation (5 folds)
- **No look-ahead bias**: strict feature engineering
- Feature importance ranking via SHAP-equivalent

### Engine 12 — Options Flow
- GLD options chain: put/call ratio, implied volatility
- **Max Pain**: price where option sellers lose least (gravitational pull)
- **GEX**: Gamma Exposure (positive = range, negative = trending)
- **IV Surface**: term structure + volatility skew
- **GVIX**: Gold Volatility Index proxy from options

---

## 📈 Verdict Engine

The Verdict Engine uses **Bayesian log-odds fusion** to combine all 12 engine signals:

```
Prior: 50/50 (neutral)
For each engine:
  → Compute likelihood ratio from signal + confidence
  → Update posterior using log-odds
Final posterior → BUY/SELL/NEUTRAL + calibrated confidence
```

**Risk Fortress (10 hard blocks):**
1. Confidence < 58%
2. No directional majority
3. Daily loss limit hit (2%)
4. Consecutive losses ≥ 3
5. Pre-news blackout (15min)
6. Post-news blackout (10min)
7. Spread too wide (> 3 pips)
8. R:R below minimum (< 1.5)
9. Signal is NEUTRAL
10. Critical data missing

---

## 💹 Paper Trading

TITAN includes a built-in paper trader that:
- Opens positions automatically from live verdicts
- Tracks P&L, win rate, drawdown in real-time
- Auto-closes when SL or TP is hit
- Logs all trades to SQLite
- Respects daily loss limits and consecutive loss limits

```bash
# Check paper trading status
curl http://localhost:8000/paper
```

---

## 📱 Telegram Alerts

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` to receive:
- 🟢/🔴 Trade signals with entry/SL/TP
- 📋 Daily performance reports
- 🌐 Macro alerts (DXY moves, VIX spikes)

---

## 🧪 Testing

```bash
# Run all 40 tests
python main.py test

# Or directly with pytest
pytest tests/test_engines.py -v
```

**Test coverage:**
- FVG detection logic
- Market structure (HH/HL/LL/LH)
- Liquidity pool detection
- Stop hunt pattern recognition
- Bayesian fusion math
- Risk Fortress blocks
- Kelly criterion calculation
- Volume profile (POC/VAH/VAL)
- Hurst exponent
- Market Efficiency Ratio
- Wyckoff phase detection
- Backtester metrics
- Monte Carlo simulation
- Seasonal bias
- Economic calendar

---

## 📁 Project Structure

```
titan-v2/
├── main.py                          ← Entry point (all modes)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── core/
│   ├── config.py                    ← All settings
│   ├── db.py                        ← SQLAlchemy database layer
│   ├── logger.py                    ← Logging setup
│   ├── titan_runner.py              ← Engine orchestrator
│   ├── verdict_engine.py            ← Bayesian fusion + risk
│   ├── economic_calendar.py         ← 2026 event calendar
│   ├── paper_trader.py              ← Paper trading engine
│   └── telegram_alerts.py           ← Telegram integration
│
├── engines/
│   ├── engine1_price_matrix.py      ← ICT price analysis
│   ├── engine2_sentiment_fusion.py  ← NLP + macro sentiment
│   ├── engine3_volume_cot.py        ← CFTC COT + volume
│   ├── engine4_macro_correlation.py ← Cross-asset analysis
│   ├── engine5_liquidity_hunt.py    ← ICT liquidity
│   ├── engine6_regime_detection.py  ← HMM + GARCH + Wyckoff
│   ├── engine7_adversarial_trap.py  ← Institutional traps
│   ├── engine8_memory_learning.py   ← Performance learning
│   ├── engine9_meta_learning.py     ← Dynamic weight optimization
│   ├── engine10_smc_advanced.py     ← Advanced SMC/ICT
│   ├── engine11_ml_predictor.py     ← ML ensemble predictor
│   └── engine12_options_flow.py     ← Options + derivatives
│
├── api/
│   └── routes.py                    ← FastAPI REST API
│
├── backtesting/
│   └── backtester.py                ← Historical simulation
│
├── alerts/
│   └── alert_engine.py              ← Alert system
│
├── dashboard/
│   └── app.py                       ← Streamlit UI
│
├── tests/
│   └── test_engines.py              ← 40 unit tests
│
├── .github/
│   └── workflows/ci.yml             ← GitHub Actions CI/CD
│
├── data/                            ← Verdicts, ML models (gitignored)
├── db/                              ← SQLite database (gitignored)
└── logs/                            ← Application logs (gitignored)
```

---

## 🆓 Free Stack

| Component | Library / Source | Cost |
|-----------|-----------------|------|
| Price data | `yfinance` (XAUUSD, GLD, DXY, VIX) | FREE |
| COT data | CFTC.gov direct download | FREE |
| Macro data | FRED API (register for key) | FREE |
| News sentiment | RSS feeds (Kitco, Reuters) | FREE |
| Technical analysis | `ta` library | FREE |
| Regime detection | `hmmlearn` + `arch` | FREE |
| ML predictor | `scikit-learn` | FREE |
| Options data | `yfinance` GLD options | FREE |
| Database | SQLite | FREE |
| API | FastAPI + uvicorn | FREE |
| Dashboard | Streamlit | FREE |

> The only optional paid component: FinBERT (requires `transformers` + `torch`, ~500MB download, free to run locally)

---

## ⚙️ Configuration

Key settings in `core/config.py`:

```python
RISK = {
    "max_daily_loss_pct":     2.0,   # Stop trading after 2% daily loss
    "max_position_pct":       1.0,   # Risk 1% per trade
    "max_trades_per_day":     5,     # Max 5 trades per day
    "max_consecutive_losses": 3,     # Stop after 3 consecutive losses
    "min_rr_ratio":           1.5,   # Minimum 1.5:1 R:R
    "min_confidence":         0.58,  # Minimum 58% confidence to trade
}
```

---

## ⚠️ Disclaimer

TITAN v2.0 is for **educational purposes only**. It is not financial advice. Trading involves substantial risk of loss. Always use proper risk management. Past performance does not guarantee future results.

---

*Built by tsk — class 12, 18 years old, with the ambition to build something serious.*
