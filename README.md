# ⚡ TITAN v2.0 — 9-Engine Gold Intelligence System

> Built by tsk | Full free-stack gold analysis framework

---

## Architecture

```
TITAN v2.0
├── Engine 1: Price Matrix          → Multi-TF structure, FVG, key levels
├── Engine 2: Sentiment Fusion      → FinBERT + FRED + RSS news
├── Engine 3: Volume & COT          → CFTC COT, volume profile, ETF flows
├── Engine 4: Macro Correlation     → DXY, yields, VIX, cross-asset
├── Engine 5: Liquidity Hunt ★      → Stop mapping, manipulation detection
├── Engine 6: Regime Detection      → HMM, GARCH, Wyckoff cycle
├── Engine 7: Adversarial Trap ★    → Institutional playbook matching
├── Engine 8: Memory & Learning ★   → Trade journal, mistake classification
├── Engine 9: Meta-Learning ★       → Dynamic weight optimization
│
├── Verdict Engine                  → Bayesian fusion, calibrated confidence
└── Risk Fortress                   → 10 hard blocks + 5 soft warnings
```

---

## Quick Start

### 1. Install dependencies
```bash
cd titan
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your free API keys:
# - FRED: https://fred.stlouisfed.org/docs/api/api_key.html
# - NewsAPI: https://newsapi.org/register
```

### 3. Run analysis
```bash
# One-shot analysis
python main.py analyze

# Start API server
python main.py api

# Launch dashboard
python main.py dashboard

# Start scheduler (every 15 min)
python main.py schedule 15
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analyze` | Full 9-engine analysis |
| GET | `/engine/{1-9}` | Run single engine |
| GET | `/performance` | System performance stats |
| GET | `/weights` | Current engine weights |
| POST | `/trades` | Log a trade |
| GET | `/trades` | Get trade history |
| GET | `/signals/history` | Historical signals |

---

## Free APIs Used

| Service | What For | Key Required |
|---------|----------|--------------|
| yfinance | Price data (gold, DXY, VIX, yields) | No |
| CFTC.gov | COT positioning data | No |
| FRED | Fed rates, real yields, CPI | Yes (free) |
| NewsAPI | News headlines | Yes (free) |
| HuggingFace | FinBERT sentiment | No |
| RSS feeds | Kitco, Reuters gold news | No |

---

## Project Structure

```
titan/
├── main.py                    ← Entry point
├── requirements.txt
├── .env.example
│
├── core/
│   ├── config.py              ← All settings
│   ├── db.py                  ← SQLite storage
│   ├── logger.py              ← Logging
│   ├── titan_runner.py        ← Engine orchestrator
│   └── verdict_engine.py      ← Bayesian fusion + risk
│
├── engines/
│   ├── engine1_price_matrix.py
│   ├── engine2_sentiment_fusion.py
│   ├── engine3_volume_cot.py
│   ├── engine4_macro_correlation.py
│   ├── engine5_liquidity_hunt.py
│   ├── engine6_regime_detection.py
│   ├── engine7_adversarial_trap.py
│   ├── engine8_memory_learning.py
│   └── engine9_meta_learning.py
│
├── api/
│   └── routes.py              ← FastAPI endpoints
│
├── dashboard/
│   └── app.py                 ← Streamlit UI
│
├── db/                        ← SQLite database (auto-created)
├── data/                      ← Saved verdicts (auto-created)
├── logs/                      ← Log files (auto-created)
└── journal/                   ← Trade journal (auto-created)
```

---

## Adding Trades (for Engine 8 & 9 to learn)

```bash
# Via API
curl -X POST http://localhost:8000/trades \
  -H "Content-Type: application/json" \
  -d '{
    "direction": "BUY",
    "entry": 2345.50,
    "sl": 2338.00,
    "tp": 2360.00,
    "exit_price": 2359.00,
    "pnl": 13.50,
    "outcome": "WIN",
    "notes": "London breakout, clean FVG entry"
  }'
```

---

## Notes

- Engine 9 (Meta-Learning) needs **50+ trades** before weights become meaningful
- FinBERT runs locally — first run downloads ~500MB model (cached after)
- GARCH volatility requires `arch` package
- HMM regime detection requires `hmmlearn`
- All data is stored in `db/titan.db` (SQLite)

---

*TITAN v2.0 — Built for tsk | Not financial advice*
