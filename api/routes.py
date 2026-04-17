"""
TITAN v2.0 — FastAPI Routes
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
from datetime import datetime

app = FastAPI(
    title="TITAN v2.0 — Gold Intelligence API",
    description="9-Engine Gold Analysis Framework",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ────────────────────────────────────────────────────────────────────

class TradeInput(BaseModel):
    direction:   str
    entry:       float
    sl:          float
    tp:          float
    exit_price:  Optional[float] = None
    pnl:         Optional[float] = None
    outcome:     Optional[str]   = None
    notes:       Optional[str]   = ""


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "TITAN v2.0",
        "status":  "online",
        "engines": 9,
        "asset":   "GOLD (XAUUSD)"
    }


@app.get("/analyze")
def analyze(background_tasks: BackgroundTasks):
    """Run full TITAN analysis — all 9 engines"""
    from core.titan_runner import titan_analyze
    try:
        verdict = titan_analyze(verbose=False)
        return {"status": "ok", "verdict": verdict, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engine/{engine_id}")
def run_single_engine(engine_id: int):
    """Run a single engine (1-9)"""
    engine_map = {
        1: ("engines.engine1_price_matrix",    "run"),
        2: ("engines.engine2_sentiment_fusion", "run"),
        3: ("engines.engine3_volume_cot",       "run"),
        4: ("engines.engine4_macro_correlation","run"),
        5: ("engines.engine5_liquidity_hunt",   "run"),
        6: ("engines.engine6_regime_detection", "run"),
        7: ("engines.engine7_adversarial_trap", "run"),
        8: ("engines.engine8_memory_learning",  "run"),
        9: ("engines.engine9_meta_learning",    "run"),
    }

    if engine_id not in engine_map:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")

    module_path, fn_name = engine_map[engine_id]
    try:
        import importlib
        module = importlib.import_module(module_path)
        fn     = getattr(module, fn_name)
        result = fn()
        return {"status": "ok", "engine": engine_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trades")
def add_trade(trade: TradeInput):
    """Log a trade to the journal"""
    from core.db import log_trade
    try:
        log_trade(trade.dict())
        return {"status": "ok", "message": "Trade logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades")
def get_trades():
    """Get trade history"""
    from core.db import get_conn
    import pandas as pd
    try:
        conn   = get_conn()
        df     = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()
        return {"status": "ok", "trades": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
def get_performance():
    """Get system performance metrics"""
    from engines.engine8_memory_learning import compute_performance_metrics, get_trade_history
    try:
        trades  = get_trade_history()
        metrics = compute_performance_metrics(trades)
        return {"status": "ok", "performance": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weights")
def get_weights():
    """Get current engine weights"""
    from engines.engine9_meta_learning import run as e9
    try:
        result = e9()
        return {"status": "ok", "weights": result.get("final_weights", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/history")
def get_signal_history(limit: int = 50):
    """Get historical engine signals"""
    from core.db import get_conn
    import pandas as pd
    try:
        conn = get_conn()
        df   = pd.read_sql(
            f"SELECT * FROM engine_signals ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return {"status": "ok", "signals": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
