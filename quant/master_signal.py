# FILE: quant/master_signal.py
"""
TITAN v2.0 — Master Quant Signal
Combines ALL signals: macro model + technical model + geopolitical + 12 engines
Final output: conviction score, direction, confidence, full attribution
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from core.logger import get_logger
from core.db import init_db

log = get_logger("MasterSignal")

MASTER_WEIGHTS = {
    "quant_macro":        0.30,  # Fundamental macro model
    "quant_technical":    0.25,  # Technical analysis
    "quant_geo":          0.10,  # Geopolitical model
    "engine_composite":   0.35,  # All 12 TITAN engines combined
}


def _run_quant_models() -> Dict:
    """Run all quant models"""
    results = {}

    try:
        from quant.macro_model import run as macro_run
        results["quant_macro"] = macro_run()
        log.info(f"  ✓ Macro model: {results['quant_macro'].get('macro_bias')}")
    except Exception as e:
        log.warning(f"  ✗ Macro model error: {e}")
        results["quant_macro"] = {"signal": "NEUTRAL", "confidence": 0.0, "composite_score": 0.0}

    try:
        from quant.technical_model import run as tech_run
        results["quant_technical"] = tech_run()
        log.info(f"  ✓ Technical model: {results['quant_technical'].get('signal')}")
    except Exception as e:
        log.warning(f"  ✗ Technical model error: {e}")
        results["quant_technical"] = {"signal": "NEUTRAL", "confidence": 0.0, "composite": 0.0}

    try:
        from quant.geopolitical_model import run as geo_run
        results["quant_geo"] = geo_run()
        log.info(f"  ✓ Geo model: {results['quant_geo'].get('signal')}")
    except Exception as e:
        log.warning(f"  ✗ Geo model error: {e}")
        results["quant_geo"] = {"signal": "NEUTRAL", "confidence": 0.0, "composite": 0.0}

    return results


def _run_engine_models() -> Dict:
    """Run all 12 TITAN engines and get composite signal"""
    from core.titan_runner import run_all_engines
    from engines.engine9_meta_learning import run as e9
    from core.config import ENGINE_WEIGHTS

    engine_results = run_all_engines()

    # Get E6 regime
    e6 = engine_results.get("engine6_regime_detection", {})
    regime = e6.get("current_regime", "RANGING")

    # Get weights from E9
    meta = e9(engine_results=engine_results, current_regime=regime)
    weights = meta.get("final_weights", ENGINE_WEIGHTS)

    # Also run new engines
    for eng_name, eng_module in [
        ("engine10_smc_advanced",  "engines.engine10_smc_advanced"),
        ("engine11_ml_predictor",  "engines.engine11_ml_predictor"),
        ("engine12_options_flow",  "engines.engine12_options_flow"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(eng_module)
            engine_results[eng_name] = mod.run()
            sig  = engine_results[eng_name].get("signal", "?")
            conf = engine_results[eng_name].get("confidence", 0)
            log.info(f"  ✓ {eng_name}: {sig} @ {conf:.1%}")
        except Exception as ex:
            log.warning(f"  ✗ {eng_name}: {ex}")
            engine_results[eng_name] = {"signal": "ERROR", "confidence": 0.0}

    # Add new engine weights
    weights["engine10_smc_advanced"]  = 0.08
    weights["engine11_ml_predictor"]  = 0.07
    weights["engine12_options_flow"]  = 0.05

    # Renormalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    return engine_results, weights, regime


def score_to_signal(score: float) -> str:
    if score > 0.20:   return "BUY"
    if score < -0.20:  return "SELL"
    return "NEUTRAL"


def run(verbose: bool = True) -> Dict:
    """Run the complete master signal combining everything"""
    log.info("=" * 60)
    log.info("TITAN MASTER QUANT SIGNAL — STARTING")
    log.info("=" * 60)

    start = time.time()
    init_db()

    # Run quant models
    log.info("\n[Phase 1] Running Quant Models...")
    quant = _run_quant_models()

    # Run TITAN engines
    log.info("\n[Phase 2] Running TITAN Engines...")
    engine_results, weights, regime = _run_engine_models()

    # Extract score from each quant model
    SIGNAL_MAP = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0}

    macro_score   = quant.get("quant_macro", {}).get("composite_score", 0)
    tech_score    = quant.get("quant_technical", {}).get("composite", 0)
    geo_score     = quant.get("quant_geo", {}).get("composite", 0)

    # Engine composite from Bayesian fusion
    from core.verdict_engine import log_odds_fusion
    engine_fusion = log_odds_fusion(engine_results, weights)
    engine_score  = engine_fusion.get("posterior_buy", 0.5) - engine_fusion.get("posterior_sell", 0.5)

    # Master score
    master_score = (
        macro_score  * MASTER_WEIGHTS["quant_macro"] +
        tech_score   * MASTER_WEIGHTS["quant_technical"] +
        geo_score    * MASTER_WEIGHTS["quant_geo"] +
        engine_score * MASTER_WEIGHTS["engine_composite"]
    )
    master_score = round(max(-1.0, min(1.0, master_score)), 4)

    # Signal + confidence
    signal = score_to_signal(master_score)
    raw_confidence = 0.50 + abs(master_score) * 0.40

    # Agreement bonus
    individual_signals = [
        score_to_signal(macro_score),
        score_to_signal(tech_score),
        score_to_signal(geo_score),
        engine_fusion.get("signal", "NEUTRAL"),
    ]
    agreement = sum(1 for s in individual_signals if s == signal)
    if agreement == 4:
        raw_confidence = min(raw_confidence + 0.06, 0.95)
        conviction = "MAXIMUM"
    elif agreement == 3:
        raw_confidence = min(raw_confidence + 0.03, 0.92)
        conviction = "HIGH"
    elif agreement == 2:
        conviction = "MODERATE"
    else:
        conviction = "LOW"
        raw_confidence = max(raw_confidence - 0.05, 0.35)

    confidence = round(raw_confidence, 3)

    # Full verdict (with risk fortress)
    e1 = engine_results.get("engine1_price_matrix", {})
    e6 = engine_results.get("engine6_regime_detection", {})
    from core.economic_calendar import is_blackout_active, minutes_to_next_event
    from core.verdict_engine import compute_verdict

    context = {
        "current_price":    e1.get("current_price"),
        "atr":              e1.get("atr"),
        "session":          e1.get("session", "UNKNOWN"),
        "regime":           regime,
        "volatility_regime": e6.get("volatility", {}).get("regime", "NORMAL"),
        "mins_to_next_event": minutes_to_next_event(),
        "blackout_active":  is_blackout_active()["active"],
    }

    verdict = compute_verdict(engine_results, weights, context)
    verdict["master_score"]   = master_score
    verdict["conviction"]     = conviction
    verdict["quant_models"]   = {
        "macro_score": round(macro_score, 4),
        "tech_score":  round(tech_score, 4),
        "geo_score":   round(geo_score, 4),
        "engine_score": round(engine_score, 4),
    }
    verdict["elapsed"] = round(time.time() - start, 2)

    if verbose:
        _print_master_verdict(verdict, quant, conviction, individual_signals)

    # Save full output
    os.makedirs("data", exist_ok=True)
    try:
        with open("data/master_signal.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
    except Exception:
        pass

    return verdict


def _print_master_verdict(verdict: Dict, quant: Dict, conviction: str, signals: list):
    direction  = verdict.get("direction", "?")
    confidence = verdict.get("confidence", 0)
    tradeable  = verdict.get("tradeable", False)
    entry      = verdict.get("entry")
    sl         = verdict.get("sl")
    tp         = verdict.get("tp")

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║          TITAN v2.0 — MASTER QUANT SIGNAL                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    emoji = {"BUY": "📈 🟢", "SELL": "📉 🔴", "NEUTRAL": "➡️ ⚪"}.get(direction, "")
    print(f"║  {emoji}  {direction:<8}  |  Confidence: {confidence:.1%}  |  {conviction} conviction  ║")
    print(f"║  Master Score: {verdict.get('master_score',0):+.4f}                                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    qm = verdict.get("quant_models", {})
    print(f"║  Macro Score:    {qm.get('macro_score',0):+.4f}  (30% weight)                    ║")
    print(f"║  Technical:      {qm.get('tech_score',0):+.4f}  (25% weight)                    ║")
    print(f"║  Geopolitical:   {qm.get('geo_score',0):+.4f}  (10% weight)                    ║")
    print(f"║  Engine Fusion:  {qm.get('engine_score',0):+.4f}  (35% weight)                    ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    macro_bias = quant.get("quant_macro", {}).get("macro_bias", "?")
    ry         = quant.get("quant_macro", {}).get("models", {}).get("real_yield", {})
    inf        = quant.get("quant_macro", {}).get("models", {}).get("inflation", {})
    dollar     = quant.get("quant_macro", {}).get("models", {}).get("dollar", {})

    print(f"║  Macro Regime:   {macro_bias:<20}                       ║")
    print(f"║  Real Yield:     {ry.get('current_ry', '?'):.2f}%  |  Regime: {ry.get('regime','?'):<12}         ║")
    print(f"║  BEI Inflation:  {inf.get('breakeven_10y','?'):.2f}%  |  CPI: {inf.get('cpi_yoy','?'):.2f}%              ║")
    print(f"║  Dollar:         {dollar.get('regime','?'):<20}                       ║")

    if entry:
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Entry:     ${entry:<10.2f}                                      ║")
        if sl: print(f"║  Stop Loss: ${sl:<10.2f}                                      ║")
        if tp: print(f"║  Take Profit:${tp:<10.2f}                                     ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    t_color = "✅" if tradeable else "🚫"
    print(f"║  Tradeable: {t_color} {'YES' if tradeable else 'NO — BLOCKED'}{'                                    '[:36]}  ║")

    if not tradeable:
        for b in verdict.get("hard_blocks", [])[:3]:
            print(f"║    🚫 {b[:56]:<56} ║")

    print(f"║  Elapsed: {verdict.get('elapsed','?')}s                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    result = run(verbose=True)
    print(f"\nSaved to data/master_signal.json")
