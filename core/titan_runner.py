# FILE: core/titan_runner.py
"""
TITAN v2.0 — Main Orchestrator
Runs all 9 engines concurrently, fuses verdicts, logs everything
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict

from core.config import ENGINE_WEIGHTS, VERDICT_PATH
from core.logger import get_logger
from core.db import init_db, log_snapshot
from core.verdict_engine import compute_verdict
from core.economic_calendar import run as get_calendar, is_blackout_active, minutes_to_next_event

log = get_logger("TitanRunner")

ENGINE_TIMEOUT = 35  # seconds per engine
_last_run_ts   = 0
_last_verdict  = None

BANNER = """
╔══════════════════════════════════════════════════════════╗
║  ████████╗██╗████████╗ █████╗ ███╗   ██╗                ║
║     ██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║                ║
║     ██║   ██║   ██║   ███████║██╔██╗ ██║                ║
║     ██║   ██║   ██║   ██╔══██║██║╚██╗██║                ║
║     ██║   ██║   ██║   ██║  ██║██║ ╚████║                ║
║     ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝                ║
║         v2.0 — 9-Engine Gold Intelligence                ║
╚══════════════════════════════════════════════════════════╝
"""


def _import_engines():
    from engines.engine1_price_matrix     import run as e1
    from engines.engine2_sentiment_fusion import run as e2
    from engines.engine3_volume_cot       import run as e3
    from engines.engine4_macro_correlation import run as e4
    from engines.engine5_liquidity_hunt   import run as e5
    from engines.engine6_regime_detection import run as e6
    from engines.engine7_adversarial_trap import run as e7
    from engines.engine8_memory_learning  import run as e8
    return {"engine1_price_matrix": e1, "engine2_sentiment_fusion": e2,
            "engine3_volume_cot": e3, "engine4_macro_correlation": e4,
            "engine5_liquidity_hunt": e5, "engine6_regime_detection": e6,
            "engine7_adversarial_trap": e7, "engine8_memory_learning": e8}


def run_all_engines() -> Dict:
    """Run engines 1-8 concurrently, return results dict"""
    engine_fns = _import_engines()
    results    = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fn): name for name, fn in engine_fns.items()}
        for future in as_completed(futures, timeout=ENGINE_TIMEOUT * 2):
            name = futures[future]
            try:
                result = future.result(timeout=ENGINE_TIMEOUT)
                results[name] = result
                sig  = result.get("signal", "?")
                conf = result.get("confidence", 0)
                log.info(f"  ✓ {name:35s} → {sig:7s} @ {conf:.1%}")
            except Exception as ex:
                log.warning(f"  ✗ {name} timed out / failed: {ex}")
                results[name] = {"signal": "ERROR", "confidence": 0.0, "error": str(ex)}

    return results


def titan_analyze(verbose: bool = True, force: bool = False) -> Dict:
    """Full TITAN analysis — all 9 engines → verdict"""
    global _last_run_ts, _last_verdict

    # Cache: skip if run < 5 min ago and not forced
    from core.config import CACHE_TTL_MINUTES
    if not force and _last_verdict and (time.time() - _last_run_ts) < CACHE_TTL_MINUTES * 60:
        log.info(f"Returning cached verdict (age: {int(time.time()-_last_run_ts)}s)")
        return _last_verdict

    if verbose:
        print(BANNER)

    start = time.time()
    init_db()
    log.info("═" * 60)
    log.info("TITAN v2.0 ANALYSIS STARTED")
    log.info("═" * 60)

    # Run engines 1-8
    engine_results = run_all_engines()

    # Extract regime from E6
    e6 = engine_results.get("engine6_regime_detection", {})
    current_regime  = e6.get("current_regime", "RANGING")
    vol_regime      = e6.get("volatility", {}).get("regime", "NORMAL_VOLATILITY")

    # Run E9 with regime context
    from engines.engine9_meta_learning import run as e9
    meta = e9(engine_results=engine_results, current_regime=current_regime)
    engine_results["engine9_meta_learning"] = meta
    weights = meta.get("final_weights", ENGINE_WEIGHTS)

    # Build context from E1 + E6 + calendar
    e1 = engine_results.get("engine1_price_matrix", {})
    calendar = get_calendar()
    blackout = is_blackout_active()
    mins_to_news = minutes_to_next_event()

    context = {
        "current_price":       e1.get("current_price"),
        "atr":                 e1.get("atr"),
        "session":             e1.get("session", "UNKNOWN"),
        "regime":              current_regime,
        "volatility_regime":   vol_regime,
        "mins_to_next_event":  mins_to_news,
        "blackout_active":     blackout["active"],
        "calendar":            calendar,
    }

    # Add ATR average for flash crash detection (use 20-day ATR proxy)
    if context.get("atr"):
        context["atr_avg"] = context["atr"]  # Simplified; would improve with historical avg

    # Compute final verdict
    verdict = compute_verdict(engine_results, weights, context)
    verdict["engine_results"]  = engine_results
    verdict["weights"]         = weights
    verdict["elapsed_seconds"] = round(time.time() - start, 2)
    verdict["analyzed_at"]     = datetime.now(timezone.utc).isoformat()

    # Log market snapshot
    if context.get("current_price") and context.get("atr"):
        log_snapshot(
            price=context["current_price"], atr=context["atr"],
            session=context["session"],     regime=current_regime,
            details={"vol_regime": vol_regime},
        )

    # Save to disk
    os.makedirs("data", exist_ok=True)
    try:
        with open(VERDICT_PATH, "w") as f:
            json.dump(verdict, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"Could not save verdict: {e}")

    # Update cache
    _last_run_ts  = time.time()
    _last_verdict = verdict

    if verbose:
        _print_verdict(verdict)

    return verdict


def _print_verdict(v: Dict):
    direction  = v.get("direction", "?")
    confidence = v.get("confidence", 0)
    tradeable  = v.get("tradeable", False)
    entry      = v.get("entry")
    sl         = v.get("sl")
    tp         = v.get("tp")

    emoji = {"BUY": "📈", "SELL": "📉", "NEUTRAL": "➡️"}.get(direction, "?")
    tcolor = "✅" if tradeable else "🚫"

    print("\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  {emoji}  TITAN VERDICT                                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Direction:   {direction:<10}                                 ║")
    print(f"║  Confidence:  {confidence:.1%:<10}                                 ║")
    print(f"║  Tradeable:   {tcolor}  {'YES' if tradeable else 'NO':<8}                                 ║")
    if entry:
        print(f"║  Entry:       {entry:<10.2f}                                 ║")
    if sl:
        print(f"║  Stop Loss:   {sl:<10.2f}                                 ║")
    if tp:
        print(f"║  Take Profit: {tp:<10.2f}                                 ║")
    if not tradeable:
        print("╠══════════════════════════════════════════════════════════╣")
        print("║  BLOCKS:                                                 ║")
        for b in v.get("hard_blocks", []):
            print(f"║    🚫 {b[:52]:<52} ║")
    if v.get("soft_warnings"):
        print("╠══════════════════════════════════════════════════════════╣")
        print("║  WARNINGS:                                               ║")
        for w in v.get("soft_warnings", []):
            print(f"║    ⚠️  {w[:51]:<51} ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    vb = v.get("vote_breakdown", {})
    print(f"║  Votes: BUY={vb.get('BUY',0):.3f}  SELL={vb.get('SELL',0):.3f}  NEU={vb.get('NEUTRAL',0):.3f}        ║")
    print(f"║  Elapsed: {v.get('elapsed_seconds','?')}s | Regime: {v.get('regime','?'):<16}         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    result = titan_analyze(verbose=True)
    print(f"Saved to {VERDICT_PATH}")
