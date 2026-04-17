"""
TITAN v2.0 - Engine 9: Meta-Learning
Dynamic weight optimization, regime-based weighting, trust scoring.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import ENGINE_WEIGHTS
from core.db import get_signals_df, get_trades_df

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Regime-Based Weights
# ─────────────────────────────────────────────

# How much to weight each engine in different regimes
REGIME_WEIGHT_PROFILES = {
    "TRENDING_STRONG": {
        "engine1_price_matrix":      0.25,
        "engine2_sentiment_fusion":  0.07,
        "engine3_volume_cot":        0.10,
        "engine4_macro_correlation": 0.12,
        "engine5_liquidity_hunt":    0.18,
        "engine6_regime_detection":  0.12,
        "engine7_adversarial_trap":  0.08,
        "engine8_memory_learning":   0.05,
        "engine9_meta_learning":     0.03,
    },
    "TRENDING_WEAK": {
        "engine1_price_matrix":      0.22,
        "engine2_sentiment_fusion":  0.09,
        "engine3_volume_cot":        0.11,
        "engine4_macro_correlation": 0.13,
        "engine5_liquidity_hunt":    0.15,
        "engine6_regime_detection":  0.12,
        "engine7_adversarial_trap":  0.10,
        "engine8_memory_learning":   0.05,
        "engine9_meta_learning":     0.03,
    },
    "RANGING_ACCUMULATION": {
        "engine1_price_matrix":      0.18,
        "engine2_sentiment_fusion":  0.10,
        "engine3_volume_cot":        0.15,
        "engine4_macro_correlation": 0.12,
        "engine5_liquidity_hunt":    0.18,
        "engine6_regime_detection":  0.10,
        "engine7_adversarial_trap":  0.09,
        "engine8_memory_learning":   0.05,
        "engine9_meta_learning":     0.03,
    },
    "RANGING_DISTRIBUTION": {
        "engine1_price_matrix":      0.18,
        "engine2_sentiment_fusion":  0.08,
        "engine3_volume_cot":        0.14,
        "engine4_macro_correlation": 0.13,
        "engine5_liquidity_hunt":    0.15,
        "engine6_regime_detection":  0.12,
        "engine7_adversarial_trap":  0.12,
        "engine8_memory_learning":   0.05,
        "engine9_meta_learning":     0.03,
    },
    "MEAN_REVERTING": {
        "engine1_price_matrix":      0.15,
        "engine2_sentiment_fusion":  0.09,
        "engine3_volume_cot":        0.12,
        "engine4_macro_correlation": 0.11,
        "engine5_liquidity_hunt":    0.20,
        "engine6_regime_detection":  0.13,
        "engine7_adversarial_trap":  0.12,
        "engine8_memory_learning":   0.05,
        "engine9_meta_learning":     0.03,
    },
    "HIGH_VOL": {
        "engine1_price_matrix":      0.16,
        "engine2_sentiment_fusion":  0.12,
        "engine3_volume_cot":        0.10,
        "engine4_macro_correlation": 0.15,
        "engine5_liquidity_hunt":    0.13,
        "engine6_regime_detection":  0.14,
        "engine7_adversarial_trap":  0.12,
        "engine8_memory_learning":   0.05,
        "engine9_meta_learning":     0.03,
    },
}

DEFAULT_WEIGHTS = ENGINE_WEIGHTS.copy()


def compute_regime_weights(regime: str) -> Dict[str, float]:
    """
    Get engine weights for a given market regime.
    Falls back to default weights if regime unknown.
    """
    profile = REGIME_WEIGHT_PROFILES.get(regime, DEFAULT_WEIGHTS)

    # Normalize to ensure sum = 1.0
    total = sum(profile.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()

    return {k: float(v / total) for k, v in profile.items()}


# ─────────────────────────────────────────────
# Performance-Based Weights
# ─────────────────────────────────────────────

def performance_based_weights(accuracy_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Compute weights based on each engine's historical accuracy.
    Higher accuracy = higher weight.
    Returns normalized weights dict.
    """
    if not accuracy_dict:
        return DEFAULT_WEIGHTS.copy()

    # Start from default and adjust based on accuracy
    weights = DEFAULT_WEIGHTS.copy()

    for engine_id, accuracy in accuracy_dict.items():
        if engine_id in weights:
            # Accuracy of 0.5 = no change, >0.5 = boost, <0.5 = reduce
            adjustment = (accuracy - 0.5) * 0.5  # max +/-25% adjustment
            weights[engine_id] = max(0.01, weights[engine_id] * (1 + adjustment))

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: float(v / total) for k, v in weights.items()}

    return weights


# ─────────────────────────────────────────────
# Weight Blending
# ─────────────────────────────────────────────

def blend_weights(regime_w: Dict[str, float], perf_w: Dict[str, float],
                  alpha: float = 0.6) -> Dict[str, float]:
    """
    Blend regime weights with performance weights.
    alpha: weight given to regime weights (0.0-1.0).
    Returns blended, normalized weights.
    """
    blended = {}
    all_engines = set(regime_w.keys()) | set(perf_w.keys())

    for engine in all_engines:
        rw = regime_w.get(engine, DEFAULT_WEIGHTS.get(engine, 0.05))
        pw = perf_w.get(engine, DEFAULT_WEIGHTS.get(engine, 0.05))
        blended[engine] = alpha * rw + (1 - alpha) * pw

    # Normalize
    total = sum(blended.values())
    if total > 0:
        blended = {k: float(v / total) for k, v in blended.items()}

    return blended


# ─────────────────────────────────────────────
# Trust Scoring
# ─────────────────────────────────────────────

def compute_trust_scores(engine_results: Dict[str, Dict],
                          accuracy_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Compute trust score for each engine (0.0-1.0).
    Based on: historical accuracy, signal confidence, absence of errors.
    """
    trust_scores = {}

    for engine_id, result in engine_results.items():
        base_trust = 0.7  # start with moderate trust

        # Penalize if engine errored
        if result.get("error"):
            base_trust -= 0.4

        # Adjust by historical accuracy
        acc = accuracy_dict.get(engine_id, 0.5)
        acc_adjustment = (acc - 0.5) * 0.4  # +/-20% adjustment
        base_trust += acc_adjustment

        # Adjust by signal confidence
        conf = result.get("confidence", 0.5)
        if conf >= 0.75:
            base_trust += 0.05
        elif conf < 0.55:
            base_trust -= 0.05

        # NEUTRAL signals slightly less trusted (may be noise)
        if result.get("signal") == "NEUTRAL":
            base_trust -= 0.05

        trust_scores[engine_id] = float(max(0.0, min(1.0, base_trust)))

    return trust_scores


# ─────────────────────────────────────────────
# Optuna Weight Optimization
# ─────────────────────────────────────────────

def optuna_optimize_weights(engine_results: Dict[str, Dict],
                              trades_df=None,
                              n_trials: int = 100) -> Dict[str, float]:
    """
    Use Optuna to optimize engine weights by minimizing (1 - win_rate)
    over last 50 trades.
    Falls back gracefully if Optuna unavailable.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if trades_df is None or len(trades_df) < 10:
            logger.debug("Not enough trades for Optuna optimization")
            return DEFAULT_WEIGHTS.copy()

        recent_trades = trades_df.tail(50)
        engine_names  = list(ENGINE_WEIGHTS.keys())

        def objective(trial):
            # Sample weights
            raw_weights = {}
            for eng in engine_names:
                raw_weights[eng] = trial.suggest_float(eng, 0.02, 0.35)

            # Normalize
            total = sum(raw_weights.values())
            weights = {k: v / total for k, v in raw_weights.items()}

            # Simulate "win rate" based on signal alignment
            wins = 0
            total_count = 0
            for _, trade in recent_trades.iterrows():
                if trade.get("outcome") not in ["WIN", "LOSS"]:
                    continue

                # Weighted vote
                bull = 0.0
                bear = 0.0
                for eng, w in weights.items():
                    eng_result = engine_results.get(eng, {})
                    sig = eng_result.get("signal", "NEUTRAL")
                    if sig == "BUY":
                        bull += w
                    elif sig == "SELL":
                        bear += w

                pred_dir = "LONG" if bull >= bear else "SHORT"
                actual_dir = trade.get("direction", "")
                actual_out = trade.get("outcome", "LOSS")

                if pred_dir == actual_dir and actual_out == "WIN":
                    wins += 1
                total_count += 1

            win_rate = wins / (total_count + 1e-9)
            return 1.0 - win_rate  # minimize loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params  = study.best_params
        total_weight = sum(best_params.values())
        optimized    = {k: float(v / total_weight) for k, v in best_params.items()}

        logger.info("Optuna optimization complete. Best win rate: %.2f%%",
                    (1 - study.best_value) * 100)
        return optimized

    except ImportError:
        logger.debug("Optuna not available, using default weights")
        return DEFAULT_WEIGHTS.copy()
    except Exception as e:
        logger.warning("Optuna optimization failed: %s", e)
        return DEFAULT_WEIGHTS.copy()


# ─────────────────────────────────────────────
# EWMA Weight Smoothing
# ─────────────────────────────────────────────

def ewma_smooth_weights(new_weights: Dict[str, float],
                         old_weights: Dict[str, float],
                         alpha: float = 0.3) -> Dict[str, float]:
    """
    Smooth weight transitions using EWMA.
    new_weight_smoothed = alpha * new + (1-alpha) * old
    Prevents sudden large shifts in weights.
    """
    smoothed = {}
    all_engines = set(new_weights.keys()) | set(old_weights.keys())

    for eng in all_engines:
        n = new_weights.get(eng, DEFAULT_WEIGHTS.get(eng, 0.05))
        o = old_weights.get(eng, DEFAULT_WEIGHTS.get(eng, 0.05))
        smoothed[eng] = float(alpha * n + (1 - alpha) * o)

    # Normalize
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {k: float(v / total) for k, v in smoothed.items()}

    return smoothed


# ─────────────────────────────────────────────
# Main Run
# ─────────────────────────────────────────────

def run(engine_results: Dict[str, Dict], current_regime: str = "NEUTRAL_RANGING") -> Dict:
    """
    Meta-learning engine: compute optimal weights and trust scores.
    Returns {weights, trust_scores, regime_weights, performance_weights, ...}
    """
    try:
        # Get historical performance
        from core.db import get_engine_accuracy
        accuracy_dict = get_engine_accuracy()

        trades_df  = get_trades_df(limit=100)
        signals_df = get_signals_df(limit=500)

        # Compute weights from different sources
        regime_w = compute_regime_weights(current_regime)
        perf_w   = performance_based_weights(accuracy_dict)
        blended  = blend_weights(regime_w, perf_w, alpha=0.6)

        # Optuna refinement (if enough data)
        optuna_w = optuna_optimize_weights(engine_results, trades_df, n_trials=50)

        # Final blend: 70% blended + 30% optuna
        final_w = blend_weights(blended, optuna_w, alpha=0.7)

        # EWMA smoothing against defaults
        final_w = ewma_smooth_weights(final_w, DEFAULT_WEIGHTS, alpha=0.4)

        # Trust scores
        trust = compute_trust_scores(engine_results, accuracy_dict)

        # Apply trust scores to final weights
        trust_adjusted = {}
        for eng, w in final_w.items():
            t = trust.get(eng, 0.7)
            trust_adjusted[eng] = float(w * t)

        # Normalize
        total = sum(trust_adjusted.values())
        if total > 0:
            trust_adjusted = {k: float(v / total) for k, v in trust_adjusted.items()}

        return {
            "engine":             "engine9_meta_learning",
            "signal":             "NEUTRAL",   # meta engine doesn't give directional signal
            "confidence":         0.7,
            "final_weights":      trust_adjusted,
            "regime_weights":     regime_w,
            "performance_weights":perf_w,
            "trust_scores":       trust,
            "current_regime":     current_regime,
            "engine_accuracy":    accuracy_dict,
            "ts":                 datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("Engine9 run error: %s", e, exc_info=True)
        return {
            "engine":       "engine9_meta_learning",
            "signal":       "NEUTRAL",
            "confidence":   0.5,
            "final_weights":DEFAULT_WEIGHTS.copy(),
            "trust_scores": {},
            "error":        str(e),
        }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    # Test with mock engine results
    mock_results = {k: {"signal": "BUY", "confidence": 0.65} for k in ENGINE_WEIGHTS}
    result = run(mock_results, "TRENDING_STRONG")
    print(json.dumps(result, indent=2, default=str))
