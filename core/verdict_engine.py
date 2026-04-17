"""
TITAN v2.0 - Verdict Engine
Bayesian fusion of all engine signals into a final trading verdict.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import RISK, SESSION_TIMES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Bayesian Log-Odds Fusion
# ─────────────────────────────────────────────

def log_odds_fusion(engine_results: Dict[str, Dict], weights: Dict[str, float]) -> Dict:
    """
    Proper log-odds Bayesian update:
    For each engine, compute likelihood ratio and update posterior.
    Prior: 50% BUY, 50% SELL (neutral prior).
    Returns {log_odds_buy, log_odds_sell, posterior_buy, signal}
    """
    # Prior log-odds = 0 (50/50)
    log_odds = 0.0

    signal_contributions = []

    for engine_id, result in engine_results.items():
        signal     = result.get("signal", "NEUTRAL")
        confidence = result.get("confidence", 0.5)
        weight     = weights.get(engine_id, 0.05)

        if signal == "NEUTRAL" or confidence <= 0.5:
            continue

        # Likelihood ratio: P(evidence | BUY) / P(evidence | SELL)
        # If signal is BUY with confidence c, LR_buy = c / (1-c)
        # If signal is SELL with confidence c, LR_sell = (1-c) / c
        c = float(np.clip(confidence, 0.501, 0.999))

        if signal == "BUY":
            lr = c / (1 - c)
            contribution = weight * math.log(lr)
        elif signal == "SELL":
            lr = (1 - c) / c
            contribution = weight * math.log(lr)  # negative for SELL
        else:
            contribution = 0.0

        log_odds += contribution
        signal_contributions.append({
            "engine": engine_id,
            "signal": signal,
            "confidence": float(confidence),
            "weight": float(weight),
            "contribution": float(contribution),
        })

    # Convert log-odds to probability
    # P(BUY) = sigmoid(log_odds)
    posterior_buy = float(1 / (1 + math.exp(-log_odds)))
    posterior_sell = 1 - posterior_buy

    if posterior_buy > 0.5:
        signal    = "BUY"
        raw_score = posterior_buy
    elif posterior_sell > 0.5:
        signal    = "SELL"
        raw_score = posterior_sell
    else:
        signal    = "NEUTRAL"
        raw_score = 0.5

    return {
        "signal":              signal,
        "raw_score":           float(raw_score),
        "log_odds":            float(log_odds),
        "posterior_buy":       float(posterior_buy),
        "posterior_sell":      float(posterior_sell),
        "contributions":       sorted(signal_contributions, key=lambda x: abs(x["contribution"]), reverse=True),
    }


# ─────────────────────────────────────────────
# Confidence Calibration
# ─────────────────────────────────────────────

def calibrate_confidence(raw_score: float) -> float:
    """
    Sigmoid-based confidence calibration.
    Maps raw Bayesian posterior [0.5, 1.0] to calibrated confidence.
    Applies conservative compression (never extreme).
    """
    # Scale to [-3, 3] for sigmoid
    scaled = (raw_score - 0.5) * 6
    calibrated = 1 / (1 + math.exp(-scaled))

    # Compress to [0.50, 0.92] range
    return float(0.50 + (calibrated - 0.5) * 0.84)


# ─────────────────────────────────────────────
# Agreement Bonus
# ─────────────────────────────────────────────

def agreement_bonus(engine_results: Dict[str, Dict]) -> float:
    """
    If 6+ engines agree on the same direction, add +0.05 confidence bonus.
    Returns bonus to add.
    """
    signals = [r.get("signal", "NEUTRAL") for r in engine_results.values()
               if r.get("signal") != "NEUTRAL"]

    if not signals:
        return 0.0

    buy_count  = signals.count("BUY")
    sell_count = signals.count("SELL")
    max_agree  = max(buy_count, sell_count)

    if max_agree >= 6:
        return 0.05
    elif max_agree >= 5:
        return 0.03
    elif max_agree >= 4:
        return 0.01
    return 0.0


# ─────────────────────────────────────────────
# RiskFortress
# ─────────────────────────────────────────────

class RiskFortress:
    """
    Hard and soft risk controls for trade execution.
    10 hard blocks (prevent trade), 5 soft warnings (reduce size/confidence).
    """

    def __init__(self, config: Dict = None):
        self.config = config or RISK

    def check_all(self, context: Dict, proposed_confidence: float) -> Dict:
        """
        Run all risk checks.
        Returns {blocked: bool, reason: str, warnings: list, adjusted_confidence: float}
        """
        hard_blocks = self._run_hard_blocks(context, proposed_confidence)
        if hard_blocks:
            return {
                "blocked":              True,
                "reason":               hard_blocks[0],
                "all_blocks":           hard_blocks,
                "warnings":             [],
                "adjusted_confidence":  0.0,
            }

        warnings = self._run_soft_warnings(context, proposed_confidence)
        adjusted = proposed_confidence
        for w in warnings:
            adjusted -= w.get("confidence_reduction", 0.0)
        adjusted = float(max(0.0, adjusted))

        return {
            "blocked":              False,
            "reason":               None,
            "all_blocks":           [],
            "warnings":             warnings,
            "adjusted_confidence":  adjusted,
        }

    def _run_hard_blocks(self, ctx: Dict, confidence: float) -> List[str]:
        blocks = []

        # 1. Minimum confidence threshold
        if confidence < self.config.get("min_confidence", 0.58):
            blocks.append(f"CONFIDENCE_TOO_LOW: {confidence:.3f} < {self.config['min_confidence']}")

        # 2. Max daily loss
        daily_pnl = ctx.get("daily_pnl_pct", 0.0)
        max_loss  = self.config.get("max_daily_loss_pct", 2.0)
        if daily_pnl <= -max_loss:
            blocks.append(f"MAX_DAILY_LOSS_HIT: {daily_pnl:.2f}% <= -{max_loss}%")

        # 3. Max trades per day
        trades_today = ctx.get("trades_today", 0)
        max_trades   = self.config.get("max_trades_per_day", 5)
        if trades_today >= max_trades:
            blocks.append(f"MAX_TRADES_REACHED: {trades_today} >= {max_trades}")

        # 4. Max consecutive losses
        consec_losses = ctx.get("consecutive_losses", 0)
        max_consec    = self.config.get("max_consecutive_losses", 3)
        if consec_losses >= max_consec:
            blocks.append(f"MAX_CONSECUTIVE_LOSSES: {consec_losses} >= {max_consec}")

        # 5. News blackout (pre-news)
        mins_to_news = ctx.get("minutes_to_next_high_impact", 9999)
        blackout_pre = self.config.get("blackout_pre_news", 15)
        if mins_to_news is not None and mins_to_news <= blackout_pre:
            blocks.append(f"PRE_NEWS_BLACKOUT: {mins_to_news} min to high-impact event")

        # 6. News blackout (post-news)
        mins_since_news = ctx.get("minutes_since_last_news", 9999)
        blackout_post   = self.config.get("blackout_post_news", 10)
        if mins_since_news is not None and mins_since_news <= blackout_post:
            blocks.append(f"POST_NEWS_BLACKOUT: {mins_since_news} min since high-impact event")

        # 7. Spread too wide
        current_spread = ctx.get("spread_pips", 0.0)
        max_spread     = self.config.get("max_spread_pips", 3.0)
        if current_spread > max_spread:
            blocks.append(f"SPREAD_TOO_WIDE: {current_spread:.2f} > {max_spread}")

        # 8. Minimum R:R ratio
        proposed_rr = ctx.get("proposed_rr", 9999)
        min_rr = self.config.get("min_rr_ratio", 1.5)
        if proposed_rr is not None and proposed_rr < min_rr:
            blocks.append(f"RR_TOO_LOW: {proposed_rr:.2f} < {min_rr}")

        # 9. NEUTRAL signal
        signal = ctx.get("signal", "NEUTRAL")
        if signal == "NEUTRAL":
            blocks.append("SIGNAL_IS_NEUTRAL: No directional conviction")

        # 10. Missing critical data
        if ctx.get("data_missing", False):
            blocks.append("CRITICAL_DATA_MISSING: Cannot trade without complete data")

        return blocks

    def _run_soft_warnings(self, ctx: Dict, confidence: float) -> List[Dict]:
        warnings = []

        # W1. High volatility regime
        if ctx.get("vol_regime") == "HIGH_VOL":
            warnings.append({
                "type":                 "HIGH_VOLATILITY",
                "message":              "High volatility regime — reduce position size by 50%",
                "confidence_reduction": 0.03,
            })

        # W2. Low liquidity session (dead zone)
        current_session = ctx.get("current_session", "")
        if current_session == "DEAD_ZONE":
            warnings.append({
                "type":                 "LOW_LIQUIDITY_SESSION",
                "message":              "Dead zone (22:00-08:00 UTC) — wider spreads, low volume",
                "confidence_reduction": 0.05,
            })

        # W3. Not in kill zone
        in_killzone = ctx.get("in_killzone", False)
        if not in_killzone:
            warnings.append({
                "type":                 "NOT_IN_KILLZONE",
                "message":              "Not in ICT kill zone — timing suboptimal",
                "confidence_reduction": 0.02,
            })

        # W4. Weak confluence
        confluence_score = ctx.get("confluence_score", 1.0)
        if confluence_score < 0.5:
            warnings.append({
                "type":                 "WEAK_CONFLUENCE",
                "message":              f"Low timeframe confluence: {confluence_score:.2f}",
                "confidence_reduction": 0.03,
            })

        # W5. Approaching daily loss limit
        daily_pnl = ctx.get("daily_pnl_pct", 0.0)
        max_loss  = self.config.get("max_daily_loss_pct", 2.0)
        if daily_pnl <= -(max_loss * 0.7):
            warnings.append({
                "type":                 "APPROACHING_DAILY_LIMIT",
                "message":              f"Daily PnL {daily_pnl:.2f}% approaching limit",
                "confidence_reduction": 0.04,
            })

        return warnings


# ─────────────────────────────────────────────
# Kelly Criterion
# ─────────────────────────────────────────────

def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                    half_kelly: bool = True) -> float:
    """
    Kelly Criterion: f* = (W * b - L) / b
    Where b = avg_win/avg_loss (reward ratio).
    half_kelly: use 50% of Kelly fraction for conservative sizing.
    Returns Kelly fraction (0.0-1.0).
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.01

    b = float(avg_win / avg_loss)  # reward ratio

    if b <= 0:
        return 0.01

    kelly_f = (win_rate * b - (1 - win_rate)) / b
    kelly_f = max(0.0, min(kelly_f, 0.25))  # cap at 25%

    if half_kelly:
        kelly_f *= 0.5

    return float(kelly_f)


# ─────────────────────────────────────────────
# Position Sizing
# ─────────────────────────────────────────────

def compute_position_size(capital: float, kelly_f: float,
                           entry: float, sl: float,
                           max_position_pct: float = None) -> Dict:
    """
    Compute position size in lots/units based on Kelly fraction and risk.
    Returns {risk_amount, position_value, lots_approx, risk_pct}
    """
    if max_position_pct is None:
        max_position_pct = RISK.get("max_position_pct", 1.0) / 100

    # Risk per trade
    risk_pct    = min(kelly_f, max_position_pct)
    risk_amount = capital * risk_pct

    # Stop distance in price units
    sl_distance = abs(entry - sl)
    if sl_distance < 1e-9:
        sl_distance = entry * 0.005  # 0.5% fallback

    # Position value
    position_value = risk_amount / sl_distance * entry

    # Gold: 1 lot = 100 troy oz
    # Approximate lots
    gold_per_lot = 100  # troy oz per standard lot
    lots_approx  = position_value / (entry * gold_per_lot) if entry > 0 else 0

    return {
        "risk_pct":       float(risk_pct * 100),
        "risk_amount":    float(risk_amount),
        "position_value": float(position_value),
        "lots_approx":    float(lots_approx),
        "sl_distance":    float(sl_distance),
    }


# ─────────────────────────────────────────────
# SL/TP Computation
# ─────────────────────────────────────────────

def compute_sl_tp(entry: float, direction: str, atr: float,
                   rr: float = 1.5, session_adjusted: bool = True,
                   current_session: str = "LONDON") -> Dict:
    """
    Compute stop-loss and take-profit levels.
    SL = 1.5x ATR from entry (session-adjusted).
    TP = SL distance * rr.
    """
    if atr <= 0:
        atr = entry * 0.005  # 0.5% fallback

    # Session multiplier
    session_mult = 1.0
    if session_adjusted:
        session_mult = {
            "ASIA":      0.8,   # tighter ranges
            "LONDON":    1.0,
            "NEW_YORK":  1.1,   # wider ranges
            "DEAD_ZONE": 1.3,   # very wide (low liquidity)
        }.get(current_session, 1.0)

    sl_distance = atr * 1.5 * session_mult
    tp_distance = sl_distance * rr

    if direction == "BUY" or direction == "LONG":
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:  # SELL / SHORT
        sl = entry + sl_distance
        tp = entry - tp_distance

    return {
        "entry":       float(entry),
        "sl":          float(sl),
        "tp":          float(tp),
        "sl_distance": float(sl_distance),
        "tp_distance": float(tp_distance),
        "rr_ratio":    float(rr),
        "direction":   direction,
    }


# ─────────────────────────────────────────────
# Full Verdict
# ─────────────────────────────────────────────

def compute_verdict(engine_results: Dict[str, Dict],
                     weights: Dict[str, float],
                     context: Dict) -> Dict:
    """
    Compute final trading verdict from all engine results.
    Returns comprehensive verdict dict.
    """
    try:
        # Bayesian fusion
        fusion = log_odds_fusion(engine_results, weights)
        signal = fusion["signal"]
        raw    = fusion["raw_score"]

        # Base confidence from calibration
        confidence = calibrate_confidence(raw)

        # Agreement bonus
        bonus      = agreement_bonus(engine_results)
        confidence = min(confidence + bonus, 0.95)

        # Confidence modifier from memory engine
        conf_modifier = 0.0
        for eng_id, res in engine_results.items():
            if "memory" in eng_id or "engine8" in eng_id:
                conf_modifier = float(res.get("confidence_modifier", 0.0))
                break
        confidence = float(min(0.95, max(0.0, confidence + conf_modifier)))

        # Risk checks
        context_full = {
            "signal":     signal,
            "confidence": confidence,
            **context,
        }

        risk_fortress = RiskFortress()
        risk_result   = risk_fortress.check_all(context_full, confidence)

        if risk_result["blocked"]:
            return {
                "signal":     "BLOCKED",
                "direction":  signal,
                "confidence": 0.0,
                "blocked":    True,
                "block_reason": risk_result["reason"],
                "all_blocks": risk_result["all_blocks"],
                "warnings":   [],
                "fusion":     fusion,
                "ts":         datetime.utcnow().isoformat(),
            }

        adjusted_confidence = risk_result["adjusted_confidence"]

        # Compute trade levels
        entry     = float(context.get("current_price", 0.0))
        atr       = float(context.get("atr", entry * 0.005 if entry > 0 else 10.0))
        session   = str(context.get("current_session", "LONDON"))
        rr        = float(RISK.get("min_rr_ratio", 1.5))

        levels = compute_sl_tp(entry, signal, atr, rr=rr, session_adjusted=True,
                                current_session=session)

        # Position sizing (assume 10k capital if not provided)
        capital = float(context.get("account_capital", 10000.0))
        trades_df_metrics = context.get("performance_metrics", {})
        win_rate = float(trades_df_metrics.get("win_rate", 0.5))
        avg_win  = float(trades_df_metrics.get("avg_win", 1.5))
        avg_loss = float(trades_df_metrics.get("avg_loss", 1.0))

        kelly_f  = kelly_criterion(win_rate, avg_win, avg_loss,
                                    half_kelly=RISK.get("half_kelly", True))
        position = compute_position_size(capital, kelly_f, entry, levels["sl"])

        # RR check
        if levels.get("rr_ratio", 0) < RISK.get("min_rr_ratio", 1.5):
            logger.warning("RR %.2f below minimum %.2f", levels["rr_ratio"], RISK["min_rr_ratio"])

        return {
            "signal":             signal,
            "direction":          signal,
            "confidence":         float(adjusted_confidence),
            "raw_confidence":     float(confidence),
            "blocked":            False,
            "block_reason":       None,
            "warnings":           risk_result["warnings"],
            "entry":              float(entry),
            "sl":                 float(levels["sl"]),
            "tp":                 float(levels["tp"]),
            "sl_distance":        float(levels["sl_distance"]),
            "tp_distance":        float(levels["tp_distance"]),
            "rr_ratio":           float(rr),
            "position_size":      position,
            "kelly_f":            float(kelly_f),
            "fusion":             fusion,
            "agreement_bonus":    float(bonus),
            "conf_modifier":      float(conf_modifier),
            "atr":                float(atr),
            "session":            session,
            "engine_count":       len(engine_results),
            "ts":                 datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("compute_verdict error: %s", e, exc_info=True)
        return {
            "signal":     "BLOCKED",
            "direction":  "NEUTRAL",
            "confidence": 0.0,
            "blocked":    True,
            "block_reason": f"VERDICT_ERROR: {str(e)}",
            "ts":         datetime.utcnow().isoformat(),
        }
