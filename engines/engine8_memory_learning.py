"""
TITAN v2.0 - Engine 8: Memory & Learning
Historical performance analysis, confidence calibration, and adaptive adjustments.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import SESSION_TIMES
from core.db import get_signals_df, get_trades_df

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core Performance Metrics
# ─────────────────────────────────────────────

def compute_all_metrics(trades_df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive performance metrics from trades DataFrame.
    Returns {win_rate, avg_rr, profit_factor, expectancy, sharpe,
             calmar, max_dd, total_trades, ...}
    """
    if trades_df.empty or "outcome" not in trades_df.columns:
        return _empty_metrics()

    df = trades_df.copy()

    # Filter completed trades
    if "pnl_pct" in df.columns:
        df = df[df["pnl_pct"].notna()]

    if df.empty:
        return _empty_metrics()

    total_trades = len(df)

    # Win rate
    wins   = df[df["outcome"] == "WIN"] if "outcome" in df.columns else pd.DataFrame()
    losses = df[df["outcome"] == "LOSS"] if "outcome" in df.columns else pd.DataFrame()
    bes    = df[df["outcome"] == "BE"] if "outcome" in df.columns else pd.DataFrame()

    win_rate   = len(wins) / total_trades if total_trades > 0 else 0.0
    loss_rate  = len(losses) / total_trades if total_trades > 0 else 0.0

    # Average R:R
    avg_win  = float(df.loc[df["outcome"] == "WIN", "rr_achieved"].mean()) if not wins.empty and "rr_achieved" in df.columns else 0.0
    avg_loss = float(df.loc[df["outcome"] == "LOSS", "rr_achieved"].abs().mean()) if not losses.empty and "rr_achieved" in df.columns else 1.0

    if avg_win == 0 or pd.isna(avg_win):
        # Fall back to pnl_pct
        if "pnl_pct" in df.columns:
            avg_win  = float(df.loc[df["pnl_pct"] > 0, "pnl_pct"].mean()) if len(df[df["pnl_pct"] > 0]) > 0 else 0.0
            avg_loss = float(df.loc[df["pnl_pct"] < 0, "pnl_pct"].abs().mean()) if len(df[df["pnl_pct"] < 0]) > 0 else 1.0
        avg_rr = avg_win / (avg_loss + 1e-9)
    else:
        avg_rr = avg_win

    # Profit factor
    gross_profit = float(df.loc[df.get("pnl_pct", pd.Series()) > 0, "pnl_pct"].sum()) if "pnl_pct" in df.columns else 0.0
    gross_loss   = abs(float(df.loc[df.get("pnl_pct", pd.Series()) < 0, "pnl_pct"].sum())) if "pnl_pct" in df.columns else 0.0
    profit_factor = gross_profit / (gross_loss + 1e-9)

    # Expectancy
    expectancy = win_rate * avg_win - loss_rate * avg_loss

    # Returns series
    returns = pd.Series(dtype=float)
    if "pnl_pct" in df.columns:
        returns = df["pnl_pct"].dropna()

    # Sharpe ratio (annualized, assuming ~250 trades/year)
    sharpe = 0.0
    if len(returns) > 5:
        mean_ret  = float(returns.mean())
        std_ret   = float(returns.std())
        sharpe    = (mean_ret / (std_ret + 1e-9)) * np.sqrt(250) if std_ret > 0 else 0.0

    # Max drawdown
    max_dd = 0.0
    if len(returns) > 0:
        equity = (1 + returns / 100).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / (rolling_max + 1e-9)
        max_dd = float(drawdown.min()) * 100  # as percentage

    # Calmar ratio
    ann_return  = float(returns.mean()) * 250 if len(returns) > 0 else 0.0
    calmar      = ann_return / (abs(max_dd) + 1e-9) if max_dd != 0 else 0.0

    # Consecutive losses
    max_consec_losses = 0
    current_streak    = 0
    if "outcome" in df.columns:
        for out in df["outcome"].values:
            if out == "LOSS":
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0

    return {
        "total_trades":        int(total_trades),
        "win_rate":            float(win_rate),
        "loss_rate":           float(loss_rate),
        "avg_rr":              float(avg_rr),
        "avg_win":             float(avg_win),
        "avg_loss":            float(avg_loss),
        "profit_factor":       float(profit_factor),
        "expectancy":          float(expectancy),
        "sharpe":              float(sharpe),
        "calmar":              float(calmar),
        "max_drawdown_pct":    float(max_dd),
        "max_consecutive_losses": int(max_consec_losses),
        "current_streak":      int(current_streak),
    }


def _empty_metrics() -> Dict:
    return {
        "total_trades": 0, "win_rate": 0.0, "loss_rate": 0.0,
        "avg_rr": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0, "expectancy": 0.0, "sharpe": 0.0,
        "calmar": 0.0, "max_drawdown_pct": 0.0,
        "max_consecutive_losses": 0, "current_streak": 0,
    }


# ─────────────────────────────────────────────
# Session Breakdown
# ─────────────────────────────────────────────

def session_breakdown(trades_df: pd.DataFrame) -> Dict:
    """
    Break down performance by trading session (ASIA/LONDON/NEW_YORK).
    Returns {session: {win_rate, total, avg_pnl}}
    """
    if trades_df.empty or "session" not in trades_df.columns:
        return {}

    result = {}
    for session in ["ASIA", "LONDON", "NEW_YORK", "DEAD_ZONE"]:
        sub = trades_df[trades_df["session"] == session]
        if sub.empty:
            continue

        total  = len(sub)
        wins   = len(sub[sub["outcome"] == "WIN"]) if "outcome" in sub.columns else 0
        wr     = wins / total
        avg_pnl = float(sub["pnl_pct"].mean()) if "pnl_pct" in sub.columns else 0.0

        result[session] = {
            "total":    total,
            "wins":     wins,
            "win_rate": float(wr),
            "avg_pnl":  avg_pnl,
        }

    return result


# ─────────────────────────────────────────────
# Regime Breakdown
# ─────────────────────────────────────────────

def regime_breakdown(trades_df: pd.DataFrame) -> Dict:
    """
    Break down performance by market regime.
    Returns {regime: {win_rate, total, avg_pnl}}
    """
    if trades_df.empty or "regime" not in trades_df.columns:
        return {}

    result = {}
    for regime in trades_df["regime"].dropna().unique():
        sub  = trades_df[trades_df["regime"] == regime]
        total  = len(sub)
        wins   = len(sub[sub["outcome"] == "WIN"]) if "outcome" in sub.columns else 0
        wr     = wins / total
        avg_pnl = float(sub["pnl_pct"].mean()) if "pnl_pct" in sub.columns else 0.0

        result[str(regime)] = {
            "total":    total,
            "wins":     wins,
            "win_rate": float(wr),
            "avg_pnl":  avg_pnl,
        }

    return result


# ─────────────────────────────────────────────
# Mistake Classifier
# ─────────────────────────────────────────────

def mistake_classifier(trades_df: pd.DataFrame) -> List[Dict]:
    """
    Classify common trading mistakes from the trade journal.
    Patterns: early exit, chased entry, ignored SL, low confidence wins,
              high confidence losses.
    """
    if trades_df.empty:
        return []

    mistakes = []

    # High confidence losses (confidence > 0.75 but LOSS)
    if "confidence" in trades_df.columns and "outcome" in trades_df.columns:
        high_conf_losses = trades_df[
            (trades_df["confidence"] > 0.75) & (trades_df["outcome"] == "LOSS")
        ]
        if len(high_conf_losses) > 0:
            mistakes.append({
                "type":       "HIGH_CONFIDENCE_LOSS",
                "count":      int(len(high_conf_losses)),
                "severity":   "HIGH",
                "suggestion": "Review entry timing and level selection — overconfidence detected",
            })

    # Low confidence wins (confidence < 0.60 but WIN)
    if "confidence" in trades_df.columns and "outcome" in trades_df.columns:
        low_conf_wins = trades_df[
            (trades_df["confidence"] < 0.60) & (trades_df["outcome"] == "WIN")
        ]
        if len(low_conf_wins) > 0:
            mistakes.append({
                "type":       "LOW_CONFIDENCE_WIN",
                "count":      int(len(low_conf_wins)),
                "severity":   "MEDIUM",
                "suggestion": "Some luck in low-confidence trades — do not size up on these",
            })

    # Poor RR achieved (rr < 0.5 on wins)
    if "rr_achieved" in trades_df.columns and "outcome" in trades_df.columns:
        early_exits = trades_df[
            (trades_df["outcome"] == "WIN") & (trades_df["rr_achieved"] < 0.8)
        ]
        if len(early_exits) > 0:
            mistakes.append({
                "type":       "EARLY_EXIT",
                "count":      int(len(early_exits)),
                "severity":   "MEDIUM",
                "suggestion": "Exiting too early — let winners run to full TP",
            })

    # Losses in DEAD_ZONE
    if "session" in trades_df.columns and "outcome" in trades_df.columns:
        dead_zone = trades_df[
            (trades_df["session"] == "DEAD_ZONE") & (trades_df["outcome"] == "LOSS")
        ]
        if len(dead_zone) > 0:
            mistakes.append({
                "type":       "TRADING_DEAD_ZONE",
                "count":      int(len(dead_zone)),
                "severity":   "HIGH",
                "suggestion": "Avoid trading during 22:00-08:00 UTC — low liquidity, spreads wide",
            })

    return mistakes


# ─────────────────────────────────────────────
# Confidence Calibration
# ─────────────────────────────────────────────

def confidence_calibration(signals_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
    """
    Check if stated confidence aligns with actual outcomes.
    e.g., does 80% confidence actually win 80% of the time?
    Returns calibration curve data.
    """
    if signals_df.empty or trades_df.empty:
        return {"calibration": [], "is_well_calibrated": False}

    # Merge signals with trade outcomes
    # Simplified: bucket trades by confidence level
    if "confidence" not in trades_df.columns or "outcome" not in trades_df.columns:
        return {"calibration": [], "is_well_calibrated": False}

    df = trades_df[trades_df["outcome"].isin(["WIN", "LOSS"])].copy()
    if df.empty:
        return {"calibration": [], "is_well_calibrated": False}

    bins = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    calibration = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        subset = df[(df["confidence"] >= low) & (df["confidence"] < high)]
        if len(subset) == 0:
            continue
        actual_wr = float(len(subset[subset["outcome"] == "WIN"]) / len(subset))
        mid_conf  = (low + high) / 2
        calibration.append({
            "confidence_bucket": f"{int(low*100)}-{int(high*100)}%",
            "stated_confidence": float(mid_conf),
            "actual_win_rate":   float(actual_wr),
            "n_trades":          int(len(subset)),
            "calibration_error": float(abs(mid_conf - actual_wr)),
        })

    # Well-calibrated if avg error < 10%
    if calibration:
        avg_error = float(np.mean([c["calibration_error"] for c in calibration]))
        well_calibrated = avg_error < 0.10
    else:
        avg_error = 0.0
        well_calibrated = False

    return {
        "calibration":        calibration,
        "avg_calibration_error": float(avg_error),
        "is_well_calibrated": bool(well_calibrated),
    }


# ─────────────────────────────────────────────
# Engine Accuracy Tracker
# ─────────────────────────────────────────────

def engine_accuracy_tracker(signals_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
    """
    Track which engines' signals were most predictive.
    Matches signals to trade outcomes by timestamp proximity.
    Returns {engine_id: accuracy}
    """
    if signals_df.empty or trades_df.empty:
        return {}

    if "engine_id" not in signals_df.columns:
        return {}

    accuracy_by_engine = {}

    for engine_id in signals_df["engine_id"].unique():
        eng_signals = signals_df[signals_df["engine_id"] == engine_id].copy()
        if eng_signals.empty:
            continue

        correct = 0
        total   = 0

        for _, sig in eng_signals.iterrows():
            sig_direction = sig.get("signal", "NEUTRAL")
            if sig_direction == "NEUTRAL":
                continue

            # Find nearest trade
            if "ts_open" in trades_df.columns:
                sig_ts = pd.to_datetime(sig.get("ts", ""))
                if pd.isna(sig_ts):
                    continue

                trades_df_copy = trades_df.copy()
                trades_df_copy["ts_open"] = pd.to_datetime(trades_df_copy["ts_open"])
                time_diffs = (trades_df_copy["ts_open"] - sig_ts).abs()
                if time_diffs.empty:
                    continue

                nearest_idx = time_diffs.idxmin()
                nearest_trade = trades_df.loc[nearest_idx]

                trade_dir = nearest_trade.get("direction", "")
                outcome   = nearest_trade.get("outcome", "")

                # Correct if signal direction matched and trade won
                signal_to_dir = {"BUY": "LONG", "SELL": "SHORT"}
                if signal_to_dir.get(sig_direction) == trade_dir and outcome == "WIN":
                    correct += 1
                total += 1

        if total > 0:
            accuracy_by_engine[str(engine_id)] = float(correct / total)

    return accuracy_by_engine


# ─────────────────────────────────────────────
# Confidence Modifier
# ─────────────────────────────────────────────

def compute_confidence_modifier(metrics: Dict) -> float:
    """
    Compute a confidence modifier [-0.1, +0.1] based on recent performance.
    Good performance = boost confidence; poor performance = reduce.
    """
    if not metrics or metrics.get("total_trades", 0) < 5:
        return 0.0

    win_rate   = metrics.get("win_rate", 0.5)
    profit_fac = metrics.get("profit_factor", 1.0)
    max_dd     = abs(metrics.get("max_drawdown_pct", 0.0))
    consec_loss = metrics.get("max_consecutive_losses", 0)

    modifier = 0.0

    # Win rate bonus/penalty
    if win_rate > 0.65:
        modifier += 0.05
    elif win_rate > 0.55:
        modifier += 0.02
    elif win_rate < 0.40:
        modifier -= 0.08
    elif win_rate < 0.50:
        modifier -= 0.04

    # Profit factor
    if profit_fac > 1.8:
        modifier += 0.03
    elif profit_fac < 0.8:
        modifier -= 0.05

    # Drawdown penalty
    if max_dd > 10:
        modifier -= 0.07
    elif max_dd > 5:
        modifier -= 0.03

    # Consecutive losses penalty
    if consec_loss >= 4:
        modifier -= 0.05
    elif consec_loss >= 2:
        modifier -= 0.02

    return float(max(-0.10, min(0.10, modifier)))


# ─────────────────────────────────────────────
# Main Run
# ─────────────────────────────────────────────

def run() -> Dict:
    """
    Full memory and learning analysis.
    Returns performance summary + confidence_modifier.
    """
    try:
        trades_df  = get_trades_df(limit=500)
        signals_df = get_signals_df(limit=1000)

        metrics       = compute_all_metrics(trades_df)
        session_bd    = session_breakdown(trades_df)
        regime_bd     = regime_breakdown(trades_df)
        mistakes      = mistake_classifier(trades_df)
        calibration   = confidence_calibration(signals_df, trades_df)
        eng_accuracy  = engine_accuracy_tracker(signals_df, trades_df)
        conf_modifier = compute_confidence_modifier(metrics)

        # Performance signal
        signal = "NEUTRAL"
        confidence = 0.5

        if metrics["total_trades"] >= 10:
            wr = metrics["win_rate"]
            pf = metrics["profit_factor"]
            if wr > 0.55 and pf > 1.2:
                signal     = "BUY"   # System performing well, trade with it
                confidence = 0.6 + (wr - 0.55) * 0.5
            elif wr < 0.40 or pf < 0.8:
                signal     = "NEUTRAL"  # System underperforming, be cautious
                confidence = 0.3
            else:
                confidence = 0.55

        return {
            "engine":            "engine8_memory_learning",
            "signal":            signal,
            "confidence":        float(min(confidence, 0.85)),
            "confidence_modifier": float(conf_modifier),
            "metrics":           metrics,
            "session_breakdown": session_bd,
            "regime_breakdown":  regime_bd,
            "mistakes":          mistakes,
            "calibration":       calibration,
            "engine_accuracy":   eng_accuracy,
            "ts":                datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("Engine8 run error: %s", e, exc_info=True)
        return {
            "engine":             "engine8_memory_learning",
            "signal":             "NEUTRAL",
            "confidence":         0.5,
            "confidence_modifier":0.0,
            "error":              str(e),
        }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    result = run()
    print(json.dumps(result, indent=2, default=str))
