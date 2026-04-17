# FILE: backtesting/backtester.py
"""
TITAN v2.0 — Backtester
Historical simulation using simplified signal logic, full metrics, Monte Carlo
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from core.config import GOLD_SYMBOL, RISK
from core.logger import get_logger

log = get_logger("Backtester")


@dataclass
class Trade:
    direction:  str
    entry:      float
    sl:         float
    tp:         float
    exit_price: float = None
    pnl:        float = None
    rr:         float = None
    outcome:    str   = None
    entry_idx:  int   = 0
    exit_idx:   int   = 0
    bars:       int   = 0


@dataclass
class BacktestResult:
    symbol:        str
    start:         str
    end:           str
    capital:       float
    total_trades:  int       = 0
    win_rate:      float     = 0.0
    profit_factor: float     = 0.0
    expectancy:    float     = 0.0
    sharpe:        float     = 0.0
    calmar:        float     = 0.0
    max_drawdown:  float     = 0.0
    total_return:  float     = 0.0
    total_pct:     float     = 0.0
    avg_win:       float     = 0.0
    avg_loss:      float     = 0.0
    trades:        List[Trade] = field(default_factory=list)
    equity_curve:  List[float] = field(default_factory=list)
    monte_carlo:   Dict      = field(default_factory=dict)


def fetch_backtest_data(symbol: str = GOLD_SYMBOL,
                         start: str = "2023-01-01",
                         end:   str = "2024-01-01") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {symbol} {start}-{end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def simple_signal_engine(df: pd.DataFrame, i: int) -> Tuple[str, float, float]:
    """
    Simplified signal logic for backtesting:
    - Uses EMA crossover + RSI confirmation
    - Returns (direction, entry, atr)
    """
    if i < 30:
        return "NEUTRAL", 0, 0

    window = df.iloc[max(0, i-50):i+1]

    close  = window["close"]
    ema20  = close.ewm(span=20).mean().iloc[-1]
    ema50  = close.ewm(span=50).mean().iloc[-1]

    # RSI
    delta   = close.diff()
    gain    = delta.clip(lower=0).ewm(span=14).mean()
    loss    = (-delta.clip(upper=0)).ewm(span=14).mean()
    rs      = gain / loss.replace(0, np.nan)
    rsi     = float((100 - 100 / (1 + rs)).iloc[-1])

    # ATR
    h, l, c = window["high"], window["low"], close
    tr      = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr     = float(tr.ewm(span=14).mean().iloc[-1])

    price   = float(df["close"].iloc[i])

    if ema20 > ema50 and rsi > 55 and rsi < 75:
        return "BUY", price, atr
    elif ema20 < ema50 and rsi < 45 and rsi > 25:
        return "SELL", price, atr
    return "NEUTRAL", price, atr


def simulate_trades(df: pd.DataFrame, risk_pct: float = 1.0,
                     capital: float = 10000.0) -> List[Trade]:
    """Simulate trades using simple signal engine"""
    trades    = []
    equity    = capital
    in_trade  = False
    current_trade: Optional[Trade] = None

    for i in range(30, len(df)):
        bar = df.iloc[i]
        h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])

        # Check if current trade hit SL or TP
        if in_trade and current_trade:
            sl, tp = current_trade.sl, current_trade.tp

            if current_trade.direction == "BUY":
                if l <= sl:  # Stop hit
                    current_trade.exit_price = sl
                    current_trade.pnl  = (sl - current_trade.entry) / current_trade.entry * equity * risk_pct / 100
                    current_trade.rr   = -1.0
                    current_trade.outcome = "LOSS"
                    current_trade.exit_idx = i
                    current_trade.bars = i - current_trade.entry_idx
                    equity += current_trade.pnl
                    trades.append(current_trade)
                    in_trade = False

                elif h >= tp:  # TP hit
                    current_trade.exit_price = tp
                    current_trade.pnl  = (tp - current_trade.entry) / current_trade.entry * equity * risk_pct / 100
                    current_trade.rr   = RISK["min_rr_ratio"]
                    current_trade.outcome = "WIN"
                    current_trade.exit_idx = i
                    current_trade.bars = i - current_trade.entry_idx
                    equity += current_trade.pnl
                    trades.append(current_trade)
                    in_trade = False

            elif current_trade.direction == "SELL":
                if h >= sl:
                    current_trade.exit_price = sl
                    current_trade.pnl  = (current_trade.entry - sl) / current_trade.entry * equity * risk_pct / 100 * -1
                    current_trade.rr   = -1.0
                    current_trade.outcome = "LOSS"
                    current_trade.exit_idx = i
                    current_trade.bars = i - current_trade.entry_idx
                    equity += current_trade.pnl
                    trades.append(current_trade)
                    in_trade = False

                elif l <= tp:
                    current_trade.exit_price = tp
                    current_trade.pnl  = (current_trade.entry - tp) / current_trade.entry * equity * risk_pct / 100
                    current_trade.rr   = RISK["min_rr_ratio"]
                    current_trade.outcome = "WIN"
                    current_trade.exit_idx = i
                    current_trade.bars = i - current_trade.entry_idx
                    equity += current_trade.pnl
                    trades.append(current_trade)
                    in_trade = False

        # Enter new trade if not in one
        if not in_trade and i < len(df) - 5:
            signal, entry, atr = simple_signal_engine(df, i)
            if signal != "NEUTRAL" and atr > 0:
                sl_dist = atr * RISK["atr_sl_multiplier"]
                tp_dist = sl_dist * RISK["min_rr_ratio"]
                if signal == "BUY":
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                current_trade = Trade(
                    direction=signal, entry=entry, sl=sl, tp=tp,
                    entry_idx=i
                )
                in_trade = True

    return trades, equity


def compute_metrics(trades: List[Trade], capital: float,
                     final_equity: float) -> Dict:
    if not trades:
        return {"error": "No trades"}

    pnl_list = [t.pnl for t in trades if t.pnl is not None]
    wins     = [p for p in pnl_list if p > 0]
    losses   = [p for p in pnl_list if p < 0]

    if not pnl_list:
        return {"error": "No closed trades"}

    win_rate      = len(wins) / len(pnl_list)
    avg_win       = float(np.mean(wins))  if wins   else 0
    avg_loss      = float(np.mean(losses)) if losses else 0
    profit_factor = float(sum(wins) / abs(sum(losses))) if sum(losses) != 0 else 999
    expectancy    = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    pnl_arr   = np.array(pnl_list)
    sharpe    = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)) if np.std(pnl_arr) > 0 else 0

    # Drawdown
    cumulative  = np.cumsum(pnl_arr)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown    = cumulative - rolling_max
    max_dd      = float(np.min(drawdown))
    calmar      = float(sum(pnl_list) / abs(max_dd)) if max_dd != 0 else 0

    total_return = final_equity - capital
    total_pct    = total_return / capital * 100

    return {
        "total_trades":  len(pnl_list),
        "win_rate":      round(win_rate, 3),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "expectancy":    round(expectancy, 2),
        "sharpe":        round(sharpe, 3),
        "calmar":        round(calmar, 3),
        "max_drawdown":  round(max_dd, 2),
        "total_return":  round(total_return, 2),
        "total_pct":     round(total_pct, 2),
        "start_capital": capital,
        "end_capital":   round(final_equity, 2),
    }


def monte_carlo(trades: List[Trade], n: int = 1000,
                 initial_capital: float = 10000) -> Dict:
    """Bootstrap resample trade outcomes for confidence intervals"""
    if len(trades) < 5:
        return {"note": "Need at least 5 trades for Monte Carlo"}

    pnl_list = [t.pnl for t in trades if t.pnl is not None]
    if not pnl_list:
        return {"note": "No PnL data"}

    final_equities = []
    max_drawdowns  = []

    for _ in range(n):
        # Resample with replacement
        sampled = np.random.choice(pnl_list, size=len(pnl_list), replace=True)
        equity  = initial_capital + np.cumsum(sampled)
        final_equities.append(float(equity[-1]))

        # Max drawdown for this sample
        rolling_max = np.maximum.accumulate(equity)
        dd          = equity - rolling_max
        max_drawdowns.append(float(np.min(dd)))

    fe = sorted(final_equities)
    dd = sorted(max_drawdowns)

    return {
        "n_simulations":  n,
        "final_equity": {
            "p5":    round(np.percentile(fe, 5), 2),
            "p25":   round(np.percentile(fe, 25), 2),
            "p50":   round(np.median(fe), 2),
            "p75":   round(np.percentile(fe, 75), 2),
            "p95":   round(np.percentile(fe, 95), 2),
        },
        "max_drawdown": {
            "p5":  round(np.percentile(dd, 5), 2),
            "p50": round(np.median(dd), 2),
            "p95": round(np.percentile(dd, 95), 2),
        },
        "probability_of_profit": round(sum(1 for e in fe if e > initial_capital) / n, 3),
    }


def run_backtest(start: str = "2023-01-01", end: str = "2024-01-01",
                  capital: float = 10000.0, risk_pct: float = 1.0) -> BacktestResult:
    log.info(f"Running backtest: {start} → {end} | Capital: ${capital:,.0f}")

    df      = fetch_backtest_data(start=start, end=end)
    trades, final_equity = simulate_trades(df, risk_pct=risk_pct, capital=capital)
    metrics = compute_metrics(trades, capital, final_equity)
    mc      = monte_carlo(trades, n=1000, initial_capital=capital)

    result = BacktestResult(
        symbol       = GOLD_SYMBOL,
        start        = start,
        end          = end,
        capital      = capital,
        trades       = trades,
        monte_carlo  = mc,
        **{k: v for k, v in metrics.items() if k not in ("start_capital", "end_capital", "error")}
    )

    # Equity curve
    equity = capital
    curve  = [equity]
    for t in trades:
        if t.pnl:
            equity += t.pnl
            curve.append(round(equity, 2))
    result.equity_curve = curve

    log.info(f"Backtest complete: {metrics.get('total_trades',0)} trades | WR: {metrics.get('win_rate',0):.1%} | PF: {metrics.get('profit_factor',0):.2f} | Return: {metrics.get('total_pct',0):.1f}%")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TITAN v2.0 Backtester")
    parser.add_argument("--start",   default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2024-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", default=10000, type=float, help="Starting capital")
    parser.add_argument("--risk",    default=1.0, type=float, help="Risk per trade (%)")
    parser.add_argument("--json",    action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_backtest(args.start, args.end, args.capital, args.risk)

    if args.json:
        output = {
            "symbol": result.symbol, "start": result.start, "end": result.end,
            "capital": result.capital, "total_trades": result.total_trades,
            "win_rate": result.win_rate, "profit_factor": result.profit_factor,
            "expectancy": result.expectancy, "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown, "total_return": result.total_return,
            "total_pct": result.total_pct, "monte_carlo": result.monte_carlo,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"TITAN BACKTEST RESULTS — {result.start} to {result.end}")
        print(f"{'='*50}")
        print(f"Trades:        {result.total_trades}")
        print(f"Win Rate:      {result.win_rate:.1%}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Expectancy:    ${result.expectancy:.2f}")
        print(f"Sharpe:        {result.sharpe:.2f}")
        print(f"Max Drawdown:  ${result.max_drawdown:.2f}")
        print(f"Total Return:  ${result.total_return:.2f} ({result.total_pct:.1f}%)")
        print(f"\nMonte Carlo (1000 sims):")
        mc = result.monte_carlo
        if "final_equity" in mc:
            fe = mc["final_equity"]
            print(f"  P5 equity:   ${fe['p5']:,.0f}")
            print(f"  P50 equity:  ${fe['p50']:,.0f}")
            print(f"  P95 equity:  ${fe['p95']:,.0f}")
            print(f"  Prob profit: {mc.get('probability_of_profit',0):.1%}")
        print(f"{'='*50}\n")
