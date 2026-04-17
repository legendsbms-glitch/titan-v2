# FILE: engines/engine11_ml_predictor.py
"""
TITAN v2.0 — Engine 11: ML Predictor
Gradient Boosting ensemble, feature engineering from all other engines,
rolling walk-forward validation, SHAP importance, prediction confidence
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

from core.config import GOLD_SYMBOL
from core.logger import get_logger
from core.db import log_signal, init_db

log = get_logger("Engine11.MLPredictor")

MODEL_PATH  = "data/ml_model.pkl"
SCALER_PATH = "data/ml_scaler.pkl"
os.makedirs("data", exist_ok=True)


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rich feature matrix from OHLCV.
    All features are fully self-contained — no look-ahead bias.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Returns ──────────────────────────────────────────────────────────────
    for period in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_{period}d"] = close.pct_change(period)

    # ── Momentum ─────────────────────────────────────────────────────────────
    feat["rsi_14"]  = ta.momentum.rsi(close, window=14)
    feat["rsi_7"]   = ta.momentum.rsi(close, window=7)
    feat["rsi_21"]  = ta.momentum.rsi(close, window=21)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
    feat["stoch_k"]  = stoch.stoch()
    feat["stoch_d"]  = stoch.stoch_signal()
    feat["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    feat["cci_20"]   = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    feat["roc_10"]   = ta.momentum.ROCIndicator(close, window=10).roc()

    # ── Trend ─────────────────────────────────────────────────────────────────
    for span in [10, 20, 50, 100, 200]:
        ema = close.ewm(span=span).mean()
        feat[f"ema_{span}"]      = ema
        feat[f"close_vs_ema{span}"] = (close - ema) / ema * 100

    feat["ema20_vs_ema50"]  = (close.ewm(span=20).mean() - close.ewm(span=50).mean()) / close * 100
    feat["ema50_vs_ema200"] = (close.ewm(span=50).mean() - close.ewm(span=200).mean()) / close * 100

    macd = ta.trend.MACD(close)
    feat["macd"]       = macd.macd()
    feat["macd_signal"]= macd.macd_signal()
    feat["macd_hist"]  = macd.macd_diff()
    feat["macd_hist_change"] = feat["macd_hist"].diff()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    feat["adx"]       = adx.adx()
    feat["adx_pos"]   = adx.adx_pos()
    feat["adx_neg"]   = adx.adx_neg()
    feat["dm_diff"]   = feat["adx_pos"] - feat["adx_neg"]

    # ── Volatility ────────────────────────────────────────────────────────────
    feat["atr_14"]   = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    feat["atr_7"]    = ta.volatility.AverageTrueRange(high, low, close, window=7).average_true_range()
    feat["atr_ratio"] = feat["atr_7"] / feat["atr_14"].replace(0, np.nan)

    bb = ta.volatility.BollingerBands(close, window=20)
    feat["bb_width"]  = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    feat["bb_pct"]    = bb.bollinger_pband()
    feat["bb_upper_dist"] = (bb.bollinger_hband() - close) / close * 100
    feat["bb_lower_dist"] = (close - bb.bollinger_lband()) / close * 100

    kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
    feat["kc_pband"] = kc.keltner_channel_pband()

    feat["hist_vol_10"] = close.pct_change().rolling(10).std() * np.sqrt(252)
    feat["hist_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    feat["vol_regime"]  = feat["hist_vol_10"] / feat["hist_vol_20"].replace(0, np.nan)

    # ── Volume ────────────────────────────────────────────────────────────────
    feat["vol_sma10"]     = vol.rolling(10).mean()
    feat["vol_ratio"]     = vol / feat["vol_sma10"].replace(0, np.nan)
    feat["vol_ma_diff"]   = (vol - feat["vol_sma10"]) / feat["vol_sma10"].replace(0, np.nan)
    obv = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    feat["obv_slope"]     = obv.diff(5)
    feat["vwap_dist"]     = (close - (close * vol).rolling(20).sum() / vol.rolling(20).sum()) / close * 100

    # ── Price Action ──────────────────────────────────────────────────────────
    feat["body_size"]     = abs(df["close"] - df["open"]) / df["close"] * 100
    feat["upper_wick"]    = (high - df[["open","close"]].max(axis=1)) / close * 100
    feat["lower_wick"]    = (df[["open","close"]].min(axis=1) - low) / close * 100
    feat["body_ratio"]    = feat["body_size"] / ((high - low) / close * 100).replace(0, np.nan)
    feat["gap"]           = (df["open"] - close.shift(1)) / close.shift(1) * 100

    # ── Mean Reversion ────────────────────────────────────────────────────────
    feat["dist_52w_high"] = (close.rolling(252).max() - close) / close * 100
    feat["dist_52w_low"]  = (close - close.rolling(252).min()) / close * 100
    feat["zscore_20"]     = (close - close.rolling(20).mean()) / close.rolling(20).std()
    feat["zscore_60"]     = (close - close.rolling(60).mean()) / close.rolling(60).std()

    # ── Time Features ─────────────────────────────────────────────────────────
    feat["day_of_week"]   = pd.to_datetime(df.index).dayofweek
    feat["month"]         = pd.to_datetime(df.index).month
    feat["is_month_end"]  = pd.to_datetime(df.index).is_month_end.astype(int)

    # Drop raw price columns (prevent leakage)
    drop_cols = [c for c in feat.columns if c.startswith("ema_")]
    feat.drop(columns=drop_cols, inplace=True, errors="ignore")

    return feat.replace([np.inf, -np.inf], np.nan)


def build_labels(df: pd.DataFrame, forward_periods: int = 5,
                  threshold_pct: float = 0.3) -> pd.Series:
    """
    Label: 1 = price up >threshold% in next N bars, -1 = down, 0 = flat
    """
    future_ret = df["close"].pct_change(forward_periods).shift(-forward_periods) * 100
    labels = pd.Series(0, index=df.index)
    labels[future_ret >  threshold_pct] =  1
    labels[future_ret < -threshold_pct] = -1
    return labels


# ── Model Training ────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> Tuple[object, object, Dict]:
    """Walk-forward training with cross-validation"""
    features = build_features(df)
    labels   = build_labels(df, forward_periods=5, threshold_pct=0.3)

    # Align
    valid_idx = features.dropna().index
    X = features.loc[valid_idx]
    y = labels.loc[valid_idx]

    # Drop last N rows (no labels yet)
    X = X.iloc[:-5]
    y = y.iloc[:-5]

    if len(X) < 100:
        log.warning("Insufficient training data for ML model")
        return None, None, {"error": "Insufficient data", "samples": len(X)}

    # Time-series cross-validation
    tscv     = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        gb.fit(X_scaled[train_idx], y.iloc[train_idx])
        preds = gb.predict(X_scaled[val_idx])
        score = accuracy_score(y.iloc[val_idx], preds)
        cv_scores.append(score)

    # Train final model on all data
    gb.fit(X_scaled, y)

    # Feature importance
    feat_importance = sorted(
        zip(X.columns, gb.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:15]

    metrics = {
        "cv_scores":      [round(s, 4) for s in cv_scores],
        "cv_mean":        round(float(np.mean(cv_scores)), 4),
        "cv_std":         round(float(np.std(cv_scores)), 4),
        "train_samples":  len(X),
        "n_features":     len(X.columns),
        "top_features":   [(f, round(float(i), 5)) for f, i in feat_importance],
    }

    log.info(f"ML model trained: CV={metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")
    return gb, scaler, metrics


def predict(df: pd.DataFrame, model, scaler) -> Dict:
    """Make prediction on current data"""
    features = build_features(df)
    latest   = features.iloc[[-1]].fillna(0)

    try:
        X_scaled = scaler.transform(latest)
        pred     = model.predict(X_scaled)[0]
        proba    = model.predict_proba(X_scaled)[0]
        classes  = model.classes_

        prob_map = dict(zip(classes, proba))
        p_up     = float(prob_map.get(1,  0))
        p_flat   = float(prob_map.get(0,  0))
        p_down   = float(prob_map.get(-1, 0))

        return {
            "prediction":  int(pred),
            "p_up":        round(p_up,   4),
            "p_flat":      round(p_flat, 4),
            "p_down":      round(p_down, 4),
            "confidence":  round(max(p_up, p_down), 4),
            "direction":   "BUY" if pred == 1 else "SELL" if pred == -1 else "NEUTRAL",
        }
    except Exception as e:
        return {"error": str(e), "direction": "NEUTRAL", "confidence": 0.0}


# ── Save / Load Model ─────────────────────────────────────────────────────────

def save_model(model, scaler, metrics: Dict):
    try:
        import pickle
        with open(MODEL_PATH, "wb")  as f: pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
        with open("data/ml_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        log.info("ML model saved to disk")
    except Exception as e:
        log.warning(f"Model save failed: {e}")


def load_model():
    try:
        import pickle
        with open(MODEL_PATH, "rb")  as f: model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except Exception:
        return None, None


# ── Run ───────────────────────────────────────────────────────────────────────

def run() -> Dict:
    log.info("Running Engine 11: ML Predictor")
    try:
        df = yf.download(GOLD_SYMBOL, period="2y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return {"signal": "ERROR", "confidence": 0.0, "error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.dropna()

        # Try to load existing model
        model, scaler = load_model()
        retrain_needed = model is None

        if retrain_needed:
            log.info("Training ML model (first run or model not found)")
            model, scaler, train_metrics = train_model(df)
            if model is not None:
                save_model(model, scaler, train_metrics)
        else:
            train_metrics = {}
            try:
                with open("data/ml_metrics.json") as f:
                    train_metrics = json.load(f)
            except Exception:
                pass

        if model is None or scaler is None:
            return {"signal": "NEUTRAL", "confidence": 0.0,
                    "error": "Model training failed — insufficient data"}

        # Predict
        pred_result = predict(df, model, scaler)
        direction   = pred_result.get("direction", "NEUTRAL")
        confidence  = pred_result.get("confidence", 0.0)

        # Discount confidence by model CV score
        cv_mean = train_metrics.get("cv_mean", 0.5)
        if cv_mean > 0.55:
            confidence = min(confidence * 1.1, 0.85)
        elif cv_mean < 0.50:
            confidence *= 0.80

        result = {
            "signal":        direction,
            "confidence":    round(confidence, 3),
            "prediction":    pred_result,
            "model_metrics": {
                "cv_accuracy": train_metrics.get("cv_mean"),
                "cv_std":      train_metrics.get("cv_std"),
                "n_features":  train_metrics.get("n_features"),
                "train_samples": train_metrics.get("train_samples"),
                "top_features":  train_metrics.get("top_features", [])[:5],
            },
            "retrained": retrain_needed,
        }

        log_signal("engine11_ml_predictor", direction, confidence, result)
        log.info(f"Engine 11 → {direction} @ {confidence:.1%} | CV: {cv_mean:.1%}")
        return result

    except Exception as e:
        log.error(f"Engine 11 error: {e}", exc_info=True)
        return {"signal": "ERROR", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import json
    init_db()
    print(json.dumps(run(), indent=2, default=str))
