"""
Simple short-term post-earnings direction model.
This module builds a tiny feature vector from recent returns and optional sentiment to predict next-half-day direction.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import joblib


def build_features(prices: pd.DataFrame) -> np.ndarray:
    """Build simple features from price series (assumes datetime index, 'Close' column)."""
    df = prices.copy()
    df = df.sort_index()
    df['ret1'] = df['Close'].pct_change(1)
    df['ret5'] = df['Close'].pct_change(5)
    df['ret10'] = df['Close'].pct_change(10)
    df['volatility5'] = df['ret1'].rolling(5).std()
    df['volatility10'] = df['ret1'].rolling(10).std()
    feats = df[['ret1','ret5','ret10','volatility5','volatility10']].iloc[-1:].fillna(0.0).to_numpy()
    return feats


def fetch_recent_prices_yf(symbol: str, end: dt.datetime, days: int = 30) -> pd.DataFrame:
    """Fetch recent prices with yfinance to construct quick features."""
    import yfinance as yf
    start = end - dt.timedelta(days=days)
    data = yf.download(symbol, start=start.date().isoformat(), end=end.date().isoformat(), progress=False)
    if data is None or data.empty:
        raise RuntimeError("No price data")
    data = data.rename(columns={"Adj Close":"Close"})
    if 'Close' not in data.columns:
        data['Close'] = data['Adj Close'] if 'Adj Close' in data.columns else data['close']
    return data


def predict_half_day_direction(
    symbol: str,
    event_time: dt.datetime,
    sentiment_score: Optional[float] = None,
    blend_weight: float = 0.2,
) -> Dict[str, Any]:
    """Train a tiny logistic model on synthetic past windows and score the current feature.

    Note: This is a heuristic demo; for production use, train offline with proper labels.
    """
    # Try saved model first
    model = load_saved_model(symbol)
    prices = fetch_recent_prices_yf(symbol, end=event_time)
    X_curr = build_features(prices)

    # Build a small synthetic training set from rolling windows as a placeholder
    df = prices.copy().sort_index()
    df['ret1'] = df['Close'].pct_change(1)
    df['ret5'] = df['Close'].pct_change(5)
    df['ret10'] = df['Close'].pct_change(10)
    df['volatility5'] = df['ret1'].rolling(5).std()
    df['volatility10'] = df['ret1'].rolling(10).std()
    # Next-half-day proxy label: sign of next 1-step return (very rough)
    df['y'] = (df['ret1'].shift(-1) > 0).astype(int)
    df = df.dropna()
    X = df[['ret1','ret5','ret10','volatility5','volatility10']].to_numpy()
    y = df['y'].to_numpy()

    if len(df) < 30:
        # Too small; return neutral
        return {"symbol": symbol, "prob_up": 0.5, "note": "insufficient history"}

    if model is None:
        model = LogisticRegression(max_iter=500)
        try:
            model.fit(X, y)
        except Exception:
            model = None
    try:
        if model is not None:
            prob_up = float(model.predict_proba(X_curr)[0,1])
        else:
            prob_up = 0.5
    except Exception:
        prob_up = 0.5

    # Blend with sentiment if provided
    if sentiment_score is not None and blend_weight > 0:
        # map sentiment [-1,1] to [0,1]
        sent_p = max(0.0, min(1.0, 0.5 * (sentiment_score + 1.0)))
        w = max(0.0, min(1.0, float(blend_weight)))
        prob_up = (1.0 - w) * prob_up + w * sent_p

    return {"symbol": symbol, "prob_up": prob_up}


# ---- Persistence helpers ----

def _model_dir() -> Path:
    p = Path("models") / "earnings"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _model_path(symbol: str) -> Path:
    return _model_dir() / f"lr_{symbol.upper()}.joblib"

def save_model(symbol: str, model: LogisticRegression) -> Path:
    path = _model_path(symbol)
    joblib.dump(model, path)
    return path

def load_saved_model(symbol: str) -> Optional[LogisticRegression]:
    path = _model_path(symbol)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def train_symbol_model(symbol: str, end_time: dt.datetime, lookback_days: int = 180) -> Dict[str, Any]:
    """Train and persist a per-symbol logistic model using recent history."""
    # fetch longer window
    import yfinance as yf
    start = end_time - dt.timedelta(days=lookback_days)
    data = yf.download(symbol, start=start.date().isoformat(), end=end_time.date().isoformat(), progress=False)
    if data is None or data.empty:
        return {"ok": False, "error": "No price data"}
    data = data.rename(columns={"Adj Close":"Close"})
    if 'Close' not in data.columns:
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        else:
            return {"ok": False, "error": "Missing Close column"}

    df = data.copy().sort_index()
    df['ret1'] = df['Close'].pct_change(1)
    df['ret5'] = df['Close'].pct_change(5)
    df['ret10'] = df['Close'].pct_change(10)
    df['volatility5'] = df['ret1'].rolling(5).std()
    df['volatility10'] = df['ret1'].rolling(10).std()
    df['y'] = (df['ret1'].shift(-1) > 0).astype(int)
    df = df.dropna()
    if len(df) < 60:
        return {"ok": False, "error": "Insufficient history"}
    X = df[['ret1','ret5','ret10','volatility5','volatility10']].to_numpy()
    y = df['y'].to_numpy()
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    path = save_model(symbol, model)
    return {"ok": True, "path": str(path), "samples": int(len(df))}
