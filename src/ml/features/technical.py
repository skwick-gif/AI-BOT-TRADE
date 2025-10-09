from __future__ import annotations

import pandas as pd
try:
    import pandas_ta as ta
except Exception:
    ta = None
    # When pandas_ta is unavailable or incompatible, feature building will skip TA functions
    # and return the input DataFrame unchanged where applicable.
from typing import Dict, Any


def build_technical_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build basic technical features on a per-symbol daily OHLCV DataFrame.
    Assumes columns: [ticker, date, open, high, low, close, adj_close, volume].
    Returns a DataFrame aligned by date with new feature columns; avoids lookahead.
    """
    out = df.copy()
    price = out["adj_close"].fillna(out["close"]) if "adj_close" in out.columns else out["close"]

    # returns
    for w in cfg.get("ret_windows", [1, 5, 10]):
        out[f"ret_{w}"] = price.pct_change(w)

    # EMA/SMA (if available)
    if ta is not None:
        for w in cfg.get("ema_windows", [5, 20]):
            try:
                out[f"ema_{w}"] = ta.ema(price, length=w)
                out[f"sma_{w}"] = ta.sma(price, length=w)
            except Exception:
                out[f"ema_{w}"] = pd.NA
                out[f"sma_{w}"] = pd.NA

    # MACD
    fast, slow, sig = cfg.get("macd", [12, 26, 9])
    if ta is not None:
        try:
            macd = ta.macd(price, fast=fast, slow=slow, signal=sig)
            if macd is not None and isinstance(macd, pd.DataFrame):
                out = out.join(macd)
        except Exception:
            pass

    # RSI
    rsi_w = int(cfg.get("rsi_window", 14))
    if ta is not None:
        try:
            out["rsi"] = ta.rsi(price, length=rsi_w)
        except Exception:
            out["rsi"] = pd.NA

    # ATR
    atr_w = int(cfg.get("atr_window", 14))
    if ta is not None:
        try:
            out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=atr_w)
        except Exception:
            out["atr"] = pd.NA

    # Bollinger Bands
    bb_w = int(cfg.get("bbands_window", 20))
    if ta is not None:
        try:
            bb = ta.bbands(price, length=bb_w)
            if bb is not None and isinstance(bb, pd.DataFrame):
                out = out.join(bb)
        except Exception:
            pass

    return out
