from __future__ import annotations

import pandas as pd
import pandas_ta as ta
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

    # EMA/SMA
    for w in cfg.get("ema_windows", [5, 20]):
        out[f"ema_{w}"] = ta.ema(price, length=w)
        out[f"sma_{w}"] = ta.sma(price, length=w)

    # MACD
    fast, slow, sig = cfg.get("macd", [12, 26, 9])
    macd = ta.macd(price, fast=fast, slow=slow, signal=sig)
    if macd is not None and isinstance(macd, pd.DataFrame):
        out = out.join(macd)

    # RSI
    rsi_w = int(cfg.get("rsi_window", 14))
    out["rsi"] = ta.rsi(price, length=rsi_w)

    # ATR
    atr_w = int(cfg.get("atr_window", 14))
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=atr_w)

    # Bollinger Bands
    bb_w = int(cfg.get("bbands_window", 20))
    bb = ta.bbands(price, length=bb_w)
    if bb is not None and isinstance(bb, pd.DataFrame):
        out = out.join(bb)

    return out
