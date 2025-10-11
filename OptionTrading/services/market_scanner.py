from __future__ import annotations
from typing import List, Dict, Any, Optional
import time

from .pricing import price_greeks_pyvollib

class HybridScanner:
    """Hybrid market scanner.
    - Uses IBKRClient for authoritative snapshots (and optionally scanner presets later).
    - Applies local scoring (on-prem) with py_vollib when available.
    """
    def __init__(self, ib_client) -> None:
        self.ib = ib_client

    def scan_bank(self, symbols: List[str], max_symbols: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        symbols = symbols[:max_symbols]
        for sym in symbols:
            try:
                snap = self.ib.fetch_stock_snapshot(sym)
                price = float(snap.get("price", 0.0) or 0.0)
                # simplistic local "potential" scoring: favor liquid-ish & mid price range
                # (placeholder - can be replaced with IV/ATR/Spread-based rules)
                potential = price > 5.0
                out.append({"symbol": sym, "sector": "-", "signal": "Bullish" if potential else "Neutral", "potential": potential})
            except Exception:
                out.append({"symbol": sym, "sector": "-", "signal": "Neutral", "potential": False})
            # polite pacing; IBKR has pacing rules
            time.sleep(0.05)
        return out
