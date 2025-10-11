from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import time, math, statistics

try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv
    from py_vollib.black_scholes import black_scholes as bs_price
    from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
    HAS_VOLLIB = True
except Exception:
    HAS_VOLLIB = False

class LocalVectorScanner:
    """On-prem scanner focusing on per-symbol metrics using limited IBKR pulls.
    Goals:
      - Low pacing: sample ~6-10 options per symbol (near ATM, ±OTM)
      - Compute: IV (snapshot), IV Rank (vs rolling cache), liquidity (bid/ask spread proxy), POP approx.
    Notes:
      - POP approx is heuristic: for short premium ideas, ~1 - |delta_short|; refined later per strategy.
      - IV Rank requires a rolling history. We keep/update a simple cache file externally (not handled here).
    """
    def __init__(self, ib_client, iv_history: Optional[Dict[str, List[float]]] = None, cfg: Optional[dict] = None) -> None:
        self.ib = ib_client
        self.iv_history = iv_history or {}
        self.cfg = cfg or {}
        self.weights = self.cfg.get('scanner', {}).get('weights', {'pop':0.5,'ivrank':0.3,'liquidity':0.2})
        self.thresholds = self.cfg.get('scanner', {}).get('thresholds', {'pop':0.55,'spread_max':0.15})

    # ---------- helpers ----------
    def _nearest_monthly(self, expirations: List[str], min_days=20, max_days=60) -> str:
        # naive: pick the first; in production compute real DTE from dates
        return sorted(expirations)[0] if expirations else ""

    def _iv_rank(self, sym: str, iv_now: float) -> Optional[float]:
        hist = self.iv_history.get(sym)
        if not hist or len(hist) < 10:
            return None
        lo, hi = min(hist), max(hist)
        if hi <= lo:
            return None
        return (iv_now - lo) / (hi - lo)

    def _pop_from_delta(self, delta_abs: float) -> float:
        # rough mapping: POP ≈ 1 - delta_abs for a one-sided short; clamp [0,1]
        return max(0.0, min(1.0, 1.0 - float(delta_abs)))

    # ---------- main ----------
    def scan_symbol(self, symbol: str, spot: Optional[float] = None, sample=3) -> Dict[str, Any]:
        # spot
        try:
            snap = self.ib.fetch_stock_snapshot(symbol)
            spot = float(snap.get("price", 0.0) or 0.0) if spot is None else float(spot)
        except Exception:
            spot = float(spot or 0.0)
        # secdef
        try:
            sd = self.ib.get_secdef_params(symbol)
            strikes = sorted(sd.get("strikes", []))
            expirations = list(sd.get("expirations", []))
        except Exception:
            strikes = []
            expirations = []
        expiry = self._nearest_monthly(expirations)

        # choose a few strikes around spot (±1%, ±3%, ±5% approx)
        def snap_strike(x: float) -> float:
            return min(strikes, key=lambda k: abs(k-x)) if strikes else round(x,2)
        samples = [spot*0.99, spot*1.01, spot*0.97, spot*1.03, spot*0.95, spot*1.05]
        ks = [snap_strike(v) for v in samples][:max(2, sample*2)]

        # request a small set of options (both C and P) and compute bid/ask spread proxy & IV if available
        ivs = []
        spreads = []
        deltas = []
        for i,k in enumerate(ks):
            for right in ("C","P"):
                try:
                    opt = self.ib.build_option(symbol, right, float(k), expiry)
                    # place a market data request; then sleep a bit to let modelGreeks populate
                    # in production, use ib.sleep or await event; here we assume ib_insync env
                    self.ib.qualify_options([opt])
                    ticker = self.ib.ib.reqMktData(opt, "", False, False) if hasattr(self.ib, "ib") and self.ib.ib else None
                    if hasattr(self.ib, "ib") and self.ib.ib:
                        self.ib.ib.sleep(0.3)
                    iv = None
                    bid, ask = None, None
                    if ticker is not None:
                        # try modelGreeks impliedVol if ready
                        if getattr(ticker, "modelGreeks", None) is not None and getattr(ticker.modelGreeks, "impliedVol", None) is not None:
                            iv = float(ticker.modelGreeks.impliedVol)
                        # fallback to implied vol from mid price if possible
                        mid = None
                        try:
                            bid = float(ticker.bid) if ticker.bid is not None else None
                            ask = float(ticker.ask) if ticker.ask is not None else None
                            if bid is not None and ask is not None and ask > 0:
                                mid = (bid + ask) / 2.0
                        except Exception:
                            pass
                        if iv is None and HAS_VOLLIB and mid is not None and spot and k and expiry:
                            # heuristic T = 30/365 for now
                            try:
                                iv = max(1e-4, min(4.0, bs_iv(mid, spot, float(k), 30/365, 0.0, right)))
                            except Exception:
                                iv = None
                    if iv is not None:
                        ivs.append(iv)
                    if bid is not None and ask is not None and ask > 0:
                        spreads.append((ask - bid) / ask)
                    # compute approx delta if we have iv
                    if HAS_VOLLIB and iv is not None and spot and k:
                        try:
                            d = abs(bs_delta(right, spot, float(k), 30/365, 0.0, float(iv)))
                            deltas.append(d)
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(0.05)  # gentle pace
        iv_now = statistics.median(ivs) if ivs else None
        iv_rank = self._iv_rank(symbol, iv_now) if iv_now is not None else None
        spread_med = statistics.median(spreads) if spreads else None
        delta_med = statistics.median(deltas) if deltas else None
        pop = self._pop_from_delta(delta_med) if delta_med is not None else None

        # simple potential score
        score = 0.0
        if pop is not None: score += pop * float(self.weights.get('pop',0.5))
        if iv_rank is not None:
                score += (1 - abs(0.5 - iv_rank)) * float(self.weights.get('ivrank',0.3))  # prefer middle-to-high IV rank
        if spread_med is not None:
                score += (1.0 - min(1.0, spread_med*5)) * float(self.weights.get('liquidity',0.2))  # penalize wide spreads
        return {
            "symbol": symbol,
            "iv": iv_now,
            "iv_rank": iv_rank,
            "spread": spread_med,
            "pop": pop,
            "score": score,
            "potential": (score >= float(self.thresholds.get('pop', 0.55))),
            "signal": ("Bullish" if pop and pop > 0.55 else "Neutral"),
            "sector": "-"
        }

    def scan_bank(self, symbols: List[str], max_symbols: int = 50) -> List[Dict[str, Any]]:
        out = []
        for sym in symbols[:max_symbols]:
            out.append(self.scan_symbol(sym))
        # sort by score desc
        out.sort(key=lambda r: (r.get("score") or 0.0), reverse=True)
        return out
