from __future__ import annotations
from typing import List, Dict, Any, Literal, Tuple
import math

try:
    import numpy as np
    from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
    from py_vollib.black_scholes import black_scholes as bs_price
    HAS_VOLLIB = True
except Exception:
    HAS_VOLLIB = False

Right = Literal["C","P"]

def _bs_flag(pos: str) -> Right:
    return "C" if "Call" in pos else "P"

def price_greeks_pyvollib(legs: List[Dict[str, Any]], spot: float, r: float = 0.0, q: float = 0.0, iv_hint: float | None = None) -> Dict[str, Any]:
    """Compute per-leg fair value and Greeks using py_vollib (if installed). 
    legs: [{pos, strike, qty, premium (optional), expiry (YYYYMMDD)}]
    Returns: {legs:[{... with price, delta, gamma, theta, vega}], totals:{delta,gamma,theta,vega, debit}}
    """
    out_legs = []
    tot_delta = tot_gamma = tot_theta = tot_vega = 0.0
    debit = 0.0

    if not HAS_VOLLIB:
        # Fallback: naive approximations
        for leg in legs:
            qty = int(leg.get("qty",1))
            price = float(leg.get("premium") or 1.0)
            d = 0.5 if _bs_flag(leg["pos"])=="C" else -0.5
            g = 0.01; t = -0.02; v = 0.1
            out_legs.append({**leg, "price": price, "delta": d*qty, "gamma": g*qty, "theta": t*qty, "vega": v*qty})
            debit += price * qty
            tot_delta += d*qty; tot_gamma += g*qty; tot_theta += t*qty; tot_vega += v*qty
        return {"legs": out_legs, "totals": {"delta": tot_delta, "gamma": tot_gamma, "theta": tot_theta, "vega": tot_vega, "debit": debit}}

    # With vollib: assume T (time to expiry) = 30/365 as placeholder unless premium present to back out IV.
    # In production, compute actual T from expiry string and current date.
    T_default = 30/365

    for leg in legs:
        right = _bs_flag(leg["pos"])
        K = float(leg["strike"])
        qty = int(leg.get("qty",1))
        mkt = leg.get("premium")
        T = T_default
        # If a market premium given, try to back out IV (guard for failures)
        try:
            if mkt is not None and mkt > 0:
                iv = max(1e-4, min(4.0, bs_iv(mkt, spot, K, T, r, right)))
            else:
                iv = iv_hint if iv_hint is not None else 0.25
        except Exception:
            iv = iv_hint if iv_hint is not None else 0.25
        # Price via BS
        px = bs_price(right, spot, K, T, r, iv)
        # Greeks
        d = delta(right, spot, K, T, r, iv)
        g = gamma(right, spot, K, T, r, iv)
        t = theta(right, spot, K, T, r, iv) / 365.0  # per-day
        v = vega(right, spot, K, T, r, iv)
        # Apply position (Long/Short sign)
        sign = 1 if "Long" in leg["pos"] else -1
        out_legs.append({**leg, "price": px, "delta": sign*d*qty, "gamma": sign*g*qty, "theta": sign*t*qty, "vega": sign*v*qty, "iv": iv})
        debit += px * sign * qty
        tot_delta += sign*d*qty; tot_gamma += sign*g*qty; tot_theta += sign*t*qty; tot_vega += sign*v*qty

    return {"legs": out_legs, "totals": {"delta": tot_delta, "gamma": tot_gamma, "theta": tot_theta, "vega": tot_vega, "debit": debit}}
