from __future__ import annotations
from typing import List, Dict, Any, Literal
from datetime import datetime

try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
    from py_vollib.black_scholes import black_scholes as bs_price
    HAS_VOLLIB = True
except Exception:
    HAS_VOLLIB = False

Right = Literal["C", "P"]


def _right_from_pos(pos: str) -> Right:
    return "C" if "Call" in pos else "P"


def _to_T(expiry: Any) -> float:
    """Convert expiry (YYYYMMDD or 'Mon DD, YYYY') to year fraction."""
    try:
        if not expiry:
            return 30/365
        s = str(expiry)
        dt = None
        for fmt in ("%Y%m%d", "%b %d, %Y", "%b %d, %y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except Exception:
                continue
        if not dt:
            return 30/365
        days = max((dt.date() - datetime.utcnow().date()).days, 1)
        return days/365
    except Exception:
        return 30/365


def price_greeks(legs: List[Dict[str, Any]], spot: float, r: float = 0.0, q: float = 0.0, iv_hint: float | None = None) -> Dict[str, Any]:
    """Compute per-leg fair value and Greeks using py_vollib if available.
    legs: [{pos, strike, qty, premium (optional), expiry (YYYYMMDD)}]
    Returns: {legs:[{... with price, delta, gamma, theta, vega}], totals:{delta,gamma,theta,vega, debit}}
    """
    out_legs = []
    tot = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    debit = 0.0

    if not HAS_VOLLIB:
        # Fallback heuristics
        for leg in legs:
            qty = int(leg.get("qty", 1))
            price = float(leg.get("premium") or 1.0)
            d = 0.5 if _right_from_pos(leg["pos"]) == "C" else -0.5
            g = 0.01
            t = -0.02
            v = 0.1
            out_legs.append({**leg, "price": price, "delta": d * qty, "gamma": g * qty, "theta": t * qty, "vega": v * qty})
            debit += price * qty
            tot["delta"] += d * qty
            tot["gamma"] += g * qty
            tot["theta"] += t * qty
            tot["vega"] += v * qty
        return {"legs": out_legs, "totals": {**tot, "debit": debit}}

    # With vollib
    for leg in legs:
        right = _right_from_pos(leg["pos"])
        K = float(leg["strike"]) if leg.get("strike") is not None else 0.0
        qty = int(leg.get("qty", 1))
        mkt = leg.get("premium")
        T = _to_T(leg.get("expiry"))
        try:
            if mkt is not None and float(mkt) > 0:
                iv = max(1e-4, min(4.0, bs_iv(float(mkt), float(spot), K, T, r, right)))
            else:
                iv = iv_hint if iv_hint is not None else 0.25
        except Exception:
            iv = iv_hint if iv_hint is not None else 0.25
        px = bs_price(right, float(spot), K, T, r, iv)
        d = delta(right, float(spot), K, T, r, iv)
        g = gamma(right, float(spot), K, T, r, iv)
        t = theta(right, float(spot), K, T, r, iv) / 365.0
        v = vega(right, float(spot), K, T, r, iv)
        sign = 1 if "Long" in leg["pos"] else -1
        out_legs.append({**leg, "price": px, "delta": sign * d * qty, "gamma": sign * g * qty, "theta": sign * t * qty, "vega": sign * v * qty, "iv": iv})
        debit += px * sign * qty
        tot["delta"] += sign * d * qty
        tot["gamma"] += sign * g * qty
        tot["theta"] += sign * t * qty
        tot["vega"] += sign * v * qty

    return {"legs": out_legs, "totals": {**tot, "debit": debit}}
