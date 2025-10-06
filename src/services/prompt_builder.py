"""
PromptBuilder: compact, profile-aware prompts for numeric-only AI scoring.
"""
from __future__ import annotations
import json
from typing import Optional, Dict, Any


def _trim_context(market_context: Optional[Dict[str, Any]], limit: int = 300) -> str:
    if not market_context:
        return ""
    try:
        s = json.dumps(market_context, separators=(",", ":"))
        return s[:limit]
    except Exception:
        return ""


def build_numeric_score_prompt(
    symbol: str,
    *,
    profile: Optional[str] = None,
    market_context: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """
    Return a minimal prompt that asks for ONLY a number 0-10, with profile-aware guidance.
    Profiles: Intraday | Swing | Long-term
    """
    sym = symbol.upper().strip()
    p = (profile or "").strip().lower()

    if p == "intraday":
        focus = "for intraday trading horizon (today, next hours). Focus on price action, momentum, and near-term catalysts. Ignore long-term factors."
    elif p == "swing":
        focus = "for swing trading horizon (days to weeks). Consider trend, momentum, and upcoming catalysts."
    elif p == "long-term" or p == "long term":
        focus = "for long-term horizon (months+). Emphasize fundamentals and macro context."
    else:
        focus = "for short-term trading horizon."

    ctx = _trim_context(market_context)
    ctx_line = f"Context:{ctx}\n" if ctx else ""

    thr_line = ""
    if thresholds:
        try:
            bt = thresholds.get("buy_threshold")
            st = thresholds.get("sell_threshold")
            if bt is not None and st is not None:
                thr_line = f" Calibrate internally with thresholds BUY≥{float(bt):.1f} and SELL≤{float(st):.1f}."
        except Exception:
            thr_line = ""

    # Keep it ultra-compact to minimize tokens
    return (
        f"{ctx_line}Score the opportunity for '{sym}' on a 0-10 scale {focus}{thr_line} "
        f"Output ONLY the number (0-10). No text, no code, no units."
    )
