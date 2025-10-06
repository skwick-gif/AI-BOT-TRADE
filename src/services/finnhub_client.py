"""
Finnhub REST client for calendar-related endpoints.
Uses requests directly to avoid extra dependencies.
"""

from __future__ import annotations

import requests
from typing import Dict, Any, List, Optional

BASE_URL = "https://finnhub.io/api/v1"


def _get(url: str, params: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def fetch_earnings_calendar(
    api_key: str,
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch earnings calendar. If symbols provided, query per symbol to limit payload."""
    out: List[Dict[str, Any]] = []
    if symbols:
        for sym in symbols:
            params = {"from": start_date, "to": end_date, "symbol": sym, "token": api_key}
            try:
                data = _get(f"{BASE_URL}/calendar/earnings", params)
                items = data.get("earningsCalendar") or data.get("earnings", []) or []
                for it in items:
                    it["_source"] = "earnings"
                    out.append(it)
            except Exception:
                # Skip symbol on error
                continue
    else:
        params = {"from": start_date, "to": end_date, "token": api_key}
        data = _get(f"{BASE_URL}/calendar/earnings", params)
        items = data.get("earningsCalendar") or data.get("earnings", []) or []
        for it in items:
            it["_source"] = "earnings"
        out.extend(items)
    return out


def fetch_dividends(
    api_key: str,
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch dividends per symbol within date range."""
    out: List[Dict[str, Any]] = []
    if not symbols:
        return out
    for sym in symbols:
        params = {"symbol": sym, "from": start_date, "to": end_date, "token": api_key}
        try:
            data = _get(f"{BASE_URL}/stock/dividend", params)
            # data is a list of dividends
            for it in data or []:
                it["symbol"] = sym
                it["_source"] = "dividend"
                out.append(it)
        except Exception:
            continue
    return out


def fetch_economic_calendar(api_key: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetch economic calendar if permitted by plan; otherwise return empty on error."""
    params = {"from": start_date, "to": end_date, "token": api_key}
    try:
        data = _get(f"{BASE_URL}/calendar/economic", params)
        # Finnhub returns { economicCalendar: [ ... ] }
        items = data.get("economicCalendar") or data.get("economic", []) or []
        for it in items:
            it["_source"] = "economic"
        return items
    except Exception:
        return []


def fetch_ipo_calendar(
    api_key: str,
    start_date: str,
    end_date: str,
    us_only: bool = True,
) -> List[Dict[str, Any]]:
    """Fetch IPO calendar and optionally filter to US exchanges only."""
    params = {"from": start_date, "to": end_date, "token": api_key}
    try:
        data = _get(f"{BASE_URL}/calendar/ipo", params)
        items = data.get("ipoCalendar") or data.get("ipo", []) or []
        us_exchanges = {
            "NASDAQ",
            "NYSE",
            "AMEX",
            "NYSE American",
            "NYSE Arca",
            "BATS",
        }
        out: List[Dict[str, Any]] = []
        for it in items:
            it["_source"] = "ipo"
            ex = (it.get("exchange") or it.get("market") or "").upper()
            if (not us_only) or (ex in us_exchanges):
                out.append(it)
        return out
    except Exception:
        return []
