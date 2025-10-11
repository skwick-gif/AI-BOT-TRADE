from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json, os

class StrategyDSL:
    def __init__(self, path: str) -> None:
        self.path = path
        self._data: Dict[str, Any] = json.load(open(path, "r", encoding="utf-8"))

    def list(self) -> List[str]:
        return list(self._data.keys())

    def spec(self, key: str) -> Dict[str, Any]:
        if key not in self._data:
            raise KeyError(f"Strategy '{key}' not in DSL file: {self.path}")
        return self._data[key]

    def build_ui_legs(self, key: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return UI legs list dicts: {pos,strike,qty,premium,expiry} using provided params (K1,K2,...,E1,E_near,E_far)."""
        spec = self.spec(key)
        legs_out: List[Dict[str, Any]] = []
        for leg in spec.get("legs", []):
            strike_token = str(leg.get("strike"))
            expiry_token = str(leg.get("expiry"))
            strike = float(params.get(strike_token, 0))
            expiry = str(params.get(expiry_token, ""))
            qty = int(leg.get("qty", 1))
            sign = int(leg.get("sign", 1))
            pos = leg.get("pos", "Long Call")  # display text
            # Convert sign to Long/Short if not aligned with pos text
            if sign > 0 and pos.startswith("Short"): pos = pos.replace("Short","Long")
            if sign < 0 and pos.startswith("Long"):  pos = pos.replace("Long","Short")
            legs_out.append({"pos": pos, "strike": strike, "qty": qty, "premium": None, "expiry": expiry})
        return legs_out
