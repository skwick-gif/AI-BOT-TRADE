from __future__ import annotations
from typing import Dict, Any, List
import json


class StrategyDSL:
    """Lightweight Strategy DSL loader used by the OPTIONS tab.

    Expects a JSON mapping { strategy_key: { legs: [...], params: [...] } }.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = json.load(f)

    def list(self) -> List[str]:
        return list(self._data.keys())

    def spec(self, key: str) -> Dict[str, Any]:
        if key not in self._data:
            raise KeyError(f"Strategy '{key}' not in DSL file: {self.path}")
        return self._data[key]

    def build_ui_legs(self, key: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return UI legs list dicts: {pos,strike,qty,premium,expiry} using provided params.
        Params keys are placeholders like K1,K2,E1,E_near,E_far.
        """
        spec = self.spec(key)
        legs_out: List[Dict[str, Any]] = []
        for leg in spec.get("legs", []):
            strike_token = str(leg.get("strike"))
            expiry_token = str(leg.get("expiry"))
            strike = float(params.get(strike_token, 0))
            expiry = str(params.get(expiry_token, ""))
            qty = int(leg.get("qty", 1))
            sign = int(leg.get("sign", 1))
            pos = leg.get("pos", "Long Call")
            right = leg.get("right", "C")
            if sign > 0 and pos.startswith("Short"):
                pos = pos.replace("Short", "Long")
            if sign < 0 and pos.startswith("Long"):
                pos = pos.replace("Long", "Short")
            legs_out.append({"pos": pos, "right": right, "strike": strike, "qty": qty, "premium": None, "expiry": expiry})
        return legs_out
