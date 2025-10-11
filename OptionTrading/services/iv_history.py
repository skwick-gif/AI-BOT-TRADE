from __future__ import annotations
from typing import Dict, List
import json, os, time

class IVHistoryStore:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, List[float]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                self.data = json.load(open(self.path, "r", encoding="utf-8"))
            except Exception:
                self.data = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def update(self, symbol: str, iv_value: float, keep_last: int = 365):
        arr = self.data.get(symbol, [])
        arr.append(float(iv_value))
        if len(arr) > keep_last:
            arr = arr[-keep_last:]
        self.data[symbol] = arr
        self.save()
