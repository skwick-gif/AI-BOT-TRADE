from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional
import requests
from pydantic import BaseModel, Field, ValidationError, model_validator

from core.config_manager import ConfigManager


class StrategyOut(BaseModel):
    key: str
    name: str
    blurb: str = ""
    confidence: float = 0.0
    expected_profit: float = 0.0
    max_loss: float = 0.0
    success_prob: float = 0.0
    timeframe: str = ""

    @model_validator(mode="after")
    def _bounds(self):
        self.confidence = float(max(0.0, min(1.0, float(self.confidence or 0.0))))
        self.success_prob = float(max(0.0, min(1.0, float(self.success_prob or 0.0))))
        self.expected_profit = float(max(0.0, float(self.expected_profit or 0.0)))
        self.max_loss = float(max(0.0, float(self.max_loss or 0.0)))
        if not self.key or not self.name:
            raise ValueError("Strategy 'key' and 'name' must be non-empty")
        return self


class AnalyzeOut(BaseModel):
    strategies: List[StrategyOut] = Field(default_factory=list)


class PerplexityOptionsClient:
    """Minimal client for options analysis via Perplexity returning structured JSON."""

    def __init__(self, config: Optional[ConfigManager] = None) -> None:
        self.config = config or ConfigManager()
        self.base_url = "https://api.perplexity.ai/chat/completions"
        if not self.config.perplexity.api_key:
            raise RuntimeError("Missing PERPLEXITY_API_KEY in environment/.env")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.perplexity.api_key}",
            "Content-Type": "application/json",
        }

    def analyze_symbol(self, symbol: str, snapshot: Dict[str, Any], template: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tpl = template or {
            "system": "You are an elite options strategist. Respond only in strict JSON for strategies.",
            "user": (
                "Analyze {symbol} based on snapshot {snapshot}. "
                "Recommend up to 3 option strategies with fields: key, name, blurb, confidence (0..1), "
                "expected_profit, max_loss, success_prob (0..1), timeframe. Return only JSON: {\"strategies\":[...]}"
            ),
            "temperature": 0.2,
            "max_tokens": 700,
        }
        system_prompt = tpl.get("system", "Respond only in JSON.")
        user_prompt = tpl.get("user", "").format(symbol=symbol, snapshot=json.dumps(snapshot))
        model_to_use = (
            self.config.perplexity.finance_model
            if getattr(self.config.perplexity, "force_finance", False)
            else self.config.perplexity.model
        )
        payload: Dict[str, Any] = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": tpl.get("temperature", 0.2),
            "max_tokens": tpl.get("max_tokens", 700),
            "stream": False,
        }
        # Optional finance filters
        try:
            if getattr(self.config.perplexity, "force_finance", False):
                domains = getattr(self.config.perplexity, "search_domains", None)
                if domains:
                    payload["search_domain_filter"] = [d.strip() for d in str(domains).split(",") if d.strip()]
                recency = getattr(self.config.perplexity, "search_recency", None)
                if recency:
                    payload["search_recency_filter"] = recency
        except Exception:
            pass

        resp = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        # Parse strictly with pydantic; handle fenced JSON gracefully
        try:
            parsed = AnalyzeOut.model_validate_json(text)
        except ValidationError:
            txt = text.strip().strip("`").strip()
            parsed = AnalyzeOut.model_validate_json(txt)

        out = parsed.model_dump()
        # Minimal audit log under logs/
        try:
            from pathlib import Path
            logd = Path("logs")
            logd.mkdir(parents=True, exist_ok=True)
            with (logd / "perplexity_options_analyze.log").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"symbol": symbol, "out": out}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return out
