import os, json, requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

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
        # clamp/validate ranges for probabilities etc.
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.success_prob = float(max(0.0, min(1.0, self.success_prob)))
        # non-negative P/L fields
        self.expected_profit = float(max(0.0, self.expected_profit))
        self.max_loss = float(max(0.0, self.max_loss))
        # require non-empty key/name
        if not self.key or not self.name:
            raise ValueError("Strategy 'key' and 'name' must be non-empty")
        return self

    key: str
    name: str
    blurb: str = ""
    confidence: float = 0.0
    expected_profit: float = 0.0
    max_loss: float = 0.0
    success_prob: float = 0.0
    timeframe: str = ""

class AnalyzeOut(BaseModel):
    strategies: List[StrategyOut] = Field(default_factory=list)

class PerplexityClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "PERPLEXITY_API_KEY"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = os.getenv(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"Missing API key in env var {api_key_env}")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=45)
        r.raise_for_status()
        return r.json()

        r = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=45)
        r.raise_for_status()
        return r.json()

    def analyze_symbol(self, symbol: str, snapshot: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = template.get("system", "You are Perplexity Finance. Respond only in JSON.")
        user_prompt = template.get("user", "").format(symbol=symbol, snapshot=json.dumps(snapshot))
        payload = {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": template.get("temperature", 0.2), "max_tokens": template.get("max_tokens", 700)}
        data = self._post(payload)
        text = data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        # strict validation with logging
        try:
            parsed = AnalyzeOut.model_validate_json(text)
        except ValidationError:
            txt = text.strip().strip('`').strip()
            parsed = AnalyzeOut.model_validate_json(txt)
        out = parsed.model_dump()
        try:
            # minimal audit log
            import os, time
            logp = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(logp, exist_ok=True)
            with open(os.path.join(logp, "perplexity_analyze.log"), "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} -> {json.dumps(out, ensure_ascii=False)}\n")
        except Exception:
            pass
        return out

        system_prompt = template.get("system", "You are Perplexity Finance. Respond only in JSON.")
        user_prompt = template.get("user", "").format(symbol=symbol, snapshot=json.dumps(snapshot))
        payload = {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": template.get("temperature", 0.2), "max_tokens": template.get("max_tokens", 700)}
        data = self._post(payload)
        text = data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        # strict validation
        try:
            parsed = AnalyzeOut.model_validate_json(text)
        except ValidationError:
            # try to coerce if response is not pure JSON (e.g., fenced code)
            txt = text.strip().strip('`').strip()
            parsed = AnalyzeOut.model_validate_json(txt)
        return parsed.model_dump()
