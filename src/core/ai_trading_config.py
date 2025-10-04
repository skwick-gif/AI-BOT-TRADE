"""
AI Trading configuration manager
Persists global settings, profiles/strategies, and assets to JSON under config/ai_trading.json
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class Strategy:
    name: str
    buy_threshold: float = 8.0
    sell_threshold: float = 4.0
    hysteresis: int = 2
    cooldown_min: int = 3
    sl_pct: float = 3.0
    tp_pct: float = 6.0
    interval: str = "1m"  # 1m,2m,5m,15m,60m


@dataclass
class AssetEntry:
    symbol: str
    quantity: int = 100
    strategy: str = "Intraday"
    use_global_interval: bool = True
    custom_interval: str = "1m"
    enabled: bool = False


@dataclass
class GlobalSettings:
    trading_hours_only: bool = True
    mode: str = "Paper"  # Paper/Live
    interval: str = "1m"


@dataclass
class AiTradingConfig:
    globals: GlobalSettings = field(default_factory=GlobalSettings)
    strategies: Dict[str, Strategy] = field(default_factory=dict)
    assets: List[AssetEntry] = field(default_factory=list)


class AiTradingConfigManager:
    def __init__(self, path: Optional[Path] = None):
        project_root = Path(__file__).resolve().parents[2]
        self.path = path or (project_root / "config" / "ai_trading.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_or_init()

    def _load_or_init(self) -> AiTradingConfig:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return self._from_dict(data)
            except Exception:
                pass
        # default
        cfg = AiTradingConfig()
        # built-in strategies (act like profiles)
        builtin = [
            Strategy("Intraday", 8.0, 4.0, 2, 3, 3.0, 6.0, "1m"),
            Strategy("Swing", 7.0, 5.0, 3, 10, 5.0, 10.0, "15m"),
            Strategy("Long-term", 6.5, 5.5, 3, 60, 8.0, 20.0, "60m"),
        ]
        cfg.strategies = {s.name: s for s in builtin}
        self._save(cfg)
        return cfg

    def _from_dict(self, data: Dict[str, Any]) -> AiTradingConfig:
        g = data.get("globals", {})
        globals_cfg = GlobalSettings(
            trading_hours_only=bool(g.get("trading_hours_only", True)),
            mode=str(g.get("mode", "Paper")),
            interval=str(g.get("interval", "1m")),
        )
        strategies_dict = {}
        for name, s in (data.get("strategies", {}) or {}).items():
            strategies_dict[name] = Strategy(
                name=name,
                buy_threshold=float(s.get("buy_threshold", 8.0)),
                sell_threshold=float(s.get("sell_threshold", 4.0)),
                hysteresis=int(s.get("hysteresis", 2)),
                cooldown_min=int(s.get("cooldown_min", 3)),
                sl_pct=float(s.get("sl_pct", 3.0)),
                tp_pct=float(s.get("tp_pct", 6.0)),
                interval=str(s.get("interval", "1m")),
            )
        assets = []
        for a in (data.get("assets", []) or []):
            assets.append(AssetEntry(
                symbol=str(a.get("symbol", "")),
                quantity=int(a.get("quantity", 100)),
                strategy=str(a.get("strategy", "Intraday")),
                use_global_interval=bool(a.get("use_global_interval", True)),
                custom_interval=str(a.get("custom_interval", "1m")),
                enabled=bool(a.get("enabled", False)),
            ))
        return AiTradingConfig(globals=globals_cfg, strategies=strategies_dict, assets=assets)

    def _save(self, cfg: Optional[AiTradingConfig] = None) -> None:
        cfg = cfg or self.config
        data = {
            "globals": asdict(cfg.globals),
            "strategies": {k: asdict(v) for k, v in cfg.strategies.items()},
            "assets": [asdict(a) for a in cfg.assets],
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Public API
    def save(self) -> None:
        self._save(self.config)

    def list_strategies(self) -> List[str]:
        return list(self.config.strategies.keys())

    def get_strategy(self, name: str) -> Optional[Strategy]:
        return self.config.strategies.get(name)

    def upsert_strategy(self, s: Strategy) -> None:
        self.config.strategies[s.name] = s
        self.save()

    def delete_strategy(self, name: str) -> None:
        if name in self.config.strategies:
            del self.config.strategies[name]
            # also detach assets referencing it (fallback to Intraday)
            for a in self.config.assets:
                if a.strategy == name:
                    a.strategy = "Intraday"
            self.save()

    def add_asset(self, entry: AssetEntry) -> None:
        # coalesce by symbol if exists (update)
        for a in self.config.assets:
            if a.symbol.upper() == entry.symbol.upper():
                a.quantity = entry.quantity
                a.strategy = entry.strategy
                a.use_global_interval = entry.use_global_interval
                a.custom_interval = entry.custom_interval
                self.save()
                return
        self.config.assets.append(entry)
        self.save()

    def remove_asset(self, symbol: str) -> None:
        self.config.assets = [a for a in self.config.assets if a.symbol.upper() != symbol.upper()]
        self.save()

    def set_asset_enabled(self, symbol: str, enabled: bool) -> None:
        for a in self.config.assets:
            if a.symbol.upper() == symbol.upper():
                a.enabled = enabled
                break
        self.save()

    def update_globals(self, *, trading_hours_only: Optional[bool] = None, mode: Optional[str] = None, interval: Optional[str] = None) -> None:
        if trading_hours_only is not None:
            self.config.globals.trading_hours_only = bool(trading_hours_only)
        if mode is not None:
            self.config.globals.mode = str(mode)
        if interval is not None:
            self.config.globals.interval = str(interval)
        self.save()
