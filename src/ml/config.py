from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class FeatureTechnicalConfig:
    enabled: bool = True
    ema_windows: List[int] = field(default_factory=lambda: [5, 20])
    rsi_window: int = 14
    macd: List[int] = field(default_factory=lambda: [12, 26, 9])
    atr_window: int = 14
    bbands_window: int = 20
    ret_windows: List[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass
class TargetConfig:
    type: Literal["classification", "regression"] = "classification"
    tau: float = 0.005  # ±0.5% HOLD band for classification


@dataclass
class TrainingConfig:
    holdout_last_days: int = 30
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10])
    step_days: int = 5
    lookback_days: int = 500
    window_mode: Literal["expanding", "rolling"] = "expanding"
    pooled_mode: Literal["pooled", "per_ticker"] = "pooled"
    metric: str = "F1_macro"
    models: List[dict] = field(default_factory=lambda: [
        {"name": "RandomForest", "tuning_budget": 30},
        {"name": "XGBoost", "tuning_budget": 30},
        {"name": "LogisticRegression", "tuning_budget": 15},
    ])
    features: FeatureTechnicalConfig = field(default_factory=FeatureTechnicalConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
