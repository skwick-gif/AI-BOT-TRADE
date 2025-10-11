from dataclasses import dataclass
from typing import List, Literal

@dataclass
class Recommendation:
    key: str
    name: str
    confidence: float
    blurb: str
    timeframe: str
    expected_profit: float
    max_loss: float
    success_prob: float

@dataclass
class StrategyLeg:
    pos: Literal["Long Call","Short Call","Long Put","Short Put"]
    strike: float
    qty: int
    premium: float | None
    expiry: str  # 'YYYYMMDD'

@dataclass
class StrategyDetails:
    name: str
    current_price: float
    confidence: float
    expected_profit: float
    max_loss: float
    success_prob: float
    risk_reward: float
    legs: List[StrategyLeg]
    delta: float
    gamma: float
    theta: float
    vega: float
