from dataclasses import dataclass
from typing import List, Optional, Literal
from .strategy import StrategyLeg

@dataclass
class OrderTicket:
    symbol: str
    quantity: int
    pricing_mode: Literal["MID","LAST","MARK","LIMIT"]
    limit_price: Optional[float]
    tif: Literal["DAY","GTC"]
    slippage_bps: int
    legs: List[StrategyLeg]
    mode: Literal["PAPER","LIVE"]
