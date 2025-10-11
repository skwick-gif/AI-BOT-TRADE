from dataclasses import dataclass

@dataclass
class SymbolSnapshot:
    symbol: str
    price: float
    iv: float | None
    trend: str | None
    sentiment: str | None
