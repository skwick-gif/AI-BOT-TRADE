from dataclasses import dataclass

@dataclass
class Account:
    balance: float
    margin_used: float
    margin_available: float
