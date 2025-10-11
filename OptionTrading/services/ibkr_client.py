from __future__ import annotations
from typing import List, Dict, Any, Optional
import time, math

try:
    from ib_insync import IB, Stock, Option, Contract, ComboLeg, ContractDescription, Order, LimitOrder, MarketOrder
except Exception as e:
    IB = None

class IBKRClient:
    def __init__(self, host: str, port: int, clientId: int, paper: bool = True):
        self.host = host; self.port = port; self.clientId = clientId; self.paper = paper
        self.ib: Optional[IB] = None
        # pacing controls
        self.min_delay = 0.2  # base wait between calls
        self.last_call_ts = 0.0

    # ---------- connectivity ----------
    def connect(self):
        if IB is None:
            raise RuntimeError("ib_insync not installed")
        if self.ib and self.ib.isConnected():
            return
        self.ib = IB()
        self.ib.connect(self.host, self.port, clientId=self.clientId)

    def _pace(self, mult: float = 1.0):
        now = time.time()
        to_wait = self.min_delay * mult - max(0.0, now - self.last_call_ts)
        if to_wait > 0:
            time.sleep(to_wait)
        self.last_call_ts = time.time()

    # ---------- contracts & market data ----------
    def _stock(self, symbol: str, primaryExchange: str = "SMART", currency: str = "USD") -> Stock:
        return Stock(symbol, primaryExchange, currency)

    def _option(self, symbol: str, lastTradeDateOrContractMonth: str, strike: float, right: str, exchange: str = "SMART", currency: str = "USD") -> Option:
        # expiry expected as YYYYMMDD
        return Option(symbol, lastTradeDateOrContractMonth, float(strike), right, exchange, currency)

    def qualify_stock(self, symbol: str) -> Stock:
        self.connect()
        con = self._stock(symbol)
        [q] = self.ib.qualifyContracts(con)
        return q

    def qualify_options(self, opts: List[Option]) -> List[Option]:
        self.connect()
        qs = self.ib.qualifyContracts(*opts)
        return qs

    def fetch_stock_snapshot(self, symbol: str) -> Dict[str, Any]:
        self.connect()
        stk = self.qualify_stock(symbol)
        self._pace()
        t = self.ib.reqMktData(stk, "", False, False)
        self.ib.sleep(0.5)
        price = None
        if t.last is not None: price = float(t.last)
        elif t.close is not None: price = float(t.close)
        elif t.marketPrice() is not None: price = float(t.marketPrice())
        return {"symbol": symbol, "conId": getattr(stk, "conId", None), "price": price}

    def get_secdef_params(self, symbol: str) -> Dict[str, Any]:
        # Pull strikes/expirations for USD/SMART
        self.connect()
        descs = self.ib.reqContractDetails(self._option(symbol, "00000000", 0.0, "C"))
        # Extract unique expirations/strikes
        expirations, strikes = set(), set()
        for d in descs:
            try:
                sd = d.secIdList or []
                expirations.update(d.contract.lastTradeDateOrContractMonth for _ in [0] if d.contract.lastTradeDateOrContractMonth)
                strikes.add(float(d.contract.strike))
            except Exception:
                pass
        return {"expirations": sorted(expirations), "strikes": sorted(strikes), "multiplier": 100}

    # ---------- combo build & place ----------
    def build_combo_from_ui_legs(self, symbol: str, ui_legs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Translate UI legs [{pos,strike,qty,expiry}] to (options[], comboLegs[]).
        pos: 'Long Call' / 'Short Put' etc. expiry: YYYYMMDD. right inferred from pos.
        sign: Long=+1, Short=-1
        """
        self.connect()
        options: List[Option] = []
        legs: List[ComboLeg] = []
        for leg in ui_legs:
            pos = str(leg.get("pos",""))
            right = "C" if "Call" in pos else "P"
            sign = +1 if "Long" in pos else -1
            k = float(leg.get("strike"))
            e = str(leg.get("expiry"))
            qty = int(leg.get("qty",1))
            opt = self._option(symbol, e, k, right)
            [q] = self.ib.qualifyContracts(opt)
            options.append(q)
            cl = ComboLeg()
            cl.conId = q.conId
            cl.ratio = abs(qty)
            cl.action = "BUY" if sign > 0 else "SELL"
            cl.exchange = q.exchange or "SMART"
            legs.append(cl)
            self._pace(1.2)
        return {"options": options, "combo_legs": legs}

    def place_combo_order(self, symbol: str, ui_legs: List[Dict[str, Any]], pricing: str = "MID", limit_price: Optional[float] = None, tif: str = "DAY", qty: int = 1) -> Dict[str, Any]:
        self.connect()
        built = self.build_combo_from_ui_legs(symbol, ui_legs)
        bag = Contract()
        bag.symbol = symbol
        bag.secType = "BAG"
        bag.currency = "USD"
        bag.exchange = "SMART"
        bag.comboLegs = built["combo_legs"]
        # order
        if pricing == "LIMIT" and (limit_price is not None):
            order = LimitOrder("BUY", qty, float(limit_price))
        else:
            # For combos, a MarketOrder may be risky; prefer Limit based on mids.
            order = MarketOrder("BUY", qty) if pricing in ("MARK","LAST") else LimitOrder("BUY", qty, float(limit_price or 0.0))
        order.tif = tif
        trade = self.ib.placeOrder(bag, order)
        # wait a bit for an orderId/status
        self.ib.sleep(0.2)
        return {"orderId": getattr(trade, "order", None).orderId if getattr(trade,"order",None) else None, "status": getattr(trade, "orderStatus", None).status if getattr(trade,"orderStatus",None) else "Submitted"}
