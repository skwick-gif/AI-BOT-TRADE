from __future__ import annotations
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit, QStackedWidget
from PyQt6.QtCore import Qt

from services.perplexity_options import PerplexityOptionsClient
from services.strategies_dsl import StrategyDSL
from ui.widgets.options.recommendations_view import RecommendationsView
from ui.widgets.options.strategy_details_view import StrategyDetailsView
from ui.dialogs.order_ticket_dialog import OrderTicketDialog
from ui.dialogs.option_chain_selector import OptionChainSelectorDialog
from services.ibkr_service import IBKRService
from services.options_pricing import price_greeks


class OptionsMainPanel(QWidget):
    """MAIN sub-tab: search bar + Analyze -> list of recommendations -> strategy details -> order ticket."""

    def __init__(self, strategies_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("OptionsMainPanel")

        try:
            self.ai = PerplexityOptionsClient()
        except Exception:
            self.ai = None

        # Strategy DSL
        self._strategies_path = strategies_path
        try:
            self.dsl = StrategyDSL(strategies_path)
        except Exception:
            self.dsl = None

        # View 1: search bar
        self.symbol_edit = QLineEdit(self)
        self.symbol_edit.setPlaceholderText("Enter symbol, e.g., SPY")
        self.analyze_btn = QPushButton("Analyze", self)

        search_row = QHBoxLayout()
        search_row.addWidget(self.symbol_edit, 1)
        search_row.addWidget(self.analyze_btn)

        # View 2: recommendations list
        self.reco_view = RecommendationsView(self)
        # View 3: strategy details
        self.details_view = StrategyDetailsView(self)

        # stack
        self.stack = QStackedWidget(self)
        self.page_search = QWidget(self)
        ps_layout = QVBoxLayout(self.page_search)
        ps_layout.addLayout(search_row)
        ps_layout.addStretch(1)
        self.stack.addWidget(self.page_search)  # index 0
        self.stack.addWidget(self.reco_view)    # index 1
        self.stack.addWidget(self.details_view) # index 2

        layout = QVBoxLayout(self)
        layout.addWidget(self.stack, 1)
        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.status_lbl)

        # state
        self._current_symbol = None  # type: Optional[str]
        self._current_price = None   # type: Optional[float]
        self._current_strategy = None  # type: Optional[dict]
        self._current_params = {}  # type: Dict[str, Any]
        self._ibkr: Optional[IBKRService] = None
        # Keep last computed legs/kpis for pre-execute confirmation
        self._last_legs: list[dict] | None = None
        self._last_kpis: dict | None = None

        # signals
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.reco_view.strategySelected.connect(self._on_select_strategy)
        self.reco_view.backRequested.connect(self._show_recommendations)
        self.details_view.backRequested.connect(self._show_recommendations)
        self.details_view.executeRequested.connect(self._on_execute)
        self.details_view.selectChainRequested.connect(self._on_select_chain)

    def _on_analyze(self) -> None:
        sym = (self.symbol_edit.text() or "").strip().upper()
        if not sym:
            return
        self._current_symbol = sym
        # Prefer IBKR price, fallback to local parquet
        price = None
        ib = self._get_ibkr()
        if ib and ib.is_connected():
            price = ib.get_last_price(sym)
        if price is None:
            price = self._load_current_price(sym)
        if price is None:
            price = 178.45
        self._current_price = float(price)
        # call AI
        strategies = []  # type: list[dict]
        if self.ai:
            try:
                out = self.ai.analyze_symbol(sym, {"price": self._current_price}, None)
                strategies = list(out.get("strategies", []))
            except Exception:
                strategies = []
        else:
            strategies = []

        # demo fallback if no strategies
        if not strategies:
            strategies = [
                {"key": "bull_call_spread", "name": "Bull Call Spread", "blurb": "Strong bullish momentum with controlled risk", "confidence": 0.85, "expected_profit": 850, "max_loss": 450, "success_prob": 0.68, "timeframe": "30-45 days"},
                {"key": "iron_condor", "name": "Iron Condor", "blurb": "Price consolidation expected in range", "confidence": 0.72, "expected_profit": 420, "max_loss": 580, "success_prob": 0.65, "timeframe": "20-30 days"},
                {"key": "long_strangle", "name": "Long Strangle", "blurb": "High volatility expected, direction uncertain", "confidence": 0.65, "expected_profit": 1200, "max_loss": 800, "success_prob": 0.45, "timeframe": "15-30 days"},
            ]
        # show recommendations list
        self.reco_view.set_analysis_context(current_price=self._current_price)
        self.reco_view.set_recommendations(strategies)
        self.stack.setCurrentIndex(1)

    def _on_select_strategy(self, strategy: dict) -> None:
        self._current_strategy = dict(strategy)
        # Build legs from DSL with defaults, fallback to examples
        legs = self._build_legs_for_strategy(strategy)
        self.details_view.set_context(current_price=self._current_price)
        self.details_view.set_strategy(strategy, legs)
        self.stack.setCurrentIndex(2)

    def _on_execute(self, payload: dict) -> None:
        # Before opening an order ticket, show a pre-execute confirmation with KPIs
        from ui.dialogs.pre_execute_confirm import PreExecuteConfirmDialog

        strat_name = self._current_strategy.get("name") if self._current_strategy else "Strategy"
        # Use last computed legs/kpis if available, else build fresh
        legs = self._last_legs or (self._build_legs_for_strategy(self._current_strategy) if self._current_strategy else [])
        kpis = self._last_kpis

        confirm = PreExecuteConfirmDialog(kpis=kpis, legs=legs, parent=self)
        if not confirm.exec():
            self.status_lbl.setText("Execution cancelled by user.")
            return
        res = confirm.result()
        dry_run = res.get('dry_run', True)

        # Open order ticket dialog to collect order params (quantity/limit etc.)
        dlg = OrderTicketDialog(symbol=self._current_symbol or "", strategy_name=strat_name, parent=self)
        if dlg.exec():
            params = dlg.result_payload() or {}
            ib = self._get_ibkr()
            if dry_run:
                self.status_lbl.setText("Order submitted (simulated - dry run).")
                return

            # If not dry run, require IBKR connection
            if not (ib and ib.is_connected() and self._current_strategy):
                self.status_lbl.setText("IBKR not connected. Cannot send real order.")
                return

            # Build leg specs from the legs used for pricing
            leg_specs = []
            est_total = 0.0
            for leg in legs:
                side = 'BUY' if 'Long' in leg.get('pos', '') else 'SELL'
                right = str(leg.get('right', 'C'))
                strike = float(leg.get('strike', 0) or 0)
                expiry = str(leg.get('expiry', ''))
                qty = int(leg.get('qty', 1))
                # Use market mid if we have it, else try fetching now
                mid = leg.get('premium')
                if mid is None:
                    try:
                        mid = ib.get_option_mid_price(self._current_symbol or '', right, strike, expiry)
                    except Exception:
                        mid = None
                if mid is not None:
                    est_total += (float(mid) * qty) * (1 if side == 'BUY' else -1)
                leg_specs.append({'right': right, 'strike': strike, 'expiry': expiry, 'ratio': qty, 'side': side})

            limit_price = float(params.get('limit', 0.0))
            if limit_price <= 0 and est_total != 0:
                cushion = 0.03
                if est_total > 0:
                    limit_price = est_total + cushion
                else:
                    limit_price = max(0.01, est_total - cushion)
            if limit_price <= 0:
                limit_price = 0.05

            try:
                trade = ib.place_option_combo_limit(
                    symbol=self._current_symbol or '',
                    legs_specs=leg_specs,
                    total_qty=max(1, int(params.get('quantity', 1))),
                    limit_price=limit_price,
                    action='BUY' if limit_price > 0 else 'SELL'
                )
                if trade:
                    self.status_lbl.setText(f"Order submitted to IBKR (combo) at limit {limit_price:.2f}.")
                    return
            except Exception as e:
                self.status_lbl.setText(f"IBKR order failed: {e}")
                return
            self.status_lbl.setText("Order submitted (simulated). IBKR not connected.")

    def _on_select_chain(self) -> None:
        # Try to pre-fill from IBKR secdef and show dialog with populated choices
        prefill = {}
        expirations = None
        strikes = None
        ib = self._get_ibkr()
        if ib and ib.is_connected() and self._current_symbol:
            try:
                sec = ib.get_option_secdef_params(self._current_symbol)
                expirations = sec.get('expirations') or None
                strikes = sec.get('strikes') or None
                if expirations:
                    prefill['E1'] = sorted(expirations)[0]
                if strikes and self._current_price:
                    ks = list(strikes)
                    k_atm = min(ks, key=lambda k: abs(k - self._current_price))
                    k_above = min([k for k in ks if k >= k_atm] or [k_atm])
                    prefill['K1'] = k_atm
                    prefill['K2'] = k_above
            except Exception:
                pass

        # Determine how many leg inputs the chain selector should show by previewing the strategy legs
        leg_preview_count = 2
        try:
            if self._current_strategy:
                preview_legs = self._build_legs_for_strategy(self._current_strategy)
                leg_preview_count = max(1, len(preview_legs))
        except Exception:
            preview_legs = None

        dlg = OptionChainSelectorDialog(self)
        dlg.configure_leg_count(leg_preview_count)
        if expirations or strikes:
            dlg.set_choices(expirations, strikes, leg_count=leg_preview_count)
        if prefill.get('E1'):
            # set in combo
            idx = dlg.expiry_combo.findText(str(prefill['E1']))
            if idx >= 0:
                dlg.expiry_combo.setCurrentIndex(idx)
            else:
                dlg.expiry_combo.setEditText(str(prefill['E1']))
        # Map prefill K values to the available k_combos
        for i, combo in enumerate(dlg.k_combos, start=1):
            key = f'K{i}'
            if prefill.get(key) is not None:
                txt = str(prefill[key])
                idx = combo.findText(txt)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                if idx < 0:
                    combo.setEditText(txt)
        if dlg.exec():
            vals = dlg.values()
            if self._current_strategy:
                self._current_params.update({k: v for k, v in vals.items() if v})
                key = (self._current_strategy.get("key") or "").lower()
                if self.dsl and key:
                    try:
                        legs = self.dsl.build_ui_legs(key, self._current_params)
                    except Exception:
                        legs = self._build_example_legs(self._current_strategy)
                else:
                    legs = self._build_example_legs(self._current_strategy)

                # If IBKR connected, try to fetch market mid for each leg and attach as 'premium'
                ib = self._get_ibkr()
                if ib and ib.is_connected():
                    for leg in legs:
                        try:
                            right = str(leg.get('right', 'C'))
                            strike = float(leg.get('strike', 0) or 0)
                            expiry = str(leg.get('expiry', ''))
                            mid = ib.get_option_mid_price(self._current_symbol or '', right, strike, expiry)
                            if mid is not None:
                                # store market mid as premium for pricing/Greeks
                                leg['premium'] = float(mid)
                        except Exception:
                            # ignore failures per-leg
                            continue

                # Compute Greeks/pricing using available premiums (market mids if fetched)
                try:
                    kpis = price_greeks(legs, float(self._current_price or 0.0))
                except Exception:
                    kpis = None

                # Prefer the detailed legs returned by the pricing function (which include computed price/iv)
                legs_to_show = legs
                try:
                    if kpis and isinstance(kpis, dict) and 'legs' in kpis:
                        legs_to_show = kpis.get('legs') or legs
                except Exception:
                    legs_to_show = legs

                # Update UI: refresh legs table and show pricing KPIs when available
                self.details_view.set_strategy(self._current_strategy, legs_to_show)
                # Persist last computed legs/kpis for confirmation & execution
                try:
                    self._last_legs = legs_to_show
                    self._last_kpis = kpis
                except Exception:
                    pass
                if kpis:
                    try:
                        self.details_view.set_pricing_kpis(kpis)
                    except Exception:
                        pass

    def _show_recommendations(self) -> None:
        self.stack.setCurrentIndex(1)

    def _build_legs_for_strategy(self, strategy: dict) -> list[dict]:
        key = (strategy.get("key") or "").lower()
        # Seed default params
        self._current_params = self._default_params_for_strategy(key)
        if self.dsl and key:
            try:
                return self.dsl.build_ui_legs(key, self._current_params)
            except Exception:
                pass
        return self._build_example_legs(strategy)

    def _default_params_for_strategy(self, key: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        px = float(self._current_price or 0)
        k_atm = max(int(round(px)), 1)
        params.update({
            "K": k_atm,
            "K1": max(k_atm - 5, 1),
            "K2": k_atm + 5,
            "K3": k_atm + 10,
            "K_put": max(k_atm - 5, 1),
            "K_call": k_atm + 5,
            "K_put_low": max(k_atm - 10, 1),
            "K_put_high": max(k_atm - 5, 1),
            "K_call_low": k_atm + 5,
            "K_call_high": k_atm + 10,
            "K_short": k_atm + 5,
            "K_long": k_atm + 10,
            "E1": "20251130",
            "E_near": "20251115",
            "E_far": "20251220",
        })
        return params

    def _build_example_legs(self, strategy: dict) -> list[dict]:
        # Until OptionChainSelector + live chain are wired, create example legs matching the screenshots.
        key = (strategy.get("key") or "").lower()
        if "bull_call" in key:
            return [
                {"pos": "Long Call", "strike": 180, "qty": 2, "premium": 5.5, "expiry": "Nov 30, 2025"},
                {"pos": "Short Call", "strike": 190, "qty": 2, "premium": 2.2, "expiry": "Nov 30, 2025"},
            ]
        elif "iron_condor" in key:
            return [
                {"pos": "Short Put", "strike": 170, "qty": 1, "premium": 1.8, "expiry": "Nov 30, 2025"},
                {"pos": "Long Put", "strike": 165, "qty": 1, "premium": 1.0, "expiry": "Nov 30, 2025"},
                {"pos": "Short Call", "strike": 185, "qty": 1, "premium": 1.7, "expiry": "Nov 30, 2025"},
                {"pos": "Long Call", "strike": 190, "qty": 1, "premium": 1.1, "expiry": "Nov 30, 2025"},
            ]
        else:
            return [
                {"pos": "Long Call", "strike": 105, "qty": 1, "premium": 3.0, "expiry": "Nov 30, 2025"},
                {"pos": "Long Put", "strike": 95, "qty": 1, "premium": 2.8, "expiry": "Nov 30, 2025"},
            ]

    def _load_current_price(self, symbol: str) -> Optional[float]:
        # Prefer local parquet in data/bronze/daily, then stock_data, else optional yfinance via helper
        try:
            from pathlib import Path
            import pandas as pd
            fp = Path("data/bronze/daily") / f"{symbol}.parquet"
            if fp.exists():
                df = pd.read_parquet(fp)
                if not df.empty:
                    row = df.iloc[-1]
                    if 'close' in row.index and pd.notna(row['close']):
                        return float(row['close'])
                    if 'adj_close' in row.index and pd.notna(row['adj_close']):
                        return float(row['adj_close'])
            root = Path("stock_data")
            if root.exists():
                for p in root.rglob("*"):
                    if not p.is_file():
                        continue
                    stem = p.stem.upper()
                    if stem == symbol or stem.startswith(symbol + "_"):
                        try:
                            if p.suffix.lower() == ".parquet":
                                df = pd.read_parquet(p)
                            elif p.suffix.lower() in (".csv", ".txt"):
                                df = pd.read_csv(p)
                            else:
                                continue
                            if not df.empty:
                                if 'date' in df.columns:
                                    df = df.sort_values('date')
                                row = df.iloc[-1]
                                if 'close' in row.index and pd.notna(row['close']):
                                    return float(row['close'])
                                if 'adj_close' in row.index and pd.notna(row['adj_close']):
                                    return float(row['adj_close'])
                        except Exception:
                            continue
        except Exception:
            pass
        try:
            from utils.trading_helpers import get_current_price as _gcp
            return _gcp(symbol)
        except Exception:
            return None

    def _get_ibkr(self) -> Optional[IBKRService]:
        """Locate the shared IBKR service from the MainWindow if available."""
        try:
            w = self.window()
            if hasattr(w, 'ibkr_service'):
                self._ibkr = getattr(w, 'ibkr_service', None)
        except Exception:
            pass
        return self._ibkr
