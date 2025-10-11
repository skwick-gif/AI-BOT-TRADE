from __future__ import annotations
from typing import List
import json
import os
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QGroupBox, QGridLayout, QMessageBox, QMenuBar, QStatusBar, QToolBar, QComboBox
from PyQt6.QtCore import Qt
from models.strategy import Recommendation, StrategyLeg, StrategyDetails
from models.ticket import OrderTicket
from ui.strategy_details_panel import StrategyDetailsPanel
from ui.scanner_panel import ScannerPanel
from ui.order_ticket_dialog import OrderTicketDialog, StrategyLeg as TicketLeg, StrategyDetails as TicketDetails
from ui.orders_panel import OrdersPanel, OrderRow
from ui.order_details_dialog import OrderDetailsDialog
from ui.learning_center import LearningCenter
from ui.prompt_manager import PromptManager
from ui.option_chain_selector import OptionChainSelectorDialog
from services.ibkr_client import IBKRClient
from services.perplexity_client import PerplexityClient
from services.strategies import StrategyDSL
from services.market_scanner import HybridScanner
from services.local_vector_scanner import LocalVectorScanner
from services.iv_history import IVHistoryStore


class TradingWindow(QMainWindow):
    def __init__(self, offline: bool = False):
        super().__init__()
        self.setWindowTitle("Options Trading — PyQt6 (Real Services)")
        self.resize(1150, 820)

        # Menu
        menu = QMenuBar(self)
        file_menu = menu.addMenu("File")
        edit_menu = menu.addMenu("Edit")
        view_menu = menu.addMenu("View")
        about_menu = menu.addMenu("About")
        self.setMenuBar(menu)

        # Top health toolbar (visual only)
        top_tb = QToolBar("health", self)
        top_tb.setObjectName("topHealthBar")
        top_tb.setMovable(False)
        self._lbl_conn_dot = QLabel()
        self._lbl_conn_dot.setFixedSize(8, 8)
        self._lbl_conn_dot.setStyleSheet("background:#16a34a;border-radius:4px;")
        top_tb.addWidget(self._lbl_conn_dot)
        top_tb.addWidget(QLabel(" IBKR / Perplexity "))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Paper", "Live"])
        self._mode_combo.setCurrentText("Paper")
        top_tb.addWidget(self._mode_combo)
        self.addToolBar(top_tb)

        # Wire menus
        from PyQt6.QtGui import QAction
        act_learn = QAction("Learning Center", self)
        act_learn.triggered.connect(lambda: self._open_learning_center())
        about_menu.addAction(act_learn)
        act_prompts = QAction("Prompt Templates", self)
        act_prompts.triggered.connect(lambda: self._open_prompt_manager())
        edit_menu.addAction(act_prompts)

        # Status
        status = QStatusBar(self)
        status.setObjectName("bottomStatus")
        self.lbl_conn = QLabel("Disconnected")
        status.addWidget(self.lbl_conn)
        status.addPermanentWidget(QLabel("PAPER"))
        self.setStatusBar(status)

        # Central area
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Header with title + main tabs (visual only)
        header_widget = QWidget(self)
        header_widget.setObjectName("headerTabs")
        hh = QHBoxLayout(header_widget)
        hh.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Options Trading")
        title.setStyleSheet("font-weight:600; margin-right:12px;")
        hh.addWidget(title)
        tabs = ["Dashboard", "Stocks", "Options", "ML", "Bot Trading"]
        for t in tabs:
            b = QPushButton(t)
            b.setProperty("header", True)
            if t == "Options":
                b.setProperty("selected", True)
            hh.addWidget(b)
        hh.addStretch(1)
        root.addWidget(header_widget)

        # Secondary tabs
        secondary = QWidget(self)
        secondary.setObjectName("secondaryTabs")
        sh = QHBoxLayout(secondary)
        sh.setContentsMargins(6, 4, 6, 4)
        for s in ["Options", "Analyze", "Bank"]:
            sb = QPushButton(s)
            sb.setProperty("secondary", True)
            if s == "Analyze":
                sb.setProperty("selected", True)
            sh.addWidget(sb)
        sh.addStretch(1)
        root.addWidget(secondary)

        # Analyze Box
        analyze_box = QGroupBox("Find Trading Opportunity")
        analyze_box.setObjectName("analyzeCard")
        ab = QHBoxLayout(analyze_box)
        lbl_search = QLabel("🔍")
        lbl_search.setFixedWidth(24)
        lbl_search.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ab.addWidget(lbl_search)
        self.ed_symbol = QLineEdit()
        self.ed_symbol.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
        ab.addWidget(self.ed_symbol, 1)
        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setObjectName("analyzeButton")
        self.btn_analyze.setProperty("primary", True)
        self.btn_analyze.clicked.connect(self._on_analyze)
        if offline:
            self.btn_analyze.setDisabled(True)
            self.lbl_conn.setText("Offline")
            self._lbl_conn_dot.setStyleSheet("background:#94a3b8;border-radius:4px;")
        ab.addWidget(self.btn_analyze)
        root.addWidget(analyze_box)

        # Summary
        self.box_summary = QGroupBox("Analysis")
        self.box_summary.setVisible(False)
        g = QGridLayout(self.box_summary)
        self.lbl_sent = QLabel("")
        g.addWidget(QLabel("Sentiment"), 0, 0)
        g.addWidget(self.lbl_sent, 0, 1)
        self.lbl_iv = QLabel("")
        g.addWidget(QLabel("Implied Volatility"), 0, 2)
        g.addWidget(self.lbl_iv, 0, 3)
        self.lbl_trend = QLabel("")
        g.addWidget(QLabel("Trend"), 1, 0)
        g.addWidget(self.lbl_trend, 1, 1)
        self.lbl_price = QLabel("")
        g.addWidget(QLabel("Current Price"), 1, 2)
        g.addWidget(self.lbl_price, 1, 3)
        root.addWidget(self.box_summary)

        # Recommendations
        self.box_recs = QGroupBox("Recommended Strategies")
        self.box_recs.setVisible(False)
        v = QVBoxLayout(self.box_recs)
        self.tbl_recs = QTableWidget(0, 6)
        self.tbl_recs.setHorizontalHeaderLabels(["Name", "Confidence", "Blurb", "Timeframe", "Profit/Loss", "Action"]) 
        self.tbl_recs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.tbl_recs)
        root.addWidget(self.box_recs)

        # Details panel
        self.details_panel = StrategyDetailsPanel()
        self.details_panel.setVisible(False)
        self.details_panel.requestOrderTicket.connect(self._open_ticket)
        root.addWidget(self.details_panel)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # Scanner + Orders panels
        self.scanner = ScannerPanel(self)
        root.addWidget(self.scanner)
        self.orders = OrdersPanel(self)
        root.addWidget(self.orders)
        self.orders.cancelRequested.connect(self._on_cancel_order)
        self.orders.replaceRequested.connect(self._on_replace_order)
        self.orders.detailsRequested.connect(self._on_order_details)

        # Load configuration and initialize services (guarded by offline)
        cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json"), "r"))
        self._docs_dir = cfg.get("paths", {}).get("docs_dir", "docs")
        self._templates_path = cfg.get("paths", {}).get("prompt_templates", "prompt_templates.json")

        if not offline:
            # IBKR
            self.ib = IBKRClient(cfg["ibkr"]["host"], cfg["ibkr"]["port"], cfg["ibkr"]["clientId"])
            try:
                self.ib.connect()
                self.lbl_conn.setText("Connected to IBKR")
                self.lbl_conn.setStyleSheet("color:white;background:#2f855a;padding:2px 6px;border-radius:3px")
            except Exception as e:
                self.lbl_conn.setText(f"IBKR connect error: {e}")
                self.lbl_conn.setStyleSheet("color:white;background:#c53030;padding:2px 6px;border-radius:3px")

            # Scanners / IV history
            self._scanner = HybridScanner(self.ib)
            _iv_path = cfg.get('paths', {}).get('iv_history', 'data/iv_history.json')
            self._iv_store = IVHistoryStore(_iv_path)
            self._scanner_local = LocalVectorScanner(self.ib, iv_history=self._iv_store.data, cfg=cfg)

            # Perplexity (optional)
            try:
                self.perp = PerplexityClient(cfg["perplexity"]["base_url"], cfg["perplexity"]["model"])
            except Exception:
                self.perp = None
                self.lbl_conn.setText(self.lbl_conn.text() + " | Perplexity not ready")

            # Optional bank preload
            banks_file = os.path.join(os.path.dirname(__file__), "..", "banks.json")
            if os.path.exists(banks_file):
                import json as _json
                bank = _json.load(open(banks_file)).get("Main", [])[:50]
                rows = [{"symbol": s, "sector": "-", "signal": "-", "potential": True} for s in bank]
                try:
                    self.scanner.load_rows(rows)
                except Exception:
                    pass
        else:
            # Offline mode
            self.ib = None
            self._scanner = None
            self._iv_store = None
            self._scanner_local = None
            self.perp = None

        # Apply visual theme (QSS) - purely visual, no functional changes
        try:
            self._apply_visual_theme()
        except Exception:
            # don't break functionality if styling fails
            pass

    def _open_ticket(self, details: StrategyDetails):
        t_legs = [TicketLeg(pos=l.pos, strike=l.strike, qty=l.qty, premium=l.premium, expiry=l.expiry) for l in details.legs]
        t_details = TicketDetails(name=details.name, current_price=details.current_price, confidence=details.confidence, expected_profit=details.expected_profit, max_loss=details.max_loss, success_prob=details.success_prob, risk_reward=details.risk_reward, legs=t_legs, delta=details.delta, gamma=details.gamma, theta=details.theta, vega=details.vega)
        def on_send(ticket: OrderTicket):
            # translate legs to IBKR combo and place
            # map pos -> (right, sign)
            def map_leg(leg): return ('C' if 'Call' in leg.pos else 'P', +1 if 'Long' in leg.pos else -1, leg.strike, leg.expiry)
            mapped = [map_leg(l) for l in ticket.legs]
            # build ib_insync options
            ib_legs = []
            for right_sign_strike_exp in mapped:
                right, sign, k, exp = right_sign_strike_exp
                opt = self.ib.build_option(ticket.symbol, right, k, exp)
                ib_legs.append((opt, sign * ticket.quantity))
            order_type = 'LMT' if ticket.pricing_mode == 'LIMIT' else 'MKT'
            oid = self.ib.place_combo_order(ticket.symbol, ib_legs, action='BUY', tif=ticket.tif, pricing=order_type, limit_price=ticket.limit_price)
            self.orders.add_order(OrderRow(order_id=oid, symbol=ticket.symbol, strategy_name=details.name, qty=ticket.quantity, pricing=ticket.pricing_mode, tif=ticket.tif, status="Submitted", filled=0, remaining=ticket.quantity))
        dlg = OrderTicketDialog(symbol=self.ed_symbol.text().upper(), strategy=t_details, account_margin_available=50_000.0, mode="PAPER", on_send=on_send)
        dlg.exec()

    def set_symbol_from_scanner(self, symbol: str): self.ed_symbol.setText(symbol.upper())

    def _on_analyze(self):
        sym = (self.ed_symbol.text() or "").upper().strip()
        if not sym: QMessageBox.warning(self,"Analyze","נא להזין סימבול"); return
        # Pull snapshot from IBKR
        snap = self.ib.fetch_stock_snapshot(sym)
        self.lbl_price.setText(f"${snap['price']:.2f}")
        self.lbl_iv.setText("-"); self.lbl_trend.setText("-")
        # Ask Perplexity (optional)
        strategies = []
        if self.perp:
            try:
                import json as _json
                templates = _json.load(open(self._templates_path, 'r', encoding='utf-8'))
                tpl = templates.get('analyze_symbol', {})
                resp = self.perp.analyze_symbol(sym, snap, tpl)
                strategies = resp.get("strategies", [])
                self.lbl_sent.setText("Perplexity: " + (strategies[0].get("blurb","") if strategies else "No suggestion"))
            except Exception as e:
                self.lbl_sent.setText(f"Perplexity error: {e}")
        else:
            self.lbl_sent.setText("Perplexity disabled (no API key)")
        self.box_summary.setVisible(True)
        # Fill recs (from AI or fallback)
        recs = []
        for s in strategies[:3]:
            recs.append(Recommendation(
                key=s.get("key","custom"),
                name=s.get("name","Strategy"),
                confidence=float(s.get("confidence",0.6)),
                blurb=s.get("blurb",""),
                timeframe=s.get("timeframe","30-45 days"),
                expected_profit=float(s.get("expected_profit",500)),
                max_loss=float(s.get("max_loss",400)),
                success_prob=float(s.get("success_prob",0.5)),
            ))
        if not recs:
            recs = [Recommendation(key="bull_call_spread", name="Bull Call Spread", confidence=0.75, blurb="Uptrend with controlled risk", timeframe="30-45d", expected_profit=600, max_loss=400, success_prob=0.6)]
        self._set_recommendations(recs)

    def _set_recommendations(self, recs: List[Recommendation]):
        self.tbl_recs.setRowCount(0)
        for r in recs:
            row = self.tbl_recs.rowCount(); self.tbl_recs.insertRow(row)
            self.tbl_recs.setItem(row,0,QTableWidgetItem(r.name)); self.tbl_recs.setItem(row,1,QTableWidgetItem(f"{int(r.confidence*100)}%")); self.tbl_recs.setItem(row,2,QTableWidgetItem(r.blurb)); self.tbl_recs.setItem(row,3,QTableWidgetItem(r.timeframe)); self.tbl_recs.setItem(row,4,QTableWidgetItem(f"+${r.expected_profit} / -${r.max_loss}"))
            btn = QPushButton("Select Strategy →"); btn.clicked.connect(lambda _, rec=r: self._on_select_strategy(rec)); self.tbl_recs.setCellWidget(row,5,btn)
        self.box_recs.setVisible(True)

    def _on_select_strategy(self, rec: Recommendation):
        sym = (self.ed_symbol.text() or "").upper().strip()
        price = float(self.lbl_price.text().strip('$') or 0.0)
        try:
            sd = self.ib.get_secdef_params(sym)
            expirations = list(sd.get("expirations", []))
            strikes = list(sd.get("strikes", []))
        except Exception as e:
            expirations = ["20251219"]
            strikes = [round(price* (1 + i*0.01),2) for i in range(-10, 11)]
    # determine if strategy requires two expiries
        key = rec.key
        spec = None
        try:
            spec = self._dsl.spec(key)
        except Exception:
            spec = {"mode": "vertical"}
        need_two = spec.get("mode") in ("calendar","diagonal")
        mode = spec.get("mode","vertical")
        dlg = OptionChainSelectorDialog(spot=price, expirations=expirations, strikes=strikes, mode=mode, need_two_expiries=need_two, parent=self)
        if not dlg.exec():
            return
        # Build params from chosen strikes
        if mode == "vertical":
            # expect two strikes (K1,K2) same expiry E1
            chosen = dlg.result_strikes or [round(price*1.01,2), round(price*1.05,2)]
            params = {"K1": float(chosen[0]), "K2": float(chosen[min(1,len(chosen)-1)]), "E1": dlg.result_expiry}
        elif mode == "condor":
            # expect four strikes: put low/high, call low/high
            chosen = dlg.result_strikes
            if not chosen or len(chosen) < 4:
                # simple heuristic: 2%/4% OTM each side
                kpl, kph = round(price*0.96,2), round(price*0.98,2)
                kcl, kch = round(price*1.02,2), round(price*1.04,2)
            else:
                kpl, kph, kcl, kch = map(float, chosen[:4])
            params = {"K_put_low": kpl, "K_put_high": kph, "K_call_low": kcl, "K_call_high": kch, "E1": dlg.result_expiry}
        elif mode in ("calendar","diagonal"):
            # two expiries
            chosen = dlg.result_strikes or [round(price,2), round(price*1.02,2)]
            e_near = getattr(dlg, "result_expiry", None) or expirations[0]
            e_far  = getattr(dlg, "result_expiry_far", None) or (expirations[1] if len(expirations)>1 else expirations[0])
            if key.startswith("calendar"):
                # one strike K
                K = float(chosen[0])
                params = {"K": K, "E_near": e_near, "E_far": e_far}
            else:
                # diagonal_call: two strikes
                k_short = float(chosen[0]); k_long = float(chosen[min(1,len(chosen)-1)])
                params = {"K_short": k_short, "K_long": k_long, "E_near": e_near, "E_far": e_far}
        elif mode == "covered_call":
            chosen = dlg.result_strikes or [round(price*1.05,2)]
            params = {"K1": float(chosen[0]), "E1": dlg.result_expiry}
        else:
            # strangle default
            chosen = dlg.result_strikes or [round(price*0.98,2), round(price*1.02,2)]
            params = {"K_put": float(chosen[0]), "K_call": float(chosen[min(1,len(chosen)-1)]), "E1": dlg.result_expiry}
        # Build UI legs via DSL
        try:
            ui_legs = self._dsl.build_ui_legs(key, params)
        except Exception:
            # fallback simple vertical
            ui_legs = [{"pos":"Long Call","strike":round(price*1.01,2),"qty":1,"premium":None,"expiry":dlg.result_expiry},
                       {"pos":"Short Call","strike":round(price*1.05,2),"qty":1,"premium":None,"expiry":dlg.result_expiry}]
        details = StrategyDetails(name=rec.name, current_price=price, confidence=rec.confidence, expected_profit=rec.expected_profit, max_loss=rec.max_loss, success_prob=rec.success_prob, risk_reward=round(rec.expected_profit/max(1,rec.max_loss),2), legs=[StrategyLeg(**l) for l in ui_legs], delta=0.0, gamma=0.0, theta=0.0, vega=0.0)
        self.details_panel.set_details(details); self.details_panel.setVisible(True)

    # Orders actions
    def _on_cancel_order(self, order_id: str):
        try: self.ib.cancel_order(order_id)
        except Exception as e: QMessageBox.warning(self,"Cancel",str(e))
    def _on_replace_order(self, order_id: str): pass
    def _on_order_details(self, order_id: str):
        hist = [{"ts":"now","status":"Submitted","info":""}]
        dlg = OrderDetailsDialog(order_id, [], hist, self); dlg.exec()

    def _apply_visual_theme(self):
        """Apply a QSS stylesheet to make the UI visually similar to the provided design.
        This function only changes appearance (colors, paddings, badges) and must not
        alter any logic or widget wiring.
        """
        qss = r"""
        /* Base */
        QMainWindow, QWidget { background: #f6f7fb; color: #1f2937; font-family: Segoe UI, Arial, sans-serif; }

        /* Header area */
        QWidget#headerTabs { background: #ffffff; border-bottom: 1px solid #e6e9ef; padding: 12px 18px; }
        QWidget#headerTabs QLabel { font-size: 18px; }
        QPushButton[header="true"] { background: transparent; border: none; padding: 8px 12px; color: #374151; font-weight: 500; }
        QPushButton[header="true"][selected="true"] { color: #0f172a; border-bottom: 3px solid #111827; padding-bottom: 5px; }

        /* Secondary tabs */
        QWidget#secondaryTabs { background: transparent; padding: 8px 18px; }
        QPushButton[secondary="true"] { background: transparent; border: 1px solid transparent; padding: 6px 12px; border-radius: 6px; color: #4b5563; }
        QPushButton[secondary="true"][selected="true"] { background: #111827; color: white; }

        /* Cards */
        QGroupBox, QFrame { background: #ffffff; border: 1px solid #e8eef8; border-radius: 10px; }
        QGroupBox { margin-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; font-weight: 600; }

        /* Analyze card specifics */
        QGroupBox#analyzeCard { padding: 10px; }
        QLabel#searchIcon { background: transparent; font-size: 16px; }
        QLineEdit { background: #ffffff; border: 1px solid #dbe4ef; padding: 10px 12px; border-radius: 8px; font-size: 14px; }
        QLineEdit:disabled { background: #f3f4f6; }

        /* Primary button */
        QPushButton#analyzeButton { background: #111827; color: white; padding: 10px 14px; border-radius: 8px; font-weight: 600; }
        QPushButton#analyzeButton:disabled { background: #94a3b8; color: white; }

        /* Tables */
        QTableWidget { background: #ffffff; gridline-color: #f1f5f9; }
        QHeaderView::section { background: #f8fafc; padding: 8px; border: none; color: #6b7280; }
        QTableWidget::item { padding: 8px 10px; }

        /* Status bar */
        QStatusBar { background: transparent; color: #374151; padding: 6px 10px; }
        QStatusBar QLabel { color: #374151; }

        /* Small badges */
        QLabel[badge="true"] { background: #eef2ff; color: #1e3a8a; border: 1px solid #e0e7ff; padding: 3px 8px; border-radius: 12px; font-size: 12px; }

        /* Misc */
        QPushButton:hover { filter: brightness(0.98); }
        """

        # Apply stylesheet to the main window
        self.setStyleSheet(qss)

def _open_learning_center(self):
    dlg = LearningCenter(self._docs_dir, self)
    dlg.exec()

def _open_prompt_manager(self):
    dlg = PromptManager(self._templates_path, self)
    dlg.exec()


def on_scan_bank_request(self, source: str, max_items: int):
    # Load bank symbols (Main by default)
    import os, json as _json
    banks_file = os.path.join(os.path.dirname(__file__),"..","banks.json")
    bank = ["AAPL","MSFT","NVDA","GOOG","IBM","META","TSLA","AMD","AMZN"]
    if os.path.exists(banks_file):
        try:
            data = _json.load(open(banks_file))
            bank = data.get("Main", bank)
        except Exception:
            pass
    rows = self._scanner.scan_bank(bank, max_symbols=max_items)
    self.scanner.load_rows(rows)


def _on_send_order(self, symbol: str, details: StrategyDetails, pricing: str, limit_price: float, tif: str, qty: int):
    # build ui legs dicts and send to IBKR
    ui_legs = [{"pos": l.pos, "strike": float(l.strike), "qty": int(l.qty), "premium": (None if l.premium is None else float(l.premium)), "expiry": l.expiry} for l in details.legs]
    try:
        resp = self.ib.place_combo_order(symbol, ui_legs, pricing=pricing, limit_price=limit_price, tif=tif, qty=qty)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Order Sent", f"OrderId: {resp.get('orderId')} Status: {resp.get('status')}")
    except Exception as e:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Order Error", str(e))
