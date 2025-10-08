"""
Scanner Widget
For scanning and screening stocks based on various criteria
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QFrame, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QFormLayout, QCheckBox, QGroupBox, QHeaderView,
    QProgressBar, QTextEdit, QSplitter, QMessageBox, QDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QColor

from core.config_manager import ConfigManager
from utils.logger import get_logger

# Optional: matplotlib for charting
try:
    try:
        # Preferred in Matplotlib 3.5+
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        # Fallback for older versions
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

class ChartDialog(QDialog):
    def __init__(self, symbol: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Chart - {symbol}")
        # Set reasonable size for chart dialog
        self.setMinimumSize(900, 600)
        self.resize(1000, 650)
        self.symbol = symbol
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        v.addLayout(top)
        top.addWidget(QLabel(f"{symbol}"))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["1Y", "6M", "3M", "1M", "1W", "1D"])  # timeframes
        top.addWidget(self.tf_combo)
        # Chart type selector
        self.chart_type = QComboBox()
        self.chart_type.addItems(["Line", "Candlestick"])
        # Default to Candlestick for better readability
        try:
            self.chart_type.setCurrentText("Candlestick")
        except Exception:
            pass
        top.addWidget(self.chart_type)
        # Overlays
        self.show_ema5 = QCheckBox("EMA5")
        self.show_ema5.setChecked(True)
        self.show_ema10 = QCheckBox("EMA10")
        self.show_ema10.setChecked(True)
        top.addWidget(self.show_ema5)
        top.addWidget(self.show_ema10)
        top.addStretch(1)
        if MATPLOTLIB_OK:
            self.fig = Figure(figsize=(6,3))
            self.canvas = FigureCanvas(self.fig)
            v.addWidget(self.canvas)
        else:
            v.addWidget(QLabel("Matplotlib not available."))
        self.tf_combo.currentTextChanged.connect(self._refresh)
        self.chart_type.currentTextChanged.connect(self._refresh)
        self.show_ema5.toggled.connect(self._refresh)
        self.show_ema10.toggled.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        if not MATPLOTLIB_OK:
            return
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.dates as mdates
            from pathlib import Path
            fp = Path("data/bronze/daily") / f"{self.symbol}.parquet"
            if not fp.exists():
                QMessageBox.warning(self, "Chart", f"No data found for {self.symbol}")
                return
            df = pd.read_parquet(fp)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            # Ensure OHLC presence; fallback to adj_close/close if needed
            if 'close' not in df.columns and 'adj_close' in df.columns:
                df['close'] = df['adj_close']
            for col in ['open','high','low']:
                if col not in df.columns and 'close' in df.columns:
                    df[col] = df['close']
            # Resample for 1W/1M; otherwise daily slice
            tf = self.tf_combo.currentText()
            dfr = df.copy()
            if tf in ("1W", "1M"):
                rule = 'W' if tf == '1W' else 'M'
                dfi = dfr.set_index('date')
                agg = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                dfr = dfi.resample(rule).apply(agg).dropna().reset_index()
                window = 52 if tf == '1W' else 24
                plot = dfr.tail(window)
            else:
                # Daily bars window lengths
                days = {'1Y': 252, '6M': 126, '3M': 90, '1M': 30, '1W': 10, '1D': 120}.get(tf, 252)
                plot = dfr.tail(days)

            # Build subplots: price, volume, RSI
            self.fig.clear()
            gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
            ax = self.fig.add_subplot(gs[0])
            axv = self.fig.add_subplot(gs[1], sharex=ax)
            axr = self.fig.add_subplot(gs[2], sharex=ax)

            dates = plot['date']
            closes = plot['close'].astype(float)
            ctype = self.chart_type.currentText()

            if ctype == 'Candlestick' and {'open','high','low','close'}.issubset(set(plot.columns)):
                # Manual candlesticks
                try:
                    x = mdates.date2num(pd.to_datetime(dates))
                    # Slightly larger candles for clarity
                    width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 0.8
                    up_color = '#00C851'    # Brighter green for up moves
                    down_color = '#FF4444'  # Brighter red for down moves
                    for xi, o, h, l, c in zip(x, plot['open'], plot['high'], plot['low'], plot['close']):
                        color = up_color if c >= o else down_color
                        # Thicker wicks for better visibility
                        ax.vlines(xi, l, h, color=color, linewidth=1.5, alpha=0.8)
                        y = min(o, c)
                        height = abs(c - o) if abs(c - o) > 1e-9 else 1e-9
                        # More opaque candle bodies
                        ax.add_patch(
                            __import__('matplotlib').patches.Rectangle((xi - width/2, y), width, height, 
                                                                     color=color, alpha=0.95, 
                                                                     edgecolor=color, linewidth=0.5)
                        )
                except Exception:
                    ax.plot(dates, closes, color='steelblue', label='Close')
            else:
                # Line chart
                ax.plot(dates, closes, color='steelblue', label='Close')

            # EMA overlays on price
            if self.show_ema5.isChecked():
                ax.plot(dates, closes.ewm(span=5, adjust=False).mean(), color='orange', label='EMA5')
            if self.show_ema10.isChecked():
                ax.plot(dates, closes.ewm(span=10, adjust=False).mean(), color='magenta', label='EMA10')
            # Build legend including candle colors and overlays
            try:
                from matplotlib.patches import Patch
                handles, labels = ax.get_legend_handles_labels()
                handles = list(handles)
                labels = list(labels)
                handles.extend([Patch(color=up_color if 'up_color' in locals() else '#2ECC71', label='Up Candle'),
                                Patch(color=down_color if 'down_color' in locals() else '#E74C3C', label='Down Candle')])
                ax.legend(handles=handles, loc='upper left', fontsize=8)
            except Exception:
                ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{self.symbol} ({tf})")

            # Volume subplot
            axv.bar(dates, plot['volume'].astype(float), color='lightgray', width=0.8)
            axv.set_ylabel('Vol', fontsize=8)
            axv.grid(True, alpha=0.2)

            # RSI subplot
            try:
                c = closes
                delta = c.diff()
                gain = delta.clip(lower=0.0)
                loss = -delta.clip(upper=0.0)
                roll = 14
                avg_gain = gain.ewm(alpha=1/roll, adjust=False, min_periods=roll).mean()
                avg_loss = loss.ewm(alpha=1/roll, adjust=False, min_periods=roll).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.dropna()
                axr.plot(dates.iloc[-len(rsi):], rsi, color='purple')
                axr.axhline(30, color='red', alpha=0.3, linestyle='--')
                axr.axhline(70, color='green', alpha=0.3, linestyle='--')
                axr.set_ylim(0, 100)
                axr.set_ylabel('RSI', fontsize=8)
                axr.grid(True, alpha=0.2)
            except Exception:
                pass

            self.fig.autofmt_xdate()
            self.canvas.draw_idle()
        except Exception:
            pass

class ExplainDialog(QDialog):
    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Explanation")
        self.setMinimumSize(560, 420)
        v = QVBoxLayout(self)
        sym = result.get('symbol', '')
        v.addWidget(QLabel(f"Symbol: {sym}"))
        # Build detailed text
        txt = QTextEdit()
        txt.setReadOnly(True)
        lines = []
        bd = result.get('breakdown', {}) or {}
        for name in ['Momentum','Value','Growth','Oversold']:
            items = bd.get(name) or []
            if not items:
                continue
            lines.append(f"{name}:")
            for it in items:
                ok = '✔' if it.get('ok') else '✘'
                wt = it.get('weight', 0)
                rule = it.get('name', '')
                det = it.get('details', {}) or {}
                det_str = ", ".join(f"{k}={v}" for k,v in det.items() if v is not None)
                lines.append(f"  {ok} ({wt}) {rule} {('- ' + det_str) if det_str else ''}")
        if not lines:
            # Helpful hint when no strategies are active
            strategy_text = str(result.get('strategy', '')).strip()
            if not strategy_text:
                lines = ["No breakdown available (no strategy selected).\nTip: select one or more strategies to see why a symbol matches."]
            else:
                lines = ["No breakdown available"]
        txt.setText("\n".join(lines))
        v.addWidget(txt)


class ScannerWorker(QObject):
    """Worker thread for stock scanning"""
    
    progress_updated = pyqtSignal(int)
    result_found = pyqtSignal(dict)
    results_updated = pyqtSignal(list)
    scan_completed = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.logger = get_logger("ScannerWorker")
        self.is_scanning = False
    
    def start_scan(self, criteria: dict):
        """Scan local bronze parquet data and apply filters. If ml_preds_path provided, limit to those tickers."""
        try:
            from pathlib import Path
            import math
            import pandas as pd
            import time
            self.is_scanning = True
            self.progress_updated.emit(0)
            self.status_updated.emit("Initializing scanner…")

            bronze_dir = Path("data/bronze/daily")
            if not bronze_dir.exists() or not any(bronze_dir.glob("*.parquet")):
                alt = Path("data/bronze")
                if alt.exists() and any(alt.glob("*.parquet")):
                    self.status_updated.emit("Falling back to data/bronze")
                    bronze_dir = alt
                else:
                    raise RuntimeError("Bronze data directory not found: data/bronze/daily or data/bronze")

            # Build universe from stock_data folders if available (reflect full >10k),
            # but read OHLCV from existing Parquet files.
            stock_dir = Path("stock_data")
            stock_tickers = []
            try:
                if stock_dir.exists():
                    stock_tickers = [p.name for p in stock_dir.iterdir() if p.is_dir()]
            except Exception:
                stock_tickers = []
            parquet_files = list(bronze_dir.glob("*.parquet"))
            parquet_tickers = [fp.stem for fp in parquet_files]

            # Prefer stock_data enumeration when it yields a larger universe; otherwise use parquet list.
            tickers_universe = stock_tickers if len(stock_tickers) >= len(parquet_tickers) else parquet_tickers
            self.logger.info(f"Scanner universe: stock_data={len(stock_tickers)} parquet={len(parquet_tickers)} using={len(tickers_universe)}")
            total = len(tickers_universe)
            if total == 0:
                raise RuntimeError("No symbols found (stock_data empty and no parquet files)")
            # Inform user when many symbols lack parquet (suggest Daily Update)
            if stock_tickers and len(parquet_tickers) < len(stock_tickers):
                self.status_updated.emit(
                    f"Loading {total} symbols (parquet available for {len(parquet_tickers)}/{len(stock_tickers)}). Tip: run Daily Update to convert missing."
                )
            else:
                self.status_updated.emit(f"Loading {total} tickers…")

            # Optional ML predictions limiting
            allowed_tickers = None
            pred_info_map = {}
            ml_preds_path = criteria.get("ml_preds_path")
            # Consider ML universe CSV first if requested
            try:
                if bool(criteria.get("use_universe_csv")):
                    from pathlib import Path as _P
                    uni = _P("data/silver/universe.csv")
                    if uni.exists():
                        udf = pd.read_csv(uni)
                        if 'ticker' in udf.columns:
                            allowed_tickers = set(udf['ticker'].astype(str).str.upper().unique())
                            self.status_updated.emit(f"Using ML universe.csv: {len(allowed_tickers)} tickers")
            except Exception:
                pass
            if ml_preds_path:
                try:
                    preds_all = pd.read_parquet(ml_preds_path)
                    # Build prediction info per symbol (latest row if duplicates)
                    if 'ticker' in preds_all.columns:
                        tmp = preds_all.copy()
                        tmp['ticker'] = tmp['ticker'].astype(str).str.upper()
                        tmp = tmp.drop_duplicates(subset=['ticker'], keep='last')
                        for _, r in tmp.iterrows():
                            t = str(r['ticker']).upper()
                            info = {}
                            # Flexible field names
                            for k in ['pred_close','pred_price','target_price']:
                                if k in r and pd.notna(r[k]):
                                    try:
                                        info['pred_price'] = float(r[k])
                                        break
                                    except Exception:
                                        pass
                            # Predicted return, could be ratio or percent
                            for k in ['pred_return','y_pred_return','pred_change','pred_change_pct']:
                                if k in r and pd.notna(r[k]):
                                    try:
                                        info['pred_return'] = float(r[k])
                                        break
                                    except Exception:
                                        pass
                            if 'y_pred' in r:
                                info['pred_label'] = str(r['y_pred'])
                            if info:
                                pred_info_map[t] = info
                    # Limit universe to UP if available
                    if 'y_pred' in preds_all.columns:
                        up_df = preds_all[preds_all['y_pred'].astype(str) == 'UP']
                        allowed_set = set(up_df['ticker'].astype(str).str.upper().unique())
                    else:
                        allowed_set = set(preds_all['ticker'].astype(str).str.upper().unique())
                    # If universe already set from CSV, intersect; otherwise take allowed_set
                    allowed_tickers = allowed_set if allowed_tickers is None else (allowed_tickers & allowed_set)
                    self.status_updated.emit(f"Using ML predictions universe: {len(allowed_tickers)} tickers")
                except Exception as ie:
                    self.logger.warning(f"Failed to load ML preds {ml_preds_path}: {ie}")

            # optional RSI via pandas_ta
            try:
                import pandas_ta as ta  # type: ignore
            except Exception:
                ta = None

            # Load fundamentals (MarketCap) mapping if available
            marketcap_map = {}
            try:
                from pathlib import Path as _Path
                fund_agg = _Path("data/bronze/fundamentals.parquet")
                if fund_agg.exists():
                    fdf = pd.read_parquet(fund_agg)
                    # Try common column variants
                    cols = {c.lower(): c for c in fdf.columns}
                    sym_col = cols.get('symbol')
                    mcap_col = cols.get('marketcap') or cols.get('market_cap') or cols.get('marketcapitalization')
                    if sym_col and mcap_col:
                        tmp = fdf[[sym_col, mcap_col]].dropna()
                        tmp[sym_col] = tmp[sym_col].astype(str).str.upper()
                        marketcap_map = dict(zip(tmp[sym_col].tolist(), tmp[mcap_col].astype(float).tolist()))
            except Exception:
                marketcap_map = {}

            # Load broader fundamentals into per-field symbol maps (defensive on column names)
            fund_maps = {}
            try:
                fund_agg = Path("data/bronze/fundamentals.parquet")
                if fund_agg.exists():
                    fdf_all = pd.read_parquet(fund_agg)
                    cols_l = {c.lower(): c for c in fdf_all.columns}
                    sym_col = cols_l.get('symbol') or cols_l.get('ticker')
                    if sym_col:
                        fdf_all[sym_col] = fdf_all[sym_col].astype(str).str.upper()
                        # Deduplicate by symbol (keep last)
                        fdf_all = fdf_all.drop_duplicates(subset=[sym_col], keep='last').set_index(sym_col)
                        def pick(candidates):
                            for nm in candidates:
                                if nm in cols_l:
                                    return cols_l[nm]
                            return None
                        col_defs = {
                            'pe': ['pe', 'pe_ratio', 'price_to_earnings', 'priceearningsratio', 'trailingpe', 'pe_ttm'],
                            'pb': ['pb', 'price_to_book', 'pricebookratio', 'ptb'],
                            'de': ['de', 'debt_to_equity', 'totaldebttoequity', 'debttoequity'],
                            'current_ratio': ['current_ratio', 'currentratio'],
                            'roe': ['roe', 'return_on_equity'],
                            'book_value': ['book_value', 'book_value_per_share', 'bookvalue', 'bookvaluepershare'],
                            'div_yield': ['dividend_yield', 'dividendyield'],
                            'industry_pe': ['industry_pe', 'industrype'],
                            'fcf_yield': ['fcf_yield', 'free_cash_flow_yield', 'freecashflowyield'],
                            'ev_ebitda': ['ev_ebitda', 'evtoebitda', 'enterprisevalue_to_ebitda', 'enterprisevaluetoebitda'],
                            'piotroski': ['piotroski', 'piotroski_f_score', 'piotroskifscore'],
                            'insider_buying': ['insider_buying', 'insiderbuying', 'insidertrend'],
                        }
                        for key, cands in col_defs.items():
                            colname = pick(cands)
                            if colname and colname in fdf_all.columns:
                                ser = pd.to_numeric(fdf_all[colname], errors='coerce')
                                fund_maps[key] = ser.to_dict()
                        # Ensure marketcap map present too
                        if 'market_cap' not in fund_maps:
                            fund_maps['market_cap'] = marketcap_map.copy()
            except Exception:
                fund_maps = {}

            min_price = float(criteria.get('min_price', 0) or 0)
            max_price = float(criteria.get('max_price', 1e9) or 1e9)
            min_volume = int(criteria.get('min_volume', 0) or 0)
            min_change = float(criteria.get('min_change', -100) or -100)
            max_change = float(criteria.get('max_change', 100) or 100)
            min_rsi = int(criteria.get('min_rsi', 0) or 0)
            max_rsi = int(criteria.get('max_rsi', 100) or 100)
            above_sma20 = bool(criteria.get('above_sma20', False))
            above_sma50 = bool(criteria.get('above_sma50', False))
            above_sma200 = bool(criteria.get('above_sma200', False))

            # Strategy toggles
            want_momentum = bool(criteria.get('strategy_momentum') or criteria.get('momentum_enabled'))
            want_value = bool(criteria.get('strategy_value') or criteria.get('value_enabled'))
            want_growth = bool(criteria.get('strategy_growth') or criteria.get('growth_enabled'))
            want_oversold = bool(criteria.get('strategy_oversold') or criteria.get('oversold_enabled'))

            results = []
            processed = 0
            errors = 0
            err_samples = []
            start_ts = time.time()
            for idx, sym in enumerate(tickers_universe, start=1):
                if not self.is_scanning:
                    break
                try:
                    symbol_upper = str(sym).upper()
                    if allowed_tickers is not None and symbol_upper not in allowed_tickers:
                        continue
                    # Read from parquet if present; skip if missing
                    fp = bronze_dir / f"{sym}.parquet"
                    if not fp.exists():
                        continue
                    df = pd.read_parquet(fp)
                    if df.empty:
                        continue
                    # Ensure date sorted
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.dropna(subset=['date']).sort_values('date')
                    # Use last 250 rows for indicators
                    tail = df.tail(250) if len(df) > 250 else df
                    last = tail.iloc[-1]
                    prev = tail.iloc[-2] if len(tail) > 1 else None
                    price = float(last.get('close', last.get('adj_close', math.nan)))
                    if math.isnan(price):
                        continue
                    volume = int(last.get('volume', 0) or 0)
                    prev_close = float(prev.get('close')) if prev is not None and 'close' in tail.columns else price
                    change = price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close else 0.0
                    # RSI (14)
                    rsi_val = 50.0
                    try:
                        if 'close' in tail.columns:
                            if ta is not None:
                                rsi_series = ta.rsi(tail['close'], length=14)
                                if rsi_series is not None and len(rsi_series.dropna()) > 0:
                                    rsi_val = float(rsi_series.iloc[-1])
                            else:
                                # Pure-Pandas RSI (Wilder's smoothing)
                                c = tail['close'].astype(float)
                                delta = c.diff()
                                gain = delta.clip(lower=0.0)
                                loss = -delta.clip(upper=0.0)
                                roll = 14
                                avg_gain = gain.ewm(alpha=1/roll, adjust=False, min_periods=roll).mean()
                                avg_loss = loss.ewm(alpha=1/roll, adjust=False, min_periods=roll).mean()
                                rs = avg_gain / avg_loss.replace(0, float('nan'))
                                rsi = 100 - (100 / (1 + rs))
                                rsi = rsi.dropna()
                                if len(rsi) > 0:
                                    rsi_val = float(rsi.iloc[-1])
                    except Exception:
                        pass
                    # SMAs
                    sma20_ok = sma50_ok = sma200_ok = True
                    try:
                        close = tail['close']
                        sma20 = close.rolling(20).mean().iloc[-1]
                        sma50 = close.rolling(50).mean().iloc[-1]
                        sma200 = close.rolling(200).mean().iloc[-1]
                        sma20_ok = (not above_sma20) or (price >= float(sma20 or price))
                        sma50_ok = (not above_sma50) or (price >= float(sma50 or price))
                        sma200_ok = (not above_sma200) or (price >= float(sma200 or price))
                    except Exception:
                        sma20_ok = (not above_sma20)
                        sma50_ok = (not above_sma50)
                        sma200_ok = (not above_sma200)

                    # Base filters (apply to all strategies)
                    passed = True
                    if not (min_price <= price <= max_price):
                        passed = False
                    elif not (min_volume <= volume):
                        passed = False
                    elif not (min_change <= change_pct <= max_change):
                        passed = False
                    elif not (min_rsi <= rsi_val <= max_rsi):
                        passed = False
                    elif not (sma20_ok and sma50_ok and sma200_ok):
                        passed = False

                    # 🔍 Enhanced Data Quality Filters
                    if passed:
                        # Skip penny stocks with unreliable data
                        if price < 1.0:
                            passed = False
                        
                        # Check for abnormal daily spread (liquidity proxy)
                        try:
                            daily_spread_pct = ((latest['high'] - latest['low']) / price) * 100
                            if daily_spread_pct > 15.0:  # Skip extremely volatile/illiquid stocks
                                passed = False
                        except:
                            pass
                        
                        # Volume consistency check - avoid pump & dump
                        try:
                            if len(tail) >= 20:
                                avg_volume_20d = tail['volume'].tail(20).mean()
                                volume_ratio = volume / avg_volume_20d if avg_volume_20d > 0 else 1
                                # Skip volume spikes > 10x (potential manipulation)
                                if volume_ratio > 10.0:
                                    passed = False
                                # Also skip very low relative volume (< 0.1x average)
                                elif volume_ratio < 0.1:
                                    passed = False
                        except:
                            pass
                        
                        # Price action quality check
                        try:
                            # Skip stocks with extreme gap ups/downs (> 20%)
                            if len(tail) >= 2:
                                prev_close = tail['close'].iloc[-2]
                                gap_pct = ((latest['open'] - prev_close) / prev_close) * 100
                                if abs(gap_pct) > 20.0:
                                    passed = False
                        except:
                            pass

                    # Early continue if base filters fail (keeps performance)
                    if not passed:
                        continue

                    # 🚀 Enhanced Base Score Calculation
                    score = 0.0
                    crit_met = 0
                    matched_strategies = []
                    
                    # Calculate advanced technical indicators for better scoring
                    relative_volume = 1.0
                    price_strength = 0.0
                    liquidity_score = 0.0
                    
                    try:
                        # Relative Volume Score (0-2)
                        if len(tail) >= 20:
                            avg_vol_20d = tail['volume'].tail(20).mean()
                            relative_volume = min(volume / avg_vol_20d, 5.0) if avg_vol_20d > 0 else 1.0
                        
                        # Price Strength vs Multiple Timeframes (0-3)
                        if close_series is not None and len(close_series) >= 50:
                            price_vs_sma20 = (price / close_series.tail(20).mean() - 1) * 100
                            price_vs_sma50 = (price / close_series.tail(50).mean() - 1) * 100
                            
                            # Reward consistent strength across timeframes
                            if price_vs_sma20 > 2 and price_vs_sma50 > 5:
                                price_strength = 3.0
                            elif price_vs_sma20 > 0 and price_vs_sma50 > 0:
                                price_strength = 2.0
                            elif price_vs_sma20 > 0 or price_vs_sma50 > 0:
                                price_strength = 1.0
                        
                        # Liquidity Score based on spread and volume (0-2)  
                        daily_spread_pct = ((latest['high'] - latest['low']) / price) * 100
                        if daily_spread_pct < 2.0 and relative_volume > 1.2:
                            liquidity_score = 2.0
                        elif daily_spread_pct < 5.0 and relative_volume > 0.8:
                            liquidity_score = 1.0
                            
                    except Exception:
                        pass
                    
                    # Base score incorporates multiple factors
                    base_score = (
                        (rsi_val / 100.0 * 2.0) +  # RSI component (0-2)
                        (relative_volume * 0.4) +    # Volume component (0-2) 
                        price_strength +             # Price strength (0-3)
                        liquidity_score             # Liquidity (0-2)
                    ) / 9.0 * 10.0  # Normalize to 0-10 scale
                    
                    score = base_score

                    # Common technicals used by multiple strategies
                    # Compute once defensively
                    try:
                        close_series = tail['close'].astype(float)
                    except Exception:
                        close_series = None
                    sma10 = sma20_val = None
                    try:
                        if close_series is not None:
                            sma10 = float(close_series.rolling(10).mean().iloc[-1])
                            sma20_val = float(close_series.rolling(20).mean().iloc[-1])
                    except Exception:
                        sma10 = sma20_val = None
                    vma20 = None
                    try:
                        vma20 = float(tail['volume'].rolling(20).mean().iloc[-1])
                    except Exception:
                        pass
                    # ATR 14 (percent)
                    atr_ok = False; atr_val_pct = None
                    try:
                        if {'high','low','close'}.issubset(set(tail.columns)):
                            if ta is not None:
                                atr_series = ta.atr(tail['high'], tail['low'], tail['close'], length=14)
                            else:
                                high = tail['high'].astype(float)
                                low = tail['low'].astype(float)
                                close_series2 = tail['close'].astype(float)
                                prev_close = close_series2.shift(1)
                                tr = (high - low).to_frame('tr')
                                tr['hc'] = (high - prev_close).abs()
                                tr['lc'] = (low - prev_close).abs()
                                tr_val = tr.max(axis=1)
                                atr_series = tr_val.rolling(14, min_periods=14).mean()
                            atr_series = atr_series.dropna() if atr_series is not None else atr_series
                            if atr_series is not None and len(atr_series) > 0:
                                atr_val = float(atr_series.iloc[-1])
                                atr_val_pct = (atr_val / price) * 100.0 if price else None
                                atr_ok = atr_val_pct is not None and 2.0 <= atr_val_pct <= 5.0
                    except Exception:
                        pass
                    # MACD histogram positive
                    macd_pos = False
                    try:
                        if close_series is not None:
                            if ta is not None:
                                macd = ta.macd(close_series)
                                if macd is not None:
                                    # Support common column variants
                                    for col in macd.columns:
                                        if 'macdh' in str(col).lower():
                                            macd_pos = float(macd[col].iloc[-1]) > 0
                                            break
                            else:
                                ema12 = close_series.ewm(span=12, adjust=False).mean()
                                ema26 = close_series.ewm(span=26, adjust=False).mean()
                                macd_line = ema12 - ema26
                                signal = macd_line.ewm(span=9, adjust=False).mean()
                                hist = (macd_line - signal).dropna()
                                if len(hist) > 0:
                                    macd_pos = float(hist.iloc[-1]) > 0
                    except Exception:
                        pass
                    # EMA cross with configurable spans
                    ema_ok = False
                    try:
                        if close_series is not None:
                            ema_short_len = int(criteria.get('momentum_ema_short') or 5)
                            ema_long_len = int(criteria.get('momentum_ema_long') or 10)
                            # Guard: ensure long > short
                            if ema_long_len <= ema_short_len:
                                ema_long_len = max(ema_short_len + 1, 2)
                            if ta is not None:
                                ema_s = float(ta.ema(close_series, length=ema_short_len).iloc[-1])
                                ema_l = float(ta.ema(close_series, length=ema_long_len).iloc[-1])
                            else:
                                ema_s = float(close_series.ewm(span=ema_short_len, adjust=False).mean().iloc[-1])
                                ema_l = float(close_series.ewm(span=ema_long_len, adjust=False).mean().iloc[-1])
                            ema_ok = ema_s > ema_l
                    except Exception:
                        pass
                    # 30d volatility window
                    vol30_ok = False
                    try:
                        if close_series is not None and len(close_series) >= 30:
                            ret = close_series.pct_change().dropna()
                            if len(ret) >= 30:
                                vol_30d = float(ret.tail(30).std() * (252 ** 0.5) * 100.0)
                                vol30_ok = 15.0 <= vol_30d <= 40.0
                    except Exception:
                        pass

                    # 52-week high/low from close
                    low_52w = high_52w = None
                    try:
                        if close_series is not None and len(close_series) >= 50:
                            window = close_series.tail(252) if len(close_series) > 252 else close_series
                            low_52w = float(window.min())
                            high_52w = float(window.max())
                    except Exception:
                        pass

                    # Strategy: Momentum
                    if want_momentum:
                        w = criteria.get('momentum_weights', {})
                        # Compute required indicators
                        # Use precomputed technicals when available
                        if sma10 is None:
                            sma10_eff = price
                        else:
                            sma10_eff = sma10
                        sma20_eff = sma20_val if sma20_val is not None else price
                        vma20_eff = vma20 if vma20 is not None else volume

                        # MarketCap from fundamentals mapping (> 100M)
                        mcap_ok = False
                        try:
                            mval = float(marketcap_map.get(symbol_upper, float('nan')))
                            if not math.isnan(mval):
                                mcap_ok = mval >= 100_000_000.0
                        except Exception:
                            mcap_ok = False
                        vol30_ok = vol30_ok

                        # 🚀 Enhanced Momentum Rules with Advanced Criteria
                        
                        # Calculate advanced momentum indicators
                        momentum_strength = 0
                        volume_momentum = False
                        price_acceleration = False
                        trend_consistency = False
                        
                        try:
                            # Multi-timeframe momentum strength
                            if close_series is not None and len(close_series) >= 50:
                                returns_5d = (price / close_series.iloc[-6] - 1) * 100 if len(close_series) > 5 else 0
                                returns_20d = (price / close_series.iloc[-21] - 1) * 100 if len(close_series) > 20 else 0
                                returns_50d = (price / close_series.iloc[-51] - 1) * 100 if len(close_series) > 50 else 0
                                
                                # Reward accelerating momentum
                                if returns_5d > 0 and returns_20d > 0 and returns_50d > 0:
                                    if returns_5d > returns_20d > returns_50d:
                                        momentum_strength = 3  # Accelerating
                                    elif returns_5d > 0 and returns_20d > 0:
                                        momentum_strength = 2  # Consistent
                                    else:
                                        momentum_strength = 1  # Basic uptrend
                            
                            # Volume momentum (increasing volume trend)
                            if len(tail) >= 10 and 'volume' in tail.columns:
                                vol_recent = tail['volume'].tail(5).mean()
                                vol_earlier = tail['volume'].iloc[-10:-5].mean()
                                volume_momentum = vol_recent > vol_earlier * 1.1
                            
                            # Price acceleration (rate of change increasing)
                            if close_series is not None and len(close_series) >= 10:
                                roc_recent = close_series.pct_change(5).tail(1).iloc[0] * 100
                                roc_earlier = close_series.pct_change(5).iloc[-6] * 100
                                price_acceleration = roc_recent > roc_earlier
                            
                            # Trend consistency (higher highs, higher lows)
                            if len(tail) >= 10:
                                recent_highs = tail['high'].tail(5)
                                recent_lows = tail['low'].tail(5)
                                earlier_highs = tail['high'].iloc[-10:-5]
                                earlier_lows = tail['low'].iloc[-10:-5]
                                trend_consistency = (recent_highs.min() > earlier_highs.max() and 
                                                   recent_lows.min() > earlier_lows.max())
                        except:
                            pass
                        
                        # Enhanced momentum rules with better weighting
                        rules = [
                            # Core price momentum (higher weights)
                            (price > (sma10_eff * 1.015), w.get('price_gt_sma10_1015', 2.0), 'Price > SMA10*1.015', {'price': price, 'sma10': sma10_eff}),
                            (price > (sma20_eff * 1.02), w.get('price_gt_sma20_1020', 2.5), 'Price > SMA20*1.02', {'price': price, 'sma20': sma20_eff}),
                            ((sma10_eff > (sma20_eff * 1.005)), w.get('sma10_gt_sma20_1005', 2.0), 'SMA10 > SMA20*1.005', {'sma10': sma10_eff, 'sma20': sma20_eff}),
                            
                            # Enhanced momentum indicators (NEW!)
                            (momentum_strength >= 2, w.get('momentum_strength', 3.0), 'Multi-timeframe Momentum', {'strength': momentum_strength}),
                            (volume_momentum, w.get('volume_momentum', 1.5), 'Increasing Volume Momentum', {}),
                            (price_acceleration, w.get('price_acceleration', 1.5), 'Price Acceleration', {}),
                            (trend_consistency, w.get('trend_consistency', 2.0), 'Trend Consistency (HH,HL)', {}),
                            
                            # Technical indicators (reweighted)
                            (50.0 <= rsi_val <= 75.0, w.get('rsi_50_75', 1.5), 'RSI in [50,75]', {'rsi': rsi_val}),
                            (bool(vma20_eff) and volume > (vma20_eff * 1.2), w.get('vol_gt_vma20_1_2', 1.8), 'Vol > VMA20*1.2', {'vol': volume, 'vma20': vma20_eff}),
                            (atr_ok, w.get('atr14_2_5pct', 1.0), 'ATR14 in [2%,5%]', {'atr_pct': atr_val_pct if atr_val_pct else 0}),
                            (change_pct > 1.0, w.get('daily_change_gt_1', 1.2), 'Daily %Change > 1%', {'chg_pct': change_pct}),
                            (macd_pos, w.get('macd_hist_pos', 1.5), 'MACD Histogram > 0', {}),
                            (ema_ok, w.get('ema_short_gt_long', 1.3), 'EMA(short) > EMA(long)', {}),
                            
                            # Quality filters (lower weights but important)
                            (mcap_ok, w.get('mcap_gt_100m', 1.0), 'MarketCap > 100M', {'mcap': marketcap_map.get(symbol_upper)}),
                            (vol30_ok, w.get('vol30_15_40', 0.8), 'Volatility 30d in [15%,40%]', {}),
                        ]
                        crit_m = sum(1 for ok, *_ in rules if ok)
                        score_m = float(sum(weight for ok, weight, *_ in rules if ok))
                        min_needed = int(criteria.get('momentum_min_criteria') or 0)
                        if crit_m >= min_needed:
                            matched_strategies.append('Momentum')
                            score = max(score, score_m)
                        # Breakdown
                        breakdown_m = [{'name': name, 'ok': bool(ok), 'weight': wt, 'details': det} for (ok, wt, name, det) in rules]

                    # Strategy: Value - 🔍 Enhanced with Quality Value Metrics
                    if want_value:
                        vw = criteria.get('value_weights', {})
                        def fval(key):
                            try:
                                v = fund_maps.get(key, {}).get(symbol_upper, float('nan'))
                                v = float(v)
                                return v if v == v else None
                            except Exception:
                                return None
                        pe = fval('pe')
                        pb = fval('pb')
                        de = fval('de')
                        curr = fval('current_ratio')
                        roe = fval('roe')
                        bv = fval('book_value')
                        dy = fval('div_yield')
                        ind_pe = fval('industry_pe')
                        fcfy = fval('fcf_yield')
                        eve = fval('ev_ebitda')
                        pio = fval('piotroski')
                        insider = fval('insider_buying')
                        mcap_val = marketcap_map.get(symbol_upper)
                        
                        # Normalize percent-like fields
                        def pct_norm(v):
                            if v is None:
                                return None
                            try:
                                if abs(v) <= 1.0:
                                    return v * 100.0
                                return v
                            except Exception:
                                return None
                        roe_n = pct_norm(roe)
                        dy_n = pct_norm(dy)
                        
                        # 🚀 Enhanced Value Calculations
                        value_quality_score = 0
                        price_discount = 0
                        financial_strength = False
                        earnings_quality = False
                        
                        try:
                            # Multi-metric value assessment
                            value_metrics = []
                            if pe is not None and pe > 0:
                                value_metrics.append(min(20/pe, 3.0))  # Higher score for lower PE
                            if pb is not None and pb > 0:
                                value_metrics.append(min(2/pb, 3.0))   # Higher score for lower P/B
                            if eve is not None and eve > 0:
                                value_metrics.append(min(15/eve, 2.0)) # Higher score for lower EV/EBITDA
                            
                            # Calculate composite value quality (0-3 scale)
                            if value_metrics:
                                value_quality_score = sum(value_metrics) / len(value_metrics)
                            
                            # Price discount from historical levels
                            if high_52w is not None and low_52w is not None and high_52w > low_52w:
                                price_range_position = (price - low_52w) / (high_52w - low_52w)
                                price_discount = 3.0 * (1 - price_range_position)  # Higher score when closer to 52w low
                            
                            # Financial strength composite
                            strength_factors = []
                            if curr is not None and curr > 1.5:
                                strength_factors.append(True)
                            if de is not None and de < 0.3:
                                strength_factors.append(True)
                            if roe_n is not None and roe_n > 15:
                                strength_factors.append(True)
                            financial_strength = len(strength_factors) >= 2
                            
                            # Earnings quality indicators
                            quality_factors = []
                            if fcfy is not None and pct_norm(fcfy) > 8:
                                quality_factors.append(True)
                            if pio is not None and pio >= 7:
                                quality_factors.append(True)
                            if insider is not None and float(insider) > 0:
                                quality_factors.append(True)
                            earnings_quality = len(quality_factors) >= 2
                            
                        except Exception:
                            pass
                        
                        # 52w low from closes
                        near_52w_low = False
                        if low_52w is not None:
                            near_52w_low = price < (low_52w * 1.3)
                        # Enhanced value rules with better weighting
                        rules_v = [
                            # Core value metrics (higher weights)
                            (pe is not None and 0 < pe < 12.0, vw.get('pe_lt_12', 3.0), 'PE < 12 & > 0', {'pe': pe}),
                            (pb is not None and 0 < pb < 1.2, vw.get('pb_lt_1_2', 2.5), 'P/B < 1.2 & > 0', {'pb': pb}),
                            (eve is not None and eve < 8.0, vw.get('ev_ebitda_lt_8', 2.0), 'EV/EBITDA < 8', {'ev/ebitda': eve}),
                            
                            # Enhanced value indicators (NEW!)
                            (value_quality_score >= 2.0, vw.get('value_quality_high', 3.5), 'High Value Quality Score', {'score': value_quality_score}),
                            (price_discount >= 2.0, vw.get('price_discount_high', 2.5), 'High Price Discount', {'discount': price_discount}),
                            (financial_strength, vw.get('financial_strength', 2.0), 'Strong Financial Position', {}),
                            (earnings_quality, vw.get('earnings_quality', 2.0), 'High Earnings Quality', {}),
                            
                            # Price positioning
                            (near_52w_low, vw.get('price_lt_52wlow_1_3', 1.8), 'Price < 1.3×52w Low', {'price': price, '52w_low': low_52w}),
                            # Financial health (reweighted)
                            (de is not None and 0 <= de < 0.4, vw.get('de_lt_0_4', 1.5), 'Debt/Equity < 0.4', {'de': de}),
                            (curr is not None and curr > 1.5, vw.get('curr_ratio_gt_1_5', 1.5), 'Current Ratio > 1.5', {'current_ratio': curr}),
                            (roe_n is not None and roe_n > 12.0, vw.get('roe_gt_12', 1.8), 'ROE > 12%', {'roe%': roe_n}),
                            
                            # Income and yield
                            (dy_n is not None and dy_n > 3.0, vw.get('div_yield_gt_3', 1.5), 'Dividend Yield > 3%', {'div_yield%': dy_n}),
                            (fcfy is not None and pct_norm(fcfy) > 8.0, vw.get('fcf_yield_gt_8', 2.0), 'FCF Yield > 8%', {'fcf_yield%': pct_norm(fcfy) if fcfy is not None else None}),
                            
                            # Quality indicators
                            (bv is not None and bv > 0, vw.get('bookvalue_gt_0', 1.0), 'Book Value > 0', {'book_value': bv}),
                            (pio is not None and pio >= 7, vw.get('piotroski_gt_7', 1.8), 'Piotroski >= 7', {'piotroski': pio}),
                            (insider is not None and float(insider) > 0, vw.get('insider_buying_pos', 1.2), 'Insider Buying +', {'insider_trend': insider}),
                            
                            # Market positioning  
                            (mcap_val is not None and 100_000_000 <= float(mcap_val) <= 10_000_000_000, vw.get('mcap_in_range', 1.0), 'MarketCap 100M–10B', {'mcap': mcap_val}),
                            (pe is not None and ind_pe is not None and pe < ind_pe * 0.8, vw.get('pe_lt_industry', 1.5), 'PE < IndustryPE*0.8', {'pe': pe, 'ind_pe': ind_pe}),
                        ]
                        crit_v = sum(1 for ok, _wt, *_ in rules_v if ok)
                        score_v = float(sum(weight for ok, weight, *_ in rules_v if ok))
                        min_v = int(criteria.get('value_min_criteria') or 0)
                        if crit_v >= min_v:
                            matched_strategies.append('Value')
                            score = max(score, score_v)
                        breakdown_v = [{'name': name, 'ok': bool(ok), 'weight': wt, 'details': det} for (ok, wt, name, det) in rules_v]

                    # Strategy: Growth - 🚀 Enhanced with Growth Quality Metrics
                    if want_growth:
                        gw = criteria.get('growth_weights', {})
                        
                        # 🔍 Enhanced Growth Calculations
                        growth_momentum_score = 0
                        growth_acceleration = False
                        technical_strength = False
                        volume_growth = False
                        
                        try:
                            # Multi-timeframe growth momentum (0-4 scale)
                            if close_series is not None and len(close_series) >= 120:
                                returns_1m = (price / close_series.iloc[-21] - 1) * 100 if len(close_series) > 20 else 0
                                returns_3m = (price / close_series.iloc[-64] - 1) * 100 if len(close_series) > 63 else 0
                                returns_6m = (price / close_series.iloc[-126] - 1) * 100 if len(close_series) > 125 else 0
                                
                                # Reward sustained growth acceleration
                                if returns_1m > 8 and returns_3m > 15 and returns_6m > 25:
                                    growth_momentum_score = 4  # Exceptional
                                elif returns_1m > 5 and returns_3m > 10 and returns_6m > 15:
                                    growth_momentum_score = 3  # Strong
                                elif returns_1m > 2 and returns_3m > 5:
                                    growth_momentum_score = 2  # Good
                                elif returns_1m > 0 and returns_3m > 0:
                                    growth_momentum_score = 1  # Basic
                            
                            # Growth acceleration (rate increasing)
                            if close_series is not None and len(close_series) >= 60:
                                returns_recent = close_series.pct_change(20).iloc[-1] * 100
                                returns_earlier = close_series.pct_change(20).iloc[-21] * 100 
                                growth_acceleration = returns_recent > returns_earlier and returns_recent > 3
                            
                            # Technical strength (multiple indicators aligned)
                            strength_count = 0
                            if rsi_val > 55:
                                strength_count += 1
                            if macd_pos:
                                strength_count += 1
                            if ema_ok:
                                strength_count += 1
                            technical_strength = strength_count >= 2
                            
                            # Volume growth trend
                            if len(tail) >= 60 and 'volume' in tail.columns:
                                vol_recent = tail['volume'].tail(20).mean()
                                vol_earlier = tail['volume'].iloc[-40:-20].mean()
                                volume_growth = vol_recent > vol_earlier * 1.15
                                
                        except Exception:
                            pass
                        
                        # Classic growth indicators
                        try:
                            # above SMA50/200
                            g_sma50 = price >= float(tail['close'].rolling(50).mean().iloc[-1]) if close_series is not None else False
                        except Exception:
                            g_sma50 = False
                        try:
                            g_sma200 = price >= float(tail['close'].rolling(200).mean().iloc[-1]) if close_series is not None else False
                        except Exception:
                            g_sma200 = False
                        
                        # EMA cross, MACD, RSI band
                        g_ema = ema_ok
                        g_macd = macd_pos
                        # RSI band from config
                        g_rsi_min = int(criteria.get('growth_rsi_min') or 50)
                        g_rsi_max = int(criteria.get('growth_rsi_max') or 75)
                        g_rsi = (g_rsi_min <= rsi_val <= g_rsi_max)
                        
                        # 3-month momentum
                        g_mom3m = False
                        try:
                            lookback = int(criteria.get('growth_mom_lookback') or 60)
                            if close_series is not None and len(close_series) >= lookback:
                                g_mom3m = price > float(close_series.iloc[-lookback])
                        except Exception:
                            pass
                        
                        # Near 52w high (within 20%)
                        g_high = False
                        if high_52w is not None and high_52w > 0:
                            try:
                                within_pct = float(criteria.get('growth_within_high_pct') or 20.0)
                                factor = (100.0 - within_pct) / 100.0
                                g_high = price >= (factor * high_52w)
                            except Exception:
                                g_high = False
                        
                        # Volume trend up: vma20 rising vs 20 days ago
                        g_vol_up = False
                        try:
                            if 'volume' in tail.columns and len(tail) >= 40:
                                vma = tail['volume'].rolling(20).mean()
                                g_vol_up = float(vma.iloc[-1]) > float(vma.iloc[-20])
                        except Exception:
                            pass
                        # Enhanced growth rules with weighted scoring
                        rules_g = [
                            # Core growth indicators (higher weights)
                            (g_sma50, gw.get('price_above_sma50', 2.0), 'Price above SMA50', {}),
                            (g_sma200, gw.get('price_above_sma200', 2.5), 'Price above SMA200', {}),
                            (g_mom3m, gw.get('momentum_3m', 2.5), 'Price > 3M ago', {}),
                            
                            # Enhanced growth indicators (NEW!)
                            (growth_momentum_score >= 3, gw.get('growth_momentum_strong', 4.0), 'Strong Growth Momentum', {'score': growth_momentum_score}),
                            (growth_acceleration, gw.get('growth_acceleration', 2.5), 'Growth Acceleration', {}),
                            (technical_strength, gw.get('technical_strength', 2.0), 'Technical Strength', {}),
                            (volume_growth, gw.get('volume_growth', 1.8), 'Volume Growth Trend', {}),
                            
                            # Technical indicators (reweighted)
                            (g_ema, gw.get('ema_cross', 1.8), 'EMA Cross Bullish', {}),
                            (g_macd, gw.get('macd_positive', 1.5), 'MACD Histogram > 0', {}),
                            (g_rsi, gw.get('rsi_growth_range', 1.5), f'RSI in [{g_rsi_min},{g_rsi_max}]', {'rsi': rsi_val}),
                            
                            # Position strength
                            (g_high, gw.get('near_52w_high', 2.0), 'Near 52W High', {'high_52w': high_52w}),
                            (g_vol_up, gw.get('volume_trend_up', 1.2), 'Rising Volume Trend', {}),
                        ]
                        
                        crit_g = sum(1 for ok, _wt, *_ in rules_g if ok)
                        score_g = float(sum(weight for ok, weight, *_ in rules_g if ok))
                        min_g = int(criteria.get('growth_min_criteria') or 4)
                        if crit_g >= min_g:
                            matched_strategies.append('Growth')
                            score = max(score, score_g)
                        breakdown_g = [{'name': name, 'ok': bool(ok), 'weight': wt, 'details': det} for (ok, wt, name, det) in rules_g]

                    # Strategy: Oversold - 🔄 Enhanced with Recovery Potential Metrics
                    if want_oversold:
                        ow = criteria.get('oversold_weights', {})
                        
                        # Config-driven thresholds
                        o_rsi_max = float(criteria.get('oversold_rsi_max') or 35.0)
                        o_below_sma20_pct = float(criteria.get('oversold_below_sma20_pct') or 3.0)
                        o_daily_drop_pct = float(criteria.get('oversold_daily_drop_pct') or 3.0)
                        o_within_low_pct = float(criteria.get('oversold_within_low_pct') or 10.0)

                        # 🔍 Enhanced Oversold Calculations
                        oversold_severity = 0
                        recovery_potential = False
                        support_levels = False
                        volume_capitulation = False
                        
                        try:
                            # Multi-factor oversold severity (0-4 scale)
                            severity_factors = 0
                            if rsi_val <= 25:
                                severity_factors += 2  # Extreme oversold
                            elif rsi_val <= 35:
                                severity_factors += 1  # Moderate oversold
                                
                            # Price vs moving averages severity
                            if close_series is not None and len(close_series) >= 50:
                                sma20_val = close_series.rolling(20).mean().iloc[-1]
                                sma50_val = close_series.rolling(50).mean().iloc[-1]
                                
                                if price < sma20_val * 0.92:  # > 8% below SMA20
                                    severity_factors += 1
                                if price < sma50_val * 0.88:  # > 12% below SMA50
                                    severity_factors += 1
                                    
                            oversold_severity = min(severity_factors, 4)
                            
                            # Recovery potential indicators
                            recovery_signals = []
                            
                            # Bullish divergence proxy (price down, RSI stable/up)
                            if len(tail) >= 10 and close_series is not None:
                                price_trend = (price / close_series.iloc[-10] - 1) * 100
                                if price_trend < -5 and rsi_val > 35:  # Price down but RSI not extreme
                                    recovery_signals.append(True)
                            
                            # Support at key levels (bouncing from 52W low area)
                            if low_52w is not None and price <= low_52w * 1.05:  # Within 5% of 52W low
                                recovery_signals.append(True)
                                
                            # Institutional quality (market cap filter)
                            if mcap_val is not None and float(mcap_val) > 500_000_000:
                                recovery_signals.append(True)
                                
                            recovery_potential = len(recovery_signals) >= 2
                            
                            # Support level identification
                            if low_52w is not None and high_52w is not None:
                                support_zone = price <= low_52w * 1.15  # Near support
                                not_free_fall = price > low_52w * 0.9   # Not in free fall
                                support_levels = support_zone and not_free_fall
                            
                            # Volume capitulation (high volume on down day)
                            if len(tail) >= 20 and change_pct < -2:
                                avg_vol = tail['volume'].tail(20).mean()
                                volume_capitulation = volume > avg_vol * 1.5
                                
                        except Exception:
                            pass

                        # Classic oversold indicators
                        o_rsi = rsi_val <= o_rsi_max
                        o_sma20 = False
                        try:
                            if close_series is not None:
                                sma20_tmp = float(close_series.rolling(20).mean().iloc[-1])
                                factor = (100.0 - o_below_sma20_pct) / 100.0
                                o_sma20 = price < (sma20_tmp * factor)
                        except Exception:
                            pass
                        o_drop = change_pct <= -abs(o_daily_drop_pct)
                        o_near_low = False
                        if low_52w is not None:
                            try:
                                o_near_low = price <= (low_52w * (1.0 + (o_within_low_pct / 100.0)))
                            except Exception:
                                o_near_low = False
                        
                        # Enhanced oversold rules with weighted scoring
                        rules_o = [
                            # Core oversold indicators (higher weights)
                            (o_rsi, ow.get('rsi_oversold', 2.0), f'RSI <= {o_rsi_max}', {'rsi': rsi_val}),
                            (o_sma20, ow.get('below_sma20', 1.8), 'Price < SMA20*0.97', {}),
                            (o_near_low, ow.get('near_52w_low', 2.5), 'Near 52W Low', {'low_52w': low_52w}),
                            
                            # Enhanced oversold indicators (NEW!)
                            (oversold_severity >= 3, ow.get('severe_oversold', 3.5), 'Severe Oversold Condition', {'severity': oversold_severity}),
                            (recovery_potential, ow.get('recovery_potential', 3.0), 'High Recovery Potential', {}),
                            (support_levels, ow.get('support_levels', 2.0), 'At Support Levels', {}),
                            (volume_capitulation, ow.get('volume_capitulation', 1.8), 'Volume Capitulation', {}),
                            
                            # Price action
                            (o_drop, ow.get('daily_drop', 1.5), f'Daily Drop >= {o_daily_drop_pct}%', {'change_pct': change_pct}),
                            (change_pct <= -5.0, ow.get('big_drop', 2.0), 'Big Daily Drop >= 5%', {'change_pct': change_pct}),
                            
                            # Quality filters
                            (mcap_val is not None and float(mcap_val) > 100_000_000, ow.get('quality_stock', 1.0), 'Quality Stock (>100M)', {'mcap': mcap_val}),
                        ]
                        
                        crit_o = sum(1 for ok, _wt, *_ in rules_o if ok)
                        score_o = float(sum(weight for ok, weight, *_ in rules_o if ok))
                        min_o = int(criteria.get('oversold_min_criteria') or 3)
                        if crit_o >= min_o:
                            matched_strategies.append('Oversold')
                            score = max(score, score_o)
                        breakdown_o = [{'name': name, 'ok': bool(ok), 'weight': wt, 'details': det} for (ok, wt, name, det) in rules_o]

                    # If strategies are selected, require at least one match; otherwise keep base
                    require_strategy = (want_momentum or want_value or want_growth or want_oversold)
                    if require_strategy and not matched_strategies:
                        continue

                    if True:
                        symbol = symbol_upper
                        # Per-strategy score summary and counts
                        strategy_scores = {}
                        criteria_counts = {}
                        criteria_mins = {}
                        if want_momentum:
                            strategy_scores['Momentum'] = locals().get('score_m')
                            criteria_counts['Momentum'] = locals().get('crit_m')
                            criteria_mins['Momentum'] = int(criteria.get('momentum_min_criteria') or 0)
                        if want_value:
                            strategy_scores['Value'] = locals().get('score_v')
                            criteria_counts['Value'] = locals().get('crit_v')
                            criteria_mins['Value'] = int(criteria.get('value_min_criteria') or 0)
                        if want_growth:
                            strategy_scores['Growth'] = locals().get('score_g')
                            criteria_counts['Growth'] = locals().get('crit_g')
                            criteria_mins['Growth'] = int(criteria.get('growth_min_criteria') or 4)
                        if want_oversold:
                            strategy_scores['Oversold'] = locals().get('score_o')
                            criteria_counts['Oversold'] = locals().get('crit_o')
                            criteria_mins['Oversold'] = int(criteria.get('oversold_min_criteria') or 3)
                        # Compose prediction fields if available
                        pred_price = None
                        pred_ret = None
                        try:
                            info = pred_info_map.get(symbol_upper)
                            if info is not None:
                                pred_ret = info.get('pred_return')
                                if info.get('pred_price') is not None:
                                    pred_price = float(info.get('pred_price'))
                                elif pred_ret is not None and price is not None:
                                    try:
                                        pr = float(pred_ret)
                                        # If looks like percent (e.g., 5 or -3), convert; if ratio (0.05), keep
                                        if abs(pr) > 1.5:
                                            pr = pr / 100.0
                                        pred_price = float(price * (1.0 + pr))
                                    except Exception:
                                        pred_price = None
                        except Exception:
                            pred_price = None
                            pred_ret = None

                        result = {
                            'symbol': symbol,
                            'price': price,
                            'change_pct': change_pct,
                            'volume': volume,
                            'rsi': rsi_val,
                            'pe_ratio': float('nan'),
                            'market_cap': float('nan'),
                            'score': score,
                            'strategy': ", ".join(matched_strategies),
                            'strategy_scores': strategy_scores,
                            'criteria_counts': criteria_counts,
                            'criteria_mins': criteria_mins,
                            'pred_price': pred_price,
                            'pred_return': pred_ret,
                            'breakdown': {
                                'Momentum': locals().get('breakdown_m'),
                                'Value': locals().get('breakdown_v'),
                                'Growth': locals().get('breakdown_g'),
                                'Oversold': locals().get('breakdown_o'),
                            }
                        }
                        # חישוב Price Target, Stop Loss, Signal
                        # Price Target: תחזית מחיר (pred_price) אם קיימת, אחרת ריק
                        price_target = result.get('pred_price')
                        if price_target is None:
                            price_target = ''
                        result['price_target'] = price_target
                        # Stop Loss: מחיר נוכחי פחות 5%
                        stop_loss = price * 0.95
                        result['stop_loss'] = round(stop_loss, 2)
                        # Signal: BUY אם תחזית חיובית, SELL אם שלילית, HOLD אם קרוב לאפס
                        pred_ret = result.get('pred_return')
                        if pred_ret is not None:
                            try:
                                pr = float(pred_ret)
                                if abs(pr) > 1.5:
                                    pr = pr / 100.0
                                if pr > 0.02:
                                    result['signal'] = 'BUY'
                                elif pr < -0.02:
                                    result['signal'] = 'SELL'
                                else:
                                    result['signal'] = 'HOLD'
                            except Exception:
                                result['signal'] = '-'
                        else:
                            result['signal'] = '-'
                        results.append(result)
                        # Stream result immediately
                        try:
                            self.result_found.emit(result)
                        except Exception:
                            pass
                    processed += 1
                except Exception as ie:
                    errors += 1
                    try:
                        if len(err_samples) < 5:
                            err_samples.append(f"{sym}: {ie}")
                        self.logger.warning(f"Scan read error for {sym}: {ie}")
                    except Exception:
                        pass
                # progress
                if idx % 25 == 0 or idx == total:
                    prog = int(idx / total * 100)
                    # ETA calculation based on throughput so far
                    elapsed = max(time.time() - start_ts, 0.001)
                    rate = idx / elapsed
                    remain = max(total - idx, 0)
                    eta_sec = int(remain / rate) if rate > 0 else 0
                    mm, ss = divmod(eta_sec, 60)
                    eta_str = f"{mm:02d}:{ss:02d}"
                    self.progress_updated.emit(prog)
                    extra = (" | " + "; ".join(err_samples)) if errors and err_samples else ""
                    self.status_updated.emit(
                        f"Scanning… {idx}/{total} | processed={processed} errors={errors}{extra} | ETA {eta_str}"
                    )

            if self.is_scanning:
                # Apply post-processing filters for final quality control
                results = self._apply_post_processing_filters(results)
                
                self.results_updated.emit(results)
                self.scan_completed.emit(len(results))
                extra = (" | " + "; ".join(err_samples)) if errors and err_samples else ""
                self.status_updated.emit(f"Scan complete: {len(results)} matches | processed={processed} errors={errors}{extra}")

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.is_scanning = False
    
    def stop_scan(self):
        """Stop scanning process"""
        self.is_scanning = False


class ScanCriteriaWidget(QFrame):
    """Widget for setting scan criteria"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup criteria UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(self)
        # Create criteria panels and show side-by-side using a splitter
        self.create_price_volume_tab()
        self.create_technical_tab()
        from PyQt6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.price_volume_widget)
        splitter.addWidget(self.technical_widget)
        splitter.setSizes([380, 380])

        # Also keep a tab fallback (hidden by default) for narrow windows.
        # Do NOT add the same widgets to the QTabWidget (that would reparent them
        # and leave the splitter empty). Instead add lightweight placeholder
        # pages that point users to the compact tab view when activated.
        self.criteria_tabs = QTabWidget()
        tab_price = QWidget()
        tab_price.setLayout(QVBoxLayout(tab_price))
        tab_price.layout().addWidget(QLabel("Price & Volume (compact tab view)"))
        tab_tech = QWidget()
        tab_tech.setLayout(QVBoxLayout(tab_tech))
        tab_tech.layout().addWidget(QLabel("Technical (compact tab view)"))
        self.criteria_tabs.addTab(tab_price, "💰 Price & Volume")
        self.criteria_tabs.addTab(tab_tech, "📈 Technical")

        # Group: Scan Criteria (wrap splitter)
        criteria_group = QGroupBox("Scan Criteria")
        cg_layout = QVBoxLayout(criteria_group)
        cg_layout.addWidget(splitter)
        cg_layout.addWidget(self.criteria_tabs)
        self.criteria_tabs.setVisible(False)

        # Group: Quick Scan Presets (narrow, vertical) with per-preset strategy checkboxes
        self.create_preset_buttons()
        presets_group = QGroupBox("Quick Scan Presets")
        pg_layout = QVBoxLayout(presets_group)
        pg_layout.setContentsMargins(6, 6, 6, 6)
        pg_layout.setSpacing(6)
        pg_layout.addWidget(self.preset_frame)

        # Create strategy configuration groups (will be added into a scroll area on the right)
        self.create_momentum_group()
        self.create_value_group()
        self.create_growth_group()
        self.create_oversold_group()
        # Start hidden
        for g in [self.momentum_group, self.value_group, self.growth_group, self.oversold_group]:
            g.setVisible(False)

        # Right panel: only scroll area for strategy configs
        from PyQt6.QtWidgets import QScrollArea
        right_panel = QWidget()
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(0, 0, 0, 0)
        rp_layout.setSpacing(6)
        right_panel.setMinimumWidth(380)
        self.strategy_configs_area = QScrollArea()
        self.strategy_configs_area.setWidgetResizable(True)
        self.strategy_configs_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.strategy_configs_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.strategy_configs_area.setMinimumHeight(320)
        # Allow taller area to show full strategy configs
        self.strategy_configs_area.setMaximumHeight(900)
        cfg_container = QWidget()
        cfg_layout = QVBoxLayout(cfg_container)
        cfg_layout.setContentsMargins(6, 6, 6, 6)
        cfg_layout.setSpacing(6)
        cfg_layout.addWidget(self.momentum_group)
        cfg_layout.addWidget(self.value_group)
        cfg_layout.addWidget(self.growth_group)
        cfg_layout.addWidget(self.oversold_group)
        cfg_layout.addStretch(1)
        self.strategy_configs_area.setWidget(cfg_container)
        # Always keep visible so configuration is reachable
        self.strategy_configs_area.setVisible(True)
        rp_layout.addWidget(self.strategy_configs_area, 1)

        # Top row: three areas side by side (criteria | presets | right panel)
        top_row = QWidget()
        h = QHBoxLayout(top_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(20)
        # Make criteria/presets wider to reduce crowding
        from PyQt6.QtWidgets import QSizePolicy
        # Allow the criteria area to be wider so content isn't hidden; prefer flexible sizing
        criteria_group.setMinimumWidth(420)
        criteria_group.setMaximumWidth(720)
        presets_group.setMaximumWidth(340)
        criteria_group.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred))
        presets_group.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred))
        h.addWidget(criteria_group, 0)
        h.addWidget(presets_group, 0)
        h.addWidget(right_panel, 1)
        layout.addWidget(top_row)

        # Toggle visibility when strategy checkboxes change and manage scroll area visibility
        try:
            # These checkboxes are now the ones next to presets
            self.momentum_chk.toggled.connect(lambda v: self._on_strategy_toggled('momentum', v))
            self.value_chk.toggled.connect(lambda v: self._on_strategy_toggled('value', v))
            self.growth_chk.toggled.connect(lambda v: self._on_strategy_toggled('growth', v))
            self.oversold_chk.toggled.connect(lambda v: self._on_strategy_toggled('oversold', v))
        except Exception:
            pass

    def create_strategy_toggle_bar(self) -> QWidget:
        # Deprecated: toggles now live next to presets; keep stub for compatibility if referenced elsewhere
        bar = QWidget()
        return bar

    def _on_strategy_toggled(self, name: str, visible: bool):
        group_map = {
            'momentum': getattr(self, 'momentum_group', None),
            'value': getattr(self, 'value_group', None),
            'growth': getattr(self, 'growth_group', None),
            'oversold': getattr(self, 'oversold_group', None),
        }
        grp = group_map.get(name)
        if grp is not None:
            grp.setVisible(bool(visible))
        # Sync enable checkbox inside group for Momentum/Value
        try:
            if name == 'momentum' and hasattr(self, 'momentum_enable'):
                self.momentum_enable.setChecked(bool(visible))
            if name == 'value' and hasattr(self, 'value_enable'):
                self.value_enable.setChecked(bool(visible))
            if name == 'growth' and hasattr(self, 'growth_enable'):
                self.growth_enable.setChecked(bool(visible))
            if name == 'oversold' and hasattr(self, 'oversold_enable'):
                self.oversold_enable.setChecked(bool(visible))
        except Exception:
            pass
        # Keep the scroll area visible at all times to avoid disappearing behavior
        self.strategy_configs_area.setVisible(True)
        # Also update results table column visibility live
        try:
            parent = self.parent()
            while parent is not None and not hasattr(parent, 'results_table'):
                parent = parent.parent()
            if parent is not None and hasattr(parent, 'results_table'):
                col_idx = {'momentum':7, 'value':8, 'growth':9, 'oversold':10}.get(name)
                if col_idx is not None:
                    parent.results_table.setColumnHidden(col_idx, not bool(visible))
        except Exception:
            pass
    
    def create_price_volume_tab(self):
        """Create price and volume criteria tab"""
        self.price_volume_widget = QWidget()
        layout = QFormLayout(self.price_volume_widget)
        
        # Price range
        self.min_price_spin = QDoubleSpinBox()
        self.min_price_spin.setRange(0, 10000)
        self.min_price_spin.setValue(0)
        self.min_price_spin.setPrefix("$")
        layout.addRow("Min Price:", self.min_price_spin)
        
        self.max_price_spin = QDoubleSpinBox()
        self.max_price_spin.setRange(0, 10000)
        self.max_price_spin.setValue(1000)
        self.max_price_spin.setPrefix("$")
        layout.addRow("Max Price:", self.max_price_spin)
        
        # Volume
        self.min_volume_spin = QSpinBox()
        self.min_volume_spin.setRange(0, 1000000000)
        self.min_volume_spin.setValue(100000)
        self.min_volume_spin.setSuffix(" shares")
        layout.addRow("Min Volume:", self.min_volume_spin)
        
        # Change percentage
        self.min_change_spin = QDoubleSpinBox()
        self.min_change_spin.setRange(-100, 100)
        self.min_change_spin.setValue(-100)
        self.min_change_spin.setSuffix("%")
        layout.addRow("Min Change:", self.min_change_spin)
        
        self.max_change_spin = QDoubleSpinBox()
        self.max_change_spin.setRange(-100, 100)
        self.max_change_spin.setValue(100)
        self.max_change_spin.setSuffix("%")
        layout.addRow("Max Change:", self.max_change_spin)
    
    def create_technical_tab(self):
        """Create technical criteria tab"""
        self.technical_widget = QWidget()
        layout = QFormLayout(self.technical_widget)
        
        # RSI
        self.min_rsi_spin = QSpinBox()
        self.min_rsi_spin.setRange(0, 100)
        self.min_rsi_spin.setValue(0)
        layout.addRow("Min RSI:", self.min_rsi_spin)
        
        self.max_rsi_spin = QSpinBox()
        self.max_rsi_spin.setRange(0, 100)
        self.max_rsi_spin.setValue(100)
        layout.addRow("Max RSI:", self.max_rsi_spin)
        
        # Moving averages
        self.above_sma20 = QCheckBox("Above 20-day SMA")
        layout.addRow("", self.above_sma20)
        
        self.above_sma50 = QCheckBox("Above 50-day SMA")
        layout.addRow("", self.above_sma50)
        
        self.above_sma200 = QCheckBox("Above 200-day SMA")
        layout.addRow("", self.above_sma200)
        
    # Pattern detection not implemented; omitted
    
    def create_fundamental_tab(self):
        """Deprecated: Fundamental filters not supported in current build."""
        self.fundamental_widget = QWidget()
        layout = QFormLayout(self.fundamental_widget)
        note = QLabel("Fundamental filters are not available in this build.")
        layout.addRow(note)

    def create_growth_group(self):
        """Create Growth strategy configuration."""
        self.growth_group = QGroupBox("Growth Strategy")
        form = QFormLayout(self.growth_group)
        # Enable toggle
        self.growth_enable = QCheckBox("Enable Growth scoring")
        self.growth_enable.setChecked(False)
        form.addRow(self.growth_enable)
        # Thresholds
        self.g_min_criteria = QSpinBox(); self.g_min_criteria.setRange(0, 10); self.g_min_criteria.setValue(4)
        self.g_within_high_pct = QDoubleSpinBox(); self.g_within_high_pct.setRange(0, 100); self.g_within_high_pct.setValue(15); self.g_within_high_pct.setSuffix("%")
        self.g_rsi_min = QSpinBox(); self.g_rsi_min.setRange(0, 100); self.g_rsi_min.setValue(50)
        self.g_rsi_max = QSpinBox(); self.g_rsi_max.setRange(0, 100); self.g_rsi_max.setValue(70)
        self.g_mom_days = QSpinBox(); self.g_mom_days.setRange(20, 252); self.g_mom_days.setValue(60)
        form.addRow("Min criteria:", self.g_min_criteria)
        form.addRow("Within X% of 52w high:", self.g_within_high_pct)
        form.addRow("RSI min:", self.g_rsi_min)
        form.addRow("RSI max:", self.g_rsi_max)
        form.addRow("Momentum lookback (days):", self.g_mom_days)

    def create_oversold_group(self):
        """Create Oversold strategy configuration."""
        self.oversold_group = QGroupBox("Oversold Strategy")
        form = QFormLayout(self.oversold_group)
        # Enable toggle
        self.oversold_enable = QCheckBox("Enable Oversold scoring")
        self.oversold_enable.setChecked(False)
        form.addRow(self.oversold_enable)
        # Thresholds
        self.o_min_criteria = QSpinBox(); self.o_min_criteria.setRange(0, 10); self.o_min_criteria.setValue(3)
        self.o_rsi_max = QSpinBox(); self.o_rsi_max.setRange(0, 100); self.o_rsi_max.setValue(30)
        self.o_below_sma20_pct = QDoubleSpinBox(); self.o_below_sma20_pct.setRange(0, 100); self.o_below_sma20_pct.setValue(3); self.o_below_sma20_pct.setSuffix("%")
        self.o_daily_drop_pct = QDoubleSpinBox(); self.o_daily_drop_pct.setRange(0, 100); self.o_daily_drop_pct.setValue(3); self.o_daily_drop_pct.setSuffix("%")
        self.o_within_low_pct = QDoubleSpinBox(); self.o_within_low_pct.setRange(0, 100); self.o_within_low_pct.setValue(10); self.o_within_low_pct.setSuffix("%")
        form.addRow("Min criteria:", self.o_min_criteria)
        form.addRow("RSI ≤:", self.o_rsi_max)
        form.addRow("Below SMA20 by ≥:", self.o_below_sma20_pct)
        form.addRow("Daily drop ≥:", self.o_daily_drop_pct)
        form.addRow("Within X% of 52w low:", self.o_within_low_pct)
    
    def create_preset_buttons(self):
        """Create preset scan buttons"""
        self.preset_frame = QFrame()
        self.preset_frame.setFrameStyle(QFrame.Shape.Box)
        
        layout = QVBoxLayout(self.preset_frame)
        
        # Header row with Run All checkbox aligned to the right of the label
        header_row = QWidget()
        hr = QHBoxLayout(header_row)
        hr.setContentsMargins(0, 0, 0, 0)
        hr.setSpacing(6)
        preset_label = QLabel("Quick Scan Presets")
        preset_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        hr.addWidget(preset_label)
        hr.addStretch(1)
        self.run_all_chk = QCheckBox("Run All")
        # Match width with item checkboxes so header aligns
        try:
            self.run_all_chk.setFixedWidth(80)
        except Exception:
            pass
        hr.addWidget(self.run_all_chk)
        layout.addWidget(header_row)

        # Buttons stacked vertically with inline strategy checkboxes aligned to the right
        button_layout = QVBoxLayout()

        # Create or reuse strategy checkboxes to live next to their presets
        self.growth_chk = QCheckBox("Growth")
        self.value_chk = QCheckBox("Value")
        self.momentum_chk = QCheckBox("Momentum")
        self.oversold_chk = QCheckBox("Oversold")

        def add_preset_row(text: str, callback, chk: QCheckBox):
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            # Increase spacing between the button and the checkbox
            hl.setSpacing(10)
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setFixedHeight(30)
            btn.setMinimumWidth(160)
            btn.setMaximumWidth(220)
            btn.setStyleSheet("QPushButton { font-size: 11px; padding: 2px 8px; }")
            hl.addWidget(btn, 1)
            # Add a small fixed spacer to visibly separate button and checkbox
            hl.addSpacing(8)
            # Ensure checkboxes align vertically in a column
            chk.setFixedWidth(80)
            hl.addWidget(chk, 0, Qt.AlignmentFlag.AlignRight)
            button_layout.addWidget(row)

        add_preset_row("🚀 Growth Stocks", self.load_growth_preset, self.growth_chk)
        add_preset_row("💎 Value Stocks", self.load_value_preset, self.value_chk)
        add_preset_row("📈 Momentum", self.load_momentum_preset, self.momentum_chk)
        add_preset_row("🔻 Oversold", self.load_oversold_preset, self.oversold_chk)

        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        # Run All behavior: toggles all preset checkboxes
        def on_all(_state):
            v = self.run_all_chk.isChecked()
            for c in [self.growth_chk, self.value_chk, self.momentum_chk, self.oversold_chk]:
                c.setChecked(v)
        self.run_all_chk.stateChanged.connect(on_all)

    def create_strategies_group(self):
        """Create strategy selection group (run all or individual)."""
        self.strategies_group = QGroupBox("Strategies to run")
        hl = QHBoxLayout(self.strategies_group)
        self.run_all_chk = QCheckBox("Run All")
        self.momentum_chk = QCheckBox("Momentum")
        self.value_chk = QCheckBox("Value")
        self.growth_chk = QCheckBox("Growth")
        self.oversold_chk = QCheckBox("Oversold")
        self.momentum_chk.setChecked(False)
        self.value_chk.setChecked(False)
        self.growth_chk.setChecked(False)
        self.oversold_chk.setChecked(False)
        def on_all(state):
            v = self.run_all_chk.isChecked()
            self.momentum_chk.setChecked(v)
            self.value_chk.setChecked(v)
            self.growth_chk.setChecked(v)
            self.oversold_chk.setChecked(v)
        self.run_all_chk.stateChanged.connect(on_all)
        for w in [self.run_all_chk, self.momentum_chk, self.value_chk, self.growth_chk, self.oversold_chk]:
            hl.addWidget(w)
        hl.addStretch()

    def create_value_group(self):
        """Create Value strategy configuration with weights and threshold."""
        self.value_group = QGroupBox("Value Strategy")
        form = QFormLayout(self.value_group)
        self.value_enable = QCheckBox("Enable Value scoring")
        self.value_enable.setChecked(False)
        form.addRow(self.value_enable)

        def wspin(default: int, maxv: int = 10) -> QSpinBox:
            sb = QSpinBox(); sb.setRange(0, maxv); sb.setValue(default); sb.setFixedWidth(70); return sb

        # Define weight spinboxes per rule (defaults from spec)
        self.vw_pe_lt_15 = wspin(4)
        self.vw_pb_lt_1_5 = wspin(4)
        self.vw_price_lt_52wlow_1_3 = wspin(3)
        self.vw_de_lt_0_5 = wspin(2)
        self.vw_curr_ratio_gt_1_2 = wspin(2)
        self.vw_roe_gt_10 = wspin(3)
        self.vw_bookvalue_gt_0 = wspin(2)
        self.vw_div_yield_gt_2 = wspin(2)
        self.vw_mcap_range = wspin(1)
        self.vw_pe_lt_industry = wspin(3)
        self.vw_fcf_yield_gt_5 = wspin(2)
        self.vw_ev_ebitda_lt_10 = wspin(2)
        self.vw_piotroski_gt_6 = wspin(2)
        self.vw_insider_buying = wspin(1)

        form.addRow("PE < 15 & > 0:", self.vw_pe_lt_15)
        form.addRow("P/B < 1.5 & > 0:", self.vw_pb_lt_1_5)
        form.addRow("Price < 52w Low * 1.3:", self.vw_price_lt_52wlow_1_3)
        form.addRow("Debt/Equity < 0.5:", self.vw_de_lt_0_5)
        form.addRow("Current Ratio > 1.2:", self.vw_curr_ratio_gt_1_2)
        form.addRow("ROE > 10%:", self.vw_roe_gt_10)
        form.addRow("Book Value > 0:", self.vw_bookvalue_gt_0)
        form.addRow("Dividend Yield > 2%:", self.vw_div_yield_gt_2)
        form.addRow("MarketCap 50M–5B:", self.vw_mcap_range)
        form.addRow("PE < IndustryPE * 0.9:", self.vw_pe_lt_industry)
        form.addRow("FCF Yield > 5%:", self.vw_fcf_yield_gt_5)
        form.addRow("EV/EBITDA < 10:", self.vw_ev_ebitda_lt_10)
        form.addRow("Piotroski > 6:", self.vw_piotroski_gt_6)
        form.addRow("Insider Buying +:", self.vw_insider_buying)

        self.min_value_criteria = QSpinBox(); self.min_value_criteria.setRange(0, 20); self.min_value_criteria.setValue(8)
        form.addRow("Minimum criteria met:", self.min_value_criteria)

    def create_momentum_group(self):
        """Create Momentum strategy configuration with weights and threshold."""
        self.momentum_group = QGroupBox("Momentum Strategy")
        form = QFormLayout(self.momentum_group)

        # Enable toggle
        self.momentum_enable = QCheckBox("Enable Momentum scoring")
        self.momentum_enable.setChecked(False)
        form.addRow(self.momentum_enable)

        # Helper to create weight spinbox
        def wspin(default: int, maxv: int = 10) -> QSpinBox:
            sb = QSpinBox()
            sb.setRange(0, maxv)
            sb.setValue(default)
            return sb

        # EMA tunables (defaults: 5 and 10)
        self.m_ema_short = QSpinBox(); self.m_ema_short.setRange(2, 200); self.m_ema_short.setValue(5)
        self.m_ema_long = QSpinBox(); self.m_ema_long.setRange(3, 400); self.m_ema_long.setValue(10)
        form.addRow("EMA short length:", self.m_ema_short)
        form.addRow("EMA long length:", self.m_ema_long)

        # Weights
        self.w_price_gt_sma10 = wspin(3)
        self.w_price_gt_sma20 = wspin(3)
        self.w_sma10_gt_sma20 = wspin(2)
        self.w_rsi_50_75 = wspin(3)
        self.w_vol_gt_vma20_1_2 = wspin(2)
        self.w_atr14_2_5pct = wspin(2)
        self.w_daily_change_gt_1 = wspin(1)
        self.w_macd_hist_pos = wspin(1)
        self.w_ema_short_gt_long = wspin(1)
        self.w_mcap_gt_100m = wspin(1)
        self.w_vol30_15_40 = wspin(1)

        form.addRow("Price > SMA(10) * 1.015:", self.w_price_gt_sma10)
        form.addRow("Price > SMA(20) * 1.02:", self.w_price_gt_sma20)
        form.addRow("SMA(10) > SMA(20) * 1.005:", self.w_sma10_gt_sma20)
        form.addRow("RSI(14) ∈ [50, 75]:", self.w_rsi_50_75)
        form.addRow("Volume > SMA(Vol,20) * 1.2:", self.w_vol_gt_vma20_1_2)
        form.addRow("ATR(14) ∈ [2%, 5%]:", self.w_atr14_2_5pct)
        form.addRow("Daily %Change > 1%:", self.w_daily_change_gt_1)
        form.addRow("MACD Histogram > 0:", self.w_macd_hist_pos)
        form.addRow("EMA(short) > EMA(long):", self.w_ema_short_gt_long)
        form.addRow("MarketCap > 100M:", self.w_mcap_gt_100m)
        form.addRow("Volatility 30d ∈ [15%, 40%]:", self.w_vol30_15_40)

        self.min_momentum_criteria = QSpinBox()
        self.min_momentum_criteria.setRange(0, 11)
        self.min_momentum_criteria.setValue(8)
        form.addRow("Minimum criteria met:", self.min_momentum_criteria)
    
    def load_growth_preset(self):
        """Load growth stock preset"""
        self.min_price_spin.setValue(20)
        self.max_price_spin.setValue(1000)
        self.min_volume_spin.setValue(500000)
        self.min_change_spin.setValue(-2)
        self.max_change_spin.setValue(10)
        self.above_sma20.setChecked(True)
        self.above_sma50.setChecked(True)
    
    def load_value_preset(self):
        """Load value stock preset"""
        self.min_price_spin.setValue(5)
        self.max_price_spin.setValue(100)
        self.min_volume_spin.setValue(100000)
        # Fundamental filters not available in this build; keep technical/price presets only
    
    def load_momentum_preset(self):
        """Load momentum preset"""
        self.min_change_spin.setValue(3)
        self.max_change_spin.setValue(20)
        self.min_volume_spin.setValue(1000000)
        self.min_rsi_spin.setValue(50)
        self.max_rsi_spin.setValue(80)
    
    def load_oversold_preset(self):
        """Load oversold preset"""
        self.min_rsi_spin.setValue(20)
        self.max_rsi_spin.setValue(35)
        self.min_change_spin.setValue(-10)
        self.max_change_spin.setValue(-2)
    
    def get_criteria(self) -> dict:
        """Get current scan criteria"""
        crit = {
            'min_price': self.min_price_spin.value(),
            'max_price': self.max_price_spin.value(),
            'min_volume': self.min_volume_spin.value(),
            'min_change': self.min_change_spin.value(),
            'max_change': self.max_change_spin.value(),
            'min_rsi': self.min_rsi_spin.value(),
            'max_rsi': self.max_rsi_spin.value(),
            'above_sma20': self.above_sma20.isChecked(),
            'above_sma50': self.above_sma50.isChecked(),
            'above_sma200': self.above_sma200.isChecked(),
        }
        # Strategy selection
        crit['strategy_momentum'] = self.momentum_chk.isChecked() or self.momentum_enable.isChecked()
        crit['strategy_value'] = self.value_chk.isChecked() or self.value_enable.isChecked()
        crit['strategy_growth'] = self.growth_chk.isChecked() or getattr(self, 'growth_enable', QCheckBox()).isChecked()
        crit['strategy_oversold'] = self.oversold_chk.isChecked() or getattr(self, 'oversold_enable', QCheckBox()).isChecked()

        # Momentum config
        crit['momentum_enabled'] = self.momentum_enable.isChecked()
        crit['momentum_min_criteria'] = self.min_momentum_criteria.value()
        crit['momentum_ema_short'] = getattr(self, 'm_ema_short', QSpinBox()).value() or 5
        crit['momentum_ema_long'] = getattr(self, 'm_ema_long', QSpinBox()).value() or 10
        crit['momentum_weights'] = {
            'price_gt_sma10_1015': self.w_price_gt_sma10.value(),
            'price_gt_sma20_1020': self.w_price_gt_sma20.value(),
            'sma10_gt_sma20_1005': self.w_sma10_gt_sma20.value(),
            'rsi_50_75': self.w_rsi_50_75.value(),
            'vol_gt_vma20_1_2': self.w_vol_gt_vma20_1_2.value(),
            'atr14_2_5pct': self.w_atr14_2_5pct.value(),
            'daily_change_gt_1': self.w_daily_change_gt_1.value(),
            'macd_hist_pos': self.w_macd_hist_pos.value(),
            'ema_short_gt_long': self.w_ema_short_gt_long.value(),
            'mcap_gt_100m': self.w_mcap_gt_100m.value(),
            'vol30_15_40': self.w_vol30_15_40.value(),
        }
        # Value config
        crit['value_enabled'] = self.value_enable.isChecked()
        crit['value_min_criteria'] = self.min_value_criteria.value()
        crit['value_weights'] = {
            'pe_lt_15': self.vw_pe_lt_15.value(),
            'pb_lt_1_5': self.vw_pb_lt_1_5.value(),
            'price_lt_52wlow_1_3': self.vw_price_lt_52wlow_1_3.value(),
            'de_lt_0_5': self.vw_de_lt_0_5.value(),
            'curr_ratio_gt_1_2': self.vw_curr_ratio_gt_1_2.value(),
            'roe_gt_10': self.vw_roe_gt_10.value(),
            'bookvalue_gt_0': self.vw_bookvalue_gt_0.value(),
            'div_yield_gt_2': self.vw_div_yield_gt_2.value(),
            'mcap_in_range': self.vw_mcap_range.value(),
            'pe_lt_industry': self.vw_pe_lt_industry.value(),
            'fcf_yield_gt_5': self.vw_fcf_yield_gt_5.value(),
            'ev_ebitda_lt_10': self.vw_ev_ebitda_lt_10.value(),
            'piotroski_gt_6': self.vw_piotroski_gt_6.value(),
            'insider_buying_pos': self.vw_insider_buying.value(),
        }
        # Growth config
        crit['growth_enabled'] = getattr(self, 'growth_enable', QCheckBox()).isChecked()
        crit['growth_min_criteria'] = getattr(self, 'g_min_criteria', QSpinBox()).value()
        crit['growth_rsi_min'] = getattr(self, 'g_rsi_min', QSpinBox()).value()
        crit['growth_rsi_max'] = getattr(self, 'g_rsi_max', QSpinBox()).value()
        crit['growth_within_high_pct'] = getattr(self, 'g_within_high_pct', QDoubleSpinBox()).value()
        crit['growth_mom_lookback'] = getattr(self, 'g_mom_days', QSpinBox()).value()
        # Oversold config
        crit['oversold_enabled'] = getattr(self, 'oversold_enable', QCheckBox()).isChecked()
        crit['oversold_min_criteria'] = getattr(self, 'o_min_criteria', QSpinBox()).value()
        crit['oversold_rsi_max'] = getattr(self, 'o_rsi_max', QSpinBox()).value()
        crit['oversold_below_sma20_pct'] = getattr(self, 'o_below_sma20_pct', QDoubleSpinBox()).value()
        crit['oversold_daily_drop_pct'] = getattr(self, 'o_daily_drop_pct', QDoubleSpinBox()).value()
        crit['oversold_within_low_pct'] = getattr(self, 'o_within_low_pct', QDoubleSpinBox()).value()
        return crit


class ScanResultsTable(QTableWidget):
    """Table for displaying scan results"""
    
    symbol_selected = pyqtSignal(str)
    add_to_watchlist = pyqtSignal(str)
    open_chart = pyqtSignal(str)
    explain_requested = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup results table"""
        # Set columns: add per-strategy score columns
        self.setColumnCount(17)
        headers = [
            "Symbol", "Price", "Change %", "Volume", "RSI",
            "Score", "Strategy",
            "Mom S", "Val S", "Gro S", "Over S",
            "Pred Price", "Pred %",
            "Price Target", "Stop Loss", "Signal", "AI Rating", "Actions"
        ]
        self.setHorizontalHeaderLabels(headers)
        
        # Configure table
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        # Configure headers
        header = self.horizontalHeader()
        # Allow user to resize and reorder columns
        header.setSectionsMovable(True)
        header.setStretchLastSection(False)
        
        # Set column widths
        self.setColumnWidth(0, 70)    # Symbol
        self.setColumnWidth(1, 80)    # Price
        self.setColumnWidth(2, 80)    # Change %
        self.setColumnWidth(3, 90)    # Volume
        self.setColumnWidth(4, 54)    # RSI
        self.setColumnWidth(5, 64)    # Score
        self.setColumnWidth(6, 90)    # Strategy (narrower)
        self.setColumnWidth(7, 58)    # Momentum Score
        self.setColumnWidth(8, 58)    # Value Score
        self.setColumnWidth(9, 58)    # Growth Score
        self.setColumnWidth(10, 58)   # Oversold Score
        self.setColumnWidth(11, 90)   # Pred Price
        self.setColumnWidth(12, 64)   # Pred %
        self.setColumnWidth(13, 90)   # Price Target
        self.setColumnWidth(14, 90)   # Stop Loss
        self.setColumnWidth(15, 90)   # Signal
        self.setColumnWidth(16, 90)   # AI Rating
        self.setColumnWidth(17, 88)   # Actions
        # Removed P/E and Market Cap columns in this build
        
        # Enable sorting
        self.setSortingEnabled(True)
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Double click
        self.cellDoubleClicked.connect(self.on_cell_double_clicked)

    def append_result(self, result: dict):
        row = self.rowCount()
        self.setRowCount(row + 1)
        self._render_row(row, result)
    
    def update_results(self, results: list):
        """Update table with scan results"""
        self.setRowCount(len(results))
        for row, result in enumerate(results):
            self._render_row(row, result)
        
        # Double click
        self.cellDoubleClicked.connect(self.on_cell_double_clicked)

    def _render_row(self, row: int, result: dict):
        # Symbol
            symbol_item = QTableWidgetItem(result['symbol'])
            symbol_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.setItem(row, 0, symbol_item)
            
            # Price
            price_item = QTableWidgetItem(f"${result['price']:.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, 1, price_item)
            
            # Change %
            change_item = QTableWidgetItem(f"{result['change_pct']:+.2f}%")
            change_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            if result['change_pct'] >= 0:
                change_item.setForeground(QColor("#4CAF50"))
            else:
                change_item.setForeground(QColor("#f44336"))
            self.setItem(row, 2, change_item)
            
            # Volume
            volume_item = QTableWidgetItem(f"{result['volume']:,}")
            volume_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.setItem(row, 3, volume_item)
            
            # RSI
            rsi_item = QTableWidgetItem(f"{result['rsi']:.1f}")
            rsi_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 4, rsi_item)
            
            # Score
            score_item = QTableWidgetItem(f"{result['score']:.2f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            score_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.setItem(row, 5, score_item)

            # Strategy (optional)
            strategy_text = str(result.get('strategy', '')) if isinstance(result, dict) else ''
            strat_item = QTableWidgetItem(strategy_text)
            strat_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.setItem(row, 6, strat_item)

            # Per-strategy scores (optional)
            ss = result.get('strategy_scores', {}) if isinstance(result, dict) else {}
            def fmt(v):
                try:
                    return f"{float(v):.2f}"
                except Exception:
                    return ""
            self.setItem(row, 7, QTableWidgetItem(fmt(ss.get('Momentum'))))
            self.setItem(row, 8, QTableWidgetItem(fmt(ss.get('Value'))))
            self.setItem(row, 9, QTableWidgetItem(fmt(ss.get('Growth'))))
            self.setItem(row, 10, QTableWidgetItem(fmt(ss.get('Oversold'))))

            # Tooltip breakdown of criteria counts
            cc = result.get('criteria_counts', {}) if isinstance(result, dict) else {}
            mins = result.get('criteria_mins', {}) if isinstance(result, dict) else {}
            tip_parts = []
            for name in ['Momentum','Value','Growth','Oversold']:
                sc = ss.get(name)
                cnt = cc.get(name)
                mn = mins.get(name)
                seg = []
                if sc is not None:
                    seg.append(f"score={fmt(sc)}")
                if cnt is not None:
                    seg.append(f"met={cnt}")
                if mn is not None:
                    seg.append(f"min={mn}")
                if seg:
                    tip_parts.append(f"{name}: " + ", ".join(seg))
            tip = "\n".join(tip_parts) if tip_parts else ""
            if tip:
                for col in range(self.columnCount() - 1):  # exclude actions cell
                    item = self.item(row, col)
                    if item:
                        item.setToolTip(tip)

            # Actions cell with placeholder icons (disabled for now)
            try:
                from PyQt6.QtWidgets import QWidget, QHBoxLayout, QToolButton
                action_widget = QWidget()
                hl = QHBoxLayout(action_widget)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setSpacing(2)

                def mk_btn(text_tip: str, text_icon: str):
                    btn = QToolButton()
                    btn.setToolTip(text_tip)
                    btn.setText(text_icon)
                    btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn.setAutoRaise(True)
                    return btn

                btn_chart = mk_btn("Open price chart", "📈")
                btn_watch = mk_btn("Add to Watchlist", "⭐")
                btn_explain = mk_btn("Explain why it matched", "❓")

                # Wire buttons
                symbol = result.get('symbol', '')
                btn_watch.clicked.connect(lambda _, s=symbol: self.add_to_watchlist.emit(s))
                btn_chart.clicked.connect(lambda _, s=symbol: self.open_chart.emit(s))
                btn_explain.clicked.connect(lambda _, r=result: self.explain_requested.emit(r))

                hl.addWidget(btn_chart)
                hl.addWidget(btn_watch)
                hl.addWidget(btn_explain)

                # Predicted price and return
                pred_price = result.get('pred_price')
                pred_ret = result.get('pred_return')
                pp_text = f"${pred_price:.2f}" if isinstance(pred_price, (int, float)) else ""
                pr_text = ""
                try:
                    if pred_ret is not None:
                        val = float(pred_ret)
                        if abs(val) < 1.5:
                            val *= 100.0
                        pr_text = f"{val:+.1f}%"
                except Exception:
                    pr_text = ""
                self.setItem(row, 11, QTableWidgetItem(pp_text))
                self.setItem(row, 12, QTableWidgetItem(pr_text))

                # כפתור דירוג AI לכל שורה
                btn_ai = QPushButton("דירוג AI")
                btn_ai.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 8px; }")
                btn_ai.setFixedHeight(24)
                btn_ai.clicked.connect(lambda _, s=result['symbol'], r=row: self.request_ai_rating(s, r))
                self.setCellWidget(row, 16, btn_ai)
            except Exception:
                pass
    
    def show_context_menu(self, position):
        """Show context menu"""
        if self.itemAt(position) is None:
            return
        
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        menu = QMenu(self)
        
        # Get selected symbol
        row = self.itemAt(position).row()
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            
            add_action = QAction(f"Add {symbol} to Watchlist", self)
            add_action.triggered.connect(lambda: self.add_to_watchlist.emit(symbol))
            menu.addAction(add_action)
            
            menu.exec(self.mapToGlobal(position))
    
    def on_cell_double_clicked(self, row: int, column: int):
        """Handle cell double click"""
        symbol_item = self.item(row, 0)
        if symbol_item:
            symbol = symbol_item.text()
            self.symbol_selected.emit(symbol)

    def request_ai_rating(self, symbol, row):
        """שליחת שאילתא ל-API של Perplexity והצגת התוצאה בעמודה המתאימה"""
        # כאן יש לממש קריאה ל-API של Perplexity
        # לדוגמה:
        # rating = perplexity_api.get_rating(symbol)
        # self.setItem(row, 16, QTableWidgetItem(str(rating)))
        # כרגע נציג ערך דמה
        self.setItem(row, 16, QTableWidgetItem("A+"))


class ScannerWidget(QWidget):
    # Queue scan actions to the worker thread to avoid UI freezes
    request_scan = pyqtSignal(dict)
    request_stop = pyqtSignal()
    def on_ml_toggle(self, state):
        self.use_ml_preds = bool(state)
        self.ml_run_combo.setEnabled(self.use_ml_preds)
        if self.use_ml_preds:
            self._refresh_ml_runs()

    def on_ml_run_selected(self, idx):
        if 0 <= idx < len(self.available_ml_preds):
            self.selected_ml_pred = self.available_ml_preds[idx]
        else:
            self.selected_ml_pred = None
    """Main scanner widget"""
    
    add_to_watchlist = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Initialize logger and config
        self.logger = get_logger("Scanner")
        self.config = ConfigManager()
        
        # Initialize scanner worker
        self.scanner_thread = QThread()
        self.scanner_worker = ScannerWorker(self.config)
        self.scanner_worker.moveToThread(self.scanner_thread)
        self.scanner_worker.progress_updated.connect(self.update_progress)
        self.scanner_worker.results_updated.connect(self.update_results)
        self.scanner_worker.result_found.connect(lambda r: self.results_table.append_result(r))
        self.scanner_worker.scan_completed.connect(self.on_scan_completed)
        self.scanner_worker.status_updated.connect(self.on_status_updated)
        self.scanner_worker.error_occurred.connect(self.on_scan_error)
        self.scanner_thread.start()
        # Connect queued signals so work runs in the worker thread
        self.request_scan.connect(self.scanner_worker.start_scan)
        self.request_stop.connect(self.scanner_worker.stop_scan)
        
        # Setup UI
        self.setup_ui()
        
        self.logger.info("Scanner widget initialized")
    
    def setup_ui(self):
        """Setup the scanner UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        title_layout = QHBoxLayout()

        title = QLabel("Stock Scanner")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title_layout.addWidget(title)
        title_layout.addStretch()

        # ML predictions toggle
        self.use_ml_checkbox = QCheckBox("Use ML predictions")
        self.use_ml_checkbox.setChecked(False)
        self.use_ml_checkbox.stateChanged.connect(self.on_ml_toggle)
        title_layout.addWidget(self.use_ml_checkbox)

        # ML run selector
        self.ml_run_combo = QComboBox()
        self.ml_run_combo.setMinimumWidth(180)
        self.ml_run_combo.setEnabled(False)
        self.ml_run_combo.currentIndexChanged.connect(self.on_ml_run_selected)
        title_layout.addWidget(self.ml_run_combo)

        # Use ML Universe toggle
        self.use_universe_checkbox = QCheckBox("Use ML Universe")
        self.use_universe_checkbox.setToolTip("Limit scan to tickers from data/silver/universe.csv produced by the Pipeline")
        title_layout.addWidget(self.use_universe_checkbox)

        # Internal state for ML integration
        self.available_ml_preds = []
        self.selected_ml_pred = None
        self.use_ml_preds = False

        # Scan button
        self.scan_button = QPushButton("🔍 Start Scan")
        self.scan_button.clicked.connect(self.start_scan)
        title_layout.addWidget(self.scan_button)

        self.stop_button = QPushButton("⏹️ Stop Scan")
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)
        title_layout.addWidget(self.stop_button)

        layout.addLayout(title_layout)

        # Status and progress at the top
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready to scan")
        layout.addWidget(self.status_label)
        # Secondary progress line: remaining and ETA
        self.substatus_label = QLabel("")
        self.substatus_label.setStyleSheet("color: gray; font-size: 10px;")
        self.substatus_label.setVisible(False)
        layout.addWidget(self.substatus_label)

        # Results block (full width)
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.Shape.Box)
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(8, 8, 8, 8)
        results_layout.setSpacing(6)
        header_bar = QWidget()
        hb = QHBoxLayout(header_bar)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(6)
        results_title = QLabel("Scan Results")
        f = QFont("Arial", 11, QFont.Weight.Bold)
        results_title.setFont(f)
        hb.addWidget(results_title)
        hb.addStretch(1)
        # Quick sort buttons (Score up/down)
        btn_sort_down = QPushButton("Score ↓")
        btn_sort_up = QPushButton("Score ↑")
        for b in (btn_sort_down, btn_sort_up):
            b.setFixedHeight(26)
            b.setStyleSheet("QPushButton { font-size: 11px; padding: 2px 8px; }")
        hb.addWidget(btn_sort_down)
        hb.addWidget(btn_sort_up)
        results_layout.addWidget(header_bar)
        self.results_table = ScanResultsTable()
        self.results_table.setFont(QFont("Arial", 9))
        self.results_table.symbol_selected.connect(self.on_symbol_selected)
        self.results_table.add_to_watchlist.connect(self.add_to_watchlist.emit)
        # New: connect action signals
        try:
            self.results_table.open_chart.connect(self.on_open_chart)
            self.results_table.explain_requested.connect(self.on_explain_requested)
        except Exception:
            pass
        results_layout.addWidget(self.results_table)
        # Connect sort buttons (Score is column 5)
        btn_sort_down.clicked.connect(lambda: self.results_table.sortItems(5, Qt.SortOrder.DescendingOrder))
        btn_sort_up.clicked.connect(lambda: self.results_table.sortItems(5, Qt.SortOrder.AscendingOrder))
        layout.addWidget(results_frame, 2)

        # Criteria/presets/strategies block (below results)
        self.criteria_widget = ScanCriteriaWidget()
        self.criteria_widget.setMinimumHeight(180)
        layout.addWidget(self.criteria_widget, 0)

        # Load available ML runs (if any)
        self._refresh_ml_runs()

    def _refresh_ml_runs(self):
        """Scan data/silver/preds for saved prediction files and update combo box."""
        from pathlib import Path
        self.available_ml_preds = []
        self.ml_run_combo.clear()
        pred_dir = Path("data/silver/preds")
        if pred_dir.exists():
            files = sorted(pred_dir.glob("preds_h*.parquet"))
            for fp in files:
                self.available_ml_preds.append(str(fp))
                self.ml_run_combo.addItem(fp.stem)

    def on_status_updated(self, msg: str):
        """Update status text from worker"""
        self.status_label.setText(msg)
        # Parse known pattern to fill substatus (remaining and ETA)
        # Example: "Scanning… idx/total | processed=p errors=e | ETA mm:ss"
        if "Scanning…" in msg and "ETA" in msg:
            try:
                parts = msg.split("|")
                lead = parts[0].strip()
                eta = parts[-1].strip()
                # Extract idx/total to compute remaining
                idx = total = None
                tokens = [t for t in lead.split() if '/' in t]
                if tokens:
                    nums = tokens[-1].split('/')
                    if len(nums) == 2:
                        try:
                            idx = int(nums[0])
                            total = int(nums[1])
                        except Exception:
                            idx = total = None
                if idx is not None and total is not None and total >= idx:
                    remaining = total - idx
                    remain_str = f"Remaining: {remaining:,}"
                    self.substatus_label.setText(f"{idx}/{total} | {remain_str} | {eta}")
                else:
                    self.substatus_label.setText(f"{lead} | {eta}")
                self.substatus_label.setVisible(True)
            except Exception:
                self.substatus_label.setVisible(False)
        else:
            self.substatus_label.setVisible(False)
    
    def start_scan(self):
        """Start stock scan"""
        try:
            # Get scan criteria
            criteria = self.criteria_widget.get_criteria()
            if self.use_ml_preds and self.selected_ml_pred:
                criteria["ml_preds_path"] = self.selected_ml_pred
            # Include ML universe flag when enabled
            try:
                if getattr(self, 'use_universe_checkbox', None) is not None and self.use_universe_checkbox.isChecked():
                    criteria["use_universe_csv"] = True
            except Exception:
                pass
            
            # Update UI
            self.scan_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Scanning stocks...")
            self.substatus_label.setVisible(False)
            
            # Clear previous results
            self.results_table.setRowCount(0)
            # Auto-hide per-strategy columns based on selection
            try:
                show_m = bool(criteria.get('strategy_momentum') or criteria.get('momentum_enabled'))
                show_v = bool(criteria.get('strategy_value') or criteria.get('value_enabled'))
                show_g = bool(criteria.get('strategy_growth'))
                show_o = bool(criteria.get('strategy_oversold'))
                self.results_table.setColumnHidden(7, not show_m)
                self.results_table.setColumnHidden(8, not show_v)
                self.results_table.setColumnHidden(9, not show_g)
                self.results_table.setColumnHidden(10, not show_o)
                # Hide overall Score if no strategies at all
                self.results_table.setColumnHidden(5, not (show_m or show_v or show_g or show_o))
            except Exception:
                pass
            
            # Start scan (queued to worker thread)
            self.request_scan.emit(criteria)
            
            self.logger.info("Stock scan started")
            
        except Exception as e:
            self.logger.error(f"Error starting scan: {e}")
            self.reset_scan_ui()
    
    def stop_scan(self):
        """Stop stock scan"""
        # Request stop in the worker thread
        self.request_stop.emit()
        self.reset_scan_ui()
        self.status_label.setText("Scan stopped")
        self.logger.info("Stock scan stopped")
    
    def update_progress(self, progress: int):
        """Update scan progress"""
        self.progress_bar.setValue(progress)
    
    def update_results(self, results: list):
        """Update scan results"""
        self.results_table.update_results(results)
        self.status_label.setText(f"Found {len(results)} matching stocks")
    
    def on_scan_completed(self, count: int):
        """Handle scan completion"""
        self.reset_scan_ui()
        self.status_label.setText(f"Scan completed: {count} matches")
        self.logger.info(f"Scan completed: {count} matches")
    
    def on_scan_error(self, error: str):
        """Handle scan error"""
        self.reset_scan_ui()
        self.status_label.setText(f"Scan error: {error}")
        self.logger.error(f"Scan error: {error}")
    
    def on_symbol_selected(self, symbol: str):
        """Handle symbol selection"""
        self.logger.debug(f"Symbol selected from scanner: {symbol}")

    def on_open_chart(self, symbol: str):
        """Open chart dialog with timeframe selector."""
        try:
            dlg = ChartDialog(symbol, self)
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.show()
        except Exception as e:
            QMessageBox.warning(self, "Chart", f"Failed to open chart: {e}")

    def on_explain_requested(self, result: dict):
        """Show detailed explanation dialog with per-rule breakdown."""
        try:
            dlg = ExplainDialog(result, self)
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.show()
        except Exception as e:
            QMessageBox.warning(self, "Explain", f"Failed to open explanation: {e}")
    
    def reset_scan_ui(self):
        """Reset scan UI to default state"""
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("Scanner tab activated")
    
    def closeEvent(self, event):
        """Handle widget close"""
        # Stop scan if running
        try:
            if hasattr(self, 'stop_button') and self.stop_button.isEnabled():
                self.stop_scan()
        except Exception:
            pass
        # Stop worker thread
        try:
            if hasattr(self, 'scanner_thread'):
                # signal worker to stop
                if hasattr(self, 'scanner_worker'):
                    self.scanner_worker.stop_scan()
                self.scanner_thread.quit()
                self.scanner_thread.wait()
        except Exception:
            pass
        event.accept()

    def _apply_post_processing_filters(self, results):
        """Apply final quality control filters to scan results"""
        if not results:
            return results
        
        try:
            # Remove exact duplicates
            seen = set()
            filtered_results = []
            for result in results:
                key = result.get('symbol', '')
                if key not in seen:
                    seen.add(key)
                    filtered_results.append(result)
            
            # Filter by minimum score threshold
            strategy = self.combo_strategy.currentText().lower()
            min_score_thresholds = {
                'momentum': 65,
                'value': 60,
                'growth': 62,
                'oversold': 58
            }
            min_score = min_score_thresholds.get(strategy, 60)
            filtered_results = [r for r in filtered_results if r.get('score', 0) >= min_score]
            
            # Sort by score descending and limit results
            filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            max_results = 50  # Limit to top 50 results
            filtered_results = filtered_results[:max_results]
            
            # Final validation - ensure all required fields exist
            validated_results = []
            for result in filtered_results:
                if all(key in result for key in ['symbol', 'score', 'price', 'change_pct']):
                    validated_results.append(result)
            
            return validated_results
            
        except Exception as e:
            self.logger.warning(f"Post-processing filter error: {e}")
            return results  # Return original results if filtering fails