"""
Machine Learning Widget
For training and managing ML models for trading
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QFrame,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QProgressBar, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QCheckBox, QFileDialog, QMessageBox,
    QLineEdit, QTimeEdit, QSizePolicy, QDialog, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QRunnable, QThreadPool, pyqtSlot
from PyQt6.QtGui import QFont

from core.config_manager import ConfigManager
from utils.logger import get_logger
from ml.config import TrainingConfig
from ml.dataset import load_bronze, build_pooled_dataset
from ml.runner import walk_forward_run
from services.data_update_service import DataUpdateService, UpdateConfig

from datetime import datetime, time as dtime
from pathlib import Path
import sys
import json
import subprocess
import shlex


class ModelTrainingWorker(QObject):
    """Worker thread for ML model training"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.logger = get_logger("MLTraining")
        self.is_training = False
    
    def stop_training(self):
        """Stop training process"""
        self.is_training = False
        self.status_updated.emit("Training stopped by user")


class PipelineRunWorker(QObject):
    """Worker to run the ML pipeline (feature build + walk-forward) off the UI thread."""

    progress_updated = pyqtSignal(int)  # 0..100
    status_updated = pyqtSignal(str)
    completed = pyqtSignal(dict)  # summary dict
    error_occurred = pyqtSignal(str)

    # Trick: emit params via signal so the slot runs in this worker's thread
    run_requested = pyqtSignal(object)  # params dict

    def __init__(self, ml_widget=None):
        super().__init__()
        self.is_running = False
        self.should_stop = False
        self.ml_widget = ml_widget
        # connect signal to slot
        self.run_requested.connect(self.run)

    def stop(self):
        """Stop the running pipeline."""
        self.should_stop = True

    def _save_predictions(self, preds_by_h: dict) -> int:
        from pathlib import Path
        import pandas as pd
        outdir = Path("data/silver/preds")
        outdir.mkdir(parents=True, exist_ok=True)
        count = 0
        for h, df in preds_by_h.items():
            if df is not None and not df.empty:
                df.to_parquet(outdir / f"preds_h{h}.parquet", index=False)
                count += len(df)
        return count

    def _append_metrics_csv(self, as_of_results: list, window: str, tickers: list):
        from pathlib import Path
        import pandas as pd
        outdir = Path("data/silver")
        outdir.mkdir(parents=True, exist_ok=True)
        fp = outdir / "metrics.csv"
        rows = []
        for r in as_of_results:
            rows.append({
                "as_of": getattr(r, "as_of", None),
                "horizon": getattr(r, "horizon", None),
                "metric": getattr(r, "metric_value", None),
                "metric_name": getattr(r, "metric_name", None),
                "n_train": getattr(r, "n_train", None),
                "n_test": getattr(r, "n_test", None),
                "tickers": ",".join(tickers) if tickers else "ALL",
                "window": window,
                "timestamp": pd.Timestamp.now(),
            })
        df = pd.DataFrame(rows)
        if fp.exists():
            try:
                old = pd.read_csv(fp)
                df = pd.concat([old, df], ignore_index=True)
            except Exception:
                pass
        df.to_csv(fp, index=False)

    def run(self, params: object):  # params is a dict
        print(f"PipelineRunWorker.run() called in thread: {QThread.currentThread()}")
        if self.is_running:
            self.error_occurred.emit("Pipeline is already running")
            return
        try:
            # Import Path at the beginning of the method
            from pathlib import Path
            
            self.is_running = True
            print("PipelineRunWorker: Starting pipeline execution")
            p = params or {}
            tickers = p.get("tickers") or []
            holdout = int(p.get("holdout", 30))
            step = int(p.get("step", 5))
            lookback = int(p.get("lookback", 500))
            window = str(p.get("window", "expanding"))
            models = p.get("models") or []
            max_loops = int(p.get("max_loops", 5))
            quality_threshold = float(p.get("quality_threshold", 0.7))

            self.status_updated.emit("Initializing pipeline config...")
            if self.ml_widget:
                self.ml_widget.add_pipeline_step("Initialize Config", "running")
            self.progress_updated.emit(5)  # Step 1/19 complete
            # Force UI update
            QApplication.processEvents()

            cfg = TrainingConfig()
            cfg.holdout_last_days = holdout
            cfg.step_days = step
            cfg.lookback_days = lookback
            cfg.window_mode = window  # type: ignore
            # Apply feature toggles from UI
            cfg.use_technical = bool(p.get("use_technical", True))
            cfg.use_volume = bool(p.get("use_volume", True))
            cfg.use_sentiment = bool(p.get("use_sentiment", False))

            # Emit status about feature toggles for transparency
            feat_msg = f"Feature toggles - technical={cfg.use_technical}, volume={cfg.use_volume}, sentiment={cfg.use_sentiment}"
            self.status_updated.emit(feat_msg)
            if self.ml_widget:
                try:
                    if hasattr(self.ml_widget, 'performance_widget') and getattr(self.ml_widget, 'performance_widget'):
                        self.ml_widget.performance_widget.add_log_entry(feat_msg)
                    else:
                        # fallback if MLWidget provides add_log_entry directly
                        self.ml_widget.add_log_entry(feat_msg)  # type: ignore
                except Exception:
                    pass

            if self.ml_widget:
                self.ml_widget.add_pipeline_step("Initialize Config", "completed")
                self.ml_widget.add_pipeline_step("Loading Data", "running")

            # Load bronze data once at the beginning with progress updates
            self.status_updated.emit("Loading bronze data (Parquet) files...")
            self.progress_updated.emit(10)  # Step 2/19 complete  
            # Force UI update
            QApplication.processEvents()
            
            # Progress callback for data loading
            def data_progress_callback(progress):
                # Update data progress bar
                if self.ml_widget:
                    self.ml_widget.update_data_progress(progress)
            
            bronze = load_bronze("data/bronze/daily", tickers=tickers if tickers else None, progress_callback=data_progress_callback)
            if not bronze:
                self.error_occurred.emit("No bronze Parquet files found for selected tickers")
                if self.ml_widget:
                    self.ml_widget.add_pipeline_step("Loading Data", "error")
                return
            
            self.status_updated.emit(f"Loaded {len(bronze)} tickers successfully")
            if self.ml_widget:
                self.ml_widget.add_pipeline_step("Loading Data", "completed")
                self.ml_widget.update_data_progress(0)  # Hide data progress bar
            self.progress_updated.emit(15)  # Data loading complete
            # Force UI update
            QApplication.processEvents()

            # תהליך לולאת fine tuning לכל הורייזן
            best_result = None
            selected_cache = p.get("selected_cache")
            if selected_cache:
                cache_file = Path(f"data/cache/{selected_cache}.parquet")
                self.status_updated.emit(f"Using selected cache: {selected_cache}")
            else:
                cache_file = Path("data/cache/pooled_dataset.parquet")
                self.status_updated.emit("Using default cache")
            
            # Accumulate predictions across horizons so we can save them all at the end
            all_preds_accumulator = {}
            for horizon in [1, 5, 10]:
                if self.should_stop:
                    self.status_updated.emit("Pipeline stopped by user")
                    if self.ml_widget:
                        self.ml_widget.add_pipeline_step("Pipeline Stopped", "error")
                    break
                if self.ml_widget:
                    self.ml_widget.add_pipeline_step(f"Horizon {horizon} Setup", "running")
                cfg.horizons = [horizon]
                self.status_updated.emit(f"Building features and labels for horizon {horizon}…")
                self.progress_updated.emit(15)
                
                # Use cache file for first horizon only when running over the full universe.
                # If this run is for a single ticker, skip the cache to ensure the pooled
                # dataset includes that ticker (cache may have been built earlier without it).
                use_cache = None
                skip_cache_reason = None
                if horizon == 1 and (not tickers or len(tickers) != 1):
                    use_cache = cache_file
                else:
                    # Single-ticker run or non-first horizon: skip disk cache
                    use_cache = None
                    skip_cache_reason = 'single-ticker or non-first horizon — skipping disk cache to include latest ticker features'
                if skip_cache_reason:
                    self.status_updated.emit(skip_cache_reason)
                    if self.ml_widget:
                        try:
                            if hasattr(self.ml_widget, 'performance_widget') and getattr(self.ml_widget, 'performance_widget'):
                                self.ml_widget.performance_widget.add_log_entry(skip_cache_reason)
                            else:
                                self.ml_widget.add_log_entry(skip_cache_reason)  # type: ignore
                        except Exception:
                            pass
                use_parallel = bool(p.get("use_parallel", True))
                pooled = build_pooled_dataset(bronze, cfg, cache_file=use_cache, use_parallel=use_parallel)
                
                if pooled is None or pooled.empty:
                    self.error_occurred.emit("Pooled dataset is empty after feature/label building")
                    if self.ml_widget:
                        self.ml_widget.add_pipeline_step("Feature Building", "error")
                    return

                if self.ml_widget:
                    self.ml_widget.add_pipeline_step("Feature Building", "completed")

                for loop in range(max_loops):
                    if self.should_stop:
                        self.status_updated.emit("Pipeline stopped by user")
                        if self.ml_widget:
                            self.ml_widget.add_pipeline_step("Training Stopped", "error")
                        break
                    if self.ml_widget:
                        self.ml_widget.add_pipeline_step(f"Training Loop {loop+1}/{max_loops} (H{horizon})", "running")
                    self.status_updated.emit(f"Training loop {loop+1}/{max_loops} for horizon {horizon}")
                    # Let the step counter handle progress calculation automatically
                    # The progress will be updated when steps are marked as completed
                    
                    # Force UI update by processing events
                    if self.ml_widget and hasattr(self.ml_widget, 'parent'):
                        QApplication.processEvents()
                    
                    results, preds, model_scores, confusions = walk_forward_run(
                        pooled, cfg, selected_models=models if models else ["RandomForest"]
                    )
                    # תחזיות על תקופת holdout
                    # preds returned by walk_forward_run is a dict keyed by horizon strings
                    # accumulate non-empty dataframes into all_preds_accumulator
                    if preds:
                        for pk, pv in preds.items():
                            if pv is None or (hasattr(pv, 'empty') and pv.empty):
                                continue
                            all_preds_accumulator.setdefault(pk, []).append(pv)
                    predictions = None
                    if isinstance(all_preds_accumulator.get(str(horizon)), list):
                        # concat pieces for this horizon
                        import pandas as _pd
                        try:
                            predictions = _pd.concat(all_preds_accumulator.get(str(horizon)), ignore_index=True)
                        except Exception:
                            predictions = all_preds_accumulator.get(str(horizon))[0]

                    # הזנה לסורק (פשוט: בדיקה האם יש מניות עם תחזית UP)
                    scan_results = []
                    if predictions is not None:
                        for idx, row in predictions.iterrows():
                            if row.get(f"y_h{horizon}_pred") == "UP":
                                scan_results.append(row.get("ticker"))

                    # בדיקת איכות: השוואה בין תחזית לתוצאה אמיתית
                    quality = 0.0
                    if predictions is not None:
                        y_true = predictions.get(f"y_h{horizon}")
                        y_pred = predictions.get(f"y_h{horizon}_pred")
                        if y_true is not None and y_pred is not None:
                            from sklearn.metrics import f1_score
                            try:
                                quality = f1_score(y_true, y_pred, average="macro", labels=["DOWN","HOLD","UP"], zero_division=0)
                            except Exception:
                                quality = 0.0

                    self.status_updated.emit(f"Loop {loop+1}: Horizon {horizon} Quality={quality:.3f} | Scan matches={len(scan_results)}")

                    if best_result is None or quality > best_result.get("quality", 0):
                        best_result = {
                            "horizon": horizon,
                            "loop": loop+1,
                            "quality": quality,
                            "scan_results": scan_results,
                            "model_results": results,
                        }
                    if quality >= quality_threshold:
                        break

                if self.ml_widget:
                    self.ml_widget.add_pipeline_step(f"Horizon {horizon} Training", "completed")

                    if best_result is None or quality > best_result.get("quality", 0):
                        best_result = {
                            "horizon": horizon,
                            "loop": loop+1,
                            "quality": quality,
                            "scan_results": scan_results,
                            "model_results": results,
                        }
                    if quality >= quality_threshold:
                        break

                if self.ml_widget:
                    self.ml_widget.add_pipeline_step(f"Horizon {horizon} Training", "completed")

            # Persist predictions and metrics
            try:
                if self.ml_widget:
                    self.ml_widget.add_pipeline_step("Saving Predictions", "running")
                # Build final preds dict by concatenating accumulated dfs per horizon
                final_preds = {}
                import pandas as _pd
                for k, lst in all_preds_accumulator.items():
                    try:
                        final_preds[k] = _pd.concat(lst, ignore_index=True) if isinstance(lst, list) and len(lst) > 0 else (_pd.DataFrame() if not lst else lst)
                    except Exception:
                        # fallback - take first element
                        final_preds[k] = lst[0] if isinstance(lst, list) and lst else (_pd.DataFrame() if not lst else lst)

                saved = self._save_predictions(final_preds) if final_preds else 0
            except Exception:
                saved = 0
            try:
                # Append compact step metrics
                self._append_metrics_csv(results, window, tickers)
            except Exception:
                pass

            if self.ml_widget:
                self.ml_widget.add_pipeline_step("Saving Predictions", "completed")
                self.ml_widget.add_pipeline_step("Exporting Signals", "running")

            # Export simple signals and universe for next steps
            try:
                import pandas as pd
                from pathlib import Path
                outdir = Path("data/silver")
                outdir.mkdir(parents=True, exist_ok=True)
                # Universe: unique tickers with an UP signal at best horizon
                universe = []
                if best_result and best_result.get("scan_results"):
                    universe = sorted({t for t in best_result["scan_results"] if t})
                pd.DataFrame({"ticker": universe}).to_csv(outdir / "universe.csv", index=False)
                # Signals: last predictions for best horizon with confidence/price_target if present
                sig_rows = []
                # Use the aggregated final_preds for exporting simple signals
                if final_preds and best_result:
                    bh = str(best_result.get("horizon"))
                    dfp = final_preds.get(bh)
                    if dfp is not None and not dfp.empty:
                        # take last date per ticker
                        last = dfp.sort_values("date").groupby("ticker").tail(1)
                        for _, r in last.iterrows():
                            action = "BUY" if str(r.get("y_pred")) == "UP" else ("SELL" if str(r.get("y_pred")) == "DOWN" else "HOLD")
                            sig_rows.append({
                                "ticker": r.get("ticker"),
                                "date": r.get("date"),
                                "action": action,
                                "horizon": int(best_result.get("horizon", 0)),
                                "confidence": float(r.get("confidence", 0.5)),
                                "price_target": float(r.get("price_target", 0.0)),
                                "model": r.get("model"),
                            })
                pd.DataFrame(sig_rows).to_csv(outdir / "signals.csv", index=False)
            except Exception:
                pass

            if self.ml_widget:
                self.ml_widget.add_pipeline_step("Exporting Signals", "completed")

            # Final result
            # If this run was for a single ticker, prepare a compact in-memory payload
            compact_payload = None
            try:
                # Use the aggregated final_preds (all horizons) when building compact payload
                if isinstance(tickers, (list, tuple)) and len(tickers) == 1 and final_preds:
                    sym = tickers[0].upper()
                    signals = []
                    price_targets = {}
                    for h in [1, 5, 10]:
                        # final_preds keys may be strings like '1' or ints; try both
                        dfp = None
                        if isinstance(final_preds, dict):
                            dfp = final_preds.get(str(h)) or final_preds.get(h) or None
                        if dfp is None or dfp.empty:
                            continue
                        # Filter for ticker (case-insensitive)
                        if 'ticker' in dfp.columns:
                            row = dfp[dfp['ticker'].str.upper() == sym]
                            if row.empty:
                                continue
                            latest = row.sort_values('date').iloc[-1]
                        else:
                            latest = dfp.sort_values('date').iloc[-1]

                        pred_lbl = str(latest.get('y_pred', 'HOLD')).upper()
                        conf = float(latest.get('confidence', 0.0)) if latest.get('confidence', None) is not None else 0.0
                        pt = float(latest.get('price_target', latest.get('adj_close', latest.get('close', 0.0))))
                        model_name = latest.get('model', None)
                        signals.append((h, pred_lbl, conf))
                        # Use integer keys for horizons so UI lookups (1,5,10) work consistently
                        price_targets[int(h)] = pt
                        # store per-horizon meta so UI can show tooltips
                        if 'meta' not in price_targets:
                            price_targets['meta'] = {}
                        # include the source date so UI can show which date the price_target refers to
                        src_date = latest.get('date', None)
                        # normalize to isoformat string if it's a Timestamp or datetime-like
                        try:
                            if hasattr(src_date, 'isoformat'):
                                src_date_str = src_date.isoformat()
                            else:
                                src_date_str = str(src_date)
                        except Exception:
                            src_date_str = str(src_date)
                        price_targets['meta'][int(h)] = {
                            'confidence': conf,
                            'model': model_name,
                            'date': src_date_str,
                        }

                    # Derive overall signal via confidence-weighted vote
                    if signals:
                        score_map = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
                        for (_h, lbl, conf) in signals:
                            if lbl == 'UP':
                                score_map['BUY'] += conf
                            elif lbl == 'DOWN':
                                score_map['SELL'] += conf
                            else:
                                score_map['HOLD'] += conf
                        overall = max(score_map.keys(), key=lambda k: score_map[k])
                        compact_payload = {
                            'symbol': sym,
                            'overall_signal': overall,
                            'price_targets': price_targets,
                            # include per-horizon confidences for better UI display
                            'per_horizon': {h: {'signal': lbl, 'confidence': conf} for (h, lbl, conf) in signals}
                        }
            except Exception:
                compact_payload = None

            self.completed.emit({
                "best_result": best_result or {},
                "saved_predictions": int(saved),
                "compact_table": compact_payload,
            })
            # Persist compact payload to disk for debugging (works for headless runs too)
            try:
                if compact_payload is not None:
                    from pathlib import Path as _Path
                    import json as _json
                    _Path('data/silver').mkdir(parents=True, exist_ok=True)
                    fp = _Path('data/silver/debug_compact_from_worker.json')
                    fp.write_text(_json.dumps(compact_payload, default=str, indent=2), encoding='utf-8')
            except Exception:
                pass
            if best_result:
                self.status_updated.emit(f"Pipeline completed. Best horizon={best_result['horizon']} Quality={best_result['quality']:.3f} Scan matches={len(best_result['scan_results'])}")
            else:
                self.status_updated.emit("Pipeline completed, but no best result identified")
            # DO NOT emit progress_updated(100) here - let the step counter handle it properly

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.is_running = False
            self.should_stop = False


class ModelConfigWidget(QFrame):
    """Model configuration widget"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup model configuration UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Model Configuration")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Form layout
        form_layout = QFormLayout()
        
        # Model configuration moved to pipeline parameters section
        layout.addLayout(form_layout)
    
    def get_config(self) -> dict:
        """Get current configuration - features moved to pipeline section"""
        return {}


class ModelPerformanceWidget(QFrame):
    """Model performance metrics widget"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup performance UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Model Performance")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Metrics summary table
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        # Responsive table height
        from PyQt6.QtWidgets import QApplication as QtApp
        screen = QtApp.primaryScreen()  
        screen_height = screen.availableGeometry().height()
        
        if screen_height <= 768:  # Small screens
            self.metrics_table.setMaximumHeight(120)
        else:
            self.metrics_table.setMaximumHeight(200)
        layout.addWidget(self.metrics_table)

        # Metrics viewer controls
        viewer_box = QGroupBox("Metrics Viewer")
        vlayout = QVBoxLayout(viewer_box)
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Horizon:"))
        self.filter_horizon_combo = QComboBox()
        self.filter_horizon_combo.addItems(["All", "1", "5", "10"])
        fl.addWidget(self.filter_horizon_combo)
        fl.addWidget(QLabel("Window:"))
        self.filter_window_combo = QComboBox()
        self.filter_window_combo.addItems(["All", "expanding", "rolling"])
        fl.addWidget(self.filter_window_combo)
        self.refresh_metrics_btn = QPushButton("Refresh Metrics")
        fl.addWidget(self.refresh_metrics_btn)
        vlayout.addLayout(fl)

        # Metrics table detailed
        self.metrics_detail = QTableWidget(0, 8)
        self.metrics_detail.setHorizontalHeaderLabels([
            "as_of", "horizon", "metric", "n_train", "n_test", "tickers", "window", "timestamp"
        ])
        self.metrics_detail.horizontalHeader().setStretchLastSection(True)
        vlayout.addWidget(self.metrics_detail)
        layout.addWidget(viewer_box)

        # Predictions viewer
        preds_box = QGroupBox("Predictions Preview")
        playout = QVBoxLayout(preds_box)
        pfl = QHBoxLayout()
        pfl.addWidget(QLabel("Horizon:"))
        self.pred_h_combo = QComboBox()
        self.pred_h_combo.addItems(["1", "5", "10"])
        pfl.addWidget(self.pred_h_combo)
        self.load_preds_btn = QPushButton("Load Preds")
        pfl.addWidget(self.load_preds_btn)
        playout.addLayout(pfl)
        self.preds_table = QTableWidget(0, 7)
        self.preds_table.setHorizontalHeaderLabels(["ticker", "date", "y_true", "y_pred", "model", "confidence", "price_target"])
        self.preds_table.horizontalHeader().setStretchLastSection(True)
        playout.addWidget(self.preds_table)
        layout.addWidget(preds_box)

        # wire actions
        self.refresh_metrics_btn.clicked.connect(self.load_metrics_csv)
        self.load_preds_btn.clicked.connect(lambda: self.load_preds_parquet())
        
        # Training log
        log_label = QLabel("Training Log:")
        log_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(log_label)
        
        self.training_log = QTextEdit()
        # Responsive training log height
        if screen_height <= 768:  # Small screens
            self.training_log.setMaximumHeight(100)
        else:
            self.training_log.setMaximumHeight(150)
        self.training_log.setReadOnly(True)
        layout.addWidget(self.training_log)
    
    def update_metrics(self, metrics: dict):
        """Update performance metrics"""
        self.metrics_table.setRowCount(len(metrics))
        
        for i, (metric, value) in enumerate(metrics.items()):
            metric_item = QTableWidgetItem(metric.replace('_', ' ').title())
            self.metrics_table.setItem(i, 0, metric_item)
            
            if isinstance(value, float):
                if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    value_text = f"{value:.2%}"
                else:
                    value_text = f"{value:.4f}"
            else:
                value_text = str(value)
            
            value_item = QTableWidgetItem(value_text)
            self.metrics_table.setItem(i, 1, value_item)

    def load_metrics_csv(self):
        """Load metrics.csv and show filtered rows."""
        import pandas as pd
        from pathlib import Path
        fp = Path("data/silver/metrics.csv")
        if not fp.exists():
            QMessageBox.information(self, "No metrics", "metrics.csv not found yet.")
            return
        df = pd.read_csv(fp)
        # apply filters
        hsel = self.filter_horizon_combo.currentText()
        if hsel != "All":
            df = df[df["horizon"] == int(hsel)]
        wsel = self.filter_window_combo.currentText()
        if wsel != "All":
            df = df[df["window"] == wsel]
        cols = ["as_of", "horizon", "metric", "n_train", "n_test", "tickers", "window", "timestamp"]
        df = df[cols] if all(c in df.columns for c in cols) else df
        self.metrics_detail.setRowCount(len(df))
        self.metrics_detail.setColumnCount(len(cols))
        self.metrics_detail.setHorizontalHeaderLabels(cols)
        for i, row in df.iterrows():
            for j, col in enumerate(cols):
                self.metrics_detail.setItem(i, j, QTableWidgetItem(str(row.get(col, ""))))

    def load_preds_parquet(self, filter_tickers=None):
        """Load preds_h{h}.parquet, first 200 rows, optionally filtering tickers."""
        import pandas as pd
        from pathlib import Path
        h = self.pred_h_combo.currentText()
        fp = Path(f"data/silver/preds/preds_h{h}.parquet")
        if not fp.exists():
            QMessageBox.information(self, "No predictions", f"{fp.name} not found.")
            return
        df = pd.read_parquet(fp)
        if filter_tickers:
            want = {t.upper() for t in filter_tickers}
            if "ticker" in df.columns:
                df = df[df["ticker"].str.upper().isin(want)]
        df = df.head(200)
        # include optional columns when present
        cols = ["ticker", "date", "y_true", "y_pred", "model"]
        if "confidence" in df.columns:
            cols.append("confidence")
        if "price_target" in df.columns:
            cols.append("price_target")
        self.preds_table.setRowCount(len(df))
        self.preds_table.setColumnCount(len(cols))
        self.preds_table.setHorizontalHeaderLabels(cols)
        
        # Enhanced display with colors and symbols
        from PyQt6.QtGui import QColor
        
        for i, (idx, row) in enumerate(df.iterrows()):
            for j, col in enumerate(cols):
                value = str(row.get(col, ""))
                
                # Enhanced display for predictions
                if col == 'y_pred':
                    if value.upper() == 'UP':
                        value = "🟢 BUY"
                        item = QTableWidgetItem(value)
                        item.setBackground(QColor(200, 255, 200))  # Light green
                    elif value.upper() == 'DOWN':
                        value = "🔴 SELL"
                        item = QTableWidgetItem(value)
                        item.setBackground(QColor(255, 200, 200))  # Light red
                    else:
                        value = "🟡 HOLD"
                        item = QTableWidgetItem(value)
                        item.setBackground(QColor(255, 255, 200))  # Light yellow
                elif col == 'confidence':
                    try:
                        conf_val = float(value)
                        item = QTableWidgetItem(f"{conf_val:.1%}")
                        if conf_val > 0.7:
                            item.setBackground(QColor(200, 255, 200))  # High confidence - green
                        elif conf_val > 0.5:
                            item.setBackground(QColor(255, 255, 200))  # Medium confidence - yellow
                        else:
                            item.setBackground(QColor(255, 230, 230))  # Low confidence - light red
                    except:
                        item = QTableWidgetItem(value)
                elif col == 'price_target':
                    try:
                        price_val = float(value)
                        item = QTableWidgetItem(f"${price_val:.2f}")
                    except:
                        item = QTableWidgetItem(value)
                else:
                    item = QTableWidgetItem(value)
                
                self.preds_table.setItem(i, j, item)
    
    def add_log_entry(self, message: str):
        """Add entry to training log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] {message}")


class DataManagementWidget(QFrame):
    """Data management widget with integrated daily update functionality"""

    def __init__(self):
        super().__init__()
        self.logger = get_logger("DataManagement")
        self._cfg_path = Path("config/data_update.json")
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._service = DataUpdateService()
        self._service.progress.connect(self._on_progress)
        self._service.log.connect(self._on_log)
        self._service.completed.connect(self._on_completed)
        self._service.error.connect(self._on_error)
        self._service.next_run_changed.connect(self._on_next_run)
        self.setup_ui()

    def setup_ui(self):
        """Setup data management UI with integrated daily update"""
        self.setFrameStyle(QFrame.Shape.Box)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Data Management & Daily Updates")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # Create tab widget for data management sections
        tab_widget = QTabWidget()

        # Data Loading Tab
        data_tab = self.create_data_loading_tab()
        tab_widget.addTab(data_tab, "📥 Data Loading")

        # Daily Update Tab
        update_tab = self.create_daily_update_tab()
        tab_widget.addTab(update_tab, "🗓️ Daily Update")

        # Comprehensive Report Tab
        report_tab = self.create_comprehensive_report_tab()
        tab_widget.addTab(report_tab, "📋 Report")

        # Cache Management Tab
        cache_tab = self.create_cache_management_tab()
        tab_widget.addTab(cache_tab, "💾 Cache")

        layout.addWidget(tab_widget)

    def create_comprehensive_report_tab(self):
        """Create comprehensive report tab for data management"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("Comprehensive Data Report")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Description
        desc = QLabel("Run comprehensive system checks to verify all data components are working correctly.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.run_comprehensive_report_btn = QPushButton("📋 Generate Report")
        self.run_comprehensive_report_btn.clicked.connect(self.run_comprehensive_check)
        self.run_comprehensive_report_btn.setFixedWidth(200)
        buttons_layout.addWidget(self.run_comprehensive_report_btn)

        self.clear_data_cache_btn = QPushButton("🗑️ Clear Cache")
        self.clear_data_cache_btn.clicked.connect(self.clear_data_cache)
        self.clear_data_cache_btn.setFixedWidth(150)
        self.clear_data_cache_btn.setToolTip("Clear the processed data cache\nUse when downloading new data or changing settings")
        buttons_layout.addWidget(self.clear_data_cache_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        # Progress
        self.report_progress = QProgressBar()
        self.report_progress.setVisible(False)
        layout.addWidget(self.report_progress)

        # Results area with scroll
        results_group = QGroupBox("Report Results")
        results_layout = QVBoxLayout(results_group)

        self.report_results = QTextEdit()
        self.report_results.setReadOnly(True)
        self.report_results.setPlaceholderText("Comprehensive report will appear here...")
        self.report_results.setMinimumHeight(600)
        # Ensure scroll is enabled and text can expand
        self.report_results.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.report_results.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        results_layout.addWidget(self.report_results)

        layout.addWidget(results_group)

        return tab_widget

    def create_cache_management_tab(self):
        """Create cache management tab for multiple cache files"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("Cache Management")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Description
        desc = QLabel("Create and manage multiple cache files with different configurations for training.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        # Cache list and controls
        cache_group = QGroupBox("Cache Files")
        cache_layout = QVBoxLayout(cache_group)

        # Cache list
        self.cache_list = QTableWidget()
        self.cache_list.setColumnCount(5)
        self.cache_list.setHorizontalHeaderLabels(["Name", "Created", "Size", "Rows", "Config"])
        self.cache_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.cache_list.setAlternatingRowColors(True)
        self.cache_list.setMinimumHeight(200)
        cache_layout.addWidget(self.cache_list)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.create_cache_btn = QPushButton("➕ Create Cache")
        self.create_cache_btn.clicked.connect(self.show_create_cache_dialog)
        buttons_layout.addWidget(self.create_cache_btn)

        self.delete_cache_btn = QPushButton("🗑️ Delete Selected")
        self.delete_cache_btn.clicked.connect(self.delete_selected_cache)
        buttons_layout.addWidget(self.delete_cache_btn)

        self.refresh_cache_btn = QPushButton("🔄 Refresh List")
        self.refresh_cache_btn.clicked.connect(self.refresh_cache_list)
        buttons_layout.addWidget(self.refresh_cache_btn)

        buttons_layout.addStretch()
        cache_layout.addLayout(buttons_layout)

        layout.addWidget(cache_group)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
        layout.addWidget(self.status_label)

        # Cache selection for training
        selection_group = QGroupBox("Training Cache Selection")
        selection_layout = QVBoxLayout(selection_group)

        selection_layout.addWidget(QLabel("Select cache to use for training:"))

        self.training_cache_combo = QComboBox()
        self.training_cache_combo.addItem("Auto (default)", None)
        selection_layout.addWidget(self.training_cache_combo)

        info_label = QLabel("💡 'Auto' uses the default cache that updates with new data.\nNamed caches preserve specific configurations for testing.")
        info_label.setWordWrap(True)
        selection_layout.addWidget(info_label)

        layout.addWidget(selection_group)

        # Initial refresh
        self.refresh_cache_list()

        return tab_widget

    def create_data_loading_tab(self):
        """Create the data loading tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Data source selection
        source_layout = QHBoxLayout()

        source_label = QLabel("Data Source:")
        source_layout.addWidget(source_label)

        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems([
            "IBKR Historical Data",
            "Yahoo Finance",
            "Alpha Vantage",
            "Custom CSV File"
        ])
        source_layout.addWidget(self.data_source_combo)

        source_layout.addStretch()
        layout.addLayout(source_layout)

        # Symbols selection
        symbols_layout = QHBoxLayout()

        symbols_label = QLabel("Symbols:")
        symbols_layout.addWidget(symbols_label)

        self.symbols_input = QTextEdit()
        self.symbols_input.setMaximumHeight(60)
        self.symbols_input.setPlaceholderText("Enter symbols separated by commas (e.g., AAPL, MSFT, GOOGL)")
        symbols_layout.addWidget(self.symbols_input)

        layout.addLayout(symbols_layout)

        # Data range
        range_layout = QHBoxLayout()

        range_label = QLabel("Date Range:")
        range_layout.addWidget(range_label)

        self.date_range_combo = QComboBox()
        self.date_range_combo.addItems([
            "1 Month",
            "3 Months",
            "6 Months",
            "1 Year",
            "2 Years",
            "5 Years"
        ])
        self.date_range_combo.setCurrentText("1 Year")
        range_layout.addWidget(self.date_range_combo)

        range_layout.addStretch()
        layout.addLayout(range_layout)

        # Action buttons
        button_layout = QHBoxLayout()

        self.load_data_btn = QPushButton("📥 Load Data")
        self.load_data_btn.clicked.connect(self.load_data)
        button_layout.addWidget(self.load_data_btn)

        self.export_data_btn = QPushButton("📤 Export Data")
        self.export_data_btn.clicked.connect(self.export_data)
        button_layout.addWidget(self.export_data_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Data status
        self.data_status_label = QLabel("No data loaded")
        self.data_status_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.data_status_label)

        return tab

    def create_daily_update_tab(self):
        """Create the daily update tab with integrated functionality"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Schedule Configuration
        schedule_group = QGroupBox("Schedule Configuration")
        form = QFormLayout(schedule_group)

        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm")
        self.time_edit.setToolTip("Local time to run daily updates (e.g., a few hours after market close)")
        form.addRow("Run time:", self.time_edit)

        # Optional batch limit for testing
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(0, 10000)
        self.limit_spin.setValue(0)
        self.limit_spin.setToolTip("Process only first N tickers (0 = all)")
        form.addRow("Batch limit:", self.limit_spin)

        layout.addWidget(schedule_group)

        # Control Buttons
        buttons_layout = QHBoxLayout()

        self.run_now_btn = QPushButton("▶️ Run Now")
        self.run_now_btn.clicked.connect(self._on_run_now)
        buttons_layout.addWidget(self.run_now_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        self.save_schedule_btn = QPushButton("💾 Save Schedule")
        self.save_schedule_btn.clicked.connect(self.on_save)
        buttons_layout.addWidget(self.save_schedule_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        # Status and Progress
        self.next_run_label = QLabel("Next run: --")
        layout.addWidget(self.next_run_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

    # (Final Report removed per request)

        # Live Logs
        logs_group = QGroupBox("Live Logs")
        logs_layout = QVBoxLayout(logs_group)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logs will appear here...")
        logs_layout.addWidget(self.log_view)

        layout.addWidget(logs_group)

        # Load saved schedule or default 01:30
        saved = self._load_saved_time()
        if saved is not None:
            self.time_edit.setTime(saved)
        else:
            self.time_edit.setTime(dtime(hour=1, minute=30))
        self._service.set_scheduled_time(self.time_edit.time().toPyTime())
        self._service.start()

        return tab

    def load_data(self):
        """Load data for training"""
        symbols = self.symbols_input.toPlainText().strip()
        if not symbols:
            QMessageBox.warning(self, "Warning", "Please enter symbols to load data")
            return

        self.data_status_label.setText(f"Ready to load data for: {symbols}")
        # TODO: Implement actual data loading from selected source

    def export_data(self):
        """Export processed data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.data_status_label.setText(f"Data exported to: {file_path}")

    # --- Daily Update slots ---
    def _on_run_now(self):
        """Run the daily update now"""
        # Try to run the external small-run adapter in a background thread so UI stays responsive.
        self.progress.setValue(0)
        self.stop_btn.setEnabled(True)
        self.run_now_btn.setEnabled(False)
        limit = int(self.limit_spin.value() or 0)

        # Clear previous logs (final report removed)
        self.log_view.clear()

        # spawn a QThread to run the subprocess
        class SubprocRunner(QObject):
            finished = pyqtSignal()
            line = pyqtSignal(str)
            failed = pyqtSignal(str)

            def __init__(self, cmd):
                super().__init__()
                self.cmd = cmd
                self.proc = None

            def stop(self):
                try:
                    if self.proc and self.proc.poll() is None:
                        # Try graceful termination first
                        self.proc.terminate()
                        try:
                            self.proc.wait(timeout=3)
                        except Exception:
                            self.proc.kill()
                except Exception as e:
                    # Emit a non-fatal failure message to log
                    self.failed.emit(f"Stop failed: {e}")

            def run(self):
                try:
                    # Start subprocess and stream output
                    self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, text=True, bufsize=1)
                    for ln in self.proc.stdout:
                        ln = ln.rstrip('\n')
                        if ln:  # Only emit non-empty lines
                            self.line.emit(ln)
                    self.proc.wait()
                    self.finished.emit()
                except Exception as e:
                    self.failed.emit(str(e))

        # Build command to run the adapter
        python_exec = Path(sys.executable).as_posix() if hasattr(sys, 'executable') else 'python'
        adapter = Path(__file__).parent.parent.parent.parent / 'tools' / 'download_stocks.py'
        if adapter.exists():
            if limit == 0:
                # No limit - download all stocks
                cmd = [python_exec, str(adapter)]
            else:
                # Specific limit
                cmd = [python_exec, str(adapter), '--limit', str(limit)]
            self._append_log(f"Starting adapter: {cmd}")
            # create thread and runner
            runner = SubprocRunner(cmd)
            thread = QThread(self)
            runner.moveToThread(thread)
            thread.started.connect(runner.run)
            runner.line.connect(self._append_log)
            runner.failed.connect(lambda msg: self._append_log(f"Adapter error: {msg}"))
            runner.finished.connect(lambda: (self._append_log("Adapter finished."), thread.quit(), self._on_adapter_finished()))
            thread.finished.connect(lambda: setattr(self, "_adapter_thread", None))
            self._adapter_runner = runner
            self._adapter_thread = thread
            thread.start()
        else:
            # fallback to built-in service run_now
            self._append_log("Adapter not found; falling back to DataUpdateService.run_now")
            batch_limit = None if limit == 0 else limit
            self._service.run_now(batch_limit=batch_limit)
            self._append_log("Triggered update run…")

    def _on_stop(self):
        """Stop the running update"""
        # Stop built-in service if running
        self._service.stop()
        # Stop external adapter process/thread if present
        try:
            if getattr(self, "_adapter_runner", None):
                self._adapter_runner.stop()
            if getattr(self, "_adapter_thread", None):
                self._adapter_thread.quit()
                self._adapter_thread.wait(3000)
                self._adapter_thread = None
        except Exception:
            pass
        self.stop_btn.setEnabled(False)
        self.run_now_btn.setEnabled(True)
        self._append_log("Stop requested")

    def on_save(self):
        """Save the schedule configuration"""
        t = self.time_edit.time().toPyTime()
        self._service.set_scheduled_time(t)
        self._append_log(f"Schedule saved: {t.strftime('%H:%M')}")
        try:
            self._cfg_path.write_text(json.dumps({"time": t.strftime('%H:%M')}), encoding='utf-8')
        except Exception:
            pass

    # --- Service signal handlers ---
    def _on_progress(self, v: int):
        """Update progress bar"""
        self.progress.setValue(v)

    def _on_log(self, msg: str):
        """Append log message"""
        self._append_log(msg)

    def _on_error(self, msg: str):
        """Handle error"""
        self._append_log(f"ERROR: {msg}")
        self.stop_btn.setEnabled(False)
        self.run_now_btn.setEnabled(True)

    def _on_completed(self, payload: dict):
        """Handle completion"""
        n = payload.get("tickers", 0)
        completion_msg = f"Completed update for {n} tickers"
        self._append_log(completion_msg)

        # Log a summary to the live logs (Final Report UI was removed)
        summary = f"📊 Daily Update Report - Completed: {n} tickers at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._append_log(summary)

        self.stop_btn.setEnabled(False)
        self.run_now_btn.setEnabled(True)

    def _on_next_run(self, ts: datetime):
        """Update next run time display"""
        self.next_run_label.setText(f"Next run: {ts.strftime('%Y-%m-%d %H:%M')}")

    def _on_adapter_finished(self):
        """Handle adapter completion"""
        self.run_now_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # Clean up runner reference
        self._adapter_runner = None

    def _append_log(self, line: str):
        """Append a line to the log view"""
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_view.append(f"[{ts}] {line}")

    def closeEvent(self, event):
        """Ensure background thread is closed cleanly to avoid QThread warning"""
        try:
            if getattr(self, "_adapter_runner", None):
                self._adapter_runner.stop()
            if getattr(self, "_adapter_thread", None):
                self._adapter_thread.quit()
                self._adapter_thread.wait(3000)
                self._adapter_thread = None
        except Exception:
            pass
        super().closeEvent(event)

    # --- Persistence helpers ---
    def _load_saved_time(self):
        """Load saved schedule time"""
        try:
            if self._cfg_path.exists():
                obj = json.loads(self._cfg_path.read_text(encoding='utf-8'))
                val = obj.get("time")
                if isinstance(val, str) and len(val) == 5:
                    hh, mm = val.split(":")
                    return dtime(hour=int(hh), minute=int(mm))
        except Exception:
            return None
        return None

    def run_comprehensive_check(self):
        """Run comprehensive system report checking all files and components"""
        try:
            self.run_comprehensive_report_btn.setEnabled(False)
            self.report_progress.setVisible(True)
            self.report_progress.setValue(0)
            self.report_results.clear()
            self.report_results.append("📋 Starting Comprehensive System Report...\n")
            
            # Run comprehensive check in background thread
            class ComprehensiveChecker(QObject):
                finished = pyqtSignal(str)
                progress = pyqtSignal(int)
                log = pyqtSignal(str)
                
                def run(self):
                    try:
                        from pathlib import Path
                        import os
                        import sys
                        
                        repo_root = Path(__file__).parent.parent.parent.parent
                        self.log.emit(f"Repository root: {repo_root}")
                        self.progress.emit(10)
                        
                        # Check Python environment
                        self.log.emit("🔍 Checking Python environment...")
                        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                        self.log.emit(f"   Python version: {python_version}")
                        self.progress.emit(20)
                        
                        # Check virtual environment
                        venv_path = repo_root / ".venv"
                        if venv_path.exists():
                            self.log.emit("   ✅ Virtual environment found")
                        else:
                            self.log.emit("   ❌ Virtual environment missing")
                        
                        # Check required directories
                        self.log.emit("🔍 Checking directory structure...")
                        required_dirs = [
                            "src", "data", "data/bronze", "data/silver", "data/gold",
                            "models", "logs", "config", "scripts", "tools", "docs"
                        ]
                        missing_dirs = []
                        for dir_path in required_dirs:
                            full_path = repo_root / dir_path
                            if full_path.exists():
                                self.log.emit(f"   ✅ {dir_path}")
                            else:
                                self.log.emit(f"   ❌ {dir_path} - MISSING")
                                missing_dirs.append(dir_path)
                        
                        self.progress.emit(40)
                        
                        # Check required files
                        self.log.emit("🔍 Checking critical files...")
                        required_files = [
                            "main.py", "requirements.txt", "config/ai_trading.json",
                            "src/core/config_manager.py", "src/ui/main_window.py"
                        ]
                        missing_files = []
                        for file_path in required_files:
                            full_path = repo_root / file_path
                            if full_path.exists():
                                self.log.emit(f"   ✅ {file_path}")
                            else:
                                self.log.emit(f"   ❌ {file_path} - MISSING")
                                missing_files.append(file_path)
                        
                        self.progress.emit(60)
                        
                        # Check data files
                        self.log.emit("🔍 Checking data availability...")
                        stock_data_dir = repo_root / "stock_data"
                        if stock_data_dir.exists():
                            csv_files = list(stock_data_dir.glob("*.csv"))
                            parquet_files = list(stock_data_dir.glob("*.parquet"))
                            self.log.emit(f"   📊 Found {len(csv_files)} CSV files, {len(parquet_files)} Parquet files")
                            
                            if len(csv_files) + len(parquet_files) == 0:
                                self.log.emit("   ⚠️ No data files found in stock_data/")
                        else:
                            self.log.emit("   ❌ stock_data directory missing")
                        
                        # Check bronze data
                        bronze_dir = repo_root / "data" / "bronze" / "daily"
                        if bronze_dir.exists():
                            bronze_files = list(bronze_dir.glob("*.parquet"))
                            self.log.emit(f"   📊 Found {len(bronze_files)} bronze data files")
                        else:
                            self.log.emit("   ⚠️ Bronze data directory missing")
                        
                        # Check cache status
                        cache_file = repo_root / "data" / "cache" / "pooled_dataset.parquet"
                        if cache_file.exists():
                            cache_size = cache_file.stat().st_size / (1024 * 1024)  # MB
                            self.log.emit(f"   💾 Cache exists: {cache_size:.1f} MB")
                            
                            # Check if cache is up to date
                            metadata_file = cache_file.with_suffix('.metadata.json')
                            if metadata_file.exists():
                                import json
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                cache_created = metadata.get('created_at', 'unknown')
                                self.log.emit(f"   📅 Cache created: {cache_created}")
                                
                                # Check for newer files
                                cache_time = None
                                if cache_created != 'unknown':
                                    from datetime import datetime
                                    try:
                                        cache_time = datetime.fromisoformat(cache_created)
                                    except:
                                        pass
                                
                                if cache_time:
                                    outdated_files = []
                                    if bronze_dir.exists():
                                        for parquet_file in bronze_dir.glob("*.parquet"):
                                            file_mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime)
                                            if file_mtime > cache_time:
                                                outdated_files.append(parquet_file.name)
                                    
                                    if outdated_files:
                                        self.log.emit(f"   ⚠️ Cache outdated - {len(outdated_files)} newer files")
                                    else:
                                        self.log.emit("   ✅ Cache is up to date")
                            else:
                                self.log.emit("   ⚠️ Cache metadata missing")
                        else:
                            self.log.emit("   💾 No cache file found")
                        
                        self.progress.emit(80)
                        
                        # Check Python imports
                        self.log.emit("🔍 Checking Python imports...")
                        test_imports = [
                            ("PyQt6.QtWidgets", "PyQt6"),
                            ("pandas", "pandas"),
                            ("numpy", "numpy"),
                            ("yfinance", "yfinance"),
                            ("sklearn", "scikit-learn"),
                            ("ib_insync", "ib_insync")
                        ]
                        
                        failed_imports = []
                        for module_name, package_name in test_imports:
                            try:
                                __import__(module_name)
                                self.log.emit(f"   ✅ {package_name}")
                            except ImportError:
                                self.log.emit(f"   ❌ {package_name} - Import failed")
                                failed_imports.append(package_name)
                        
                        self.progress.emit(100)
                        
                        # Summary
                        self.log.emit("\n📋 SUMMARY:")
                        if not missing_dirs and not missing_files and not failed_imports:
                            self.log.emit("🎉 All checks passed! System is ready.")
                            self.finished.emit("SUCCESS")
                        else:
                            if missing_dirs:
                                self.log.emit(f"❌ Missing directories: {', '.join(missing_dirs)}")
                            if missing_files:
                                self.log.emit(f"❌ Missing files: {', '.join(missing_files)}")
                            if failed_imports:
                                self.log.emit(f"❌ Failed imports: {', '.join(failed_imports)}")
                            self.finished.emit("ISSUES_FOUND")
                            
                    except Exception as e:
                        self.log.emit(f"❌ Exception during comprehensive check: {str(e)}")
                        import traceback
                        self.log.emit(traceback.format_exc())
                        self.finished.emit("ERROR")
            
            # Create and run comprehensive check in thread
            self.comprehensive_checker = ComprehensiveChecker()
            self.comprehensive_thread = QThread()
            self.comprehensive_checker.moveToThread(self.comprehensive_thread)
            
            self.comprehensive_checker.finished.connect(self.on_comprehensive_check_finished)
            self.comprehensive_checker.progress.connect(self.report_progress.setValue)
            self.comprehensive_checker.log.connect(self.report_results.append)
            
            self.comprehensive_thread.started.connect(self.comprehensive_checker.run)
            self.comprehensive_thread.start()
            
        except Exception as e:
            self.report_results.append(f"❌ Failed to start comprehensive check: {str(e)}")
            self.run_comprehensive_report_btn.setEnabled(True)
            self.report_progress.setVisible(False)
    
    def on_comprehensive_check_finished(self, status):
        """Handle comprehensive check completion"""
        self.report_progress.setValue(100)
        self.run_comprehensive_report_btn.setEnabled(True)
        self.comprehensive_thread.quit()
        self.comprehensive_thread.wait()
        
        if status == "SUCCESS":
            self.report_results.append("\n🎉 Comprehensive check completed successfully!")
        else:
            self.report_results.append(f"\n⚠️ Comprehensive check completed with status: {status}")
    
    def clear_data_cache(self):
        """Clear the data cache files"""
        try:
            from pathlib import Path
            import os
            
            cache_dir = Path("data/cache")
            cache_file = cache_dir / "pooled_dataset.parquet"
            metadata_file = cache_dir / "pooled_dataset.metadata.json"
            
            files_deleted = 0
            
            # Delete cache file
            if cache_file.exists():
                cache_file.unlink()
                self.report_results.append(f"🗑️ Deleted cache file: {cache_file}")
                files_deleted += 1
            
            # Delete metadata file
            if metadata_file.exists():
                metadata_file.unlink()
                self.report_results.append(f"🗑️ Deleted metadata file: {metadata_file}")
                files_deleted += 1
            
            # Clear memory cache
            try:
                from ml.dataset import build_pooled_dataset
                if hasattr(build_pooled_dataset, '_cache'):
                    build_pooled_dataset._cache.clear()
                    self.report_results.append("🧠 Cleared memory cache")
            except:
                pass
            
            if files_deleted == 0:
                self.report_results.append("ℹ️ No cache files found to delete")
            else:
                self.report_results.append(f"\n✅ Cache cleared successfully! ({files_deleted} files)")
                self.report_results.append("\n💡 Next pipeline run will rebuild cache with fresh data")
                self.report_results.append("💡 Cache will auto-update when new data is downloaded")
                
        except Exception as e:
            self.report_results.append(f"❌ Error clearing cache: {str(e)}")
    
    def refresh_cache_list(self):
        """Refresh the cache files list in the table"""
        from pathlib import Path
        import json
        from datetime import datetime
        
        cache_dir = Path("data/cache")
        if not cache_dir.exists():
            self.cache_list.setRowCount(0)
            return
        
        # Find all cache files (parquet files with metadata)
        cache_files = []
        for parquet_file in cache_dir.glob("*.parquet"):
            metadata_file = parquet_file.with_suffix('.metadata.json')
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    cache_files.append({
                        'name': parquet_file.stem,
                        'created': metadata.get('created_at', 'Unknown'),
                        'size': parquet_file.stat().st_size,
                        'rows': metadata.get('num_rows', 0),
                        'config': self._get_cache_config_from_metadata(metadata)
                    })
                except Exception as e:
                    print(f"Error reading cache metadata for {parquet_file}: {e}")
        
        # Sort by creation date (newest first)
        cache_files.sort(key=lambda x: x['created'], reverse=True)
        
        # Update table
        self.cache_list.setRowCount(len(cache_files))
        for row, cache in enumerate(cache_files):
            self.cache_list.setItem(row, 0, QTableWidgetItem(cache['name']))
            
            # Format creation date
            try:
                created_dt = datetime.fromisoformat(cache['created'])
                created_str = created_dt.strftime("%Y-%m-%d %H:%M")
            except:
                created_str = cache['created']
            self.cache_list.setItem(row, 1, QTableWidgetItem(created_str))
            
            # Format file size
            size_mb = cache['size'] / (1024 * 1024)
            self.cache_list.setItem(row, 2, QTableWidgetItem(f"{size_mb:.1f} MB"))
            
            self.cache_list.setItem(row, 3, QTableWidgetItem(str(cache['rows'])))
            self.cache_list.setItem(row, 4, QTableWidgetItem(cache['config']))
        
        # Update training cache combo
        self.training_cache_combo.clear()
        self.training_cache_combo.addItem("Auto (default)", None)
        for cache in cache_files:
            config_desc = cache['config']
            display_name = f"{cache['name']} ({config_desc})"
            self.training_cache_combo.addItem(display_name, cache['name'])
    
    def _get_cache_config_from_metadata(self, metadata):
        """Extract configuration description from metadata"""
        # For now, return a simple description based on available info
        # In future, we could store config details in metadata
        columns = metadata.get('columns', [])
        
        features = []
        if any('ema_' in col for col in columns):
            features.append('technical')
        if any('volume_' in col for col in columns):
            features.append('volume')
        if any('sentiment' in col.lower() for col in columns):
            features.append('sentiment')
        
        if features:
            return '+'.join(features)
        else:
            return 'basic'
    
    def show_create_cache_dialog(self):
        """Show dialog to create a new cache file"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Cache")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        
        # Cache name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Cache Name:"))
        self.cache_name_input = QLineEdit()
        self.cache_name_input.setPlaceholderText("e.g., technical_only, full_features")
        name_layout.addWidget(self.cache_name_input)
        layout.addLayout(name_layout)
        
        # Configuration options
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        
        self.use_technical_cb = QCheckBox("Use Technical Features (EMA, RSI, MACD, etc.)")
        self.use_technical_cb.setChecked(True)
        config_layout.addWidget(self.use_technical_cb)
        
        self.use_volume_cb = QCheckBox("Use Volume Features")
        self.use_volume_cb.setChecked(True)
        config_layout.addWidget(self.use_volume_cb)
        
        self.use_sentiment_cb = QCheckBox("Use Sentiment Features")
        self.use_sentiment_cb.setChecked(False)
        config_layout.addWidget(self.use_sentiment_cb)
        
        layout.addWidget(config_group)
        
        # Buttons
        buttons = QHBoxLayout()
        create_btn = QPushButton("Create Cache")
        create_btn.clicked.connect(lambda: self.create_new_cache(dialog))
        buttons.addWidget(create_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_btn)
        
        layout.addLayout(buttons)
        
        dialog.exec()
    
    def create_new_cache(self, dialog):
        """Create a new cache file with specified configuration"""
        cache_name = self.cache_name_input.text().strip()
        if not cache_name:
            QMessageBox.warning(self, "Error", "Please enter a cache name")
            return
        
        # Validate cache name (no special characters that could cause file issues)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', cache_name):
            QMessageBox.warning(self, "Error", "Cache name can only contain letters, numbers, underscores, and hyphens")
            return
        
        # Check if cache already exists
        from pathlib import Path
        cache_file = Path(f"data/cache/{cache_name}.parquet")
        if cache_file.exists():
            reply = QMessageBox.question(
                self, "Cache Exists", 
                f"Cache '{cache_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Create configuration
        from ml.config import TrainingConfig
        config = TrainingConfig()
        config.use_technical = self.use_technical_cb.isChecked()
        config.use_volume = self.use_volume_cb.isChecked()
        config.use_sentiment = self.use_sentiment_cb.isChecked()
        
        dialog.accept()
        
        # Build cache in background
        self._build_cache_file(cache_name, config)
    
    def _build_cache_file(self, cache_name, config):
        """Build cache file with given configuration"""
        try:
            from pathlib import Path
            import pandas as pd
            from ml.dataset import build_pooled_dataset
            
            # Disable buttons during build
            self.create_cache_btn.setEnabled(False)
            self.status_label.setText(f"Building cache '{cache_name}'...")
            QApplication.processEvents()
            
            # Load bronze data
            bronze_dir = Path("data/bronze")
            if not bronze_dir.exists():
                raise FileNotFoundError("Bronze data directory not found")
            
            bronze = {}
            for parquet_file in bronze_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file)
                    ticker = parquet_file.stem
                    bronze[ticker] = df
                except Exception as e:
                    self.logger.warning(f"Failed to load {parquet_file}: {e}")
            
            if not bronze:
                raise ValueError("No bronze data files found")
            
            # Build dataset
            cache_file = Path(f"data/cache/{cache_name}.parquet")
            dataset = build_pooled_dataset(bronze, config, cache_file=cache_file, use_parallel=True)
            
            if dataset.empty:
                raise ValueError("Generated dataset is empty")
            
            self.status_label.setText(f"Cache '{cache_name}' created successfully ({len(dataset)} rows)")
            self.refresh_cache_list()
            
        except Exception as e:
            self.logger.error(f"Failed to create cache '{cache_name}': {e}")
            QMessageBox.critical(self, "Cache Creation Failed", str(e))
            self.status_label.setText("Cache creation failed")
        finally:
            self.create_cache_btn.setEnabled(True)
    
    def delete_selected_cache(self):
        """Delete the selected cache file"""
        selected_rows = set()
        for item in self.cache_list.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select a cache to delete")
            return
        
        if len(selected_rows) > 1:
            QMessageBox.information(self, "Multiple Selection", "Please select only one cache to delete")
            return
        
        row = list(selected_rows)[0]
        cache_name = self.cache_list.item(row, 0).text()
        
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to delete cache '{cache_name}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from pathlib import Path
                cache_file = Path(f"data/cache/{cache_name}.parquet")
                metadata_file = Path(f"data/cache/{cache_name}.metadata.json")
                
                deleted = 0
                if cache_file.exists():
                    cache_file.unlink()
                    deleted += 1
                if metadata_file.exists():
                    metadata_file.unlink()
                    deleted += 1
                
                if deleted > 0:
                    self.status_label.setText(f"Cache '{cache_name}' deleted")
                    self.refresh_cache_list()
                else:
                    QMessageBox.warning(self, "Not Found", f"Cache '{cache_name}' not found")

            except Exception as e:
                QMessageBox.critical(self, "Delete Failed", f"Failed to delete cache: {str(e)}")


# DataManagementWidget removed — now handled by top-level DataWidget
class MLWidget(QWidget):
    """Main ML widget with tabs for different ML functions"""
    # Signal emitted when user clicks the small Ask-AI icon in the One Symbol table.
    # Parameters: symbol (str), payload (dict) - payload contains compact context (price_targets, per_horizon, overall_signal)
    ask_ai_clicked = pyqtSignal(str, dict)
    
    def __init__(self):
        super().__init__()
        
        # Initialize logger and config
        self.logger = get_logger("MLWidget")
        self.config = ConfigManager()
        
        # Initialize training worker
        self.training_thread = QThread()
        self.training_worker = ModelTrainingWorker(self.config)
        self.training_worker.moveToThread(self.training_thread)
        self.training_worker.progress_updated.connect(self.update_progress)
        self.training_worker.status_updated.connect(self.update_status)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.error_occurred.connect(self.on_training_error)
        self.training_thread.start()
        
        # Initialize pipeline worker
        self.pipeline_thread = QThread()
        self.pipeline_worker = PipelineRunWorker()
        self.pipeline_worker.moveToThread(self.pipeline_thread)
        self.pipeline_worker.progress_updated.connect(self.update_progress)
        self.pipeline_worker.status_updated.connect(self.update_status)
        self.pipeline_worker.completed.connect(self.on_pipeline_completed)
        self.pipeline_worker.error_occurred.connect(self.on_pipeline_error)
        self.pipeline_thread.start()
        
        # Initialize pipeline step counter
        self.pipeline_step_counter = 0
        self.pipeline_total_steps = 0
        
        # Setup UI
        self.setup_ui()
        
    # No ML-level data_management_widget in this refactor; pipeline worker has no data_widget reference
        self.pipeline_worker.ml_widget = self
        # Connect Ask-AI signal to handler that will open a prompt dialog and request AI response
        try:
            self.ask_ai_clicked.connect(self._on_ask_ai)
        except Exception:
            pass
        
        self.logger.info("ML widget initialized")

    # --- Helpers to access data controls safely (ML-level data tab removed) ---
    def _get_symbols_from_data_widget(self):
        """Return list of tickers from the ML-level data widget if present, else empty list."""
        try:
            if hasattr(self, 'data_management_widget') and getattr(self, 'data_management_widget'):
                txt = self.data_management_widget.symbols_input.toPlainText().strip()
                if txt:
                    return [s.strip().upper() for s in txt.split(',') if s.strip()]
        except Exception:
            pass
        return []

    def _get_selected_cache_from_data_widget(self):
        """Return selected cache name from ML-level data widget if present, else None."""
        try:
            if hasattr(self, 'data_management_widget') and getattr(self, 'data_management_widget'):
                combo = getattr(self.data_management_widget, 'training_cache_combo', None)
                if combo is not None:
                    return combo.currentData()
        except Exception:
            pass
        return None
    
    def create_pipeline_tab(self):
        """Create the pipeline tab widget with responsive scroll area"""
        from PyQt6.QtWidgets import QScrollArea
        
        # Create scroll area for small screens
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)  # Reduced spacing for small screens

        # Titles row - align configuration and progress titles
        titles_layout = QHBoxLayout()
        # Reduce spacing and margins for a more compact header row
        titles_layout.setContentsMargins(0, 2, 0, 2)
        titles_layout.setSpacing(2)
        
        # Responsive font sizes - get QApplication from the import at top of file
        from PyQt6.QtWidgets import QApplication as QtApp
        screen = QtApp.primaryScreen()
        screen_height = screen.availableGeometry().height()
        title_font_size = 8 if screen_height <= 768 else 10
        
        config_title = QLabel("Pipeline Configuration")
        config_title.setFont(QFont("Arial", title_font_size, QFont.Weight.Bold))
        titles_layout.addWidget(config_title)
        
        titles_layout.addStretch()
        
        progress_title = QLabel("Pipeline Progress")
        progress_title.setFont(QFont("Arial", title_font_size, QFont.Weight.Bold))
        titles_layout.addWidget(progress_title)
        
        layout.addLayout(titles_layout)

        # Pipeline Configuration
        config_frame = QFrame()
        config_frame.setFrameStyle(QFrame.Shape.Box)
        
        config_layout = QVBoxLayout(config_frame)
        # Pull the config content up to reduce vertical gap under the titles
        config_layout.setContentsMargins(6, 2, 6, 6)
        
        # Use form layout so labels are directly adjacent to controls
        from PyQt6.QtWidgets import QFormLayout
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        # Controls
        # Put Holdout and Step on the same horizontal row
        self.holdout_spin = QSpinBox()
        self.holdout_spin.setRange(5, 250)
        self.holdout_spin.setValue(30)
        self.holdout_spin.setFixedWidth(80)
        self.holdout_spin.setToolTip(
            "Number of days to use for model performance testing.\nMore days = more accurate testing, but longer training time"
        )

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 60)
        self.step_spin.setValue(5)
        self.step_spin.setFixedWidth(70)
        self.step_spin.setToolTip(
            "Time interval between each training point.\n1 = daily training, 5 = training every 5 days.\nMore frequent = more training points, but longer runtime"
        )

        row1 = QWidget()
        row1_h = QHBoxLayout(row1)
        row1_h.setContentsMargins(0, 0, 0, 0)
        row1_h.setSpacing(8)
        row1_h.addWidget(QLabel("Holdout days:"))
        row1_h.addWidget(self.holdout_spin)
        row1_h.addSpacing(12)
        row1_h.addWidget(QLabel("Step days:"))
        row1_h.addWidget(self.step_spin)
        row1_h.addStretch()
        form.addRow(row1)

        # Put Window and Lookback on the same horizontal row
        self.window_combo = QComboBox()
        self.window_combo.addItems(["expanding", "rolling"])
        self.window_combo.setCurrentText("expanding")
        self.window_combo.setFixedWidth(110)
        self.window_combo.setToolTip(
            "Training window expansion method:\n• expanding = all data up to current point\n• rolling = fixed window (requires lookback setting)"
        )

        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(50, 2000)
        self.lookback_spin.setValue(500)
        self.lookback_spin.setEnabled(False)
        self.lookback_spin.setFixedWidth(90)
        self.lookback_spin.setToolTip(
            "Number of days to use for training when window is rolling.\nOnly active when rolling window is selected"
        )

        row2 = QWidget()
        row2_h = QHBoxLayout(row2)
        row2_h.setContentsMargins(0, 0, 0, 0)
        row2_h.setSpacing(8)
        row2_h.addWidget(QLabel("Window:"))
        row2_h.addWidget(self.window_combo)
        row2_h.addSpacing(12)
        row2_h.addWidget(QLabel("Lookback:"))
        row2_h.addWidget(self.lookback_spin)
        row2_h.addStretch()
        form.addRow(row2)

        self.window_combo.currentTextChanged.connect(self._on_window_changed)

        self.pipeline_models_combo = QComboBox()
        self.pipeline_models_combo.addItems([
            "RandomForest",
            "LightGBM",
            "CatBoost",
            "LogisticRegression",
        ])
        self.pipeline_models_combo.setToolTip("Machine learning algorithm:\n• RandomForest = stable and fast\n• LightGBM = fast and efficient\n• CatBoost = accurate but slow\n• LogisticRegression = simple and fast")
        form.addRow(QLabel("Model:"), self.pipeline_models_combo)
        
        # Single stock field for pipeline execution
        self.single_stock_input = QLineEdit()
        self.single_stock_input.setPlaceholderText("Optional: Run pipeline on single stock (e.g., AAPL)")
        self.single_stock_input.setFixedWidth(200)
        self.single_stock_input.setToolTip("Run pipeline on a specific single stock only.\nIf empty = run on all stocks from Data tab")
        form.addRow(QLabel("Single Stock:"), self.single_stock_input)
        
        # Feature selection checkboxes
        features_layout = QHBoxLayout()
        features_layout.addWidget(QLabel("Features:"))
        
        self.use_technical_indicators = QCheckBox("Technical Indicators")
        self.use_technical_indicators.setChecked(True)
        self.use_technical_indicators.setToolTip("Technical indicators: RSI, MACD, Bollinger Bands, EMA, SMA\nImprove prediction accuracy but increase training time")
        features_layout.addWidget(self.use_technical_indicators)
        
        self.use_volume_data = QCheckBox("Volume Data")
        self.use_volume_data.setChecked(True)
        self.use_volume_data.setToolTip("Trading volume data: averages and ratios\nHelp identify strong trends but may cause issues with extreme values")
        features_layout.addWidget(self.use_volume_data)
        
        self.use_sentiment_data = QCheckBox("Sentiment Data")
        self.use_sentiment_data.setChecked(False)
        self.use_sentiment_data.setToolTip("Sentiment data (if available)\nCan improve predictions but requires additional data")
        features_layout.addWidget(self.use_sentiment_data)
        
        self.use_parallel_processing = QCheckBox("Parallel Processing")
        self.use_parallel_processing.setChecked(True)
        self.use_parallel_processing.setToolTip("Parallel data processing - faster but uses more memory\n\nWhen to disable:\n• Computer with weak CPU or few cores\n• Limited RAM memory\n• System stability issues\n• Debugging - harder to track errors\n\nRecommended: Keep enabled for modern computers")
        features_layout.addWidget(self.use_parallel_processing)
        
        features_layout.addStretch()
        form.addRow(features_layout)

        config_layout.addLayout(form)

        # One Symbol Prediction small results table (placed directly under Pipeline Configuration)
        one_symbol_frame = QFrame()
        one_symbol_frame.setFrameStyle(QFrame.Shape.Box)
        one_symbol_layout = QVBoxLayout(one_symbol_frame)
        # Make title tighter to the table and reduce spacing so the compact table is fully visible
        # Minimal top margin to push title upward, small bottom margin to reduce gap to the table
        one_symbol_layout.setContentsMargins(4, 0, 4, 2)
        one_symbol_layout.setSpacing(1)

        one_title = QLabel("One Symbol Prediction")
        # Even smaller title font and slimmer weight so it takes less vertical space
        one_title.setFont(QFont("Arial", 9, QFont.Weight.DemiBold))
        # Reduce internal margins for the title label and set fixed height to keep it compact
        one_title.setContentsMargins(0, 0, 0, 1)
        try:
            one_title.setFixedHeight(18)
        except Exception:
            pass
        one_symbol_layout.addWidget(one_title)

    # Small read-only table showing a single-symbol summary (Symbol, Price, Signal, Conf, Day1, Day5, Day10) + Ask AI
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
        from PyQt6.QtGui import QColor
        self.one_symbol_table = QTableWidget()
        # Add a small Ask-AI column at the end (compact icon/button) and a Current Price column
        self.one_symbol_table.setColumnCount(8)
        self.one_symbol_table.setHorizontalHeaderLabels(["Symbol", "Price", "Signal", "Conf", "Day1", "Day5", "Day10", "7Days AI Prediction"])
        self.one_symbol_table.setRowCount(1)
        # Initialize empty cells
        for c in range(8):
            item = QTableWidgetItem("-")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.one_symbol_table.setItem(0, c, item)
        # Make the table more compact: smaller font, tighter cell padding, smaller header/row heights
        self.one_symbol_table.setFont(QFont("Segoe UI", 7))
        # Apply stylesheet to reduce padding for both header sections and items
        self.one_symbol_table.setStyleSheet(
            "QTableWidget::item{ padding:1px 4px; }"
            "QHeaderView::section{ padding:1px 4px; font-size:9px; }"
        )
        # Adjust column sizing so the long header for the AI column is fully visible
        try:
            from PyQt6.QtWidgets import QHeaderView
            header = self.one_symbol_table.horizontalHeader()
            # Small content-based sizes for first columns
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            # Day columns keep reasonable fixed width
            self.one_symbol_table.setColumnWidth(4, 70)
            self.one_symbol_table.setColumnWidth(5, 70)
            self.one_symbol_table.setColumnWidth(6, 70)
            # Make the AI column stretch and give it a comfortable minimum width
            header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
            self.one_symbol_table.setColumnWidth(7, 220)
        except Exception:
            pass
        # Hide the vertical header (row numbers) to save space
        try:
            self.one_symbol_table.verticalHeader().setVisible(False)
        except Exception:
            pass

        # Slightly smaller maximum height and row height to keep the UI compact
        # Header + row should fit within ~42-46 pixels
        self.one_symbol_table.setMaximumHeight(46)
        try:
            self.one_symbol_table.horizontalHeader().setFixedHeight(18)
            self.one_symbol_table.setRowHeight(0, 18)
        except Exception:
            pass

        # Set reasonable compact column widths so table displays without horizontal scroll
        try:
            self.one_symbol_table.setColumnWidth(0, 60)   # Symbol
            self.one_symbol_table.setColumnWidth(1, 70)   # Price
            self.one_symbol_table.setColumnWidth(2, 70)   # Signal
            self.one_symbol_table.setColumnWidth(3, 50)   # Conf
            self.one_symbol_table.setColumnWidth(4, 60)   # Day1
            self.one_symbol_table.setColumnWidth(5, 60)   # Day5
            self.one_symbol_table.setColumnWidth(6, 60)   # Day10
            self.one_symbol_table.setColumnWidth(7, 34)   # Ask (small icon button)
        except Exception:
            pass
        one_symbol_layout.addWidget(self.one_symbol_table)
        # Auto AI checkbox: when enabled, the app will automatically ask the AI after a single-stock run
        try:
            self.auto_ai_checkbox = QCheckBox("Auto AI (ask after training)")
            # Default to enabled so single-symbol runs automatically ask the AI
            self.auto_ai_checkbox.setChecked(True)
            self.auto_ai_checkbox.setToolTip("If checked, the app will automatically send the compact prompt to the AI after a single-stock pipeline run and display the response in the right-most cell.")
            # Keep it visually small and left-aligned under the compact table
            cb_row = QWidget()
            cb_row_h = QHBoxLayout(cb_row)
            cb_row_h.setContentsMargins(0, 0, 0, 0)
            cb_row_h.setSpacing(2)
            cb_row_h.addWidget(self.auto_ai_checkbox)
            cb_row_h.addStretch()
            one_symbol_layout.addWidget(cb_row)
        except Exception:
            # Best-effort: if checkboxes aren't available, skip
            pass
        config_layout.addWidget(one_symbol_frame)

        # Pipeline Progress
        progress_frame = QFrame()
        progress_frame.setFrameStyle(QFrame.Shape.Box)
        
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(10, 5, 10, 10)  # Reduced top margin

        # Data loading progress (separate from pipeline progress)
        data_progress_layout = QHBoxLayout()
        data_progress_layout.addWidget(QLabel("Data Loading:"))
        self.data_progress_bar = QProgressBar()
        self.data_progress_bar.setVisible(False)
        self.data_progress_bar.setMaximumWidth(200)
        data_progress_layout.addWidget(self.data_progress_bar)
        data_progress_layout.addStretch()
        progress_layout.addLayout(data_progress_layout)

        # Pipeline control buttons
        buttons_layout = QHBoxLayout()
        
        # Run Pipeline button
        self.run_pipeline_btn = QPushButton("🏃 Run Pipeline")
        self.run_pipeline_btn.clicked.connect(self.start_pipeline_run)
        self.run_pipeline_btn.setFixedWidth(150)
        buttons_layout.addWidget(self.run_pipeline_btn)
        
        # Stop Pipeline button
        self.stop_pipeline_btn = QPushButton("⏹️ Stop Pipeline")
        self.stop_pipeline_btn.clicked.connect(self.stop_pipeline_run)
        self.stop_pipeline_btn.setFixedWidth(150)
        self.stop_pipeline_btn.setEnabled(False)  # Disabled by default
        buttons_layout.addWidget(self.stop_pipeline_btn)
        
        buttons_layout.addStretch()
        progress_layout.addLayout(buttons_layout)

        # Status label for pipeline progress
        self.pipeline_status_label = QLabel("Ready to run pipeline")
        self.pipeline_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.pipeline_status_label)

        # Step counter label
        self.pipeline_step_label = QLabel("Steps: 0/0")
        self.pipeline_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Responsive font size for step counter
        step_font_size = 7 if screen_height <= 768 else 9
        self.pipeline_step_label.setFont(QFont("Arial", step_font_size, QFont.Weight.Bold))
        progress_layout.addWidget(self.pipeline_step_label)

        # Progress bar for pipeline
        self.pipeline_progress_bar = QProgressBar()
        self.pipeline_progress_bar.setVisible(False)
        progress_layout.addWidget(self.pipeline_progress_bar)

        # Detailed progress steps
        progress_steps_label = QLabel("Progress Details:")
        progress_steps_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        progress_layout.addWidget(progress_steps_label)
        
        self.pipeline_progress_details = QTextEdit()
        # Responsive heights based on screen size
        screen = QtApp.primaryScreen()
        screen_height = screen.availableGeometry().height()
        
        if screen_height <= 768:  # Small screens (laptop 15")
            self.pipeline_progress_details.setMinimumHeight(200)
            self.pipeline_progress_details.setMaximumHeight(250)
        elif screen_height <= 1080:  # Medium screens
            self.pipeline_progress_details.setMinimumHeight(250)
            self.pipeline_progress_details.setMaximumHeight(300)
        else:  # Large screens
            self.pipeline_progress_details.setMinimumHeight(300)
            self.pipeline_progress_details.setMaximumHeight(400)
        self.pipeline_progress_details.setReadOnly(True)
        self.pipeline_progress_details.setPlainText("Ready to run pipeline...")
        # Improve scroll behavior
        self.pipeline_progress_details.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pipeline_progress_details.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pipeline_progress_details.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        progress_layout.addWidget(self.pipeline_progress_details)

        # Create horizontal layout for configuration and progress frames
        main_h_layout = QHBoxLayout()
        main_h_layout.addWidget(config_frame)
        main_h_layout.addWidget(progress_frame)
        layout.addLayout(main_h_layout)
        
        # Set up scroll area for small screens
        scroll_area.setWidget(tab_widget)
        
        # Check screen size and return appropriate widget
        screen = QtApp.primaryScreen()
        screen_height = screen.availableGeometry().height()
        
        if screen_height <= 768:  # Small screens need scroll
            return scroll_area
        else:
            return tab_widget
    
    def _on_window_changed(self, text: str):
        """Handle window combo box changes to enable/disable lookback field"""
        self.lookback_spin.setEnabled(text == "rolling")

    def setup_ui(self):
        """Setup the ML UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Machine Learning Training Center")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
    # Data Management Tab removed from ML widget (moved to top-level DATA tab)
        
        # Pipeline Tab
        self.pipeline_widget = self.create_pipeline_tab()
        self.tab_widget.addTab(self.pipeline_widget, "🚀 Pipeline")
        
        # Performance Tab
        self.performance_widget = ModelPerformanceWidget()
        self.tab_widget.addTab(self.performance_widget, "📈 Performance")
        
        # Diagnostics Tab
        self.diagnostics_widget = self.create_diagnostics_tab()
        self.tab_widget.addTab(self.diagnostics_widget, "🔧 Diagnostics")
        
        layout.addWidget(self.tab_widget)
        
        # REMOVED: Pipeline configuration and progress moved to dedicated Pipeline tab
    
    def create_diagnostics_tab(self):
        """Create diagnostics tab for system checks"""
        # Keep a minimal diagnostics tab: only title and an empty results area placeholder.
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title (keep)
        title = QLabel("System Diagnostics & Health Checks")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Minimal placeholder description (kept for context)
        desc = QLabel("Diagnostics: basic checks available. For full reports use the DATA → Report tab.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        # Simple read-only results area (empty)
        self.diagnostics_results = QTextEdit()
        self.diagnostics_results.setReadOnly(True)
        self.diagnostics_results.setPlaceholderText("Diagnostics results will appear here (cleared to header only)...")
        self.diagnostics_results.setMinimumHeight(120)
        layout.addWidget(self.diagnostics_results)

        return tab_widget
    
    # REMOVED: Pipeline configuration and progress moved to dedicated Pipeline tab
    
    # REMOVED: start_training and stop_training functions removed as requested

    def start_pipeline_run(self):
        """Start the ML pipeline run with current UI parameters."""
        try:
            # Check for single stock first
            single_stock = self.single_stock_input.text().strip().upper()
            if single_stock:
                # Use single stock if provided
                tickers = [single_stock]
                self.performance_widget.add_log_entry(f"Running pipeline on single stock: {single_stock}")
            else:
                # Collect tickers from Data tab (safe helper)
                tickers = self._get_symbols_from_data_widget()
                # If none provided: use entire available bronze dataset
                if not tickers:
                    # Leave tickers empty so worker loads all available bronze Parquet files
                    self.performance_widget.add_log_entry("No symbols entered; running on all available symbols in data/bronze/daily")
                else:
                    self.performance_widget.add_log_entry(f"Running on {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers)>10 else ''}")

            # Parse model selection (single select, allow comma separated if user typed)
            models_text = self.pipeline_models_combo.currentText().strip()
            selected_models = [m.strip() for m in models_text.split(',') if m.strip()] if models_text else []

            params = {
                "tickers": tickers,
                "holdout": self.holdout_spin.value(),
                "step": self.step_spin.value(),
                "lookback": self.lookback_spin.value(),
                "window": self.window_combo.currentText(),
                "models": selected_models,
                # wire feature toggles
                "use_technical": self.use_technical_indicators.isChecked(),
                "use_volume": self.use_volume_data.isChecked(),
                "use_sentiment": self.use_sentiment_data.isChecked(),
                "use_parallel": self.use_parallel_processing.isChecked(),
                # cache selection
                "selected_cache": self._get_selected_cache_from_data_widget(),
            }

            # UI state
            self.run_pipeline_btn.setEnabled(False)
            self.stop_pipeline_btn.setEnabled(True)
            self.pipeline_status_label.setText("Starting pipeline...")
            self.pipeline_progress_bar.setVisible(True)
            self.pipeline_progress_bar.setValue(0)
            # REMOVED: progress_bar and status_label removed as training buttons removed
            self.performance_widget.training_log.clear()
            
            # Initialize step counter - CORRECTED to actual implementation
            self.pipeline_step_counter = 0
            # Actual steps that execute "completed":
            # Initialize Config (1) + Loading Data (1) + Feature Building per-horizon (3)
            # + Horizon Training completions (3) + Saving Predictions (1) + Exporting Signals (1) = 13 steps TOTAL
            # Keep this in sync with add_pipeline_step calls in PipelineRunWorker.run()
            self.pipeline_total_steps = 13
            self.pipeline_step_label.setText(f"Steps: 0/{self.pipeline_total_steps}")
            self.performance_widget.add_log_entry(f"Pipeline initialized: {self.pipeline_total_steps} total steps")
            
            # Clear pipeline steps in data tab
            self.clear_pipeline_steps()

            # Trigger worker in its own thread via signal
            print(f"Main thread: {QThread.currentThread()}")
            self.pipeline_worker.run_requested.emit(params)
            print("Signal emitted, returning to main thread")
            self.performance_widget.add_log_entry(f"Pipeline params: {params}")
            self.logger.info(f"ML pipeline started with params: {params}")

        except Exception as e:
            self.logger.error(f"Error starting pipeline: {e}")
            QMessageBox.critical(self, "Pipeline Error", str(e))

    def on_pipeline_completed(self, summary: dict):
        """Handle completion of pipeline run."""
        # Reset UI state and ensure step counter shows completion
        self.run_pipeline_btn.setEnabled(True)
        self.stop_pipeline_btn.setEnabled(False)
        self.pipeline_status_label.setText("Pipeline completed successfully")
        
        # Force completion - make sure step counter reaches 100%
        if self.pipeline_step_counter < self.pipeline_total_steps:
            self.pipeline_step_counter = self.pipeline_total_steps
            self.pipeline_step_label.setText(f"Steps: {self.pipeline_step_counter}/{self.pipeline_total_steps}")
            self.pipeline_progress_bar.setValue(100)
        
        self.pipeline_progress_bar.setVisible(False)
        
        # Add completion notification
        self.performance_widget.add_log_entry("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        self.performance_widget.add_log_entry("="*50)
        
        # Render compact metrics to table
        self.performance_widget.update_metrics(summary)
        
        # Check if this was a single stock run and generate special report
        single_stock = self.single_stock_input.text().strip().upper()
        if single_stock:
            # Always try to reconstruct compact payload from persisted preds parquet files.
            # This ensures the One Symbol table is populated even if the worker's in-memory
            # compact payload is incomplete.
            reconstructed = None
            try:
                import pandas as _pd
                from pathlib import Path as _Path
                ticker = single_stock.upper()
                preds_map = {}
                for h in [1, 5, 10]:
                    fp = _Path(f"data/silver/preds/preds_h{h}.parquet")
                    if fp.exists():
                        try:
                            df = _pd.read_parquet(fp)
                            if 'ticker' in df.columns:
                                df_t = df[df['ticker'].str.upper() == ticker]
                            else:
                                df_t = df
                            if not df_t.empty:
                                latest = df_t.sort_values('date').iloc[-1]
                                preds_map[h] = latest
                        except Exception:
                            pass

                if preds_map:
                    price_targets = {}
                    per_horizon = {}
                    signals = []
                    for h in [1, 5, 10]:
                        if h in preds_map:
                            r = preds_map[h]
                            pred_lbl = str(r.get(f'y_h{h}_pred', r.get('y_pred', 'HOLD'))).upper()
                            conf = float(r.get('confidence', 0.0)) if r.get('confidence', None) is not None else 0.0
                            pt = float(r.get('price_target', r.get('adj_close', r.get('close', 0.0))))
                            model_name = r.get('model', None)
                            price_targets[int(h)] = pt
                            per_horizon[int(h)] = {'signal': pred_lbl, 'confidence': conf, 'model': model_name}
                            signals.append((h, pred_lbl, conf))

                    overall = '-'
                    if signals:
                        score_map = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
                        for (_h, lbl, conf) in signals:
                            if lbl == 'UP':
                                score_map['BUY'] += conf
                            elif lbl == 'DOWN':
                                score_map['SELL'] += conf
                            else:
                                score_map['HOLD'] += conf
                        overall = max(score_map.keys(), key=lambda k: score_map[k])

                    price_targets['meta'] = {}
                    for h, info in per_horizon.items():
                        try:
                            src_date = preds_map[h].get('date', None)
                            try:
                                src_date_str = src_date.isoformat() if hasattr(src_date, 'isoformat') else str(src_date)
                            except Exception:
                                src_date_str = str(src_date)
                        except Exception:
                            src_date_str = None
                        price_targets['meta'][int(h)] = {
                            'confidence': info.get('confidence', 0.0),
                            'model': info.get('model', None),
                            'date': src_date_str,
                        }

                    reconstructed = {
                        'symbol': ticker,
                        'overall_signal': overall,
                        'price_targets': price_targets,
                        'per_horizon': {int(k): {'signal': v.get('signal'), 'confidence': v.get('confidence')} for k, v in per_horizon.items()}
                    }
            except Exception:
                reconstructed = None

            # Use reconstructed payload if available; otherwise fall back to worker compact_table
            compact = summary.get('compact_table') if isinstance(summary, dict) else None
            final_compact = reconstructed if reconstructed is not None else compact
            if final_compact:
                try:
                    try:
                        self.performance_widget.add_log_entry(
                            f"Using compact payload for UI (source: {'reconstructed' if reconstructed else 'worker'})"
                        )
                    except Exception:
                        pass
                    # Update the compact table display
                    self._update_one_symbol_table(
                        final_compact.get('symbol', single_stock),
                        final_compact.get('overall_signal', '-'),
                        final_compact.get('price_targets', {}),
                    )
                    try:
                        if getattr(self, 'auto_ai_checkbox', None) and self.auto_ai_checkbox.isChecked():
                            try:
                                self.performance_widget.add_log_entry(f"Auto AI enabled - requesting AI for {single_stock}")
                            except Exception:
                                pass
                            # Build prompt now so we can indicate to the user that the prompt was sent
                            try:
                                prompt_text = self._build_generic_prompt(single_stock, final_compact)
                            except Exception:
                                prompt_text = self._build_generic_prompt(single_stock, {})
                            # Show a small sending indicator in the AI cell with tooltip containing timestamp and prompt preview
                            try:
                                self._set_ai_sending_state(prompt_text, single_stock)
                            except Exception:
                                pass
                            # Start background AI request, pass prompt_text so worker uses the exact prompt we showed
                            self._auto_ask_ai_async(single_stock, final_compact, prompt_text)
                    except Exception:
                        pass
                except Exception:
                    self._generate_single_stock_report(single_stock)
            else:
                self._generate_single_stock_report(single_stock)
        
        # Auto-refresh metrics/preds view based on last run
        try:
            self.performance_widget.load_metrics_csv()
            # If single ticker was requested, filter preds preview to that ticker automatically
            tickers = self._get_symbols_from_data_widget()
            self.performance_widget.load_preds_parquet(filter_tickers=tickers if tickers else None)
        except Exception:
            pass
        self.performance_widget.add_log_entry("Pipeline run complete. Predictions saved under data/silver/preds")

    def on_pipeline_error(self, error: str):
        # Reset UI state
        self.run_pipeline_btn.setEnabled(True)
        self.stop_pipeline_btn.setEnabled(False)
        self.pipeline_status_label.setText("Pipeline failed - check logs")
        self.pipeline_progress_bar.setVisible(False)
        QMessageBox.critical(self, "Pipeline Error", error)
        self.logger.error(f"Pipeline error: {error}")
    
    def _generate_single_stock_report(self, ticker: str):
        """Generate a special trading report for single stock analysis."""
        try:
            import pandas as pd
            from pathlib import Path
            from datetime import datetime
            
            self.performance_widget.add_log_entry(f"Generating special trading report for {ticker}...")
            
            # Load predictions for all horizons
            preds_data = {}
            horizons = [1, 5, 10]  # Default horizons: 1, 5, 10 days
            
            for h in horizons:
                pred_file = Path(f"data/silver/preds/preds_h{h}.parquet")
                if pred_file.exists():
                    df = pd.read_parquet(pred_file)
                    # Filter for our ticker
                    ticker_df = df[df['ticker'].str.upper() == ticker.upper()].copy()
                    if not ticker_df.empty:
                        # Get latest prediction
                        latest = ticker_df.sort_values('date').iloc[-1]
                        preds_data[h] = latest
            
            if not preds_data:
                self.performance_widget.add_log_entry(f"No predictions found for {ticker}")
                return
                
            # Generate trading signals and price targets
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append(f"🎯 TRADING REPORT FOR {ticker}")
            report_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("=" * 60)
            report_lines.append("")
            
            # Current price (from latest available data)
            latest_data = None
            current_price = 0.0  # Initialize with default value
            for h in horizons:
                if h in preds_data:
                    latest_data = preds_data[h]
                    break
                    
            if latest_data is not None:
                current_price = float(latest_data.get('adj_close', latest_data.get('close', 0)))
                report_lines.append(f"💲 Current Price: ${current_price:.2f}")
                report_lines.append("")
            else:
                report_lines.append("⚠️ No current price data available")
                report_lines.append("")
            
            # Trading signals and price targets
            signals = []
            price_targets = {}
            
            for h in [1, 5, 10]:  # Specific horizons requested: 1, 5, 10 days
                if h in preds_data:
                    pred = preds_data[h]
                    signal = str(pred.get('y_pred', 'HOLD')).upper()
                    confidence = float(pred.get('confidence', 0.5))
                    price_target = float(pred.get('price_target', current_price))
                    
                    signals.append((h, signal, confidence))
                    price_targets[h] = price_target
            
            # Overall signal (majority vote with confidence weighting)
            signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            for h, signal, conf in signals:
                # Map predictions to trading signals
                if signal == 'UP':
                    signal_scores['BUY'] += conf
                elif signal == 'DOWN':
                    signal_scores['SELL'] += conf
                else:
                    signal_scores['HOLD'] += conf
            
            # Determine overall signal
            overall_signal = max(signal_scores.keys(), key=lambda k: signal_scores[k])
            overall_confidence = signal_scores[overall_signal] / len(signals) if signals else 0
            
            # Display overall recommendation with enhanced formatting
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
            signal_desc = {"BUY": "BUY (Expected to go UP)", "SELL": "SELL (Expected to go DOWN)", "HOLD": "HOLD (Sideways movement)"}
            
            report_lines.append(f"📊 OVERALL RECOMMENDATION:")
            report_lines.append(f"    {signal_emoji.get(overall_signal, '⚪')} {overall_signal} - {signal_desc.get(overall_signal, 'No clear direction')}")
            report_lines.append(f"    🎯 Confidence Level: {overall_confidence:.1%}")
            
            # Risk assessment
            risk_level = "LOW" if overall_confidence > 0.7 else "MEDIUM" if overall_confidence > 0.5 else "HIGH"
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            report_lines.append(f"    ⚠️ Risk Assessment: {risk_emoji.get(risk_level)} {risk_level} RISK")
            report_lines.append("")
            
            # Detailed predictions by horizon with enhanced formatting
            report_lines.append("📈 DETAILED PRICE TARGETS & SIGNALS:")
            report_lines.append("=" * 50)
            
            for h in [1, 5, 10]:
                if h in price_targets:
                    target = price_targets[h]
                    if latest_data:
                        change_pct = ((target - current_price) / current_price) * 100
                        change_symbol = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                        signal_info = next((s for s in signals if s[0] == h), (h, 'HOLD', 0.5))
                        
                        # Enhanced signal mapping
                        signal_text = signal_info[1]
                        if signal_text == 'UP':
                            signal_text = 'BUY'
                        elif signal_text == 'DOWN':
                            signal_text = 'SELL'
                        
                        # Color-coded confidence levels
                        conf_level = "HIGH" if signal_info[2] > 0.7 else "MED" if signal_info[2] > 0.5 else "LOW"
                        
                        report_lines.append(f"📅 {h:2d}-Day Horizon:")
                        report_lines.append(f"   🎯 Target Price: ${target:7.2f}")
                        report_lines.append(f"   📊 Expected Change: {change_pct:+6.1f}% {change_symbol}")
                        report_lines.append(f"   🔔 Signal: {signal_text} (Confidence: {signal_info[2]:.1%} - {conf_level})")
                        report_lines.append("")
            
            report_lines.append("")
            report_lines.append("=" * 60)
            
            # Save report to file
            report_dir = Path("data/silver/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"{ticker}_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_content = "\n".join(report_lines)
            
            report_file.write_text(report_content, encoding='utf-8')
            
            # Display in UI log
            for line in report_lines:
                self.performance_widget.add_log_entry(line)
            
            self.performance_widget.add_log_entry(f"📄 Full report saved to: {report_file}")
            # Also update the compact One Symbol Prediction table (Symbol, Signal, Day1, Day5, Day10)
            try:
                # Build compact values
                day_values = {h: price_targets.get(h, 0.0) for h in [1, 5, 10]}
                # Map overall_signal to UI-friendly value
                ui_signal = overall_signal
                self._update_one_symbol_table(ticker, ui_signal, day_values)
            except Exception:
                pass
            
            # Show popup with summary
            from PyQt6.QtWidgets import QMessageBox
            summary_msg = f"""🎯 TRADING REPORT FOR {ticker}

💲 Current Price: ${current_price:.2f}

📊 RECOMMENDATION: {overall_signal}
🎯 Confidence: {overall_confidence:.1%}

📈 PRICE TARGETS:
- 1 Day:  ${price_targets.get(1, 0):.2f}
- 5 Days: ${price_targets.get(5, 0):.2f}
- 10 Days: ${price_targets.get(10, 0):.2f}

Full report saved to data/silver/reports/"""
            
            QMessageBox.information(self, f"Trading Report - {ticker}", summary_msg)
            
        except Exception as e:
            self.performance_widget.add_log_entry(f"Error generating single stock report: {e}")
            import traceback
            self.logger.error(f"Single stock report error: {traceback.format_exc()}")

    def _update_one_symbol_table(self, symbol: str, overall_signal: str, price_targets: dict):
        """Update the small One Symbol Prediction table with a single row.

        price_targets should be a dict like {1: price1, 5: price5, 10: price10}
        """
        try:
            from PyQt6.QtGui import QColor

            # Normalize values
            sig = overall_signal if overall_signal else "-"
            p1 = price_targets.get(1, 0.0)
            p5 = price_targets.get(5, 0.0)
            p10 = price_targets.get(10, 0.0)
            # Compute confidence: prefer compact payload per_horizon if present, else check price_targets.meta
            conf_val = 0.0
            if isinstance(price_targets, dict):
                # If worker provided per-horizon meta inside price_targets['meta'], use it
                meta = price_targets.get('meta', {}) if isinstance(price_targets.get('meta', {}), dict) else {}
                confidences = []
                for h in (1, 5, 10):
                    ph = None
                    # first look in meta
                    if h in meta and isinstance(meta[h], dict) and meta[h].get('confidence') is not None:
                        try:
                            ph = float(meta[h].get('confidence', 0.0))
                        except Exception:
                            ph = 0.0
                    # fallback: price_targets might include per_horizon-like structure
                    if ph is None and isinstance(price_targets.get('per_horizon', {}), dict):
                        try:
                            ph = float(price_targets['per_horizon'].get(h, {}).get('confidence', 0.0))
                        except Exception:
                            ph = 0.0
                    if ph is not None:
                        confidences.append(ph)
                if confidences:
                    conf_val = sum(confidences) / len(confidences)

            # Helper to create read-only item
            def mk_item(text: str):
                it = QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                return it

            # Symbol
            self.one_symbol_table.setItem(0, 0, mk_item(symbol))

            # Current price: attempt to read last close from parquet for this ticker
            try:
                cur_price = None
                from pathlib import Path
                p = Path(f"data/bronze/daily/{symbol}.parquet")
                if not p.exists():
                    # try lowercase
                    p = Path(f"data/bronze/daily/{symbol.lower()}.parquet")
                if p.exists():
                    import pandas as _pd
                    dfp = _pd.read_parquet(p)
                    if 'adj_close' in dfp.columns:
                        cur_price = float(dfp['adj_close'].dropna().iloc[-1])
                    elif 'close' in dfp.columns:
                        cur_price = float(dfp['close'].dropna().iloc[-1])
                if cur_price is None:
                    price_item = mk_item("N/A")
                else:
                    price_item = mk_item(f"${cur_price:.2f}")
                    # add tooltip showing the date of the current price (last row)
                    try:
                        last_date = None
                        if 'date' in dfp.columns:
                            last_date = dfp['date'].dropna().iloc[-1]
                            # format
                            try:
                                last_date_str = last_date.isoformat() if hasattr(last_date, 'isoformat') else str(last_date)
                            except Exception:
                                last_date_str = str(last_date)
                            price_item.setToolTip(f"Price as of: {last_date_str}")
                    except Exception:
                        pass
                self.one_symbol_table.setItem(0, 1, price_item)
            except Exception:
                try:
                    self.one_symbol_table.setItem(0, 1, mk_item("N/A"))
                except Exception:
                    pass

            # Signal with color-coding and readable text
            sig_text = str(sig).upper()
            if sig_text in ("BUY", "UP"):
                display_sig = "🟢 BUY"
                color = QColor(200, 255, 200)
            elif sig_text in ("SELL", "DOWN"):
                display_sig = "🔴 SELL"
                color = QColor(255, 200, 200)
            else:
                display_sig = "🟡 HOLD"
                color = QColor(255, 255, 200)

            sig_item = mk_item(display_sig)
            sig_item.setBackground(color)
            # Signal column moved to index 2
            self.one_symbol_table.setItem(0, 2, sig_item)

            # Confidence cell (show percentage)
            try:
                conf_display = f"{float(conf_val):.1%}" if conf_val is not None else "-"
            except Exception:
                conf_display = str(conf_val)
            conf_item = mk_item(conf_display)
            # color confidence gradient: green>0.7, yellow>0.5
            try:
                cv = float(conf_val)
                if cv > 0.7:
                    conf_item.setBackground(QColor(200, 255, 200))
                elif cv > 0.5:
                    conf_item.setBackground(QColor(255, 255, 200))
                else:
                    conf_item.setBackground(QColor(255, 230, 230))
            except Exception:
                pass
            # Confidence now at index 3
            self.one_symbol_table.setItem(0, 3, conf_item)

            # Day1, Day5, Day10 as price strings
            # Add tooltips on day cells showing model and confidence when available
            def day_item(h, val):
                text = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                it = mk_item(text)
                # Build tooltip from per-horizon meta if present
                tip_parts = []
                # Try compact per_horizon structure
                ph = None
                if isinstance(price_targets, dict):
                    meta = price_targets.get('meta', {}) if isinstance(price_targets.get('meta', {}), dict) else {}
                    if h in meta:
                        mm = meta[h]
                        mname = mm.get('model') if isinstance(mm.get('model', None), str) else None
                        mconf = mm.get('confidence', None)
                        if mname:
                            tip_parts.append(f"Model: {mname}")
                        if mconf is not None:
                            try:
                                tip_parts.append(f"Confidence: {float(mconf):.1%}")
                            except Exception:
                                tip_parts.append(f"Confidence: {mconf}")
                        # include date if present
                        mdate = mm.get('date', None)
                        if mdate:
                            tip_parts.append(f"Source date: {mdate}")
                # Also check compact per_horizon key if present
                if isinstance(price_targets.get('per_horizon', {}), dict) and price_targets.get('per_horizon', {}).get(h):
                    ph2 = price_targets['per_horizon'][h]
                    if isinstance(ph2, dict):
                        if 'model' in ph2 and ph2['model']:
                            tip_parts.append(f"Model: {ph2['model']}")
                        if 'confidence' in ph2 and ph2['confidence'] is not None:
                            try:
                                tip_parts.append(f"Confidence: {float(ph2['confidence']):.1%}")
                            except Exception:
                                tip_parts.append(f"Confidence: {ph2['confidence']}")

                if tip_parts:
                    it.setToolTip('\n'.join(tip_parts))
                return it

            # Day1/5/10 shifted right by one due to price column
            self.one_symbol_table.setItem(0, 4, day_item(1, p1))
            self.one_symbol_table.setItem(0, 5, day_item(5, p5))
            self.one_symbol_table.setItem(0, 6, day_item(10, p10))

            # Reserve the right-most cell for AI response. If Auto AI is enabled the response
            # will be populated automatically after the pipeline completes. Otherwise show '-'.
            try:
                ai_item = mk_item("-")
                ai_item.setToolTip("AI response will appear here if Auto AI is enabled")
                # Ensure we set the last column (index 7)
                self.one_symbol_table.setItem(0, 7, ai_item)
            except Exception:
                try:
                    self.one_symbol_table.setItem(0, 7, mk_item("-"))
                except Exception:
                    pass

            # Ensure compact row height
            try:
                self.one_symbol_table.setRowHeight(0, 26)
            except Exception:
                pass

        except Exception:
            # Fail silently to avoid breaking UI flow
            return

    def stop_pipeline_run(self):
        """Stop the running pipeline."""
        if hasattr(self, 'pipeline_worker') and self.pipeline_worker.is_running:
            self.pipeline_worker.stop()
            self.pipeline_status_label.setText("Stopping pipeline...")
            self.performance_widget.add_log_entry("Pipeline stop requested by user")
        else:
            self.pipeline_status_label.setText("No pipeline running to stop")

    # ---- Ask-AI helpers ----
    def _build_generic_prompt(self, symbol: str, payload: dict) -> str:
        """Build the user-specified precise 7-day prediction prompt for the AI.

        The returned prompt is exactly the template requested by the user with the
        ticker symbol inserted in place of [TICKER].
        """
        try:
            t = str(symbol).upper() if symbol else "TICKER"
            prompt = (
                f"You are a  professional financial analyst. Provide a precise 7-day price prediction for {t} by analyzing ALL of the following factors: current financial metrics, recent earnings/guidance, upcoming earnings or corporate events (if any), overall market sentiment (VIX, fear/greed index), sector performance, trading volume patterns, analyst ratings and forecasts, insider trading activity, options flow (including short interest and put/call ratio), technical indicators, macroeconomic events, regulatory or legislative changes, ESG scores, global geopolitical events, patent filings and major product launches, and any relevant news or catalysts. If no significant upcoming events exist, focus on current market dynamics and seasonal effects. Output only the predicted price as a single number in USD."
            )
            return prompt
        except Exception:
            return f"You are a professional financial analyst. Provide a precise 7-day price prediction for {symbol} and output only the predicted price as a single number in USD."

    def _call_ai(self, prompt: str, symbol: str = None) -> str:
        """Delegate AI call to AIService so ML widget uses same Perplexity settings as Watchlist.

        Uses AIService.analyze_stock_simple (sync) to preserve behavior and config-driven model/filters.
        """
        try:
            from core.config_manager import ConfigManager
            from services.ai_service import AIService

            cfg = ConfigManager()
            svc = AIService(cfg)
            # analyze_stock_simple returns raw text or None
            try:
                text = svc.analyze_stock_simple(prompt, symbol or '', timeout=30)
                if text is None:
                    return ""
                return text
            except Exception as e:
                return f"AI call failed: {e}"
        except Exception as e:
            return f"AI call failed: {e}"

    def _on_ask_ai(self, symbol: str, payload: dict):
        """Handler for ask_ai_clicked signal: open dialog with prompt and allow sending to AI."""
        try:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QPushButton

            prompt_text = self._build_generic_prompt(symbol, payload)

            dlg = QDialog(self)
            dlg.setWindowTitle(f"Ask AI — {symbol}")
            v = QVBoxLayout(dlg)
            lbl = QLabel("Edit prompt (optional):")
            v.addWidget(lbl)
            prompt_edit = QTextEdit()
            prompt_edit.setPlainText(prompt_text)
            prompt_edit.setMinimumHeight(160)
            v.addWidget(prompt_edit)

            resp_label = QLabel("AI Response:")
            v.addWidget(resp_label)
            resp_view = QTextEdit()
            resp_view.setReadOnly(True)
            resp_view.setMinimumHeight(140)
            v.addWidget(resp_view)

            h = QHBoxLayout()
            send_btn = QPushButton("Send")
            close_btn = QPushButton("Close")
            h.addWidget(send_btn)
            h.addWidget(close_btn)
            v.addLayout(h)

            def do_send():
                send_btn.setEnabled(False)
                resp_view.setPlainText("(requesting...)")
                QApplication.processEvents()
                res = self._call_ai(prompt_edit.toPlainText())
                resp_view.setPlainText(res)
                send_btn.setEnabled(True)

            send_btn.clicked.connect(do_send)
            close_btn.clicked.connect(dlg.accept)

            dlg.exec()
        except Exception as e:
            try:
                self.performance_widget.add_log_entry(f"Ask-AI dialog failed: {e}")
            except Exception:
                pass

    def _set_ai_response_cell(self, text: str):
        """Safely update the AI response cell in the One Symbol table on the main thread."""
        try:
            def do_update():
                try:
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    # Put truncated text if it's long to keep table compact
                    disp = text if len(text) <= 64 else text[:60] + "..."
                    item.setText(disp)
                    item.setToolTip(text)
                    # Last column index may be 7
                    self.one_symbol_table.setItem(0, 7, item)
                except Exception:
                    pass
            QTimer.singleShot(0, do_update)
        except Exception:
            # Best-effort: try direct set (may fail if called off main thread)
            try:
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                disp = text if len(text) <= 64 else text[:60] + "..."
                item.setText(disp)
                item.setToolTip(text)
                self.one_symbol_table.setItem(0, 7, item)
            except Exception:
                pass

    def _set_ai_sending_state(self, prompt_text: str, symbol: str = ''):
        """Show a small sending indicator in the AI cell and set tooltip with prompt preview and timestamp."""
        try:
            from datetime import datetime

            preview = prompt_text if len(prompt_text) <= 120 else prompt_text[:116] + '...'
            sent_ts = datetime.now().isoformat(sep=' ', timespec='seconds')
            display = "(sending...)"
            item = QTableWidgetItem(display)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            tooltip = f"Prompt sent: {sent_ts}\nSymbol: {symbol}\n\nPrompt preview:\n{preview}"
            item.setToolTip(tooltip)
            # light gray background to indicate pending state
            try:
                from PyQt6.QtGui import QColor
                item.setBackground(QColor(240, 240, 240))
            except Exception:
                pass
            # Update on main thread
            try:
                def do_set():
                    try:
                        self.one_symbol_table.setItem(0, 7, item)
                    except Exception:
                        pass
                QTimer.singleShot(0, do_set)
            except Exception:
                try:
                    self.one_symbol_table.setItem(0, 7, item)
                except Exception:
                    pass
            # Schedule a timeout fallback in 30s: if still showing '(sending...)' replace with 'AI timeout'
            try:
                def _timeout_check():
                    try:
                        cur = self.one_symbol_table.item(0, 7)
                        if cur and cur.text() == '(sending...)':
                            from PyQt6.QtWidgets import QTableWidgetItem
                            it = QTableWidgetItem('AI timeout')
                            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                            it.setToolTip('No AI response within timeout period (30s). Check Perplexity API key and network connection.')
                            try:
                                from PyQt6.QtGui import QColor
                                it.setBackground(QColor(255, 230, 230))
                            except Exception:
                                pass
                            try:
                                self.one_symbol_table.setItem(0, 7, it)
                            except Exception:
                                pass
                    except Exception:
                        pass
                QTimer.singleShot(30000, _timeout_check)
            except Exception:
                pass
        except Exception:
            pass

    def _auto_ask_ai_async(self, symbol: str, compact_payload: dict, prompt_text: str = None):
        """Run _call_ai in a worker thread and update the AI cell when complete.

        If prompt_text is supplied, the worker will use it instead of rebuilding the prompt.
        """
        try:
            # Use a QThread + QObject worker to run the async Perplexity path
            # This mirrors the Watchlist pattern (create an event loop and call the async API)
            from PyQt6.QtCore import QThread, QObject, pyqtSignal

            class _AIWorker(QObject):
                finished = pyqtSignal(str)
                error = pyqtSignal(str)

                def __init__(self, prompt: str, cfg):
                    super().__init__()
                    self.prompt = prompt
                    self.cfg = cfg

                def run(self):
                    try:
                        # Import here to avoid top-level event loop issues
                        import asyncio
                        from services.ai_service import AIService

                        svc = AIService(self.cfg)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # Call the same async Perplexity path used elsewhere
                            text = loop.run_until_complete(svc._call_perplexity_api(self.prompt))
                        finally:
                            try:
                                loop.close()
                            except Exception:
                                pass

                        if text is None:
                            self.error.emit('No response from AI')
                        else:
                            self.finished.emit(text)
                    except Exception as e:
                        try:
                            self.error.emit(str(e))
                        except Exception:
                            pass

            # Build prompt if not provided
            if prompt_text:
                prompt = prompt_text
            else:
                prompt = self._build_generic_prompt(symbol, compact_payload)

            # Show sending state in UI
            try:
                self._set_ai_sending_state(prompt, symbol)
            except Exception:
                pass

            # Acquire config
            try:
                from core.config_manager import ConfigManager
                cfg = ConfigManager()
            except Exception:
                cfg = None

            worker = _AIWorker(prompt, cfg)
            thread = QThread()
            worker.moveToThread(thread)

            # Wire signals
            thread.started.connect(worker.run)

            def _on_finished(txt: str):
                try:
                    self._set_ai_response_cell(txt)
                    try:
                        if hasattr(self, 'performance_widget') and getattr(self, 'performance_widget'):
                            self.performance_widget.add_log_entry(f"Auto AI response received for {symbol}")
                    except Exception:
                        pass
                finally:
                    try:
                        thread.quit()
                    except Exception:
                        pass

            def _on_error(err: str):
                try:
                    self._set_ai_response_cell(f"AI Error: {err}")
                finally:
                    try:
                        thread.quit()
                    except Exception:
                        pass

            worker.finished.connect(_on_finished)
            worker.error.connect(_on_error)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            # Keep reference so callers/tests can wait for completion and prevent
            # QThread destruction while still running (prevents abrupt crash)
            try:
                self._last_ai_thread = thread
                self._last_ai_worker = worker
            except Exception:
                pass

            thread.start()
        except Exception as e:
            try:
                self._set_ai_response_cell(f"AI Error: {e}")
            except Exception:
                pass
    
    def update_progress(self, progress: int):
        """Update training progress"""
        # Update pipeline progress bar with more accurate calculation
        if progress > 0:
            self.pipeline_progress_bar.setVisible(True)
            # Ensure progress doesn't exceed 100% and matches step counter
            if self.pipeline_total_steps > 0:
                calculated_progress = min(100, int((self.pipeline_step_counter / self.pipeline_total_steps) * 100))
                # Use the higher of received progress or calculated progress for smoother updates
                actual_progress = max(progress, calculated_progress)
                self.pipeline_progress_bar.setValue(actual_progress)
            else:
                self.pipeline_progress_bar.setValue(progress)
        else:
            self.pipeline_progress_bar.setVisible(False)
    
    def update_data_progress(self, progress: int):
        """Update data loading progress"""
        if progress > 0:
            self.data_progress_bar.setVisible(True)
            self.data_progress_bar.setValue(progress)
        else:
            self.data_progress_bar.setVisible(False)
    
    def update_status(self, status: str):
        """Update training status"""
        # REMOVED: status_label removed as training buttons removed
        self.performance_widget.add_log_entry(status)
        # Update pipeline status label
        self.pipeline_status_label.setText(status)
    
    def add_pipeline_step(self, step: str, status: str = "running"):
        """Add or update a pipeline step in the progress details"""
        # Update counter for completed steps
        if status == "completed":
            self.pipeline_step_counter += 1
            self.pipeline_step_label.setText(f"Steps: {self.pipeline_step_counter}/{self.pipeline_total_steps}")
            # Update progress bar to match step counter
            if self.pipeline_total_steps > 0:
                progress_pct = int((self.pipeline_step_counter / self.pipeline_total_steps) * 100)
                self.pipeline_progress_bar.setValue(min(progress_pct, 100))
        
        current_text = self.pipeline_progress_details.toPlainText()
        lines = current_text.split('\n') if current_text != "Ready to run pipeline..." else []
        
        # Find if step already exists
        step_found = False
        for i, line in enumerate(lines):
            if line.startswith(f"[{step}]"):
                # Update existing step
                if status == "completed":
                    lines[i] = f"[{step}] ✅ Completed"
                elif status == "running":
                    lines[i] = f"[{step}] 🔄 Running..."
                elif status == "error":
                    lines[i] = f"[{step}] ❌ Error"
                step_found = True
                break
        
        if not step_found:
            # Add new step
            if status == "completed":
                lines.append(f"[{step}] ✅ Completed")
            elif status == "running":
                lines.append(f"[{step}] 🔄 Running...")
            elif status == "error":
                lines.append(f"[{step}] ❌ Error")
        
        # Keep only last 20 lines to avoid overflow
        if len(lines) > 20:
            lines = lines[-20:]
        
        self.pipeline_progress_details.setPlainText('\n'.join(lines))
        # Auto scroll to bottom with immediate UI update
        cursor = self.pipeline_progress_details.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.pipeline_progress_details.setTextCursor(cursor)
        self.pipeline_progress_details.ensureCursorVisible()
        
        # Force immediate UI update
        QApplication.processEvents()
    
    def clear_pipeline_steps(self):
        """Clear all pipeline steps"""
        self.pipeline_progress_details.setPlainText("Ready to run pipeline...")
        self.pipeline_step_counter = 0
        self.pipeline_step_label.setText(f"Steps: 0/{self.pipeline_total_steps}")
    
    def on_training_completed(self, results: dict):
        """Handle training completion"""
        self.performance_widget.update_metrics(results)
        # REMOVED: reset_training_ui() call removed as training buttons removed
        
        QMessageBox.information(
            self,
            "Training Complete",
            f"Model training setup completed!\\n\\n"
            f"Connect to real data sources to begin actual training.\\n"
            f"Training Time: {results['training_time']:.1f}s"
        )
        
        self.logger.info("Model training completed successfully")
    
    def on_training_error(self, error: str):
        """Handle training error"""
        # REMOVED: Training UI reset removed as training buttons removed
        QMessageBox.critical(self, "Training Error", f"Training failed: {error}")
        self.logger.error(f"Training error: {error}")
    
    # REMOVED: reset_training_ui function removed as training buttons removed
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("ML tab activated")
    
    def run_system_check(self):
        """Run comprehensive system diagnostics"""
        try:
            self.run_diagnostics_btn.setEnabled(False)
            self.diagnostics_progress.setVisible(True)
            self.diagnostics_progress.setValue(0)
            self.diagnostics_results.clear()
            self.diagnostics_results.append("🔍 Starting System Diagnostics...\n")
            
            # Run smoke test in background thread
            class DiagnosticsRunner(QObject):
                finished = pyqtSignal(str)
                progress = pyqtSignal(int)
                log = pyqtSignal(str)
                
                def run(self):
                    try:
                        import subprocess
                        import sys
                        from pathlib import Path
                        
                        self.log.emit("Running smoke test...")
                        self.progress.emit(25)
                        
                        # Run smoke_test.py
                        script_path = Path(__file__).parent.parent.parent / "scripts" / "smoke_test.py"
                        if not script_path.exists():
                            self.log.emit(f"❌ Error: smoke_test.py not found at {script_path}")
                            self.finished.emit("FAILED")
                            return
                        
                        # Run the script and capture output
                        result = subprocess.run([sys.executable, str(script_path)], 
                                              capture_output=True, text=True, cwd=script_path.parent.parent)
                        
                        self.progress.emit(75)
                        
                        if result.returncode == 0:
                            self.log.emit("✅ Smoke test completed successfully!")
                            self.log.emit("📊 Results:")
                            # Parse and format output
                            output_lines = result.stdout.strip().split('\n')
                            for line in output_lines[-10:]:  # Last 10 lines typically contain summary
                                if line.strip():
                                    self.log.emit(f"   {line}")
                            self.finished.emit("SUCCESS")
                        else:
                            self.log.emit("❌ Smoke test failed!")
                            self.log.emit("Error output:")
                            for line in result.stderr.strip().split('\n'):
                                if line.strip():
                                    self.log.emit(f"   {line}")
                            self.finished.emit("FAILED")
                            
                    except Exception as e:
                        self.log.emit(f"❌ Exception during diagnostics: {str(e)}")
                        self.finished.emit("ERROR")
            
            # Create and run diagnostics in thread
            self.diagnostics_runner = DiagnosticsRunner()
            self.diagnostics_thread = QThread()
            self.diagnostics_runner.moveToThread(self.diagnostics_thread)
            
            self.diagnostics_runner.finished.connect(self.on_diagnostics_finished)
            self.diagnostics_runner.progress.connect(self.diagnostics_progress.setValue)
            self.diagnostics_runner.log.connect(self.diagnostics_results.append)
            
            self.diagnostics_thread.started.connect(self.diagnostics_runner.run)
            self.diagnostics_thread.start()
            
        except Exception as e:
            self.diagnostics_results.append(f"❌ Failed to start diagnostics: {str(e)}")
            self.run_diagnostics_btn.setEnabled(True)
            self.diagnostics_progress.setVisible(False)
    
    def on_diagnostics_finished(self, status):
        """Handle diagnostics completion"""
        self.diagnostics_progress.setValue(100)
        self.run_diagnostics_btn.setEnabled(True)
        self.diagnostics_thread.quit()
        self.diagnostics_thread.wait()
        
        if status == "SUCCESS":
            self.diagnostics_results.append("\n🎉 All system checks passed!")
        else:
            self.diagnostics_results.append(f"\n⚠️ System check completed with status: {status}")
    
    def closeEvent(self, event):
        """Handle widget close"""
        # REMOVED: Training stop check removed as training buttons removed
        # Stop worker thread
        if hasattr(self, 'training_thread'):
            self.training_thread.quit()
            self.training_thread.wait()
        if hasattr(self, 'pipeline_thread'):
            self.pipeline_thread.quit()
            self.pipeline_thread.wait()
        
        event.accept()