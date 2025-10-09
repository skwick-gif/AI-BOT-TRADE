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
    
    def start_training(self, model_config: dict):
        """Start model training"""
        try:
            self.is_training = True
            self.status_updated.emit("Initializing training...")
            self.progress_updated.emit(0)
            
            # Simulate training process
            import time
            
            steps = [
                "Initializing...",
                "Loading configuration...", 
                "Preparing training environment...",
                "Ready for real data connection...",
                "Training process ready...",
                "Waiting for data source...",
                "Training setup complete..."
            ]
            
            for i, step in enumerate(steps):
                if not self.is_training:
                    break
                    
                self.status_updated.emit(step)
                progress = int((i + 1) / len(steps) * 100)
                self.progress_updated.emit(progress)
                
                # Simulate work
                time.sleep(1.0)
            
            # Training completed successfully
            if self.is_training:
                # Training completed - return basic results structure
                results = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'validation_loss': 0.0,
                    'training_time': 0.0
                }
                
                self.training_completed.emit(results)
                self.status_updated.emit("Training completed - Connect to data source for real training!")
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
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
                
                # Use cache file for first horizon, then reuse cached data for others
                use_cache = cache_file if horizon == 1 else None
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
                    predictions = preds.get(str(horizon)) if preds else None

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
                saved = self._save_predictions({k: v for k, v in preds.items()}) if preds else 0
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
                if preds and best_result:
                    bh = str(best_result.get("horizon"))
                    dfp = preds.get(bh)
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
            self.completed.emit({
                "best_result": best_result or {},
                "saved_predictions": int(saved),
            })
            if best_result:
                self.status_updated.emit(f"Pipeline completed. Best horizon={best_result['horizon']} Quality={best_result['quality']:.3f} Scan matches={len(best_result['scan_results'])}")
            else:
                self.status_updated.emit("Pipeline completed, but no best result identified")
            self.progress_updated.emit(100)  # Pipeline fully complete

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

        # Final Report Section
        report_group = QGroupBox("Final Report")
        report_layout = QVBoxLayout(report_group)

        self.final_report_text = QTextEdit()
        self.final_report_text.setReadOnly(True)
        self.final_report_text.setPlaceholderText("Final update report will appear here after completion...")
        self.final_report_text.setMaximumHeight(150)
        report_layout.addWidget(self.final_report_text)

        layout.addWidget(report_group)

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

        # Clear previous final report
        self.final_report_text.clear()

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

        # Update final report
        self.final_report_text.setPlainText(f"""📊 Daily Update Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Update completed successfully!
📈 Tickers processed: {n}
🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📁 Data location: data/bronze/daily/
🔄 Next scheduled run: {self.next_run_label.text().replace('Next run: ', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")

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


class MLWidget(QWidget):
    """Main ML widget with tabs for different ML functions"""
    
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
        
        # Now that data_management_widget exists, update pipeline worker
        self.pipeline_worker.data_widget = self.data_management_widget
        self.pipeline_worker.ml_widget = self
        
        self.logger.info("ML widget initialized")
    
    def create_pipeline_tab(self):
        """Create the pipeline tab widget"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)  # Reduced spacing

        # Titles row - align configuration and progress titles
        titles_layout = QHBoxLayout()
        
        config_title = QLabel("Pipeline Configuration")
        config_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        titles_layout.addWidget(config_title)
        
        titles_layout.addStretch()
        
        progress_title = QLabel("Pipeline Progress")
        progress_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        titles_layout.addWidget(progress_title)
        
        layout.addLayout(titles_layout)

        # Pipeline Configuration
        config_frame = QFrame()
        config_frame.setFrameStyle(QFrame.Shape.Box)
        
        config_layout = QVBoxLayout(config_frame)
        config_layout.setContentsMargins(10, 5, 10, 10)  # Reduced top margin
        
        # Use form layout so labels are directly adjacent to controls
        from PyQt6.QtWidgets import QFormLayout
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        # Controls
        self.holdout_spin = QSpinBox()
        self.holdout_spin.setRange(5, 250)
        self.holdout_spin.setValue(30)
        self.holdout_spin.setFixedWidth(80)
        self.holdout_spin.setToolTip("Number of days to use for model performance testing.\nMore days = more accurate testing, but longer training time")
        form.addRow(QLabel("Holdout days:"), self.holdout_spin)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 60)
        self.step_spin.setValue(5)
        self.step_spin.setFixedWidth(70)
        self.step_spin.setToolTip("Time interval between each training point.\n1 = daily training, 5 = training every 5 days.\nMore frequent = more training points, but longer runtime")
        form.addRow(QLabel("Step days:"), self.step_spin)

        self.window_combo = QComboBox()
        self.window_combo.addItems(["expanding", "rolling"])
        self.window_combo.setCurrentText("expanding")
        self.window_combo.setFixedWidth(110)
        self.window_combo.setToolTip("Training window expansion method:\n• expanding = all data up to current point\n• rolling = fixed window (requires lookback setting)")
        form.addRow(QLabel("Window:"), self.window_combo)

        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(50, 2000)
        self.lookback_spin.setValue(500)
        self.lookback_spin.setEnabled(False)
        self.lookback_spin.setFixedWidth(90)
        self.lookback_spin.setToolTip("Number of days to use for training when window is rolling.\nOnly active when rolling window is selected")
        form.addRow(QLabel("Lookback:"), self.lookback_spin)

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
        self.pipeline_step_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
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
        self.pipeline_progress_details.setMinimumHeight(300)  # Increased height
        self.pipeline_progress_details.setMaximumHeight(400)  # Set max height for better responsiveness
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
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Data Management Tab
        self.data_management_widget = DataManagementWidget()
        self.tab_widget.addTab(self.data_management_widget, "💾 Data")
        
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
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("System Diagnostics & Health Checks")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Run basic system checks to verify core components are working.\n\nFor comprehensive data reports, use the DATA tab → Report sub-tab.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.run_diagnostics_btn = QPushButton("🔍 Run System Check")
        self.run_diagnostics_btn.clicked.connect(self.run_system_check)
        self.run_diagnostics_btn.setFixedWidth(200)
        buttons_layout.addWidget(self.run_diagnostics_btn)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Progress
        self.diagnostics_progress = QProgressBar()
        self.diagnostics_progress.setVisible(False)
        layout.addWidget(self.diagnostics_progress)
        
        # Results area
        results_group = QGroupBox("Check Results")
        results_layout = QVBoxLayout(results_group)
        
        self.diagnostics_results = QTextEdit()
        self.diagnostics_results.setReadOnly(True)
        self.diagnostics_results.setPlaceholderText("System check results will appear here...")
        self.diagnostics_results.setMinimumHeight(400)
        results_layout.addWidget(self.diagnostics_results)
        
        layout.addWidget(results_group)
        
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
                # Collect tickers from Data tab
                symbols_text = self.data_management_widget.symbols_input.toPlainText().strip()
                tickers = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]
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
                "selected_cache": self.data_management_widget.training_cache_combo.currentData(),
            }

            # UI state
            self.run_pipeline_btn.setEnabled(False)
            self.stop_pipeline_btn.setEnabled(True)
            self.pipeline_status_label.setText("Starting pipeline...")
            self.pipeline_progress_bar.setVisible(True)
            self.pipeline_progress_bar.setValue(0)
            # REMOVED: progress_bar and status_label removed as training buttons removed
            self.performance_widget.training_log.clear()
            
            # Initialize step counter
            self.pipeline_step_counter = 0
            # Calculate total steps more accurately based on actual pipeline flow:
            # 1 (init config) + 1 (loading data) + 3 horizons * (1 setup + 1 feature building + 3 training loops) + 1 (save) + 1 (export) = 15 steps  
            self.pipeline_total_steps = 2 + 3 * (1 + 1 + 3) + 2  # = 19 steps
            self.pipeline_step_label.setText(f"Steps: 0/{self.pipeline_total_steps}")
            
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
        # Reset UI state
        self.run_pipeline_btn.setEnabled(True)
        self.stop_pipeline_btn.setEnabled(False)
        self.pipeline_status_label.setText("Pipeline completed successfully")
        self.pipeline_progress_bar.setVisible(False)
        # Render compact metrics to table
        self.performance_widget.update_metrics(summary)
        
        # Check if this was a single stock run and generate special report
        single_stock = self.single_stock_input.text().strip().upper()
        if single_stock:
            self._generate_single_stock_report(single_stock)
        
        # Auto-refresh metrics/preds view based on last run
        try:
            self.performance_widget.load_metrics_csv()
            # If single ticker was requested, filter preds preview to that ticker automatically
            symbols_text = self.data_management_widget.symbols_input.toPlainText().strip()
            tickers = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]
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

    def stop_pipeline_run(self):
        """Stop the running pipeline."""
        if hasattr(self, 'pipeline_worker') and self.pipeline_worker.is_running:
            self.pipeline_worker.stop()
            self.pipeline_status_label.setText("Stopping pipeline...")
            self.performance_widget.add_log_entry("Pipeline stop requested by user")
        else:
            self.pipeline_status_label.setText("No pipeline running to stop")
    
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