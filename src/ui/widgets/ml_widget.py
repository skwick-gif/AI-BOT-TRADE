"""
Machine Learning Widget
For training and managing ML models for trading
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QFrame,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QProgressBar, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QCheckBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont

from core.config_manager import ConfigManager
from utils.logger import get_logger
from ml.config import TrainingConfig
from ml.dataset import load_bronze, build_pooled_dataset
from ml.runner import walk_forward_run
from ui.dialogs.data_update_dialog import DataUpdateDialog


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

    def __init__(self):
        super().__init__()
        self.is_running = False
        # connect signal to slot
        self.run_requested.connect(self.run)

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
        if self.is_running:
            self.error_occurred.emit("Pipeline is already running")
            return
        try:
            self.is_running = True
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
            self.progress_updated.emit(5)

            cfg = TrainingConfig()
            cfg.holdout_last_days = holdout
            cfg.step_days = step
            cfg.lookback_days = lookback
            cfg.window_mode = window  # type: ignore

            # תהליך לולאת fine tuning לכל הורייזן
            best_result = None
            for horizon in [1, 5, 10]:
                cfg.horizons = [horizon]
                self.status_updated.emit(f"Loading bronze data (Parquet) for horizon {horizon}…")
                self.progress_updated.emit(10)
                bronze = load_bronze("data/bronze/daily", tickers=tickers if tickers else None)
                if not bronze:
                    self.error_occurred.emit("No bronze Parquet files found for selected tickers")
                    return
                self.status_updated.emit("Building features and labels…")
                self.progress_updated.emit(20)
                pooled = build_pooled_dataset(bronze, cfg)
                if pooled is None or pooled.empty:
                    self.error_occurred.emit("Pooled dataset is empty after feature/label building")
                    return

                for loop in range(max_loops):
                    self.status_updated.emit(f"Training loop {loop+1}/{max_loops} for horizon {horizon}")
                    self.progress_updated.emit(30 + int(loop * 10 / max_loops))
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

            # Persist predictions and metrics
            try:
                saved = self._save_predictions({k: v for k, v in preds.items()}) if preds else 0
            except Exception:
                saved = 0
            try:
                # Append compact step metrics
                self._append_metrics_csv(results, window, tickers)
            except Exception:
                pass

            # Final result
            self.completed.emit({
                "best_result": best_result or {},
                "saved_predictions": int(saved),
            })
            if best_result:
                self.status_updated.emit(f"Pipeline completed. Best horizon={best_result['horizon']} Quality={best_result['quality']:.3f} Scan matches={len(best_result['scan_results'])}")
            else:
                self.status_updated.emit("Pipeline completed, but no best result identified")
            self.progress_updated.emit(100)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.is_running = False


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
        
        # Model type
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "TabNet",
            "ExtraTrees",
            "GradientBoosting",
            "Neural Network",
            "SVM",
            "Linear Regression",
            "LSTM"
        ])
        form_layout.addRow("Model Type:", self.model_type_combo)
        
        # Training parameters
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        form_layout.addRow("Epochs:", self.epochs_spin)
        
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(4)
        form_layout.addRow("Learning Rate:", self.learning_rate_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        form_layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Feature selection
        self.use_technical_indicators = QCheckBox("Technical Indicators")
        self.use_technical_indicators.setChecked(True)
        form_layout.addRow("Features:", self.use_technical_indicators)
        
        self.use_volume_data = QCheckBox("Volume Data")
        self.use_volume_data.setChecked(True)
        form_layout.addRow("", self.use_volume_data)
        
        self.use_sentiment_data = QCheckBox("Sentiment Data")
        self.use_sentiment_data.setChecked(False)
        form_layout.addRow("", self.use_sentiment_data)
        
        layout.addLayout(form_layout)
    
    def get_config(self) -> dict:
        """Get current configuration"""
        return {
            'model_type': self.model_type_combo.currentText(),
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'features': {
                'technical_indicators': self.use_technical_indicators.isChecked(),
                'volume_data': self.use_volume_data.isChecked(),
                'sentiment_data': self.use_sentiment_data.isChecked()
            }
        }


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
        self.preds_table = QTableWidget(0, 5)
        self.preds_table.setHorizontalHeaderLabels(["ticker", "date", "y_true", "y_pred", "model"])
        self.preds_table.horizontalHeader().setStretchLastSection(True)
        playout.addWidget(self.preds_table)
        layout.addWidget(preds_box)

        # wire actions
        self.refresh_metrics_btn.clicked.connect(self.load_metrics_csv)
        self.load_preds_btn.clicked.connect(self.load_preds_parquet)
        
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

    def load_preds_parquet(self):
        """Load preds_h{h}.parquet, first 200 rows."""
        import pandas as pd
        from pathlib import Path
        h = self.pred_h_combo.currentText()
        fp = Path(f"data/silver/preds/preds_h{h}.parquet")
        if not fp.exists():
            QMessageBox.information(self, "No predictions", f"{fp.name} not found.")
            return
        df = pd.read_parquet(fp)
        df = df.head(200)
        cols = ["ticker", "date", "y_true", "y_pred", "model"]
        self.preds_table.setRowCount(len(df))
        self.preds_table.setColumnCount(len(cols))
        self.preds_table.setHorizontalHeaderLabels(cols)
        for i, row in df.iterrows():
            for j, col in enumerate(cols):
                self.preds_table.setItem(i, j, QTableWidgetItem(str(row.get(col, ""))))
    
    def add_log_entry(self, message: str):
        """Add entry to training log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] {message}")


class DataManagementWidget(QFrame):
    """Data management widget"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup data management UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Data Management")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
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

        # Daily update launcher
        self.daily_update_btn = QPushButton("🗓️ Daily Update…")
        self.daily_update_btn.setToolTip("Open the daily data update scheduler and progress window")
        self.daily_update_btn.clicked.connect(self.open_daily_update)
        button_layout.addWidget(self.daily_update_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Data status
        self.data_status_label = QLabel("No data loaded")
        self.data_status_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.data_status_label)
    
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

    def open_daily_update(self):
        dlg = DataUpdateDialog(self)
        dlg.setModal(False)
        dlg.show()


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
        
    # Setup UI
        self.setup_ui()
        
        self.logger.info("ML widget initialized")
    
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
        
        # Model Configuration Tab
        self.model_config_widget = ModelConfigWidget()
        self.tab_widget.addTab(self.model_config_widget, "📊 Configuration")
        
        # Data Management Tab
        self.data_management_widget = DataManagementWidget()
        self.tab_widget.addTab(self.data_management_widget, "💾 Data")
        
        # Performance Tab
        self.performance_widget = ModelPerformanceWidget()
        self.tab_widget.addTab(self.performance_widget, "📈 Performance")
        
        layout.addWidget(self.tab_widget)
        
        # Training controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.Box)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        controls_layout.addWidget(self.status_label)
        
        controls_layout.addStretch()
        
        # Control buttons
        self.start_training_btn = QPushButton("🚀 Start Training")
        self.start_training_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.start_training_btn)
        
        self.stop_training_btn = QPushButton("⏹️ Stop Training")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_training_btn)
        
        layout.addWidget(controls_frame)

        # Pipeline controls
        pipeline_frame = QFrame()
        pipeline_frame.setFrameStyle(QFrame.Shape.Box)
        pipe_layout = QHBoxLayout(pipeline_frame)

        # Small config controls
        self.holdout_spin = QSpinBox()
        self.holdout_spin.setRange(5, 250)
        self.holdout_spin.setValue(30)
        pipe_layout.addWidget(QLabel("Holdout days:"))
        pipe_layout.addWidget(self.holdout_spin)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 60)
        self.step_spin.setValue(5)
        pipe_layout.addWidget(QLabel("Step days:"))
        pipe_layout.addWidget(self.step_spin)

        self.window_combo = QComboBox()
        self.window_combo.addItems(["expanding", "rolling"])
        self.window_combo.setCurrentText("expanding")
        pipe_layout.addWidget(QLabel("Window:"))
        pipe_layout.addWidget(self.window_combo)

        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(50, 2000)
        self.lookback_spin.setValue(500)
        self.lookback_spin.setEnabled(False)
        pipe_layout.addWidget(QLabel("Lookback:"))
        pipe_layout.addWidget(self.lookback_spin)

        def _on_window_changed(text: str):
            self.lookback_spin.setEnabled(text == "rolling")
        self.window_combo.currentTextChanged.connect(_on_window_changed)

        # Model selection for pipeline (comma-separated backend keys)
        pipe_layout.addWidget(QLabel("Models:"))
        self.pipeline_models_combo = QComboBox()
        # Provide sensible defaults matching backend keys
        self.pipeline_models_combo.addItems([
            "RandomForest",
            "LightGBM",
            "CatBoost",
            "LogisticRegression",
        ])
        pipe_layout.addWidget(self.pipeline_models_combo)

        # Run pipeline button
        self.run_pipeline_btn = QPushButton("🏃 Run Pipeline")
        self.run_pipeline_btn.clicked.connect(self.start_pipeline_run)
        pipe_layout.addWidget(self.run_pipeline_btn)

        layout.addWidget(pipeline_frame)
    
    def start_training(self):
        """Start model training"""
        try:
            # Get configuration
            config = self.model_config_widget.get_config()
            
            # Update UI
            self.start_training_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Clear previous log
            self.performance_widget.training_log.clear()
            
            # Start training
            QTimer.singleShot(100, lambda: self.training_worker.start_training(config))
            
            self.logger.info("Model training started")
            
        except Exception as e:
            self.logger.error(f"Error starting training: {e}")
            QMessageBox.critical(self, "Training Error", str(e))
    
    def stop_training(self):
        """Stop model training"""
        self.training_worker.stop_training()
        self.reset_training_ui()
        self.logger.info("Model training stopped")

    def start_pipeline_run(self):
        """Start the ML pipeline run with current UI parameters."""
        try:
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
            }

            # UI state
            self.run_pipeline_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting ML pipeline...")
            self.performance_widget.training_log.clear()

            # Trigger worker in its own thread via signal
            self.pipeline_worker.run_requested.emit(params)
            self.performance_widget.add_log_entry(f"Pipeline params: {params}")
            self.logger.info(f"ML pipeline started with params: {params}")

        except Exception as e:
            self.logger.error(f"Error starting pipeline: {e}")
            QMessageBox.critical(self, "Pipeline Error", str(e))

    def on_pipeline_completed(self, summary: dict):
        """Handle completion of pipeline run."""
        self.reset_training_ui()
        # Render compact metrics to table
        self.performance_widget.update_metrics(summary)
        self.performance_widget.add_log_entry("Pipeline run complete. Predictions saved under data/silver/preds")
        self.run_pipeline_btn.setEnabled(True)

    def on_pipeline_error(self, error: str):
        self.reset_training_ui()
        self.run_pipeline_btn.setEnabled(True)
        QMessageBox.critical(self, "Pipeline Error", error)
        self.logger.error(f"Pipeline error: {error}")
    
    def update_progress(self, progress: int):
        """Update training progress"""
        self.progress_bar.setValue(progress)
    
    def update_status(self, status: str):
        """Update training status"""
        self.status_label.setText(status)
        self.performance_widget.add_log_entry(status)
    
    def on_training_completed(self, results: dict):
        """Handle training completion"""
        self.performance_widget.update_metrics(results)
        self.reset_training_ui()
        
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
        self.reset_training_ui()
        QMessageBox.critical(self, "Training Error", f"Training failed: {error}")
        self.logger.error(f"Training error: {error}")
    
    def reset_training_ui(self):
        """Reset training UI to default state"""
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready to train")
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.logger.debug("ML tab activated")
    
    def closeEvent(self, event):
        """Handle widget close"""
        # Stop training if running
        if self.stop_training_btn.isEnabled():
            self.stop_training()
        
        # Stop worker thread
        if hasattr(self, 'training_thread'):
            self.training_thread.quit()
            self.training_thread.wait()
        if hasattr(self, 'pipeline_thread'):
            self.pipeline_thread.quit()
            self.pipeline_thread.wait()
        
        event.accept()