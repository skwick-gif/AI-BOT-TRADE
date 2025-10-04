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

            self.status_updated.emit("Initializing pipeline config...")
            self.progress_updated.emit(5)

            cfg = TrainingConfig()
            cfg.holdout_last_days = holdout
            cfg.step_days = step
            cfg.lookback_days = lookback
            cfg.window_mode = window  # type: ignore

            self.status_updated.emit("Loading bronze data (Parquet)...")
            self.progress_updated.emit(15)
            bronze = load_bronze("data/bronze/daily", tickers=tickers if tickers else None)
            if not bronze:
                raise RuntimeError("No bronze Parquet files found for selected tickers")
            # Report loaded tickers and row counts
            try:
                n_tickers = len(bronze)
                n_rows_bronze = sum(len(df) for df in bronze.values())
                self.status_updated.emit(f"Loaded bronze: {n_tickers} tickers, {n_rows_bronze} rows")
            except Exception:
                pass

            self.status_updated.emit("Building features and labels...")
            self.progress_updated.emit(35)
            pooled = build_pooled_dataset(bronze, cfg)
            if pooled is None or pooled.empty:
                raise RuntimeError("Pooled dataset is empty after feature/label building")
            # Report pooled shape and coverage
            try:
                n_rows = len(pooled)
                dates = sorted(pd.to_datetime(pooled["date"]).dropna().unique())
                if dates:
                    self.status_updated.emit(f"Pooled dataset: {n_rows} rows, {pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()}")
                else:
                    self.status_updated.emit(f"Pooled dataset: {n_rows} rows")
            except Exception:
                pass

            self.status_updated.emit("Running walk-forward evaluation...")
            self.progress_updated.emit(60)

            # Provide a progress callback to surface step-level updates to the UI
            def _wf_progress(frac: float, msg: str):
                # Map 60-85% range to walk-forward internal progress
                p = 60 + int(25 * max(0.0, min(1.0, frac)))
                self.progress_updated.emit(p)
                if msg:
                    self.status_updated.emit(msg)

            # forward selected models if provided and run
            selected = set(models) if models else None
            results, preds, model_scores, confusions = walk_forward_run(
                pooled, cfg, selected_models=selected, progress_cb=_wf_progress
            )

            self.status_updated.emit("Saving predictions and summarizing metrics...")
            self.progress_updated.emit(85)
            rows = self._save_predictions(preds)

            # summarize metrics
            # Compute unique as_of steps for clarity
            try:
                as_of_steps = len({r.as_of for r in results}) if results else 0
            except Exception:
                as_of_steps = 0
            summary = {
                "steps": len(results),
                "as_of_steps": as_of_steps,
                "rows_pred": rows,
                "window": cfg.window_mode,
            }
            if results:
                # average metric per horizon and overall
                import pandas as pd
                rf = pd.DataFrame([{
                    "as_of": r.as_of,
                    "h": r.horizon,
                    "metric": r.metric_value,
                } for r in results])
                summary["avg_metric_overall"] = float(rf["metric"].mean())
                for h, grp in rf.groupby("h"):
                    summary[f"avg_metric_h{int(h)}"] = float(grp["metric"].mean())

                # persist metrics CSV (append)
                from datetime import datetime
                from pathlib import Path
                meta = {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "tickers": ",".join(tickers) if tickers else "",
                    "holdout": cfg.holdout_last_days,
                    "step": cfg.step_days,
                    "lookback": cfg.lookback_days,
                    "window": cfg.window_mode,
                    "models": ",".join(models) if models else "",
                }
                metrics_dir = Path("data/silver")
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_csv = metrics_dir / "metrics.csv"
                mrows = [{
                    "as_of": r.as_of,
                    "horizon": r.horizon,
                    "metric_name": r.metric_name,
                    "metric": r.metric_value,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    **meta,
                } for r in results]
                mdf = pd.DataFrame(mrows)
                header = not metrics_csv.exists()
                mdf.to_csv(metrics_csv, mode="a", header=header, index=False)

                # per-model scores
                if model_scores is not None and not model_scores.empty:
                    ms_csv = metrics_dir / "model_scores.csv"
                    ms = model_scores.copy()
                    for k, v in meta.items():
                        ms[k] = v
                    ms_header = not ms_csv.exists()
                    ms.to_csv(ms_csv, mode="a", header=ms_header, index=False)

                # confusion counts
                if confusions is not None and not confusions.empty:
                    cf_csv = metrics_dir / "confusions.csv"
                    cf = confusions.copy()
                    for k, v in meta.items():
                        cf[k] = v
                    cf_header = not cf_csv.exists()
                    cf.to_csv(cf_csv, mode="a", header=cf_header, index=False)

            self.progress_updated.emit(100)
            self.status_updated.emit("Pipeline run complete.")
            self.completed.emit(summary)
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

        # Model selector
        pipe_layout.addWidget(QLabel("Models:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        # Provide a default list; allow typing comma-separated custom list
        self.model_combo.addItems([
            "LogisticRegression,RandomForest",
            "LogisticRegression",
            "RandomForest",
            "Ridge"  # only used if regression target type is selected later
        ])
        self.model_combo.setCurrentText("LogisticRegression,RandomForest")
        pipe_layout.addWidget(self.model_combo)

        pipe_layout.addStretch()

        self.run_pipeline_btn = QPushButton("🏃 Run ML Pipeline")
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

            # Parse model selection
            models_text = self.model_combo.currentText().strip()
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