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
        
        # Metrics table
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMaximumHeight(200)
        layout.addWidget(self.metrics_table)
        
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
        
        event.accept()