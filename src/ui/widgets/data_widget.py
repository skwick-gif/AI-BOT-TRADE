from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QGroupBox, QFormLayout,
    QLabel, QTimeEdit, QSpinBox, QHBoxLayout, QPushButton, QProgressBar,
    QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QFont

from pathlib import Path
import sys
import subprocess
import json
from datetime import datetime, time as dtime

from services.data_update_service import DataUpdateService
import pandas as pd
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem


class DataWidget(QWidget):
    """Dedicated widget for the top-level DATA tab.

    Contains two sub-tabs:
    - Daily Update (owns the DataUpdateService and live logs)
    - Report Viewer
    """

    def __init__(self, parent=None, service: DataUpdateService = None):
        super().__init__(parent)

        # remember provided service instance (may be None)
        self._provided_service = service

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Daily Update tab (will host schedule controls and live logs)
        self.daily_update_tab = QWidget()
        self._build_daily_update_tab()

        # Report viewer placeholder
        self.report_viewer_tab = QWidget()
        rlayout = QVBoxLayout(self.report_viewer_tab)
        rlabel = QLabel("Report Viewer: Reports related to data updates will appear here.")
        rlabel.setWordWrap(True)
        rlayout.addWidget(rlabel)

        # Controls for scanning data files
        ctrl_layout = QHBoxLayout()
        self.scan_btn = QPushButton("🔎 Scan Data Files")
        self.scan_btn.clicked.connect(self._on_run_report_scan)
        ctrl_layout.addWidget(self.scan_btn)

        self.scan_stop_btn = QPushButton("⏹️ Stop Scan")
        self.scan_stop_btn.setEnabled(False)
        self.scan_stop_btn.clicked.connect(self._on_stop_report_scan)
        ctrl_layout.addWidget(self.scan_stop_btn)

        ctrl_layout.addStretch()
        rlayout.addLayout(ctrl_layout)

        # Progress and results
        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 100)
        self.report_progress.setValue(0)
        rlayout.addWidget(self.report_progress)

        self.report_log = QTextEdit()
        self.report_log.setReadOnly(True)
        self.report_log.setPlaceholderText("Scan output will appear here...")
        self.report_log.setMinimumHeight(160)
        rlayout.addWidget(self.report_log)

        # Simple results table: ticker, last_date, rows
        self.report_table = QTableWidget()
        self.report_table.setColumnCount(3)
        self.report_table.setHorizontalHeaderLabels(["Ticker", "Last Date", "Rows"])
        self.report_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rlayout.addWidget(self.report_table)

        self.tabs.addTab(self.daily_update_tab, "🗓️ Daily Update")
        self.tabs.addTab(self.report_viewer_tab, "📋 Report Viewer")

        # scanner thread placeholders
        self._report_thread = None
        self._report_worker = None

    def _on_run_report_scan(self):
        """Start background scan of parquet files under data/bronze/daily and update UI."""
        try:
            # prevent double-start
            if getattr(self, '_report_thread', None) and self._report_thread.isRunning():
                self._append_report_log("Scan already running")
                return

            # Create worker
            class ReportScanner(QObject):
                progress = pyqtSignal(int)
                log = pyqtSignal(str)
                finished = pyqtSignal(list)

                def __init__(self, base_dir: Path, limit: int = 0):
                    super().__init__()
                    self.base_dir = Path(base_dir)
                    self.limit = limit
                    self._is_running = True

                def stop(self):
                    self._is_running = False

                def run(self):
                    results = []
                    try:
                        # Determine bronze dir
                        bronze_dir = self.base_dir / 'daily'
                        if not bronze_dir.exists():
                            bronze_dir = self.base_dir
                        files = sorted(bronze_dir.glob('*.parquet'))
                        total = len(files)
                        if total == 0:
                            self.log.emit(f"No parquet files found in {bronze_dir}")
                            self.finished.emit(results)
                            return
                        if self.limit and self.limit > 0:
                            files = files[:self.limit]
                            total = len(files)

                        for idx, fp in enumerate(files, start=1):
                            if not self._is_running:
                                self.log.emit("Scan stopped by user")
                                break
                            ticker = fp.stem
                            try:
                                df = pd.read_parquet(fp)
                                if 'date' in df.columns:
                                    last = pd.to_datetime(df['date']).max()
                                elif 'Date' in df.columns:
                                    last = pd.to_datetime(df['Date']).max()
                                else:
                                    last = None
                                rows = len(df)
                                last_str = last.strftime('%Y-%m-%d') if last is not None else 'N/A'
                                results.append({'ticker': ticker, 'last_date': last_str, 'rows': rows})
                                self.log.emit(f"{ticker}: last={last_str}, rows={rows}")
                            except Exception as e:
                                self.log.emit(f"{ticker}: ERROR reading file: {e}")
                                results.append({'ticker': ticker, 'last_date': 'ERROR', 'rows': 0})

                            pct = int((idx / total) * 100)
                            self.progress.emit(pct)

                        self.finished.emit(results)
                    except Exception as e:
                        self.log.emit(f"Scan failed: {e}")
                        self.finished.emit(results)

            base = Path('data/bronze')
            limit = 0
            self._report_worker = ReportScanner(base, limit=limit)
            self._report_thread = QThread(self)
            self._report_worker.moveToThread(self._report_thread)
            self._report_thread.started.connect(self._report_worker.run)
            self._report_worker.progress.connect(self._on_report_progress)
            self._report_worker.log.connect(self._append_report_log)
            self._report_worker.finished.connect(self._on_report_finished)
            # ensure thread quits after finished
            self._report_worker.finished.connect(self._report_thread.quit)

            self.scan_btn.setEnabled(False)
            self.scan_stop_btn.setEnabled(True)
            self.report_progress.setValue(0)
            self.report_table.setRowCount(0)
            self.report_log.clear()

            self._report_thread.start()
        except Exception as e:
            self._append_report_log(f"Failed to start scan: {e}")

    def _on_stop_report_scan(self):
        try:
            if getattr(self, '_report_worker', None):
                try:
                    self._report_worker.stop()
                except Exception:
                    pass
            self.scan_stop_btn.setEnabled(False)
            self.scan_btn.setEnabled(True)
        except Exception:
            pass

    def _on_report_progress(self, v: int):
        try:
            self.report_progress.setValue(v)
        except Exception:
            pass

    def _append_report_log(self, line: str):
        try:
            ts = datetime.now().strftime('%H:%M:%S')
            self.report_log.append(f"[{ts}] {line}")
        except Exception:
            pass

    def _on_report_finished(self, results: list):
        try:
            # populate table
            self.report_table.setRowCount(len(results))
            for i, r in enumerate(results):
                self.report_table.setItem(i, 0, QTableWidgetItem(r.get('ticker', '')))
                self.report_table.setItem(i, 1, QTableWidgetItem(r.get('last_date', '')))
                self.report_table.setItem(i, 2, QTableWidgetItem(str(r.get('rows', 0))))

            self.scan_btn.setEnabled(True)
            self.scan_stop_btn.setEnabled(False)
            self.report_progress.setValue(100)
            self._append_report_log("Scan completed")
        except Exception as e:
            self._append_report_log(f"Error finishing scan: {e}")

    def _build_daily_update_tab(self):
        tab = self.daily_update_tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Daily Update")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        # Schedule Configuration
        schedule_group = QGroupBox("Schedule Configuration")
        form = QFormLayout(schedule_group)

        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm")
        self.time_edit.setToolTip("Local time to run daily updates (e.g., a few hours after market close)")
        form.addRow("Run time:", self.time_edit)

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

        # Live Logs
        logs_group = QGroupBox("Live Logs")
        logs_layout = QVBoxLayout(logs_group)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logs will appear here...")
        logs_layout.addWidget(self.log_view)

        layout.addWidget(logs_group)

        # Config file path
        self._cfg_path = Path("config/data_update.json")
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)

        # DataUpdateService instance - use provided service or create one
        if getattr(self, '_provided_service', None) is not None:
            self._service = self._provided_service
            # don't start a service owned externally; only connect signals
            try:
                self._service.progress.connect(self._on_progress)
                self._service.log.connect(self._on_log)
                self._service.completed.connect(self._on_completed)
                self._service.error.connect(self._on_error)
                self._service.next_run_changed.connect(self._on_next_run)
            except Exception:
                # If connecting fails, continue without throwing
                pass
            # Load saved schedule or default 01:30 for display only
            saved = self._load_saved_time()
            if saved is not None:
                self.time_edit.setTime(saved)
            else:
                self.time_edit.setTime(dtime(hour=1, minute=30))
        else:
            # Create and manage our own service instance
            self._service = DataUpdateService()
            self._service.progress.connect(self._on_progress)
            self._service.log.connect(self._on_log)
            self._service.completed.connect(self._on_completed)
            self._service.error.connect(self._on_error)
            self._service.next_run_changed.connect(self._on_next_run)

            # Load saved schedule or default 01:30
            saved = self._load_saved_time()
            if saved is not None:
                self.time_edit.setTime(saved)
            else:
                self.time_edit.setTime(dtime(hour=1, minute=30))

            # Apply scheduled time to service and start it
            try:
                self._service.set_scheduled_time(self.time_edit.time().toPyTime())
                self._service.start()
            except Exception:
                # Non-fatal if service fails to start during construction
                self._append_log("Failed to start DataUpdateService at widget init")

    # --- Slots and service handlers ---
    def _on_run_now(self):
        """Run the daily update now"""
        self.progress.setValue(0)
        self.stop_btn.setEnabled(True)
        self.run_now_btn.setEnabled(False)
        limit = int(self.limit_spin.value() or 0)

        # Clear previous logs
        self.log_view.clear()

        # spawn a subprocess runner if adapter exists, otherwise fallback to service
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
                        self.proc.terminate()
                        try:
                            self.proc.wait(timeout=3)
                        except Exception:
                            self.proc.kill()
                except Exception as e:
                    self.failed.emit(f"Stop failed: {e}")

            def run(self):
                try:
                    self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, text=True, bufsize=1)
                    for ln in self.proc.stdout:
                        ln = ln.rstrip('\n')
                        if ln:
                            self.line.emit(ln)
                    self.proc.wait()
                    self.finished.emit()
                except Exception as e:
                    self.failed.emit(str(e))

        python_exec = Path(sys.executable).as_posix() if hasattr(sys, 'executable') else 'python'
        adapter = Path(__file__).parent.parent.parent.parent / 'tools' / 'download_stocks.py'
        if adapter.exists():
            if limit == 0:
                cmd = [python_exec, str(adapter)]
            else:
                cmd = [python_exec, str(adapter), '--limit', str(limit)]
            self._append_log(f"Starting adapter: {cmd}")
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
            self._append_log("Adapter not found; falling back to DataUpdateService.run_now")
            batch_limit = None if limit == 0 else limit
            try:
                self._service.run_now(batch_limit=batch_limit)
            except Exception as e:
                self._append_log(f"Service run_now failed: {e}")
            self._append_log("Triggered update run…")

    def _on_stop(self):
        """Stop the running update"""
        try:
            if getattr(self, "_adapter_runner", None):
                self._adapter_runner.stop()
            if getattr(self, "_adapter_thread", None):
                self._adapter_thread.quit()
                self._adapter_thread.wait(3000)
                self._adapter_thread = None
        except Exception:
            pass
        try:
            self._service.stop()
        except Exception:
            pass
        self.stop_btn.setEnabled(False)
        self.run_now_btn.setEnabled(True)
        self._append_log("Stop requested")

    def on_save(self):
        """Save the schedule configuration"""
        t = self.time_edit.time().toPyTime()
        try:
            self._service.set_scheduled_time(t)
        except Exception:
            pass
        self._append_log(f"Schedule saved: {t.strftime('%H:%M')}")
        try:
            self._cfg_path.write_text(json.dumps({"time": t.strftime('%H:%M')}), encoding='utf-8')
        except Exception:
            pass

    def _on_progress(self, v: int):
        self.progress.setValue(v)

    def _on_log(self, msg: str):
        self._append_log(msg)

    def _on_error(self, msg: str):
        self._append_log(f"ERROR: {msg}")
        self.stop_btn.setEnabled(False)
        self.run_now_btn.setEnabled(True)

    def _on_completed(self, payload: dict):
        n = payload.get("tickers", 0)
        completion_msg = f"Completed update for {n} tickers"
        self._append_log(completion_msg)
        summary = f"📊 Daily Update Report - Completed: {n} tickers at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._append_log(summary)
        self.stop_btn.setEnabled(False)
        self.run_now_btn.setEnabled(True)

    def _on_next_run(self, ts: datetime):
        try:
            self.next_run_label.setText(f"Next run: {ts.strftime('%Y-%m-%d %H:%M')}")
        except Exception:
            pass

    def _on_adapter_finished(self):
        self.run_now_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._adapter_runner = None

    def _append_log(self, line: str):
        ts = datetime.now().strftime('%H:%M:%S')
        try:
            self.log_view.append(f"[{ts}] {line}")
        except Exception:
            pass

    def _load_saved_time(self):
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
