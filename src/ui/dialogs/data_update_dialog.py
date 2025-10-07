"""
Data Update Dialog
Provides UI to configure the scheduled daily update and to run it on-demand
without blocking the main UI. Shows progress and live logs.
"""

from __future__ import annotations

from datetime import datetime, time as dtime
from pathlib import Path
import sys
import json
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
	QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QTextEdit,
	QTimeEdit, QGroupBox, QFormLayout, QSpinBox
)

from services.data_update_service import DataUpdateService, UpdateConfig


class DataUpdateDialog(QDialog):
	def __init__(self, parent=None, service: DataUpdateService | None = None):
		super().__init__(parent)
		self.setWindowTitle("Daily Update")
		self.resize(760, 520)

		self._cfg_path = Path("config/data_update.json")
		self._cfg_path.parent.mkdir(parents=True, exist_ok=True)

		self._service = service or DataUpdateService()
		self._service.progress.connect(self._on_progress)
		self._service.log.connect(self._on_log)
		self._service.completed.connect(self._on_completed)
		self._service.error.connect(self._on_error)
		self._service.next_run_changed.connect(self._on_next_run)

		self._build_ui()
		# load saved schedule or default 01:30
		saved = self._load_saved_time()
		if saved is not None:
			self.time_edit.setTime(saved)
		else:
			self.time_edit.setTime(dtime(hour=1, minute=30))
		self._service.set_scheduled_time(self.time_edit.time().toPyTime())
		self._service.start()

	def _build_ui(self):
		layout = QVBoxLayout(self)

		# Controls
		ctrl_box = QGroupBox("Schedule")
		form = QFormLayout(ctrl_box)
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

		hl = QHBoxLayout()
		self.start_btn = QPushButton("Run Now")
		self.start_btn.clicked.connect(self._on_run_now)
		hl.addWidget(self.start_btn)

		self.stop_btn = QPushButton("Stop")
		self.stop_btn.clicked.connect(self._on_stop)
		self.stop_btn.setEnabled(False)
		hl.addWidget(self.stop_btn)

		self.save_btn = QPushButton("Save Schedule")
		self.save_btn.clicked.connect(self.on_save)
		hl.addWidget(self.save_btn)

		layout.addWidget(ctrl_box)
		layout.addLayout(hl)

		# Status and progress
		self.next_run_label = QLabel("Next run: --")
		layout.addWidget(self.next_run_label)

		self.progress = QProgressBar()
		self.progress.setRange(0, 100)
		self.progress.setValue(0)
		layout.addWidget(self.progress)

		# Log view
		self.log_view = QTextEdit()
		self.log_view.setReadOnly(True)
		self.log_view.setPlaceholderText("Logs will appear here…")
		layout.addWidget(self.log_view, 1)

		# Close button
		btns = QHBoxLayout()
		self.close_btn = QPushButton("Close")
		self.close_btn.clicked.connect(self.accept)
		btns.addStretch(1)
		btns.addWidget(self.close_btn)
		layout.addLayout(btns)

	# --- slots ---
	def _on_run_now(self):
		# Try to run the external small-run adapter in a background thread so UI stays responsive.
		self.progress.setValue(0)
		self.stop_btn.setEnabled(True)
		limit = int(self.limit_spin.value() or 0)
		# spawn a QThread to run the subprocess
		from PyQt6.QtCore import QThread, pyqtSignal, QObject
		import subprocess
		import shlex

		class SubprocRunner(QObject):
			finished = pyqtSignal()
			line = pyqtSignal(str)
			failed = pyqtSignal(str)

			def __init__(self, cmd):
				super().__init__()
				self.cmd = cmd

			def run(self):
				try:
					# Start subprocess and stream output
					proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, text=True)
					for ln in proc.stdout:
						self.line.emit(ln.rstrip('\n'))
					proc.wait()
					self.finished.emit()
				except Exception as e:
					self.failed.emit(str(e))

		# Build command to run the adapter
		python_exec = Path(sys.executable).as_posix() if hasattr(sys, 'executable') else 'python'
		adapter = Path(__file__).parent.parent.parent / 'tools' / 'run_stocks_small.py'
		if adapter.exists():
			cmd = [python_exec, str(adapter), '--limit', str(limit or 5)]
			self._append_log(f"Starting adapter: {cmd}")
			# create thread and runner
			runner = SubprocRunner(cmd)
			thread = QThread(self)
			runner.moveToThread(thread)
			thread.started.connect(runner.run)
			runner.line.connect(self._append_log)
			runner.failed.connect(lambda msg: self._append_log(f"Adapter error: {msg}"))
			runner.finished.connect(lambda: (self._append_log("Adapter finished."), thread.quit()))
			thread.start()
		else:
			# fallback to built-in service run_now
			self._append_log("Adapter not found; falling back to DataUpdateService.run_now")
			self._service.run_now(batch_limit=limit or None)
			self._append_log("Triggered update run…")

	def _on_stop(self):
		self._service.stop()
		self.stop_btn.setEnabled(False)
		self._append_log("Stop requested")

	def on_save(self):
		t = self.time_edit.time().toPyTime()
		self._service.set_scheduled_time(t)
		self._append_log(f"Schedule saved: {t.strftime('%H:%M')}")
		try:
			self._cfg_path.write_text(json.dumps({"time": t.strftime('%H:%M')}), encoding='utf-8')
		except Exception:
			pass

	# --- service signal handlers ---
	def _on_progress(self, v: int):
		self.progress.setValue(v)

	def _on_log(self, msg: str):
		self._append_log(msg)

	def _on_error(self, msg: str):
		self._append_log(f"ERROR: {msg}")
		self.stop_btn.setEnabled(False)

	def _on_completed(self, payload: dict):
		n = payload.get("tickers")
		self._append_log(f"Completed update for {n} tickers")
		self.stop_btn.setEnabled(False)

	def _on_next_run(self, ts: datetime):
		self.next_run_label.setText(f"Next run: {ts.strftime('%Y-%m-%d %H:%M')}")

	def _append_log(self, line: str):
		ts = datetime.now().strftime('%H:%M:%S')
		self.log_view.append(f"[{ts}] {line}")

	# --- persistence helpers ---
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

