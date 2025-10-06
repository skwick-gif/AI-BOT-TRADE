"""
Data Update Service
Runs a daily background job to update local stock_data (prices + fundamentals)
and then converts to Parquet under data/bronze/daily using the existing converter.

Design:
- DataUpdateWorker runs in a QThread, emits progress/log/completed signals.
- DataUpdateService schedules a QTimer to trigger the worker at a configured time.
- Safe to run manually via run_now(); does not block UI.

Note: This is a lightweight port that updates existing tickers found in stock_data/
or data/bronze/daily/. It uses yfinance for prices and Ticker.info for fundamentals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Iterable, List, Optional, Set

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal


@dataclass
class UpdateConfig:
	source_dir: Path = Path("stock_data")
	parquet_out: Path = Path("data/bronze/daily")
	fund_json_out: Path = Path("data/bronze/fundamentals")
	fund_parquet_out: Path = Path("data/bronze/fundamentals.parquet")
	fund_parquet_dir: Path = Path("data/bronze/fundamentals_by_symbol")
	start_date: str = "2020-01-01"
	batch_limit: Optional[int] = None  # for manual tests


class DataUpdateWorker(QObject):
	"""Background worker that downloads new data to stock_data and converts to Parquet."""

	progress = pyqtSignal(int)  # 0..100
	log = pyqtSignal(str)
	completed = pyqtSignal(dict)
	error = pyqtSignal(str)

	def __init__(self, cfg: UpdateConfig):
		super().__init__()
		self.cfg = cfg
		self._stop = False

	def stop(self):
		self._stop = True

	# --- helpers ---
	def _list_tickers(self) -> List[str]:
		src = self.cfg.source_dir
		tickers: Set[str] = set()
		if src.exists():
			for p in src.iterdir():
				if not p.is_dir():
					continue
				sym = p.name.strip()
				if not sym or sym.startswith('.'):
					continue
				# Exclude warrants/units symbols
				if self._is_excluded(sym):
					continue
				# Prefer folders that contain a *_price.csv
				price_csv = p / f"{sym}_price.csv"
				if price_csv.exists():
					tickers.add(sym.upper())
				else:
					# fallback: accept folder name as ticker
					tickers.add(sym.upper())
		# If empty, try from existing parquet files
		if not tickers and self.cfg.parquet_out.exists():
			for fp in self.cfg.parquet_out.glob("*.parquet"):
				sym = fp.stem.upper()
				if self._is_excluded(sym):
					continue
				tickers.add(sym)
		return sorted(tickers)

	def _is_excluded(self, sym: str) -> bool:
		"""Return True if symbol should be excluded (any hyphen followed by a letter, e.g., classes, warrants, units)."""
		try:
			import re
			return bool(re.search(r"-[A-Za-z]", sym))
		except Exception:
			# Fallback: conservative
			return '-' in sym

	def _update_price(self, ticker: str):
		import pandas as pd
		import yfinance as yf

		folder = self.cfg.source_dir / ticker
		folder.mkdir(parents=True, exist_ok=True)
		csv_path = folder / f"{ticker}_price.csv"
		last_date: Optional[datetime] = None
		existing: Optional[pd.DataFrame] = None
		if csv_path.exists():
			try:
				existing = pd.read_csv(csv_path, parse_dates=['Date'])
				if not existing.empty:
					last_date = existing['Date'].max()
			except Exception as e:
				self.log.emit(f"{ticker}: warn reading existing price csv: {e} -> will rewrite")
				existing = None

		start = self.cfg.start_date
		if last_date is not None and isinstance(last_date, pd.Timestamp):
			next_day = (last_date + pd.Timedelta(days=1)).date()
			if next_day <= datetime.today().date():
				start = next_day.isoformat()
			else:
				self.log.emit(f"{ticker}: prices up to date")
				return

		try:
			df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
			if df is None or df.empty:
				self.log.emit(f"{ticker}: no new price data")
				return
			df.index = pd.to_datetime(df.index)
			df.index.name = "Date"
			if existing is not None and not existing.empty:
				df = pd.concat([existing.set_index('Date'), df])
				df = df[~df.index.duplicated(keep='last')]
			df.to_csv(csv_path)
			self.log.emit(f"{ticker}: saved {len(df)} rows of price data")
		except Exception as e:
			self.log.emit(f"{ticker}: error downloading prices: {e}")

	def _update_fundamentals(self, ticker: str):
		import json
		import yfinance as yf

		folder = self.cfg.source_dir / ticker
		folder.mkdir(parents=True, exist_ok=True)
		out = folder / f"{ticker}_advanced.json"
		try:
			stock = yf.Ticker(ticker)
			info = stock.info or {}
			# keep only serializable fields (avoid sets/bytes)
			clean = {}
			for k, v in (info.items() if isinstance(info, dict) else []):
				try:
					json.dumps(v)
					clean[k] = v
				except Exception:
					clean[k] = str(v)
			if clean:
				out.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding='utf-8')
				self.log.emit(f"{ticker}: fundamentals updated ({len(clean)} fields)")
			else:
				self.log.emit(f"{ticker}: fundamentals not available")
		except Exception as e:
			self.log.emit(f"{ticker}: error fundamentals: {e}")

	def _convert_to_parquet(self):
		"""Run the existing converter script to write Parquet outputs."""
		try:
			# Call converter as a module to avoid subprocess overhead
			from scripts.convert_stock_data_to_parquet import convert_all
			convert_all(
				self.cfg.source_dir,
				self.cfg.parquet_out,
				self.cfg.fund_json_out,
				self.cfg.fund_parquet_out,
				self.cfg.fund_parquet_dir,
				None,
			)
		except Exception as e:
			self.log.emit(f"converter error: {e}")

	def run(self):
		"""Main entry to perform update: iterate tickers, update prices + fundamentals, then convert."""
		try:
			tickers = self._list_tickers()
			total = len(tickers)
			if total == 0:
				self.error.emit("No tickers found under stock_data/ or data/bronze/daily")
				return
			if self.cfg.batch_limit:
				tickers = tickers[: self.cfg.batch_limit]
				total = len(tickers)
			self.log.emit(f"Starting daily update for {total} tickers…")
			updated_prices = 0
			updated_fund = 0
			for i, t in enumerate(tickers, 1):
				if self._stop:
					self.log.emit("Update stopped by user")
					break
				before_price = datetime.now()
				self._update_price(t)
				updated_prices += 1
				# keep fundamentals light; not every day may change; update if file missing or stale (older than 14 days)
				adv = (self.cfg.source_dir / t / f"{t}_advanced.json")
				stale = True
				try:
					if adv.exists():
						mtime = datetime.fromtimestamp(adv.stat().st_mtime)
						stale = (datetime.now() - mtime) > timedelta(days=14)
				except Exception:
					stale = True
				if stale:
					self._update_fundamentals(t)
					updated_fund += 1
				self.progress.emit(int(i * 80 / max(1, total)))  # first 80% for download
			# Convert to parquet
			self.log.emit("Converting to Parquet…")
			self._convert_to_parquet()
			self.progress.emit(100)
			# Persist a simple run log (CSV)
			try:
				import pandas as pd
				from pathlib import Path
				outdir = Path("data/bronze/update_runs")
				outdir.mkdir(parents=True, exist_ok=True)
				fp = outdir / "log.csv"
				row = pd.DataFrame([{
					"timestamp": pd.Timestamp.now(),
					"tickers_total": total,
					"prices_updated": updated_prices,
					"fundamentals_updated": updated_fund,
				}])
				if fp.exists():
					old = pd.read_csv(fp)
					row = pd.concat([old, row], ignore_index=True)
				row.to_csv(fp, index=False)
			except Exception:
				pass
			self.completed.emit({"tickers": total, "prices_updated": updated_prices, "fundamentals_updated": updated_fund})
		except Exception as e:
			self.error.emit(str(e))


class DataUpdateService(QObject):
	"""Schedules and orchestrates daily data updates via a worker thread."""

	# Bubble worker signals
	progress = pyqtSignal(int)
	log = pyqtSignal(str)
	completed = pyqtSignal(dict)
	error = pyqtSignal(str)
	next_run_changed = pyqtSignal(datetime)

	def __init__(self, cfg: Optional[UpdateConfig] = None):
		super().__init__()
		self.cfg = cfg or UpdateConfig()
		self._thread: Optional[QThread] = None
		self._worker: Optional[DataUpdateWorker] = None
		self._timer = QTimer(self)
		self._timer.setSingleShot(True)
		self._timer.timeout.connect(self._on_timer)
		self._scheduled_time: dtime = dtime(hour=1, minute=30)  # default local 01:30

	# --- scheduling ---
	def set_scheduled_time(self, t: dtime):
		self._scheduled_time = t
		self._arm_timer()

	def _arm_timer(self):
		now = datetime.now()
		target = datetime.combine(now.date(), self._scheduled_time)
		if target <= now:
			target = target + timedelta(days=1)
		msec = int((target - now).total_seconds() * 1000)
		self._timer.start(max(1000, msec))
		self.next_run_changed.emit(target)

	def start(self):
		self._arm_timer()

	def stop(self):
		self._timer.stop()
		if self._worker:
			self._worker.stop()

	def _on_timer(self):
		self.run_now()
		# re-arm for next day
		self._arm_timer()

	# --- execution ---
	def run_now(self, batch_limit: Optional[int] = None):
		if self._thread is not None:
			# already running
			return
		# prepare worker
		cfg = self.cfg
		if batch_limit is not None:
			cfg = UpdateConfig(
				source_dir=self.cfg.source_dir,
				parquet_out=self.cfg.parquet_out,
				fund_json_out=self.cfg.fund_json_out,
				fund_parquet_out=self.cfg.fund_parquet_out,
				start_date=self.cfg.start_date,
				batch_limit=batch_limit,
			)
		self._thread = QThread()
		self._worker = DataUpdateWorker(cfg)
		self._worker.moveToThread(self._thread)
		# wire signals
		self._thread.started.connect(self._worker.run)
		self._worker.progress.connect(self.progress.emit)
		self._worker.log.connect(self.log.emit)
		self._worker.completed.connect(self._on_completed)
		self._worker.error.connect(self._on_error)
		self._thread.start()

	def _cleanup_thread(self):
		if self._worker:
			self._worker.stop()
		if self._thread:
			self._thread.quit()
			self._thread.wait()
		self._thread = None
		self._worker = None

	def _on_completed(self, payload: dict):
		self.completed.emit(payload)
		self._cleanup_thread()

	def _on_error(self, msg: str):
		self.error.emit(msg)
		self._cleanup_thread()

