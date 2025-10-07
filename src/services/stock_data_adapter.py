"""Adapter that wraps the user's attached stocks script (tools/stocks_attached.py)
and exposes a small API the DataUpdateWorker can call. The adapter keeps
control of the orchestration so UI callbacks (progress/log) can be used.
"""
from __future__ import annotations

import runpy
import sys
import os
from pathlib import Path
from typing import Callable, List, Optional

# Import the attached module from tools. Ensure tools is on sys.path
TOOLS_DIR = Path(__file__).parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    import stocks_attached as attached
except Exception:
    attached = None


def list_tickers_from_disk(source_dir: Path) -> List[str]:
    # Prefer folders under source_dir
    out = []
    try:
        for p in source_dir.iterdir():
            if p.is_dir() and not p.name.startswith('.'):
                out.append(p.name.upper())
    except Exception:
        pass
    # fallback to attached.all_tickers if available
    if not out and attached is not None:
        try:
            out = [t.upper() for t in getattr(attached, 'all_tickers', [])]
        except Exception:
            out = []
    return sorted(set(out))


def update_price(ticker: str, cfg) -> None:
    """Call the attached update_price_data(ticker, start_date, folder)."""
    if attached is None:
        raise RuntimeError("stocks_attached module not available")
    # attached expects folder path string
    attached.update_price_data(ticker, cfg.start_date, str(cfg.source_dir))


def update_fundamentals(ticker: str, cfg) -> None:
    if attached is None:
        raise RuntimeError("stocks_attached module not available")
    attached.scrape_all_advanced(ticker, str(cfg.source_dir))


def convert_to_parquet(cfg, log: Callable[[str], None] | None = None) -> None:
    """Use the existing converter script (scripts/convert_stock_data_to_parquet.py)
    to convert CSVs under source_dir to parquet outputs. This mirrors the
    previous worker behavior.
    """
    # scripts folder lives at repository root: src/../scripts
    try:
        script_path = Path(__file__).parent.parent.parent / "scripts" / "convert_stock_data_to_parquet.py"
        if not script_path.exists():
            if log:
                log(f"converter error: converter script not found at {script_path}")
            return
        mod_globs = runpy.run_path(str(script_path))
        convert_all = mod_globs.get('convert_all')
        if not convert_all:
            if log:
                log("converter error: convert_all not found in converter script")
            return
        # call convert_all with same signature used in the project
        # Ensure we pass Path objects (the converter expects Path-like objects)
        def _to_path(v):
            return Path(v) if v is not None else None

        convert_all(
            _to_path(cfg.source_dir),
            _to_path(cfg.parquet_out),
            _to_path(cfg.fund_json_out),
            _to_path(cfg.fund_parquet_out),
            _to_path(cfg.fund_parquet_dir),
            None,
        )
    except Exception as exc:
        if log:
            import traceback
            log(f"converter error: {exc}\n{traceback.format_exc()}")


def process_batch(cfg, batch_limit: Optional[int], progress_cb: Callable[[int], None], log_cb: Callable[[str], None], stop_flag: Callable[[], bool], tickers: Optional[List[str]] = None):
    """Orchestrate updating a batch of tickers using the attached functions.
    progress_cb: receives 0..100
    log_cb: receives strings to log
    stop_flag: callable that returns True when execution should stop
    """
    # Build list of tickers to process. If caller provided explicit tickers, use them.
    if tickers is None:
        tickers = list_tickers_from_disk(Path(cfg.source_dir))
        if not tickers and attached is not None:
            tickers = [t.upper() for t in getattr(attached, 'all_tickers', [])]

    if batch_limit is not None:
        tickers = tickers[:batch_limit]
    total = max(1, len(tickers))
    log_cb(f"Starting attached daily update for {len(tickers)} tickers…")

    prices_updated = 0
    fund_updated = 0
    for i, t in enumerate(tickers, start=1):
        if stop_flag():
            log_cb("Update stopped by user")
            break
        try:
            log_cb(f"[{i}/{total}] Updating prices: {t}")
            update_price(t, cfg)
            prices_updated += 1
        except Exception as e:
            log_cb(f"{t}: price update failed: {e}")
        # determine if we should update fundamentals by checking file mtime
        try:
            adv = Path(cfg.source_dir) / t / f"{t}_advanced.json"
            stale = True
            if adv.exists():
                mtime = adv.stat().st_mtime
                import time
                stale = (time.time() - mtime) > (14 * 24 * 3600)
            if stale:
                try:
                    log_cb(f"[{i}/{total}] Updating fundamentals: {t}")
                    update_fundamentals(t, cfg)
                    fund_updated += 1
                except Exception as e:
                    log_cb(f"{t}: fundamentals update failed: {e}")
        except Exception:
            pass
        # emit progress: 0..80 for downloads
        try:
            pct = int(i * 80 / total)
            progress_cb(pct)
        except Exception:
            pass
    # conversion
    log_cb("Converting to Parquet…")
    convert_to_parquet(cfg, log_cb)
    progress_cb(100)

    return {"tickers": len(tickers), "prices_updated": prices_updated, "fundamentals_updated": fund_updated}
