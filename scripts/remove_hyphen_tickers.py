#!/usr/bin/env python3
"""
Remove any tickers that contain a hyphen followed by a letter (e.g., -W, -WS, -U, -A, etc.)
from stock_data and delete corresponding outputs under data/bronze (daily parquet, fundamentals json,
and fundamentals_by_symbol parquet). Prints a summary of how many were removed.
"""
from __future__ import annotations
import re
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[1]
SD = BASE / 'stock_data'
DAILY = BASE / 'data' / 'bronze' / 'daily'
FUND_JSON_DIR = BASE / 'data' / 'bronze' / 'fundamentals'
FUND_PQ_DIR = BASE / 'data' / 'bronze' / 'fundamentals_by_symbol'

pattern = re.compile(r'-[A-Za-z]')

removed = []
if SD.exists():
    for p in SD.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if pattern.search(name):
            # remove stock_data folder
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
            # remove outputs
            for fp in [DAILY / f'{name}.parquet', FUND_JSON_DIR / f'{name}.json', FUND_PQ_DIR / f'{name}.parquet']:
                try:
                    if fp.exists():
                        fp.unlink()
                except Exception:
                    pass
            removed.append(name)

print(f"removed={len(removed)}")
if removed:
    print("sample:", ", ".join(removed[:20]))
