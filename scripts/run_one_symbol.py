from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `src` package imports succeed when this script is run directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.config import TrainingConfig
from src.ml.dataset import load_bronze, build_symbol_dataset, feature_label_columns
from src.ml.runner import _make_pipelines


def run_one_symbol(ticker: str, horizons: List[int] | None = None):
    ticker = ticker.upper()
    print(f"[ONE] Running single-symbol pipeline for {ticker}")
    bronze = load_bronze(Path("data/bronze/daily"), tickers=[ticker])
    if ticker not in bronze:
        print(f"[ONE] No parquet found for {ticker} in data/bronze/daily")
        return 1

    cfg = TrainingConfig()
    if horizons is None:
        horizons = cfg.horizons

    df = bronze[ticker]
    df_feat = build_symbol_dataset(df, cfg)
    if df_feat.empty:
        print(f"[ONE] Feature building produced empty frame for {ticker}")
        return 1

    # Align: drop rows without features
    x_cols, y_cols = feature_label_columns(df_feat)
    if not x_cols:
        print("[ONE] No feature columns inferred; nothing to train")
        return 1

    # Use the last available row as the prediction input
    pred_row = df_feat.dropna(subset=x_cols).iloc[[-1]]
    last_close = float(pred_row['adj_close'].iloc[0] if 'adj_close' in pred_row.columns else pred_row['close'].iloc[0])

    # Separate numeric / categorical for pipeline creation
    cat_cols = [c for c in x_cols if str(df_feat[c].dtype) == 'object']
    num_cols = [c for c in x_cols if c not in cat_cols and c != 'date']

    results = []

    for h in horizons:
        y_col = f"y_h{h}"
        if y_col not in df_feat.columns:
            print(f"[ONE] Warning: label {y_col} not present, skipping")
            continue

        train_df = df_feat.dropna(subset=[y_col] + num_cols)
        if train_df.empty or len(train_df) < 10:
            print(f"[ONE] Not enough training rows for horizon {h} (found {len(train_df)}), skipping")
            continue

        # Build pipelines (reuse runner helper)
        class_pipes = _make_pipelines('classification', num_cols, cat_cols)
        reg_pipes = _make_pipelines('regression', num_cols, cat_cols)

        # Choose a default model if present
        cls_name, cls_pipe = next(iter(class_pipes.items()))
        reg_name, reg_pipe = next(iter(reg_pipes.items()))

        # Prepare training frames: classification uses y_h{h}, regression uses fut_ret_{h}
        y_col_cls = y_col
        y_col_reg = f"fut_ret_{h}"
        # Drop rows missing features or targets
        train_required = num_cols + cat_cols + [y_col_cls, y_col_reg]
        train_df_clean = train_df.dropna(subset=train_required)
        if train_df_clean.empty:
            print(f"[ONE] After dropping NaNs, no training rows for h={h}")
            continue

        X_train = train_df_clean[num_cols + cat_cols]
        y_train_cls = train_df_clean[y_col_cls].astype(str)
        y_train_reg = train_df_clean[y_col_reg].astype(float)

        # Fit
        try:
            cls_pipe.fit(X_train, y_train_cls)
        except Exception as e:
            print(f"[ONE] Classification fit error for h={h}: {e}")
            continue
        try:
            reg_pipe.fit(X_train, y_train_reg)
        except Exception as e:
            print(f"[ONE] Regression fit error for h={h}: {e}")
            continue

        # Predict on last row
        X_pred = pred_row[num_cols + cat_cols]
        y_pred_cls = cls_pipe.predict(X_pred)[0]
        y_prob = None
        try:
            if hasattr(cls_pipe, 'predict_proba'):
                proba = cls_pipe.predict_proba(X_pred)
                # map predicted class to index
                classes = cls_pipe.named_steps[list(cls_pipe.named_steps.keys())[-1]].classes_
                idx = list(classes).index(y_pred_cls) if y_pred_cls in classes else 0
                y_prob = float(proba[0, idx])
        except Exception:
            y_prob = None

        y_pred_reg = float(reg_pipe.predict(X_pred)[0])

        # Compute price targets
        price_from_reg = last_close * (1.0 + y_pred_reg)
        # For classification, create a conservative price target scaled by confidence
        if y_prob is None:
            y_prob = 0.6
        if y_pred_cls == 'UP':
            price_from_cls = last_close * (1.0 + 0.01 * max(0.5, y_prob))
        elif y_pred_cls == 'DOWN':
            price_from_cls = last_close * (1.0 - 0.01 * max(0.5, y_prob))
        else:
            price_from_cls = last_close

        results.append({
            'horizon': h,
            'signal': y_pred_cls,
            'signal_confidence': round(float(y_prob), 4),
            'pred_return': float(y_pred_reg),
            'price_from_reg': round(float(price_from_reg), 4),
            'price_from_signal': round(float(price_from_cls), 4),
            'model_cls': cls_name,
            'model_reg': reg_name,
        })

    # Print summary
    if not results:
        print(f"[ONE] No horizons produced results for {ticker}")
        return 1

    print(f"\nSingle-symbol report for {ticker} (last_close={last_close}):")
    for r in results:
        print(f" Horizon {r['horizon']}d -> SIGNAL={r['signal']} (conf={r['signal_confidence']}) | pred_return={r['pred_return']:.4f} | price_reg={r['price_from_reg']:.2f} | price_signal={r['price_from_signal']:.2f} | models: cls={r['model_cls']} reg={r['model_reg']}")

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: run_one_symbol.py TICKER")
        sys.exit(2)
    ticker = sys.argv[1]
    sys.exit(run_one_symbol(ticker))
