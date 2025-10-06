from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure relative paths are based on repo root when run from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure we can import project modules (ml.*, etc.)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
BRONZE_DAILY = REPO_ROOT / "data" / "bronze" / "daily"
SILVER_PREDS = REPO_ROOT / "data" / "silver" / "preds"
SILVER = REPO_ROOT / "data" / "silver"


def make_sample_bronze(ticker: str = "TEST", days: int = 260) -> Path:
    BRONZE_DAILY.mkdir(parents=True, exist_ok=True)
    rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="B")
    # Simple random walk around 100
    np.random.seed(42)
    rets = np.random.normal(loc=0.0008, scale=0.02, size=len(rng))
    prices = 100 * (1 + pd.Series(rets)).cumprod().values
    df = pd.DataFrame({
        "ticker": ticker,
        "date": rng,
        "open": prices * (1 - 0.002),
        "high": prices * (1 + 0.005),
        "low": prices * (1 - 0.01),
        "close": prices,
        "adj_close": prices,
        "volume": (np.random.randint(100_000, 2_000_000, size=len(rng))).astype(int),
        "source": "smoke",
    })
    out_fp = BRONZE_DAILY / f"{ticker}.parquet"
    df.to_parquet(out_fp, index=False)
    return out_fp


def run_pipeline_and_save():
    from ml.config import TrainingConfig
    from ml.dataset import load_bronze, build_pooled_dataset
    from ml.runner import walk_forward_run

    cfg = TrainingConfig(
        holdout_last_days=30,
        horizons=[1, 5],
        step_days=5,
        lookback_days=120,
        window_mode="expanding",
    )

    bronze = load_bronze(BRONZE_DAILY)
    assert bronze, "No bronze data loaded for smoke test"
    pooled = build_pooled_dataset(bronze, cfg)
    assert not pooled.empty, "Empty pooled dataset in smoke test"

    results, preds_by_h, model_scores_df, confusions_df = walk_forward_run(
        pooled, cfg, selected_models=["LogisticRegression", "RandomForest"], progress_cb=None
    )

    # Save predictions similar to UI worker
    SILVER_PREDS.mkdir(parents=True, exist_ok=True)
    saved = 0
    for h, df in preds_by_h.items():
        if df is None or df.empty:
            continue
        fp = SILVER_PREDS / f"preds_h{h}.parquet"
        # Ensure date is string or timestamp for parquet friendliness
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            except Exception:
                pass
        df.to_parquet(fp, index=False)
        saved += len(df)

    # Append metrics
    SILVER.mkdir(parents=True, exist_ok=True)
    mfp = SILVER / "metrics.csv"
    rows = []
    for r in results:
        rows.append({
            "as_of": getattr(r, "as_of", None),
            "horizon": getattr(r, "horizon", None),
            "metric": getattr(r, "metric_value", None),
            "metric_name": getattr(r, "metric_name", None),
            "n_train": getattr(r, "n_train", None),
            "n_test": getattr(r, "n_test", None),
            "tickers": "TEST",
            "window": cfg.window_mode,
            "timestamp": pd.Timestamp.now(),
        })
    mdf = pd.DataFrame(rows)
    if mfp.exists():
        try:
            old = pd.read_csv(mfp)
            mdf = pd.concat([old, mdf], ignore_index=True)
        except Exception:
            pass
    mdf.to_csv(mfp, index=False)

    # Basic assertions for smoke test
    assert saved > 0, "No predictions saved in smoke test"
    assert any(SILVER_PREDS.glob("preds_h*.parquet")), "Preds parquet files missing"
    assert mfp.exists() and mfp.stat().st_size > 0, "metrics.csv missing or empty"

    # Print summary
    print("[SMOKE] rows saved to preds:", saved)
    print("[SMOKE] metrics rows appended:", len(rows))


if __name__ == "__main__":
    fp = make_sample_bronze()
    print(f"[SMOKE] Wrote sample bronze: {fp}")
    run_pipeline_and_save()
    print("[SMOKE] OK: pipeline ran and artifacts saved.")
