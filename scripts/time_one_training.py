import sys
from pathlib import Path
import time

# Ensure src is on path when running as a script
root = Path(__file__).resolve().parents[1]
src_path = root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from utils.logger import get_logger
from ml.config import TrainingConfig
from ml.dataset import load_bronze, build_pooled_dataset, feature_label_columns
from ml.runner import walk_forward_run


def main():
    log = get_logger("time_one_training")

    # Choose a small subset of tickers known to exist to keep runtime reasonable
    tickers = [
        "A", "AA", "AMD", "AMZN", "MSFT", "NVDA"
    ]

    log.info("Loading bronze parquet data…")
    bronze = load_bronze(Path("data/bronze/daily"), tickers=tickers)
    if not bronze:
        log.error("No bronze data found for selected tickers.")
        return 2
    rows_bronze = sum(len(df) for df in bronze.values())
    log.info(f"Bronze loaded: {len(bronze)} tickers, {rows_bronze} rows")

    cfg = TrainingConfig()
    # Configure to perform only one walk-forward step by using a large step_days
    cfg.holdout_last_days = 30
    cfg.step_days = 9999  # large to produce a single step
    cfg.lookback_days = 500
    cfg.window_mode = "expanding"  # or "rolling"

    log.info("Building pooled dataset with features and labels…")
    pooled = build_pooled_dataset(bronze, cfg)
    if pooled.empty:
        log.error("Pooled dataset is empty after feature/label building.")
        return 3
    x_cols, y_cols = feature_label_columns(pooled)
    log.info(f"Pooled shape: {pooled.shape}; features={len(x_cols)} labels={len(y_cols)}")

    # Time a single training step (walk_forward_run configured to do one step)
    log.info("Starting timed training (single walk-forward step)…")
    start = time.perf_counter()
    results, preds, model_scores, confusions = walk_forward_run(
        pooled, cfg, selected_models=["RandomForest"],
    )
    elapsed = time.perf_counter() - start

    if not results:
        log.warning("Walk-forward produced no results; possibly insufficient date range with current config.")
    else:
        r = results[-1]
        log.info(
            f"Step as_of={r.as_of.date()} horizon={r.horizon} metric={r.metric_name}={r.metric_value:.4f} "
            f"n_train={r.n_train} n_test={r.n_test}"
        )

    # Report elapsed time
    print({
        "tickers": len(bronze),
        "rows_bronze": rows_bronze,
        "pooled_rows": len(pooled),
        "features": len(x_cols),
        "labels": len(y_cols),
        "elapsed_seconds": round(elapsed, 3),
    })


if __name__ == "__main__":
    raise SystemExit(main())
