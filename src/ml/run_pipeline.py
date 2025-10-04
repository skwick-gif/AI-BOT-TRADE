from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd

from .config import TrainingConfig
from .dataset import load_bronze, build_pooled_dataset
from .runner import walk_forward_run
from src.utils.logger import get_logger


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Run ML pipeline on bronze Parquet")
    ap.add_argument("--tickers", help="Comma-separated tickers to include", default="")
    ap.add_argument("--holdout", type=int, default=None, help="Holdout last N trading days")
    ap.add_argument("--step", type=int, default=None, help="Step size in trading days")
    ap.add_argument("--lookback", type=int, default=None, help="Lookback window in trading days for rolling mode")
    ap.add_argument("--window", choices=["expanding", "rolling"], default=None, help="Window mode")
    ap.add_argument("--models", default="", help="Comma-separated model names: LogisticRegression,RandomForest or Ridge")
    args = ap.parse_args(argv)

    log = get_logger("ml_pipeline")
    cfg = TrainingConfig()
    # Overrides
    if args.holdout is not None:
        cfg.holdout_last_days = args.holdout
    if args.step is not None:
        cfg.step_days = args.step
    if args.lookback is not None:
        cfg.lookback_days = args.lookback
    if args.window is not None:
        cfg.window_mode = args.window  # type: ignore
    log.info(f"Config: {cfg}")

    ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    bronze = load_bronze(Path("data/bronze/daily"), tickers=ticker_list)
    if not bronze:
        log.error("No bronze Parquet files found. Run raw_to_parquet converter first.")
        return 2

    pooled = build_pooled_dataset(bronze, cfg)
    if pooled.empty:
        log.error("Pooled dataset is empty after feature/label building.")
        return 3

    selected_models = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else None
    results, preds, model_scores, confusions = walk_forward_run(pooled, cfg, selected_models)
    if not results:
        log.warning("Walk-forward produced no results; check data ranges and config.")
    else:
        # summarize last few results
        tail = results[-5:]
        for r in tail:
            log.info(f"as_of={r.as_of.date()} h={r.horizon} {r.metric_name}={r.metric_value:.4f} "
                     f"n_train={r.n_train} n_test={r.n_test}")

    # Persist pooled snapshot (optional reproducibility light-touch)
    snap_dir = Path("data/silver/snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"pooled_snapshot_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.parquet"
    try:
        pooled.to_parquet(snap_path, index=False)
        log.info(f"Saved pooled snapshot -> {snap_path}")
    except Exception as e:
        log.warning(f"Failed to save pooled snapshot: {e}")

    # Persist metrics CSVs (append)
    metrics_dir = Path("data/silver")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / "metrics.csv"
    run_meta = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "tickers": ",".join(ticker_list or []),
        "holdout": cfg.holdout_last_days,
        "step": cfg.step_days,
        "lookback": cfg.lookback_days,
        "window": cfg.window_mode,
        "models": ",".join(selected_models or []),
    }
    if results:
        rows = [{
            "as_of": r.as_of,
            "horizon": r.horizon,
            "metric_name": r.metric_name,
            "metric": r.metric_value,
            "n_train": r.n_train,
            "n_test": r.n_test,
            **run_meta,
        } for r in results]
        mdf = pd.DataFrame(rows)
        header = not metrics_csv.exists()
        mdf.to_csv(metrics_csv, mode="a", header=header, index=False)

    # Per-model scores
    if model_scores is not None and not model_scores.empty:
        ms_csv = metrics_dir / "model_scores.csv"
        ms = model_scores.copy()
        for k, v in run_meta.items():
            ms[k] = v
        header = not ms_csv.exists()
        ms.to_csv(ms_csv, mode="a", header=header, index=False)

    # Confusion counts (classification only)
    if confusions is not None and not confusions.empty:
        cf_csv = metrics_dir / "confusions.csv"
        cf = confusions.copy()
        for k, v in run_meta.items():
            cf[k] = v
        header = not cf_csv.exists()
        cf.to_csv(cf_csv, mode="a", header=header, index=False)

    # Optionally write predictions under data/silver/preds
    outdir = Path("data/silver/preds")
    outdir.mkdir(parents=True, exist_ok=True)
    for h, df in preds.items():
        if df is not None and not df.empty:
            df.to_parquet(outdir / f"preds_h{h}.parquet", index=False)
    log.info("ML pipeline run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
