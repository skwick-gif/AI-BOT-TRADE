from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .config import TrainingConfig
from .dataset import feature_label_columns
from .splits import compute_cutoff_dates, advance_trading_days


@dataclass
class StepResult:
    as_of: pd.Timestamp
    horizon: int
    n_train: int
    n_test: int
    metric_name: str
    metric_value: float


def _make_pipelines(task: str, numeric: List[str], categorical: List[str], selected: Optional[Set[str]] = None):
    num_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_transform, numeric),
        ("cat", cat_transform, categorical),
    ])

    if task == "classification":
        default_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        }
        # מודלים מתקדמים
        try:
            from lightgbm import LGBMClassifier
            default_models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42)
        except ImportError:
            pass
        try:
            from catboost import CatBoostClassifier
            default_models["CatBoost"] = CatBoostClassifier(iterations=200, random_state=42, verbose=0)
        except ImportError:
            pass
        try:
            from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
            default_models["ExtraTrees"] = ExtraTreesClassifier(n_estimators=200, random_state=42)
            default_models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=200, random_state=42)
        except ImportError:
            pass
        # TabNet
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            default_models["TabNet"] = TabNetClassifier()
        except ImportError:
            pass
    else:
        default_models = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        }
        try:
            from lightgbm import LGBMRegressor
            default_models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=42)
        except ImportError:
            pass
        try:
            from catboost import CatBoostRegressor
            default_models["CatBoost"] = CatBoostRegressor(iterations=200, random_state=42, verbose=0)
        except ImportError:
            pass
        try:
            from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
            default_models["ExtraTrees"] = ExtraTreesRegressor(n_estimators=200, random_state=42)
            default_models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=200, random_state=42)
        except ImportError:
            pass
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            default_models["TabNet"] = TabNetRegressor()
        except ImportError:
            pass
    models = default_models
    if selected:
        filtered = {k: v for k, v in default_models.items() if k in selected}
        # Fallback to defaults if selection yields no compatible models for the task
        models = filtered if filtered else default_models
    pipes = {name: Pipeline([("pre", pre), (name, est)]) for name, est in models.items()}
    return pipes


def _select_metric(task: str):
    if task == "classification":
        def scorer(y_true, y_pred):
            # Map labels in case of OHE categories
            labels = ["DOWN", "HOLD", "UP"]
            return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        return "F1_macro", scorer
    else:
        def scorer(y_true, y_pred):
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return "RMSE", scorer


def walk_forward_run(
    pooled: pd.DataFrame,
    cfg: TrainingConfig,
    selected_models: Optional[List[str]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[StepResult], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Run a simple walk-forward evaluation over the pooled dataset.

    Returns a list of StepResult and a dict of per-horizon predictions DataFrames.
    """
    if pooled.empty:
        return [], {}, pd.DataFrame(), pd.DataFrame()

    # Determine features/labels
    x_cols, y_cols = feature_label_columns(pooled)
    # Separate numeric/categorical
    cat_cols = [c for c in x_cols if str(pooled[c].dtype) == "object"] + ["ticker"]
    num_cols = [c for c in x_cols if c not in cat_cols and c not in ("date",)]

    # Cutoffs
    all_dates = pd.DatetimeIndex(pooled["date"].dropna().unique()).sort_values()
    max_h = max(cfg.horizons)
    cutoff_train_end, end_date = compute_cutoff_dates(all_dates, holdout_last_days=cfg.holdout_last_days, max_horizon=max_h)

    # step through time
    cur = cutoff_train_end
    # Pre-compute total number of walk-forward steps for progress reporting
    total_steps = 0
    tmp = cur
    while tmp < end_date:
        nxt_tmp = advance_trading_days(all_dates, tmp, step_days=cfg.step_days)
        if nxt_tmp <= tmp:
            break
        total_steps += 1
        tmp = nxt_tmp
    step_idx = 0
    metric_name, score_fn = _select_metric(cfg.target.type)

    results: List[StepResult] = []
    preds_by_h: Dict[str, List[pd.DataFrame]] = {str(h): [] for h in cfg.horizons}
    model_score_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []

    chosen: Optional[Set[str]] = set(m for m in selected_models) if selected_models else None
    while cur < end_date:
        # define windows
        if cfg.window_mode == "expanding":
            train_start = all_dates[0]
        else:
            # rolling window: use lookback_days worth of trading days
            idx = all_dates.get_loc(cur)
            lb_idx = max(0, idx - cfg.lookback_days)
            train_start = all_dates[lb_idx]

        train_end = cur
        test_start = advance_trading_days(all_dates, cur, step_days=1)
        test_end = advance_trading_days(all_dates, cur, step_days=cfg.step_days)

        train_mask = (pooled["date"] >= train_start) & (pooled["date"] <= train_end)
        test_mask = (pooled["date"] > test_start) & (pooled["date"] <= test_end)

        train_df = pooled.loc[train_mask].copy()
        test_df = pooled.loc[test_mask].copy()

        # Drop rows without labels in the selected horizon during fitting/prediction
        for h in cfg.horizons:
            y_col = f"y_h{h}"
            if y_col not in pooled.columns:
                continue
            t_train = train_df.dropna(subset=[y_col])
            t_test = test_df.dropna(subset=[y_col])
            if t_train.empty or t_test.empty:
                continue

            # Prepare pipeline per task
            pipes = _make_pipelines(cfg.target.type, num_cols, cat_cols, chosen)
            best_score = -np.inf if cfg.target.type == "classification" else np.inf
            best_name = None
            best_pred = None

            per_model_scores: List[Tuple[str, float, Any]] = []  # (name, score, y_hat)
            for name, pipe in pipes.items():
                    X_train = t_train[num_cols + cat_cols]
                    y_train = t_train[y_col]
                    X_test = t_test[num_cols + cat_cols]
                    y_test = t_test[y_col]

                    # For classification, ensure y is str categories
                    if cfg.target.type == "classification":
                        y_train = y_train.astype(str)
                        y_test = y_test.astype(str)
                    pipe.fit(X_train, y_train)
                    y_hat = pipe.predict(X_test)
                    score = score_fn(y_test, y_hat)
                    per_model_scores.append((name, float(score), y_hat))
                    if cfg.target.type == "classification":
                        if score > best_score:
                            best_score, best_name, best_pred = score, name, y_hat
                    else:
                        if score < best_score:
                            best_score, best_name, best_pred = score, name, y_hat
                    print(f"[ML] Horizon {h}: train rows with label={len(t_train)}, test rows with label={len(t_test)}")
            print(f"[ML] Split as_of={cur.date()}: train rows={len(train_df)}, test rows={len(test_df)}")

            if best_name is None:
                continue

            # record per-model scores
            for mname, mscore, _yh in per_model_scores:
                    model_score_rows.append({
                        "as_of": pd.Timestamp(cur),
                        "horizon": h,
                        "model": mname,
                        "metric_name": metric_name,
                        "metric": float(mscore),
                        "n_train": len(t_train),
                        "n_test": len(t_test),
                    })

            results.append(StepResult(
                as_of=pd.Timestamp(cur),
                horizon=h,
                n_train=len(t_train),
                n_test=len(t_test),
                metric_name=metric_name,
                metric_value=float(best_score),
            ))

            # store predictions frame
            out = t_test[["ticker", "date"]].copy()
            out["y_true"] = t_test[y_col].values
            out["y_pred"] = best_pred
            out["model"] = best_name
            preds_by_h[str(h)].append(out)

            # confusion counts (classification only)
            if cfg.target.type == "classification":
                try:
                    labels = ["DOWN", "HOLD", "UP"]
                    cm = confusion_matrix(out["y_true"], out["y_pred"], labels=labels)
                    confusion_rows.append({
                        "as_of": pd.Timestamp(cur),
                        "horizon": h,
                        "label": ",".join(labels),
                        "cm_down_down": int(cm[0, 0]),
                        "cm_down_hold": int(cm[0, 1]),
                        "cm_down_up": int(cm[0, 2]),
                        "cm_hold_down": int(cm[1, 0]),
                        "cm_hold_hold": int(cm[1, 1]),
                        "cm_hold_up": int(cm[1, 2]),
                        "cm_up_down": int(cm[2, 0]),
                        "cm_up_hold": int(cm[2, 1]),
                        "cm_up_up": int(cm[2, 2]),
                    })
                except Exception:
                    pass

        # advance
        nxt = advance_trading_days(all_dates, cur, step_days=cfg.step_days)
        if nxt <= cur:
            break
        # progress update (after finishing this step)
        step_idx += 1
        if progress_cb and total_steps > 0:
            frac = min(1.0, step_idx / max(1, total_steps))
            try:
                progress_cb(frac, f"Evaluated as_of={pd.Timestamp(cur).date()}")
            except Exception:
                pass
        cur = nxt

    preds_concat = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in preds_by_h.items()}
    model_scores_df = pd.DataFrame(model_score_rows)
    confusions_df = pd.DataFrame(confusion_rows)
    return results, preds_concat, model_scores_df, confusions_df
