"""
Hyperparameter tuning with Optuna for LightGBM position models.
Uses walk-forward cross-validation to prevent leakage during tuning.
"""

import logging

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _wf_cv_mae(
    params: dict,
    df: pl.DataFrame,
    position: str,
    features: list[str],
    target: str = "fantasy_points",
    min_train_seasons: int = 4,
    n_folds: int = 3,
) -> float:
    """Walk-forward CV MAE for a given parameter set."""
    pos_df = df.filter(pl.col("position") == position)
    seasons = sorted(pos_df["season"].unique().to_list())

    if len(seasons) < min_train_seasons + 1:
        return float("inf")

    fold_start = max(min_train_seasons, len(seasons) - n_folds)
    fold_maes = []

    for i in range(fold_start, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train = pos_df.filter(pl.col("season").is_in(train_seasons))
        test = pos_df.filter(pl.col("season") == test_season)

        available = [f for f in features if f in train.columns]
        train_clean = train.filter(pl.col(target).is_not_null())
        test_clean = test.filter(pl.col(target).is_not_null())

        if len(train_clean) < 50 or len(test_clean) < 10:
            continue

        X_train = train_clean.select(available).to_pandas()
        y_train = train_clean[target].to_pandas()
        X_test = test_clean.select(available).to_pandas()
        y_test = test_clean[target].to_pandas()

        # Fill nulls with training median
        for col in available:
            m = X_train[col].median()
            fill = m if m is not None and not np.isnan(m) else 0.0
            X_train[col] = X_train[col].fillna(fill)
            X_test[col] = X_test[col].fillna(fill)

        p = dict(params)
        n_est = p.pop("n_estimators", 300)
        p.update({"verbose": -1, "n_jobs": -1, "random_state": 42})
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            p, dtrain, num_boost_round=n_est,
            callbacks=[lgb.log_evaluation(period=9999)],
        )
        preds = model.predict(X_test)
        fold_maes.append(mean_absolute_error(y_test, preds))

    return float(np.mean(fold_maes)) if fold_maes else float("inf")


def tune_position(
    df: pl.DataFrame,
    position: str,
    features: list[str],
    target: str = "fantasy_points",
    n_trials: int = 40,
    n_folds: int = 3,
) -> dict:
    """Run Optuna search; return best params dict."""
    logger.info(f"  Tuning {position} ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }
        return _wf_cv_mae(params, df, position, features, target, n_folds=n_folds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"    Best MAE: {study.best_value:.3f}  lr={best.get('learning_rate', 0):.4f}  leaves={best.get('num_leaves')}")
    return best
