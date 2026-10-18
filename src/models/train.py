"""
Model training with walk-forward validation.
Trains LightGBM models per position, including quantile variants for floor/ceiling.
Ensembles LightGBM + XGBoost with a Ridge meta-learner.
Uses SHAP for explainability.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class PositionModelBundle:
    """All models for a single position."""
    position: str
    features: list[str]
    lgb_median: lgb.Booster | None = None
    lgb_q10: lgb.Booster | None = None
    lgb_q90: lgb.Booster | None = None
    xgb_median: object | None = None
    meta_learner: Ridge | None = None
    meta_scaler: StandardScaler | None = None
    shap_explainer: shap.TreeExplainer | None = None
    validation_metrics: dict = field(default_factory=dict)


def _clean_for_model(df: pl.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix and target vector.
    - Only uses features that exist in the DataFrame.
    - Drops rows where TARGET is null (we can't train on those).
    - Fills feature nulls with column median (not drop — high-null features like snap_pct
      would otherwise eliminate all rows).
    """
    available_features = [f for f in features if f in df.columns]
    # Drop rows with null target only
    subset = df.filter(pl.col(target).is_not_null())

    X_pl = subset.select(available_features)
    # Fill nulls with column median for each feature
    fill_exprs = []
    for col in available_features:
        if X_pl[col].null_count() > 0:
            median_val = X_pl[col].median()
            fill_exprs.append(pl.col(col).fill_null(median_val if median_val is not None else 0.0))
        else:
            fill_exprs.append(pl.col(col))
    X_pl = X_pl.with_columns(fill_exprs)

    X = X_pl.to_pandas()
    y = subset[target].to_pandas()
    return X, y, available_features


def _get_lgb_params(quantile: float | None = None, custom_params: dict | None = None) -> dict:
    base = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        base.update(custom_params)
    if quantile is not None:
        base["objective"] = "quantile"
        base["alpha"] = quantile
        base["metric"] = "quantile"
    else:
        base["objective"] = "regression"
        base["metric"] = "mae"
    return base


def _train_lgb(X: pd.DataFrame, y: pd.Series, quantile: float | None = None, custom_params: dict | None = None) -> lgb.Booster:
    params = _get_lgb_params(quantile, custom_params)
    n_estimators = params.pop("n_estimators")
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=999)],  # suppress per-round output
    )
    return model


def walk_forward_validate(
    df: pl.DataFrame,
    position: str,
    features: list[str],
    target: str = "fantasy_points",
    min_train_seasons: int = 4,
) -> dict:
    """
    Walk-forward validation: train on seasons 1..N, test on season N+1.
    Never shuffles data. Always trains on past, tests on future.
    """
    pos_df = df.filter(pl.col("position") == position)
    seasons = sorted(pos_df["season"].unique().to_list())

    if len(seasons) < min_train_seasons + 1:
        logger.warning(f"  Not enough seasons for {position}: {len(seasons)} available, need {min_train_seasons + 1}")
        return {}

    all_preds = []
    all_actuals = []
    all_baseline_preds = []
    fold_metrics = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train = pos_df.filter(pl.col("season").is_in(train_seasons))
        test = pos_df.filter(pl.col("season") == test_season)

        X_train, y_train, used_features = _clean_for_model(train, features, target)
        X_test, y_test, _ = _clean_for_model(test, features, target)

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        model = _train_lgb(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        spearman_r, _ = spearmanr(y_test, preds)

        # Naive baseline: 4-week rolling mean from the features
        baseline_col = "fp_roll4_mean"
        if baseline_col in used_features:
            baseline_preds = X_test[baseline_col].fillna(y_train.mean())
            baseline_mae = mean_absolute_error(y_test, baseline_preds)
        else:
            baseline_mae = float("nan")
            baseline_preds = pd.Series([y_train.mean()] * len(y_test))

        fold_metrics.append({
            "test_season": test_season,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "spearman_r": round(spearman_r, 3),
            "baseline_mae": round(baseline_mae, 3) if not np.isnan(baseline_mae) else None,
            "vs_baseline": round(mae - baseline_mae, 3) if not np.isnan(baseline_mae) else None,
        })

        all_preds.extend(preds.tolist())
        all_actuals.extend(y_test.tolist())
        all_baseline_preds.extend(baseline_preds.tolist())

        sign = "+" if mae < baseline_mae else ""
        logger.info(
            f"  {position} {test_season}: MAE={mae:.2f}  Baseline={baseline_mae:.2f}  "
            f"Rank-r={spearman_r:.3f}  n={len(y_test)}"
        )

    if not all_actuals:
        return {"folds": fold_metrics, "overall": {}}

    overall_mae = mean_absolute_error(all_actuals, all_preds)
    overall_rmse = mean_squared_error(all_actuals, all_preds) ** 0.5
    overall_spearman, _ = spearmanr(all_actuals, all_preds)
    baseline_overall_mae = mean_absolute_error(all_actuals, all_baseline_preds)

    return {
        "folds": fold_metrics,
        "overall": {
            "mae": round(overall_mae, 3),
            "rmse": round(overall_rmse, 3),
            "spearman_r": round(overall_spearman, 3),
            "baseline_mae": round(baseline_overall_mae, 3),
            "vs_baseline": round(overall_mae - baseline_overall_mae, 3),
            "n_total": len(all_actuals),
        },
    }


def train_position_bundle(
    df: pl.DataFrame,
    position: str,
    features: list[str],
    target: str = "fantasy_points",
    custom_params: dict | None = None,
) -> PositionModelBundle:
    """
    Train the full model bundle for a position on ALL available data.
    Includes: LGB median, LGB Q10/Q90, XGBoost, Ridge meta-learner, SHAP explainer.
    """
    import xgboost as xgb

    pos_df = df.filter(pl.col("position") == position)
    X, y, used_features = _clean_for_model(pos_df, features, target)
    bundle = PositionModelBundle(position=position, features=used_features)

    if len(X) < 100:
        logger.warning(f"  Only {len(X)} rows for {position} — skipping final training.")
        return bundle

    logger.info(f"  Training {position} final bundle on {len(X):,} samples, {len(used_features)} features...")

    bundle.lgb_median = _train_lgb(X, y, quantile=None, custom_params=custom_params)
    bundle.lgb_q10 = _train_lgb(X, y, quantile=0.1, custom_params=custom_params)
    bundle.lgb_q90 = _train_lgb(X, y, quantile=0.9, custom_params=custom_params)

    # XGBoost
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {
        "max_depth": 5, "eta": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "alpha": 0.1, "lambda": 0.1,
        "objective": "reg:squarederror", "eval_metric": "mae",
        "seed": 42, "nthread": -1, "verbosity": 0,
    }
    bundle.xgb_median = xgb.train(xgb_params, dtrain, num_boost_round=300)

    # Ridge meta-learner (ensemble LGB + XGB)
    lgb_preds = bundle.lgb_median.predict(X)
    xgb_preds = bundle.xgb_median.predict(xgb.DMatrix(X))
    meta_X = np.column_stack([lgb_preds, xgb_preds])
    bundle.meta_scaler = StandardScaler()
    meta_X_scaled = bundle.meta_scaler.fit_transform(meta_X)
    bundle.meta_learner = Ridge(alpha=1.0)
    bundle.meta_learner.fit(meta_X_scaled, y)

    # SHAP explainer on median LGB
    bundle.shap_explainer = shap.TreeExplainer(bundle.lgb_median)

    return bundle


def predict_player(bundle: PositionModelBundle, player_features: dict) -> dict:
    """
    Predict for a single player given a feature dict.
    Returns: predicted, floor (Q10), ceiling (Q90), shap breakdown.
    """
    import xgboost as xgb

    # Align features to what the model was trained on
    row = {f: player_features.get(f) for f in bundle.features}
    X = pd.DataFrame([row])
    # Ensure all columns are numeric — schedule features can come in as object dtype
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    lgb_median = bundle.lgb_median.predict(X)[0]
    lgb_q10 = bundle.lgb_q10.predict(X)[0]
    lgb_q90 = bundle.lgb_q90.predict(X)[0]
    xgb_pred = bundle.xgb_median.predict(xgb.DMatrix(X))[0]

    meta_X = np.array([[lgb_median, xgb_pred]])
    meta_X_scaled = bundle.meta_scaler.transform(meta_X)
    ensemble_median = float(bundle.meta_learner.predict(meta_X_scaled)[0])

    # SHAP values
    shap_vals = bundle.shap_explainer.shap_values(X)[0]
    shap_dict = dict(zip(bundle.features, shap_vals))
    top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    predicted = max(0.0, ensemble_median)
    floor = max(0.0, float(lgb_q10))
    ceiling = max(0.0, float(lgb_q90))

    # Enforce floor ≤ predicted ≤ ceiling.
    # The ensemble median (lgb + xgb meta-learner) is independent of the raw
    # lgb quantile models, so they can cross. Clamp rather than discard either.
    floor = min(floor, predicted)
    ceiling = max(ceiling, predicted)

    return {
        "predicted": round(predicted, 1),
        "floor": round(floor, 1),
        "ceiling": round(ceiling, 1),
        "shap_top_positive": [(k, round(v, 2)) for k, v in top_shap if v > 0][:5],
        "shap_top_negative": [(k, round(v, 2)) for k, v in top_shap if v < 0][:5],
    }


def save_bundle(bundle: PositionModelBundle) -> Path:
    path = MODELS_DIR / f"{bundle.position.lower()}_bundle.joblib"
    joblib.dump(bundle, path)
    logger.info(f"  Saved {bundle.position} bundle → {path}")
    return path


def load_bundle(position: str) -> PositionModelBundle | None:
    path = MODELS_DIR / f"{position.lower()}_bundle.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
