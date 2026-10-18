"""
Model evaluation utilities: metrics, baseline comparisons, calibration checks.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    spearman_r, _ = spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "spearman_r": round(spearman_r, 3),
        "r2": round(r2, 3),
        "n": len(y_true),
    }


def compute_baselines(y_true: np.ndarray, baselines: dict[str, np.ndarray]) -> dict:
    """Compare model against multiple baselines."""
    return {name: compute_metrics(y_true, preds) for name, preds in baselines.items()}


def check_interval_calibration(
    y_true: np.ndarray, y_q10: np.ndarray, y_q90: np.ndarray
) -> dict:
    """
    Check if 80% of actuals fall within the Q10–Q90 interval.
    A well-calibrated model should have coverage close to 80%.
    """
    in_interval = ((y_true >= y_q10) & (y_true <= y_q90))
    coverage = in_interval.mean()
    avg_width = (y_q90 - y_q10).mean()
    return {
        "coverage_80pct": round(float(coverage), 3),
        "target_coverage": 0.80,
        "avg_interval_width": round(float(avg_width), 2),
        "is_calibrated": abs(coverage - 0.80) < 0.05,
    }


def start_sit_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold_percentile: float = 0.5
) -> dict:
    """
    Start/sit decision accuracy: does the model correctly identify top-half players?
    threshold_percentile: players above this predicted score are "start" recommendations.
    """
    threshold = np.percentile(y_pred, threshold_percentile * 100)
    predicted_starts = y_pred >= threshold
    actual_starts = y_true >= np.median(y_true)
    accuracy = (predicted_starts == actual_starts).mean()
    return {
        "start_sit_accuracy": round(float(accuracy), 3),
        "n_recommendations": len(y_pred),
    }


def print_evaluation_report(position: str, validation_results: dict) -> None:
    """Print a formatted evaluation report to console."""
    print(f"\n{'='*60}")
    print(f"  {position} MODEL EVALUATION REPORT")
    print(f"{'='*60}")

    if "overall" in validation_results:
        overall = validation_results["overall"]
        mae = overall.get("mae")
        rmse = overall.get("rmse")
        spearman = overall.get("spearman_r")
        print(f"\nOverall (all validation seasons):")
        print(f"  MAE:       {mae:.2f} fantasy points" if mae else "  MAE:       N/A")
        print(f"  RMSE:      {rmse:.2f} fantasy points" if rmse else "  RMSE:      N/A")
        print(f"  Spearman:  {spearman:.3f}" if spearman else "  Spearman:  N/A")
        print(f"  N games:   {overall.get('n_total', 'N/A')}")

    if "folds" in validation_results:
        print(f"\nPer-season breakdown:")
        print(f"  {'Season':<8} {'MAE':>6} {'RMSE':>6} {'Spearman':>9} {'Baseline':>9} {'Diff':>7}")
        print(f"  {'-'*52}")
        for fold in validation_results["folds"]:
            vs = fold.get("vs_baseline")
            diff_str = f"{vs:+.2f}" if vs is not None else "  N/A"
            bsl = fold.get("baseline_mae")
            bsl_str = f"{bsl:.2f}" if bsl is not None else "  N/A"
            print(
                f"  {fold['test_season']:<8} "
                f"{fold.get('mae', 0):>6.2f} "
                f"{fold.get('rmse', 0):>6.2f} "
                f"{fold.get('spearman_r', 0):>9.3f} "
                f"{bsl_str:>9} "
                f"{diff_str:>7}"
            )
        print(f"\n  Diff = Model MAE - Baseline (negative = model beats 4-wk rolling avg)")

    print(f"\nTarget benchmarks for a GOOD model:")
    targets = {"QB": 2.2, "RB": 3.0, "WR": 2.8, "TE": 3.5}
    target = targets.get(position, 2.8)
    overall_mae = validation_results.get("overall", {}).get("mae")
    if overall_mae:
        status = "✓ GOOD" if overall_mae <= target else "✗ NEEDS WORK"
        print(f"  Target MAE ≤ {target:.1f} | Achieved: {overall_mae:.2f} | {status}")
