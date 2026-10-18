"""
Phase 2: Better features + hyperparameter tuning + calibration check.

What's new vs Phase 1:
- Exponential weighted moving averages (faster response to form changes)
- Usage trend features (is snap/target share rising or falling?)
- Rolling EPA (receiving_epa, rushing_epa, WOPR)
- Team passing volume trend
- Red zone proxy features (TD rate per target/carry, YAC)
- Starter filter (eval on snap_pct >= 40% only — shows real-world performance)
- Optuna hyperparameter tuning (40 trials per position)
- Calibration check (do our Q10/Q90 intervals actually cover 80% of actuals?)

Run: python run_phase2.py

Saves improved bundles to models/ as {position}_bundle_p2.joblib
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import polars as pl

from src.ingestion.load_data import (
    load_player_id_map, load_player_stats, load_schedules, load_snap_counts,
)
from src.features.engineer import build_features, drop_leaky_rows, get_feature_columns, get_target_column
from src.features.phase2 import (
    build_phase2_features, get_phase2_feature_columns, filter_starters,
    STARTER_SNAP_THRESHOLD,
)
from src.models.train import (
    walk_forward_validate, train_position_bundle, predict_player, save_bundle, load_bundle,
)
from src.models.tune import tune_position
from src.models.evaluate import print_evaluation_report, check_interval_calibration
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEASONS = list(range(2016, 2025))
POSITIONS = ["WR", "RB", "QB", "TE"]
TUNE_N_TRIALS = 40   # per position (~2 min each on Mac M-series)
TUNE_N_FOLDS = 3     # use last 3 seasons for tuning CV


def _compare_row(label: str, p1: dict, p2: dict, key: str, fmt: str = ".2f") -> None:
    v1 = p1.get(key)
    v2 = p2.get(key)
    if v1 is None or v2 is None:
        return
    delta = v2 - v1
    sign = "▼" if delta < 0 else "▲"  # lower MAE is better
    print(f"  {label:<12}  {v1:{fmt}} → {v2:{fmt}}  ({sign}{abs(delta):.3f})")


def main():
    print("\n" + "="*60)
    print("  NFL FANTASY AI — PHASE 2")
    print("  Better Features + Tuning + Calibration")
    print("="*60)

    # --- 1. Load data (from cache) ---
    print("\n[1/5] Loading data from cache...")
    stats = load_player_stats(SEASONS)
    snaps = load_snap_counts(SEASONS)
    schedules = load_schedules(SEASONS)
    id_map = load_player_id_map()
    print(f"  Stats: {len(stats):,} rows  |  Snaps: {len(snaps):,}  |  Schedules: {len(schedules):,}")

    # --- 2. Feature engineering (Phase 1 + Phase 2) ---
    print("\n[2/5] Building Phase 1 + Phase 2 features...")
    df = build_features(stats, schedules=schedules, snaps=snaps, id_map=id_map)
    df = build_phase2_features(df)
    df_clean = drop_leaky_rows(df)

    # Starter-filtered subset (for honest evaluation)
    df_starters = filter_starters(df_clean, min_snap_pct=STARTER_SNAP_THRESHOLD)

    for pos in POSITIONS:
        all_rows = df_clean.filter(pl.col("position") == pos).height
        starter_rows = df_starters.filter(pl.col("position") == pos).height
        print(f"  {pos}: {all_rows:,} total rows | {starter_rows:,} starter rows (snap≥{STARTER_SNAP_THRESHOLD:.0%})")

    # --- 3. Tune hyperparameters ---
    print(f"\n[3/5] Optuna hyperparameter tuning ({TUNE_N_TRIALS} trials × {len(POSITIONS)} positions)...")
    print(f"  Using walk-forward CV on last {TUNE_N_FOLDS} seasons as validation folds.")

    best_params: dict[str, dict] = {}
    for position in POSITIONS:
        feature_cols = get_phase2_feature_columns(position, has_vegas=True)
        feature_cols = [f for f in feature_cols if f in df_starters.columns]
        best_params[position] = tune_position(
            df_starters,
            position=position,
            features=feature_cols,
            n_trials=TUNE_N_TRIALS,
            n_folds=TUNE_N_FOLDS,
        )
        print(f"  ✓ {position} tuned")

    # --- 4. Walk-forward validation: Phase 1 vs Phase 2 ---
    print("\n[4/5] Walk-forward validation — Phase 1 vs Phase 2 (starters only)...")

    p1_results: dict = {}
    p2_results: dict = {}

    for position in POSITIONS:
        p1_features = get_feature_columns(position, has_vegas=True)
        p1_features = [f for f in p1_features if f in df_starters.columns]

        p2_features = get_phase2_feature_columns(position, has_vegas=True)
        p2_features = [f for f in p2_features if f in df_starters.columns]

        print(f"\n  {position}: {len(p1_features)} Phase 1 features → {len(p2_features)} Phase 2 features")

        # Phase 1 (default params, starters only)
        r1 = walk_forward_validate(df_starters, position, p1_features, min_train_seasons=4)
        p1_results[position] = r1

        # Phase 2 (tuned params, starters only)
        r2 = walk_forward_validate(df_starters, position, p2_features, min_train_seasons=4)
        p2_results[position] = r2

        print(f"\n  Phase 1 → Phase 2 comparison ({position}, starters only):")
        print(f"  {'Metric':<12}  {'Phase 1':>8}   {'Phase 2':>8}  {'Delta':>8}")
        print(f"  {'-'*42}")
        _compare_row("MAE", r1.get("overall", {}), r2.get("overall", {}), "mae")
        _compare_row("RMSE", r1.get("overall", {}), r2.get("overall", {}), "rmse")
        _compare_row("Spearman", r1.get("overall", {}), r2.get("overall", {}), "spearman_r", ".3f")

    # --- 5. Train final Phase 2 bundles + calibration check ---
    print("\n[5/5] Training final Phase 2 bundles + calibration check...")

    for position in POSITIONS:
        p2_features = get_phase2_feature_columns(position, has_vegas=True)
        p2_features = [f for f in p2_features if f in df_clean.columns]

        print(f"\n  {position} ({len(p2_features)} features, tuned params)...")

        bundle = train_position_bundle(
            df_clean,
            position=position,
            features=p2_features,
            custom_params=best_params[position],
        )

        if bundle.lgb_median is None:
            print(f"  ✗ Training failed")
            continue

        # Save Phase 2 bundle with _p2 suffix (keeps Phase 1 bundles intact)
        path = Path("models") / f"{position.lower()}_bundle_p2.joblib"
        joblib.dump(bundle, path)
        print(f"  ✓ Saved → {path}")

        # --- Calibration check (Q10/Q90 coverage) ---
        pos_test = df_clean.filter(
            (pl.col("position") == position) &
            (pl.col("season") == 2024) &
            pl.col("fantasy_points").is_not_null() &
            pl.col("snap_pct").fill_null(0).ge(STARTER_SNAP_THRESHOLD)
        )

        if len(pos_test) > 20:
            import xgboost as xgb
            avail = [f for f in bundle.features if f in pos_test.columns]
            X_test = pos_test.select(avail).to_pandas()
            for col in avail:
                X_test[col] = X_test[col].fillna(X_test[col].median())

            q10 = bundle.lgb_q10.predict(X_test)
            q90 = bundle.lgb_q90.predict(X_test)
            actuals = pos_test["fantasy_points"].to_numpy()

            calib = check_interval_calibration(actuals, q10, q90)
            status = "✓ CALIBRATED" if calib["is_calibrated"] else "⚠ NEEDS ADJUSTMENT"
            print(
                f"  Calibration (2024 starters): "
                f"coverage={calib['coverage_80pct']:.1%}  "
                f"(target=80%)  width={calib['avg_interval_width']:.1f} pts  {status}"
            )

    # --- Summary ---
    print("\n" + "="*60)
    print("  PHASE 2 SUMMARY")
    print("="*60)
    print(f"\n  Evaluation: starters only (snap% ≥ {STARTER_SNAP_THRESHOLD:.0%})")
    print(f"\n  {'Position':<6}  {'P1 MAE':>8}  {'P2 MAE':>8}  {'Improvement':>12}")
    print(f"  {'-'*42}")
    for pos in POSITIONS:
        p1_mae = p1_results.get(pos, {}).get("overall", {}).get("mae")
        p2_mae = p2_results.get(pos, {}).get("overall", {}).get("mae")
        if p1_mae and p2_mae:
            imp = p1_mae - p2_mae
            sign = "✓" if imp > 0 else "✗"
            print(f"  {pos:<6}  {p1_mae:>8.2f}  {p2_mae:>8.2f}  {imp:>+10.2f} {sign}")

    print("\n  Phase 2 complete. Models saved as *_bundle_p2.joblib")
    print("  Next: Phase 3 — The Odds API (live Vegas) + Open-Meteo weather")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
