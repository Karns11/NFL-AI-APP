"""
Phase 1 MVP: Load data → engineer features → walk-forward validate → example prediction.

Run: python run_phase1.py

Steps:
1. Download NFL weekly stats + snap counts + schedules (2016-2024) via nflreadpy
2. Build Phase 1 features (snap%, target share, rolling FP, opponent defense, etc.)
3. Walk-forward validation per position: train seasons 1..N → test season N+1
4. Compare model MAE vs. 4-week rolling average baseline
5. Train final models on all data, save to /models/
6. Show example prediction with SHAP for the best WR from most recent season
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import polars as pl

from src.ingestion.load_data import load_player_stats, load_snap_counts, load_schedules, load_player_id_map, get_table_info
from src.features.engineer import build_features, get_feature_columns, get_target_column, drop_leaky_rows
from src.models.train import walk_forward_validate, train_position_bundle, predict_player, save_bundle, load_bundle
from src.models.evaluate import print_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEASONS = list(range(2016, 2025))  # 2016–2024
POSITIONS = ["WR", "RB", "QB", "TE"]


def main():
    print("\n" + "="*60)
    print("  LINEUPIQ — PHASE 1 MVP")
    print("="*60)

    # --- 1. Load data ---
    print("\n[1/4] Loading NFL data (will cache in DuckDB)...")
    stats = load_player_stats(SEASONS)
    snaps = load_snap_counts(SEASONS)
    schedules = load_schedules(SEASONS)
    id_map = load_player_id_map()

    for table, meta in get_table_info().items():
        if meta.get("rows", 0) > 0:
            print(f"  {table}: {meta['rows']:,} rows  ({meta['season_min']}–{meta['season_max']})")

    print(f"\n  Stats rows:     {len(stats):,}")
    print(f"  Snap rows:      {len(snaps):,}")
    print(f"  Schedule rows:  {len(schedules):,}")

    # --- 2. Feature engineering ---
    print("\n[2/4] Engineering features...")
    features_df = build_features(stats, schedules=schedules, snaps=snaps, id_map=id_map)
    print(f"  Feature matrix: {len(features_df):,} rows × {len(features_df.columns)} columns")

    # Check if schedule features are populated (Vegas lines, weather)
    n_with_vegas = features_df.filter(pl.col("team_implied_total").is_not_null()).height
    print(f"  Rows with Vegas lines: {n_with_vegas:,}")

    # --- 3. Walk-forward validation ---
    print("\n[3/4] Walk-forward validation (train past → test future seasons)...")

    has_vegas = n_with_vegas > 0
    all_results = {}
    for position in POSITIONS:
        feature_cols = get_feature_columns(position, has_vegas=has_vegas)
        # Filter to only features that exist in df
        feature_cols = [f for f in feature_cols if f in features_df.columns]

        print(f"\n  {position} | {len(feature_cols)} features: {feature_cols[:6]}...")

        results = walk_forward_validate(
            drop_leaky_rows(features_df),
            position=position,
            features=feature_cols,
            target=get_target_column(),
            min_train_seasons=4,
        )
        all_results[position] = results
        print_evaluation_report(position, results)

    # --- 4. Final models ---
    print("\n[4/4] Training final models on all data...")
    clean_df = drop_leaky_rows(features_df)

    for position in POSITIONS:
        feature_cols = get_feature_columns(position, has_vegas=has_vegas)
        feature_cols = [f for f in feature_cols if f in clean_df.columns]

        pos_count = clean_df.filter(pl.col("position") == position).height
        print(f"\n  {position}: {pos_count:,} rows")

        if pos_count < 50:
            print(f"  Skipping — insufficient data.")
            continue

        bundle = train_position_bundle(clean_df, position=position, features=feature_cols)
        if bundle.lgb_median is not None:
            save_bundle(bundle)
            print(f"  ✓ {position} bundle saved")
        else:
            print(f"  ✗ {position} training failed (see logs)")

    # --- Example prediction ---
    print("\n" + "="*60)
    print("  EXAMPLE PREDICTION")
    print("="*60)

    wr_bundle = load_bundle("WR")
    if wr_bundle:
        recent_season = features_df["season"].max()
        recent = (
            features_df
            .filter(
                (pl.col("position") == "WR") &
                (pl.col("season") == recent_season) &
                pl.col("fp_roll4_mean").is_not_null()
            )
            .sort("fp_roll4_mean", descending=True)
            .head(1)
        )

        if len(recent) > 0:
            row = recent.row(0, named=True)
            player_name = row.get("player_display_name") or row.get("player_name") or "Unknown"
            opponent = row.get("opponent_team", "OPP")
            week = row.get("week", "?")
            team = row.get("team", "TEAM")

            pred = predict_player(wr_bundle, row)

            print(f"\n  Player:   {player_name} ({team})")
            print(f"  Week {week} vs {opponent}  |  Season {recent_season}")
            print()
            print(f"  Projected:  {pred['predicted']:>5.1f} pts")
            print(f"  Floor:      {pred['floor']:>5.1f} pts  (Q10)")
            print(f"  Ceiling:    {pred['ceiling']:>5.1f} pts  (Q90)")
            print()
            print(f"  Top positive drivers (SHAP):")
            for feat, val in pred["shap_top_positive"]:
                feat_val = row.get(feat, "?")
                feat_val_str = f"{feat_val:.2f}" if isinstance(feat_val, float) else str(feat_val)
                print(f"    +{val:>5.2f}  {feat:<30} (value: {feat_val_str})")
            print(f"\n  Top negative drivers (SHAP):")
            for feat, val in pred["shap_top_negative"]:
                feat_val = row.get(feat, "?")
                feat_val_str = f"{feat_val:.2f}" if isinstance(feat_val, float) else str(feat_val)
                print(f"    {val:>6.2f}  {feat:<30} (value: {feat_val_str})")
    else:
        print("  No WR bundle found — training may have failed.")

    print("\n" + "="*60)
    print("  Phase 1 complete!")
    print("  Next steps:")
    print("    Phase 2: Quantile ensemble, Optuna tuning, more features")
    print("    Phase 3: Vegas (Odds API) + weather integration")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
