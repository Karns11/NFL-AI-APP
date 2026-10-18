"""
Phase 2 feature engineering additions.
Builds on top of Phase 1 features — call build_phase2_features(df) after build_features().

New features:
- Starter filter utilities (snap_pct >= threshold for evaluation)
- Rolling EPA per game (receiving, rushing, passing)
- Rolling WOPR (weighted opportunity rating)
- Team passing volume trend (team's recent passing yards/attempts)
- Snap/target share trend (rising vs. falling usage)
- Red zone proxy (rec yards after catch / receiving yards ratio as TD-opportunity signal)
- Exponential weighted moving averages (alpha=0.3) as alternative to simple rolling
"""

import polars as pl


STARTER_SNAP_THRESHOLD = 0.40  # >= 40% snap share = "meaningful contributor"


def add_ewm_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add exponentially weighted moving averages (faster response to recent changes).
    EWM with alpha=0.3 weights recent games more than simple rolling average.
    All lagged by 1 game to prevent leakage.
    """
    df = df.sort(["player_id", "season", "week"])

    df = df.with_columns([
        # EWM fantasy points (lagged)
        pl.col("fantasy_points")
          .shift(1)
          .ewm_mean(alpha=0.3, ignore_nulls=True)
          .over("player_id")
          .alias("fp_ewm"),

        # EWM target share (lagged)
        pl.col("target_share")
          .shift(1)
          .ewm_mean(alpha=0.3, ignore_nulls=True)
          .over("player_id")
          .alias("target_share_ewm"),

        # EWM snap pct (lagged)
        pl.col("snap_pct")
          .fill_null(0.0)
          .shift(1)
          .ewm_mean(alpha=0.3, ignore_nulls=True)
          .over("player_id")
          .alias("snap_pct_ewm"),
    ])

    return df


def add_trend_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add trend direction features:
    - snap_pct_trend: recent 2-game avg vs. prior 4-game avg (positive = rising)
    - target_share_trend: same
    These capture momentum without requiring future data.
    All lagged.
    """
    df = df.sort(["player_id", "season", "week"])

    df = df.with_columns([
        # Snap pct recent (last 2 games) vs. prior (3-6 games ago)
        (pl.col("snap_pct").fill_null(0).shift(1).rolling_mean(2).over("player_id") -
         pl.col("snap_pct").fill_null(0).shift(3).rolling_mean(4).over("player_id"))
        .alias("snap_pct_trend"),

        # Target share recent vs. prior
        (pl.col("target_share").shift(1).rolling_mean(2).over("player_id") -
         pl.col("target_share").shift(3).rolling_mean(4).over("player_id"))
        .alias("target_share_trend"),
    ])

    return df


def add_epa_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rolling EPA features. EPA (Expected Points Added) per play is more
    informative than raw yards because it accounts for down/distance context.
    Nulls for off-position plays are filled with 0 before rolling.
    """
    df = df.sort(["player_id", "season", "week"])

    df = df.with_columns([
        # Receiving EPA rolling 4 (lagged) — most important for WR/TE
        pl.col("receiving_epa")
          .fill_null(0.0)
          .shift(1)
          .rolling_mean(4)
          .over("player_id")
          .alias("receiving_epa_roll4"),

        # Rushing EPA rolling 4 (lagged) — important for RB
        pl.col("rushing_epa")
          .fill_null(0.0)
          .shift(1)
          .rolling_mean(4)
          .over("player_id")
          .alias("rushing_epa_roll4"),

        # WOPR rolling 4 (lagged) — weighted opportunity rating, WR/TE focus
        # WOPR = 1.5 * target_share + 0.7 * air_yards_share
        pl.col("wopr")
          .fill_null(0.0)
          .shift(1)
          .rolling_mean(4)
          .over("player_id")
          .alias("wopr_roll4"),
    ])

    return df


def add_team_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add team-level passing volume trend features.
    Is the team throwing more or less than recent average?
    Computed from aggregating player stats per team-game (no PBP needed).
    """
    # Team total passing yards and attempts per game
    team_pass = (
        df.group_by(["season", "week", "team"])
        .agg([
            pl.col("passing_yards").sum().alias("team_pass_yds"),
            pl.col("attempts").sum().alias("team_pass_att"),
        ])
        .sort(["team", "season", "week"])
    )

    # Rolling team passing volume (lagged 1 game)
    team_pass = team_pass.with_columns([
        pl.col("team_pass_yds")
          .shift(1)
          .rolling_mean(4)
          .over("team")
          .alias("team_pass_yds_roll4"),
        pl.col("team_pass_att")
          .shift(1)
          .rolling_mean(4)
          .over("team")
          .alias("team_pass_att_roll4"),
    ])

    df = df.join(
        team_pass.select(["season", "week", "team", "team_pass_yds_roll4", "team_pass_att_roll4"]),
        on=["season", "week", "team"],
        how="left",
    )

    return df


def add_rz_proxy_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Red zone proxy features without PBP data.

    We don't have explicit red zone targets/carries in weekly stats, but we can
    approximate TD opportunity using:
    - receiving_tds / (targets + 1): personal TD rate per target
    - rushing_tds / (carries + 1): personal TD rate per carry

    These are lagged rolling averages — they capture a player's historical
    tendency to convert opportunities into TDs, which is a useful signal.
    """
    df = df.sort(["player_id", "season", "week"])

    # Personal TD rate per target (receiving)
    rec_td_rate = (
        pl.col("receiving_tds").fill_null(0) /
        (pl.col("targets").fill_null(0) + 1)
    )

    rush_td_rate = (
        pl.col("rushing_tds").fill_null(0) /
        (pl.col("carries").fill_null(0) + 1)
    )

    df = df.with_columns([
        rec_td_rate.alias("rec_td_rate_raw"),
        rush_td_rate.alias("rush_td_rate_raw"),
    ])

    df = df.with_columns([
        pl.col("rec_td_rate_raw")
          .shift(1)
          .rolling_mean(8)
          .over("player_id")
          .alias("rec_td_rate_roll8"),

        pl.col("rush_td_rate_raw")
          .shift(1)
          .rolling_mean(8)
          .over("player_id")
          .alias("rush_td_rate_roll8"),

        # Also add receiving_yards_after_catch share as a proxy for RZ usage
        # Players who catch short passes in RZ will have high YAC proportion
        pl.col("receiving_yards_after_catch")
          .fill_null(0)
          .shift(1)
          .rolling_mean(4)
          .over("player_id")
          .alias("yac_roll4"),
    ])

    return df


def build_phase2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply all Phase 2 feature additions."""
    df = add_ewm_features(df)
    df = add_trend_features(df)
    df = add_epa_features(df)
    df = add_team_volume_features(df)
    df = add_rz_proxy_features(df)
    return df


def get_phase2_feature_columns(position: str, has_vegas: bool = True) -> list[str]:
    """Phase 2 feature set: all Phase 1 features + new additions."""
    from src.features.engineer import get_feature_columns

    phase1 = get_feature_columns(position, has_vegas=has_vegas)

    p2_base = [
        "fp_ewm",
        "snap_pct_ewm",
        "target_share_ewm",
        "snap_pct_trend",
        "target_share_trend",
        "wopr_roll4",
        "team_pass_yds_roll4",
        "team_pass_att_roll4",
    ]

    if position in ("WR", "TE"):
        p2_pos = [
            "receiving_epa_roll4",
            "rec_td_rate_roll8",
            "yac_roll4",
        ]
    elif position == "RB":
        p2_pos = [
            "rushing_epa_roll4",
            "receiving_epa_roll4",
            "rush_td_rate_roll8",
            "rec_td_rate_roll8",
            "yac_roll4",
        ]
    elif position == "QB":
        p2_pos = [
            "rushing_epa_roll4",
        ]
    else:
        p2_pos = []

    # Deduplicate while preserving order
    seen = set()
    all_features = []
    for f in phase1 + p2_base + p2_pos:
        if f not in seen:
            seen.add(f)
            all_features.append(f)

    return all_features


def filter_starters(df: pl.DataFrame, min_snap_pct: float = STARTER_SNAP_THRESHOLD) -> pl.DataFrame:
    """
    Filter to players with meaningful snap participation.
    This improves MAE reporting to reflect real start/sit decisions
    rather than including backups and injured players' zero-point games.
    """
    return df.filter(
        pl.col("snap_pct").fill_null(0) >= min_snap_pct
    )
