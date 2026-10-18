"""
Feature engineering for fantasy football prediction.

CRITICAL LEAKAGE RULES:
- All rolling/expanding features are shifted by 1 game before rolling.
  This ensures we never include the game we are predicting in any rolling average.
- Opponent features use stats from BEFORE the target game.
- Schedule data (Vegas lines, weather, rest) is inherently pre-game — safe to use as-is.
- Never use random train/test splits — always time-based.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def join_schedule_features(
    stats: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join schedule-level features onto weekly player stats.
    Adds: is_home, away_rest, home_rest, spread_line, total_line,
          temp, wind, roof, weekday.
    """
    sched_cols = [
        "game_id", "season", "week",
        "home_team", "away_team",
        "away_rest", "home_rest",
        "spread_line", "total_line",
        "temp", "wind", "roof", "weekday",
    ]
    available = [c for c in sched_cols if c in schedules.columns]
    sched = schedules.select(available)

    # Join on game_id (already present in player stats)
    df = stats.join(sched, on=["game_id", "season", "week"], how="left")

    # Derive is_home
    df = df.with_columns(
        (pl.col("team") == pl.col("home_team")).cast(pl.Int8).alias("is_home")
    )

    # Rest days for this player's team
    df = df.with_columns(
        pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.col("home_rest"))
        .otherwise(pl.col("away_rest"))
        .alias("rest_days")
    )

    # Is outdoor game with weather concern?
    df = df.with_columns(
        pl.col("roof").is_in(["outdoors", "open"]).cast(pl.Int8).alias("is_outdoor")
    )

    # Implied team total from spread + total:
    # home_implied = (total + spread) / 2 (where spread is from home's perspective)
    # We need: player's team implied total
    df = df.with_columns(
        pl.when(pl.col("team") == pl.col("home_team"))
        .then((pl.col("total_line") - pl.col("spread_line")) / 2)
        .otherwise((pl.col("total_line") + pl.col("spread_line")) / 2)
        .alias("team_implied_total")
    )

    return df


def join_snap_counts(
    stats: pl.DataFrame,
    snaps: pl.DataFrame,
    id_map: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Join snap count data onto player stats.
    Uses pfr_player_id ↔ gsis_id bridge (via load_player_id_map) since
    name formats differ between the two tables (abbreviated vs full name).
    Falls back to game_id + team position rank matching if id_map is unavailable.
    """
    if id_map is not None and "gsis_id" in id_map.columns and "pfr_id" in id_map.columns:
        # Join snaps to id_map to get gsis_id (= player_id in stats)
        snaps_with_id = snaps.join(
            id_map.rename({"gsis_id": "player_id", "pfr_id": "pfr_player_id"}),
            on="pfr_player_id",
            how="left",
        )
        snap_cols = ["season", "week", "player_id", "team", "offense_pct"]
        available = [c for c in snap_cols if c in snaps_with_id.columns]
        snaps_slim = (
            snaps_with_id
            .select(available)
            .drop_nulls(subset=["player_id"])
            .rename({"offense_pct": "snap_pct"})
        )
        df = stats.join(snaps_slim, on=["season", "week", "player_id", "team"], how="left")
    else:
        # Fallback: no join, snap_pct will be null
        df = stats.with_columns(pl.lit(None).cast(pl.Float64).alias("snap_pct"))

    return df


def build_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build rolling features per player, sorted by (season, week).
    ALL rolling features are shifted by 1 game before rolling (leakage prevention).

    Features added:
    - snap_pct_roll3: 3-game rolling snap% (lagged)
    - target_share_roll4: 4-game rolling target share (lagged)
    - air_yards_share_roll4: 4-game rolling air yards share (lagged)
    - fp_roll4_mean: 4-game rolling fantasy points mean (lagged)
    - fp_ewm: exponentially weighted FP (lagged)
    - targets_roll4: 4-game rolling targets (lagged)
    - carries_roll4: 4-game rolling carries (lagged)
    - rec_yards_roll4: 4-game rolling receiving yards (lagged)
    - rush_yards_roll4: 4-game rolling rushing yards (lagged)
    - season_avg_fp: season-to-date FP average (expanding, lagged)
    """
    df = df.sort(["player_id", "season", "week"])

    df = df.with_columns([
        # --- Opportunity rolling means (lagged) ---
        pl.col("snap_pct")
          .shift(1)
          .rolling_mean(window_size=3)
          .over("player_id")
          .alias("snap_pct_roll3"),

        pl.col("target_share")
          .shift(1)
          .rolling_mean(window_size=4)
          .over("player_id")
          .alias("target_share_roll4"),

        pl.col("air_yards_share")
          .fill_null(0)
          .shift(1)
          .rolling_mean(window_size=4)
          .over("player_id")
          .alias("air_yards_share_roll4"),

        # --- Fantasy points rolling (lagged) ---
        pl.col("fantasy_points")
          .shift(1)
          .rolling_mean(window_size=4)
          .over("player_id")
          .alias("fp_roll4_mean"),

        pl.col("fantasy_points")
          .shift(1)
          .rolling_mean(window_size=8)
          .over("player_id")
          .alias("fp_roll8_mean"),

        # --- Volume stats rolling (lagged) ---
        pl.col("targets")
          .fill_null(0)
          .shift(1)
          .rolling_mean(window_size=4)
          .over("player_id")
          .alias("targets_roll4"),

        pl.col("carries")
          .fill_null(0)
          .shift(1)
          .rolling_mean(window_size=4)
          .over("player_id")
          .alias("carries_roll4"),

        pl.col("receiving_yards")
          .fill_null(0)
          .shift(1)
          .rolling_mean(window_size=4)
          .over("player_id")
          .alias("rec_yards_roll4"),

        pl.col("rushing_yards")
          .fill_null(0)
          .shift(1)
          .rolling_mean(window_size=4)
          .over("player_id")
          .alias("rush_yards_roll4"),
    ])

    # Season-to-date average FP (expanding mean, lagged within season)
    # cum_mean was removed in polars >= 0.19; compute as cum_sum / cum_count
    shifted_fp = pl.col("fantasy_points").shift(1)
    df = df.with_columns([
        (shifted_fp.cum_sum().over(["player_id", "season"]) /
         shifted_fp.is_not_null().cast(pl.Int32).cum_sum().over(["player_id", "season"]))
        .alias("season_avg_fp")
    ])

    return df


def build_opponent_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build opponent defensive features.
    Computes how many fantasy points each team has allowed to each position,
    then takes a rolling 4-game average (lagged) for the opponent's defense.

    This captures matchup quality better than a simple season-long rank.
    """
    # FP allowed by defense: sum across all players of that position who played vs. this team
    opp_def = (
        df.group_by(["season", "week", "opponent_team", "position"])
        .agg(pl.col("fantasy_points").sum().alias("fp_allowed_raw"))
        .sort(["opponent_team", "position", "season", "week"])
        .with_columns(
            pl.col("fp_allowed_raw")
              .shift(1)
              .rolling_mean(window_size=4)
              .over(["opponent_team", "position"])
              .alias("opp_fp_allowed_pos_roll4")
        )
    )

    df = df.join(
        opp_def.select([
            "season", "week", "opponent_team", "position", "opp_fp_allowed_pos_roll4"
        ]),
        on=["season", "week", "opponent_team", "position"],
        how="left",
    )

    return df


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """Encode categorical columns as integers for model input."""
    pos_map = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    # All weekdays including less common ones
    day_map = {
        "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
        "Thursday": 4, "Friday": 5, "Saturday": 6,
    }

    df = df.with_columns(
        pl.col("position")
          .replace(pos_map)
          .cast(pl.Int32)
          .alias("position_enc")
    )

    if "weekday" in df.columns:
        df = df.with_columns(
            pl.col("weekday")
              .replace(day_map)
              .cast(pl.Int32, strict=False)
              .fill_null(0)
              .alias("weekday_enc")
        )

    return df


def build_features(
    stats: pl.DataFrame,
    schedules: pl.DataFrame | None = None,
    snaps: pl.DataFrame | None = None,
    id_map: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Full feature engineering pipeline.

    1. Join schedule features (home/away, Vegas lines, weather, rest)
    2. Join snap counts
    3. Build rolling features (all lagged)
    4. Build opponent defensive features (lagged)
    5. Encode categoricals
    """
    df = stats.clone()

    # Join schedule features if available
    if schedules is not None:
        df = join_schedule_features(df, schedules)
    else:
        # Fallback defaults
        df = df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("team_implied_total"),
            pl.lit(None).cast(pl.Float64).alias("spread_line"),
            pl.lit(None).cast(pl.Float64).alias("total_line"),
            pl.lit(None).cast(pl.Float64).alias("temp"),
            pl.lit(None).cast(pl.Float64).alias("wind"),
            pl.lit(None).cast(pl.Int8).alias("is_home"),
            pl.lit(None).cast(pl.Float64).alias("rest_days"),
            pl.lit(0).cast(pl.Int8).alias("is_outdoor"),
        ])

    # Join snap counts if available
    if snaps is not None:
        df = join_snap_counts(df, snaps, id_map=id_map)
    elif "snap_pct" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("snap_pct"))

    # Build rolling features
    df = build_rolling_features(df)

    # Build opponent defensive features
    df = build_opponent_features(df)

    # Encode categoricals
    df = encode_categoricals(df)

    return df


def get_feature_columns(position: str, has_vegas: bool = False, has_weather: bool = False) -> list[str]:
    """Return feature columns for a given position and available data."""
    base = [
        "fp_roll4_mean",
        "fp_roll8_mean",
        "season_avg_fp",
        "opp_fp_allowed_pos_roll4",
        "is_home",
        "week",
        "position_enc",
    ]

    # Snap pct — include if it exists (may be null early on)
    snap_features = ["snap_pct_roll3"]

    if position in ("WR", "TE"):
        pos_features = [
            "target_share_roll4",
            "targets_roll4",
            "rec_yards_roll4",
            "air_yards_share_roll4",
        ]
    elif position == "RB":
        pos_features = [
            "carries_roll4",
            "rush_yards_roll4",
            "targets_roll4",
            "target_share_roll4",
            "rec_yards_roll4",
        ]
    elif position == "QB":
        pos_features = [
            "carries_roll4",
            "rush_yards_roll4",
        ]
    else:
        pos_features = []

    features = base + snap_features + pos_features

    if has_vegas:
        features += ["team_implied_total", "spread_line"]

    if has_weather:
        features += ["temp", "wind", "is_outdoor"]

    # Rest/schedule
    if "rest_days" in features:
        pass  # already there
    else:
        features.append("rest_days")

    return features


def get_target_column() -> str:
    return "fantasy_points"


def drop_leaky_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows where key rolling features couldn't be computed (first few games)."""
    return df.filter(pl.col("fp_roll4_mean").is_not_null())
