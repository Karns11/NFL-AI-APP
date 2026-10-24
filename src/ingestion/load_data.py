"""
Data loading from nflreadpy into DuckDB.
Loads weekly player stats, snap counts, and schedules for feature engineering.
"""

import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

DB_PATH = Path("data/db/nfl.duckdb")
RAW_PATH = Path("data/raw")


def get_db() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def _existing_seasons(db: duckdb.DuckDBPyConnection, table: str) -> set[int]:
    try:
        result = db.execute(f"SELECT DISTINCT season FROM {table}").fetchall()
        return {r[0] for r in result}
    except duckdb.CatalogException:
        return set()


def _upsert(db: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> None:
    existing = _existing_seasons(db, table)
    if existing:
        # Delete the seasons we're about to re-insert so we don't duplicate rows
        seasons_in_df = df.select("season").unique()["season"].to_list()
        for s in seasons_in_df:
            db.execute(f"DELETE FROM {table} WHERE season = {s}")
        db.execute(f"INSERT INTO {table} SELECT * FROM df")
    else:
        db.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")


def load_player_stats(seasons: list[int], force_refresh: bool = False) -> pl.DataFrame:
    """Load weekly player stats (offensive skill positions) from nflreadpy."""
    import nflreadpy as nfl

    db = get_db()
    existing = _existing_seasons(db, "player_stats_weekly")
    seasons_to_load = [s for s in seasons if s not in existing or force_refresh]

    if seasons_to_load:
        logger.info(f"Loading player stats for seasons: {seasons_to_load}")
        frames = []
        for season in seasons_to_load:
            logger.info(f"  Fetching {season}...")
            try:
                df = nfl.load_player_stats(season)
                # Keep only offensive skill positions
                df = df.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
                frames.append(df)
            except Exception as e:
                logger.warning(f"  Failed season {season}: {e}")
        if frames:
            new_data = pl.concat(frames, how="diagonal_relaxed")
            _upsert(db, "player_stats_weekly", new_data)
            logger.info(f"  Saved {len(new_data):,} rows.")

    return pl.from_pandas(db.execute(
        "SELECT * FROM player_stats_weekly ORDER BY season, week, player_id"
    ).fetchdf())


def load_snap_counts(seasons: list[int], force_refresh: bool = False) -> pl.DataFrame:
    """Load weekly snap counts from nflreadpy."""
    import nflreadpy as nfl

    db = get_db()
    existing = _existing_seasons(db, "snap_counts")
    seasons_to_load = [s for s in seasons if s not in existing or force_refresh]

    if seasons_to_load:
        logger.info(f"Loading snap counts for seasons: {seasons_to_load}")
        frames = []
        for season in seasons_to_load:
            logger.info(f"  Fetching snaps {season}...")
            try:
                df = nfl.load_snap_counts(season)
                df = df.with_columns(pl.lit(season).alias("season"))
                frames.append(df)
            except Exception as e:
                logger.warning(f"  Failed snaps {season}: {e}")
        if frames:
            new_data = pl.concat(frames, how="diagonal_relaxed")
            _upsert(db, "snap_counts", new_data)

    return pl.from_pandas(db.execute("SELECT * FROM snap_counts ORDER BY season, week").fetchdf())


def load_schedules(seasons: list[int], force_refresh: bool = False) -> pl.DataFrame:
    """
    Load NFL schedules. Includes: home/away teams, spread_line, total_line,
    away_rest/home_rest, temp, wind, roof, weekday.
    """
    import nflreadpy as nfl

    db = get_db()
    existing = _existing_seasons(db, "schedules")
    seasons_to_load = [s for s in seasons if s not in existing or force_refresh]

    if seasons_to_load:
        logger.info(f"Loading schedules for seasons: {seasons_to_load}")
        frames = []
        for season in seasons_to_load:
            try:
                df = nfl.load_schedules(season)
                frames.append(df)
            except Exception as e:
                logger.warning(f"  Failed schedules {season}: {e}")
        if frames:
            new_data = pl.concat(frames, how="diagonal_relaxed")
            _upsert(db, "schedules", new_data)

    return pl.from_pandas(db.execute("SELECT * FROM schedules ORDER BY season, week").fetchdf())


def load_player_id_map() -> pl.DataFrame:
    """
    Load player ID mapping table from nflreadpy.
    Maps gsis_id (player_id in stats) ↔ pfr_id (pfr_player_id in snap counts).
    """
    import nflreadpy as nfl

    db = get_db()
    try:
        count = db.execute("SELECT COUNT(*) FROM player_id_map").fetchone()[0]
        if count > 0:
            return pl.from_pandas(db.execute("SELECT * FROM player_id_map").fetchdf())
    except duckdb.CatalogException:
        pass

    players = nfl.load_players()
    id_map = players.select(["gsis_id", "pfr_id"]).drop_nulls()
    db.execute("CREATE OR REPLACE TABLE player_id_map AS SELECT * FROM id_map")
    return id_map


def build_fullname_to_gsis_map() -> dict[str, str]:
    """
    Build a lookup from full player name → gsis_id using nflreadpy.load_players().
    Keys are lowercased for case-insensitive matching.
    e.g. {"bijan robinson": "00-0038542", "brian robinson": "00-0037746"}

    This is the preferred player resolution method — bypasses abbreviated name
    ambiguity (e.g. "B.Robinson" could be Bijan or Brian).
    """
    import nflreadpy as nfl

    players = nfl.load_players()
    name_to_id: dict[str, str] = {}
    for row in players.iter_rows(named=True):
        gsis_id = row.get("gsis_id")
        if not gsis_id:
            continue
        for name_field in ("display_name", "football_name", "first_name"):
            full = row.get(name_field, "") or ""
            last = row.get("last_name", "") or ""
            if full and last:
                combined = f"{full} {last}".strip() if name_field == "first_name" else full
                name_to_id[combined.lower()] = gsis_id
    return name_to_id


def refresh_current_season(season: int) -> dict[str, int]:
    """
    Re-fetch player stats, snap counts, and schedules for the given season,
    replacing any existing rows for that season. Call this on Monday after
    the week's games finish to keep the feature data current.

    Returns a dict of table → new row count.
    """
    counts = {}
    try:
        df = load_player_stats([season], force_refresh=True)
        counts["player_stats_weekly"] = len(df.filter(__import__("polars").col("season") == season))
    except Exception as e:
        logger.warning(f"player_stats refresh failed: {e}")
        counts["player_stats_weekly"] = 0

    try:
        df = load_snap_counts([season], force_refresh=True)
        counts["snap_counts"] = len(df.filter(__import__("polars").col("season") == season))
    except Exception as e:
        logger.warning(f"snap_counts refresh failed: {e}")
        counts["snap_counts"] = 0

    try:
        df = load_schedules([season], force_refresh=True)
        counts["schedules"] = len(df.filter(__import__("polars").col("season") == season))
    except Exception as e:
        logger.warning(f"schedules refresh failed: {e}")
        counts["schedules"] = 0

    return counts


def get_table_info() -> dict:
    """Return row counts for all loaded tables."""
    db = get_db()
    tables = {}
    for table in ["player_stats_weekly", "snap_counts", "schedules"]:
        try:
            count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            season_range = db.execute(
                f"SELECT MIN(season), MAX(season) FROM {table}"
            ).fetchone()
            tables[table] = {
                "rows": count,
                "season_min": season_range[0],
                "season_max": season_range[1],
            }
        except duckdb.CatalogException:
            tables[table] = {"rows": 0, "status": "not loaded"}
    return tables
