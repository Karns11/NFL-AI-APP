"""
Player leaderboards for the Stats & Insights page.

Returns per-position season or recent-N-week leaderboards with usage metrics.
All queries run directly against DuckDB.
"""

import logging
from pathlib import Path

import duckdb

from src.stats import cache as stats_cache

logger = logging.getLogger(__name__)

DB_PATH = Path("data/db/nfl.duckdb")

POSITIONS = ("QB", "RB", "WR", "TE")
MIN_GAMES = 3


def _get_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def get_leaderboard(
    position: str,
    season: int,
    week: int,
    n_weeks: int | None = None,
) -> list[dict]:
    """
    Return a ranked player leaderboard for the given position.

    n_weeks=None  → season-to-date through `week`
    n_weeks=4     → only the last 4 weeks ending at `week`
    """
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")

    cache_key = ("leaderboard", position, season, week, n_weeks)
    cached = stats_cache.get("leaderboard", True, position, season, week, n_weeks or 0)
    if cached is not None:
        return cached

    try:
        db = _get_db()
        result = _query_leaderboard(db, position, season, week, n_weeks)
        db.close()
    except Exception as e:
        logger.error(f"Leaderboard query failed for {position} s{season}w{week}: {e}")
        return []

    stats_cache.set("leaderboard", result, position, season, week, n_weeks or 0)
    return result


def _week_filter(week: int, n_weeks: int | None) -> tuple[int, int]:
    """Return (week_min, week_max) for the query."""
    if n_weeks is None:
        return 1, week
    return max(1, week - n_weeks + 1), week


def _query_leaderboard(
    db: duckdb.DuckDBPyConnection,
    position: str,
    season: int,
    week: int,
    n_weeks: int | None,
) -> list[dict]:
    week_min, week_max = _week_filter(week, n_weeks)

    if position == "WR":
        return _leaderboard_wr(db, season, week_min, week_max)
    elif position == "TE":
        return _leaderboard_te(db, season, week_min, week_max)
    elif position == "RB":
        return _leaderboard_rb(db, season, week_min, week_max)
    elif position == "QB":
        return _leaderboard_qb(db, season, week_min, week_max)
    return []


# ── Shared snap% CTE builder ─────────────────────────────────────────────────

def _snap_cte(position: str) -> str:
    """Returns a snap_avg CTE with position hardcoded (avoids param ordering issues)."""
    return f"""
    snap_avg AS (
        SELECT p.player_id, ROUND(AVG(s.offense_pct) * 100, 1) AS snap_pct
        FROM player_stats_weekly p
        JOIN player_id_map m ON m.gsis_id = p.player_id
        JOIN snap_counts s ON s.pfr_player_id = m.pfr_id
            AND s.season = p.season AND s.week = p.week
        WHERE p.season = ? AND p.week BETWEEN ? AND ?
          AND p.position = '{position}' AND p.season_type = 'REG'
        GROUP BY p.player_id
    )
"""


def _build_row(row, keys) -> dict:
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in zip(keys, row)}


# ── WR ────────────────────────────────────────────────────────────────────────

def _leaderboard_wr(db, season, week_min, week_max):
    rows = db.execute(f"""
        WITH {_snap_cte('WR')},
        base AS (
            SELECT
                p.player_id,
                MAX(p.player_display_name)               AS name,
                MAX(p.team)                              AS team,
                MAX(p.headshot_url)                      AS headshot_url,
                COUNT(DISTINCT p.game_id)                AS games,
                ROUND(SUM(p.fantasy_points_ppr), 1)      AS fp_total,
                ROUND(AVG(p.fantasy_points_ppr), 1)      AS fp_avg,
                ROUND(AVG(p.targets), 1)                 AS tgt_avg,
                ROUND(AVG(p.receptions), 1)              AS rec_avg,
                ROUND(AVG(p.target_share) * 100, 1)      AS tgt_share,
                ROUND(AVG(p.wopr), 3)                    AS wopr,
                ROUND(AVG(p.receiving_yards), 1)         AS rec_yds_avg
            FROM player_stats_weekly p
            WHERE p.season = ? AND p.week BETWEEN ? AND ?
              AND p.position = 'WR' AND p.season_type = 'REG'
            GROUP BY p.player_id
            HAVING COUNT(DISTINCT p.game_id) >= ?
        )
        SELECT b.*, sa.snap_pct
        FROM base b
        LEFT JOIN snap_avg sa ON sa.player_id = b.player_id
        ORDER BY fp_total DESC
    """, [season, week_min, week_max,
          season, week_min, week_max,
          MIN_GAMES]).fetchall()

    keys = ["player_id","name","team","headshot_url","games","fp_total","fp_avg",
            "tgt_avg","rec_avg","tgt_share","wopr","rec_yds_avg","snap_pct"]
    result = []
    for i, row in enumerate(rows, start=1):
        d = _build_row(row, keys)
        d["rank"] = i
        result.append(d)
    return result


# ── TE ────────────────────────────────────────────────────────────────────────

def _leaderboard_te(db, season, week_min, week_max):
    rows = db.execute(f"""
        WITH {_snap_cte('TE')},
        base AS (
            SELECT
                p.player_id,
                MAX(p.player_display_name)               AS name,
                MAX(p.team)                              AS team,
                MAX(p.headshot_url)                      AS headshot_url,
                COUNT(DISTINCT p.game_id)                AS games,
                ROUND(SUM(p.fantasy_points_ppr), 1)      AS fp_total,
                ROUND(AVG(p.fantasy_points_ppr), 1)      AS fp_avg,
                ROUND(AVG(p.targets), 1)                 AS tgt_avg,
                ROUND(AVG(p.receptions), 1)              AS rec_avg,
                ROUND(AVG(p.target_share) * 100, 1)      AS tgt_share,
                ROUND(AVG(p.wopr), 3)                    AS wopr,
                ROUND(AVG(p.receiving_yards), 1)         AS rec_yds_avg
            FROM player_stats_weekly p
            WHERE p.season = ? AND p.week BETWEEN ? AND ?
              AND p.position = 'TE' AND p.season_type = 'REG'
            GROUP BY p.player_id
            HAVING COUNT(DISTINCT p.game_id) >= ?
        )
        SELECT b.*, sa.snap_pct
        FROM base b
        LEFT JOIN snap_avg sa ON sa.player_id = b.player_id
        ORDER BY fp_total DESC
    """, [season, week_min, week_max,
          season, week_min, week_max,
          MIN_GAMES]).fetchall()

    keys = ["player_id","name","team","headshot_url","games","fp_total","fp_avg",
            "tgt_avg","rec_avg","tgt_share","wopr","rec_yds_avg","snap_pct"]
    result = []
    for i, row in enumerate(rows, start=1):
        d = _build_row(row, keys)
        d["rank"] = i
        result.append(d)
    return result


# ── RB ────────────────────────────────────────────────────────────────────────

def _leaderboard_rb(db, season, week_min, week_max):
    rows = db.execute(f"""
        WITH {_snap_cte('RB')},
        base AS (
            SELECT
                p.player_id,
                MAX(p.player_display_name)               AS name,
                MAX(p.team)                              AS team,
                MAX(p.headshot_url)                      AS headshot_url,
                COUNT(DISTINCT p.game_id)                AS games,
                ROUND(SUM(p.fantasy_points_ppr), 1)      AS fp_total,
                ROUND(AVG(p.fantasy_points_ppr), 1)      AS fp_avg,
                ROUND(AVG(p.carries), 1)                 AS carries_avg,
                ROUND(AVG(p.rushing_yards), 1)           AS rush_yds_avg,
                ROUND(AVG(p.targets), 1)                 AS tgt_avg,
                ROUND(AVG(p.receiving_yards), 1)         AS rec_yds_avg
            FROM player_stats_weekly p
            WHERE p.season = ? AND p.week BETWEEN ? AND ?
              AND p.position = 'RB' AND p.season_type = 'REG'
            GROUP BY p.player_id
            HAVING COUNT(DISTINCT p.game_id) >= ?
        )
        SELECT b.*, sa.snap_pct
        FROM base b
        LEFT JOIN snap_avg sa ON sa.player_id = b.player_id
        ORDER BY fp_total DESC
    """, [season, week_min, week_max,
          season, week_min, week_max,
          MIN_GAMES]).fetchall()

    keys = ["player_id","name","team","headshot_url","games","fp_total","fp_avg",
            "carries_avg","rush_yds_avg","tgt_avg","rec_yds_avg","snap_pct"]
    result = []
    for i, row in enumerate(rows, start=1):
        d = _build_row(row, keys)
        d["rank"] = i
        result.append(d)
    return result


# ── QB ────────────────────────────────────────────────────────────────────────

def _leaderboard_qb(db, season, week_min, week_max):
    rows = db.execute(f"""
        WITH {_snap_cte('QB')},
        base AS (
            SELECT
                p.player_id,
                MAX(p.player_display_name)               AS name,
                MAX(p.team)                              AS team,
                MAX(p.headshot_url)                      AS headshot_url,
                COUNT(DISTINCT p.game_id)                AS games,
                ROUND(SUM(p.fantasy_points_ppr), 1)      AS fp_total,
                ROUND(AVG(p.fantasy_points_ppr), 1)      AS fp_avg,
                ROUND(AVG(p.attempts), 1)                AS att_avg,
                ROUND(AVG(p.completions), 1)             AS cmp_avg,
                ROUND(AVG(p.passing_yards), 1)           AS pass_yds_avg,
                ROUND(AVG(p.passing_tds), 2)             AS pass_td_avg,
                ROUND(AVG(p.passing_interceptions), 2)   AS int_avg,
                ROUND(AVG(p.rushing_yards), 1)           AS rush_yds_avg
            FROM player_stats_weekly p
            WHERE p.season = ? AND p.week BETWEEN ? AND ?
              AND p.position = 'QB' AND p.season_type = 'REG'
            GROUP BY p.player_id
            HAVING COUNT(DISTINCT p.game_id) >= ?
        )
        SELECT b.*, sa.snap_pct
        FROM base b
        LEFT JOIN snap_avg sa ON sa.player_id = b.player_id
        ORDER BY fp_total DESC
    """, [season, week_min, week_max,
          season, week_min, week_max,
          MIN_GAMES]).fetchall()

    keys = ["player_id","name","team","headshot_url","games","fp_total","fp_avg",
            "att_avg","cmp_avg","pass_yds_avg","pass_td_avg","int_avg","rush_yds_avg","snap_pct"]
    result = []
    for i, row in enumerate(rows, start=1):
        d = _build_row(row, keys)
        d["rank"] = i
        result.append(d)
    return result
