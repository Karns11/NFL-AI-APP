"""
Player trend data for the Stats & Insights page.

Returns week-by-week fantasy points for a player, plus a floor/ceiling band
derived from their historical game scores.

To remove this feature: delete this file, the /api/stats/trends and
/api/stats/player-search routes in routes.py, and clear the Trends tab
in stats.html.
"""

import logging
from pathlib import Path

import duckdb

from src.stats import cache as stats_cache

logger = logging.getLogger(__name__)

DB_PATH = Path("data/db/nfl.duckdb")


def _get_db():
    return duckdb.connect(str(DB_PATH), read_only=True)


def search_players(query: str, season: int) -> list[dict]:
    """
    Return up to 10 players whose name contains `query` (case-insensitive),
    filtered to players who have appeared in `season`.
    """
    if not query or len(query) < 2:
        return []
    try:
        db = _get_db()
        rows = db.execute("""
            SELECT DISTINCT
                player_id,
                player_display_name AS name,
                position,
                team,
                MAX(headshot_url) AS headshot_url
            FROM player_stats_weekly
            WHERE season = ?
              AND season_type = 'REG'
              AND position IN ('QB','RB','WR','TE')
              AND LOWER(player_display_name) LIKE ?
            GROUP BY player_id, player_display_name, position, team
            ORDER BY player_display_name
            LIMIT 10
        """, [season, f"%{query.lower()}%"]).fetchall()
        db.close()
    except Exception as e:
        logger.error(f"Player search failed: {e}")
        return []

    return [
        {"player_id": r[0], "name": r[1], "position": r[2],
         "team": r[3], "headshot_url": r[4]}
        for r in rows
    ]


def get_player_trend(player_id: str, season: int) -> dict:
    """
    Return week-by-week FP data for a player plus summary stats.

    Returns:
      { name, position, team, headshot_url,
        series: [{week, fp, opponent_team}],
        season_avg, floor (10th pct), ceiling (90th pct) }
    """
    cached = stats_cache.get("player_trend", True, player_id, season)
    if cached is not None:
        return cached

    try:
        db = _get_db()

        # Player info + weekly scores
        rows = db.execute("""
            SELECT
                week,
                ROUND(fantasy_points_ppr, 2) AS fp,
                opponent_team,
                player_display_name,
                position,
                team,
                headshot_url
            FROM player_stats_weekly
            WHERE player_id = ?
              AND season = ?
              AND season_type = 'REG'
            ORDER BY week
        """, [player_id, season]).fetchall()
        db.close()
    except Exception as e:
        logger.error(f"Player trend query failed: {e}")
        return {}

    if not rows:
        return {}

    name        = rows[0][3]
    position    = rows[0][4]
    team        = rows[0][5]
    headshot    = rows[0][6]
    series      = [{"week": r[0], "fp": r[1], "opp": r[2]} for r in rows]
    fps         = [r[1] for r in rows]
    season_avg  = round(sum(fps) / len(fps), 1)

    sorted_fps  = sorted(fps)
    n           = len(sorted_fps)
    floor_val   = round(sorted_fps[max(0, int(n * 0.10))], 1)
    ceiling_val = round(sorted_fps[min(n - 1, int(n * 0.90))], 1)

    result = {
        "player_id":   player_id,
        "name":        name,
        "position":    position,
        "team":        team,
        "headshot_url": headshot,
        "series":      series,
        "season_avg":  season_avg,
        "floor":       floor_val,
        "ceiling":     ceiling_val,
        "games":       n,
    }

    stats_cache.set("player_trend", result, player_id, season)
    return result
