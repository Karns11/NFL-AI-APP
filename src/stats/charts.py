"""
Quadrant chart data builders for the Stats & Insights page.

Each spec in CHART_SPECS is self-contained. To remove a chart: delete its
entry from CHART_SPECS. To remove the whole feature: delete this file,
the /api/stats/chart-data route in routes.py, and clear the tab in stats.html.

All queries run directly against DuckDB.
"""

import logging
from pathlib import Path

import duckdb

from src.stats import cache as stats_cache

logger = logging.getLogger(__name__)

DB_PATH = Path("data/db/nfl.duckdb")
MIN_GAMES = 3


def _get_db():
    return duckdb.connect(str(DB_PATH), read_only=True)


# ── Spec registry ─────────────────────────────────────────────────────────────
# Each entry defines the chart metadata shown in the UI.
# 'fn' is the builder function name (resolved below after definitions).

CHART_SPECS = {
    "defense-pass-vs-rush": {
        "title": "Defense: Pass vs. Rush Allowed",
        "x_label": "Pass FP/G Allowed (WR + TE)",
        "y_label": "Rush FP/G Allowed (RB)",
        "entity": "team",
        "quadrants": {
            "tl": "Run-heavy matchup",
            "tr": "Exploit everything",
            "bl": "Avoid",
            "br": "Pass-heavy matchup",
        },
    },
    "wr-volume-vs-efficiency": {
        "title": "WR: Volume vs. Efficiency",
        "x_label": "Target Share %",
        "y_label": "FP / Game",
        "entity": "player",
        "quadrants": {
            "tl": "Efficient, low volume",
            "tr": "Elite",
            "bl": "Avoid",
            "br": "High volume, underperforming",
        },
    },
    "rb-opportunity-vs-efficiency": {
        "title": "RB: Opportunity vs. Efficiency",
        "x_label": "Touches / Game (Car + Tgt)",
        "y_label": "FP / Game",
        "entity": "player",
        "quadrants": {
            "tl": "Efficient, low touches",
            "tr": "Workhorse",
            "bl": "Low floor",
            "br": "Volume, not converting",
        },
    },
    "qb-volume-vs-efficiency": {
        "title": "QB: Volume vs. Efficiency",
        "x_label": "Attempts / Game",
        "y_label": "Passing EPA / Game",
        "entity": "player",
        "quadrants": {
            "tl": "Efficient, game manager",
            "tr": "Elite passer",
            "bl": "Avoid",
            "br": "High volume, inefficient",
        },
    },
    "te-volume-vs-efficiency": {
        "title": "TE: Volume vs. Efficiency",
        "x_label": "Target Share %",
        "y_label": "FP / Game",
        "entity": "player",
        "quadrants": {
            "tl": "Efficient, limited role",
            "tr": "Elite TE1",
            "bl": "Streamer range",
            "br": "Targets not converting",
        },
    },
}


def get_chart_data(spec_id: str, season: int, week: int) -> dict:
    """
    Return { spec metadata + points: [{x, y, label, sublabel, image_url}] }
    for the given spec through `week`.
    """
    if spec_id not in CHART_SPECS:
        raise ValueError(f"Unknown chart spec: {spec_id}")

    cached = stats_cache.get("chart", True, spec_id, season, week)
    if cached is not None:
        return cached

    spec = CHART_SPECS[spec_id]
    try:
        db = _get_db()
        builders = {
            "defense-pass-vs-rush":       _build_defense_pass_vs_rush,
            "wr-volume-vs-efficiency":    _build_wr_volume_efficiency,
            "rb-opportunity-vs-efficiency": _build_rb_opportunity_efficiency,
            "qb-volume-vs-efficiency":    _build_qb_volume_efficiency,
            "te-volume-vs-efficiency":    _build_te_volume_efficiency,
        }
        points = builders[spec_id](db, season, week)
        db.close()
    except Exception as e:
        logger.error(f"Chart build failed for {spec_id}: {e}")
        return {}

    result = {**spec, "spec_id": spec_id, "points": points}
    stats_cache.set("chart", result, spec_id, season, week)
    return result


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_defense_pass_vs_rush(db, season, week):
    from src.stats.defense import team_logo_url
    rows = db.execute("""
        WITH game_fp AS (
            SELECT
                opponent_team AS team,
                game_id,
                SUM(CASE WHEN position IN ('WR','TE') THEN fantasy_points_ppr ELSE 0 END) AS pass_fp,
                SUM(CASE WHEN position = 'RB'         THEN fantasy_points_ppr ELSE 0 END) AS rush_fp
            FROM player_stats_weekly
            WHERE season = ? AND week <= ? AND season_type = 'REG'
              AND position IN ('WR','TE','RB') AND opponent_team IS NOT NULL
            GROUP BY opponent_team, game_id
        )
        SELECT team,
               ROUND(AVG(pass_fp), 1) AS x,
               ROUND(AVG(rush_fp), 1) AS y,
               COUNT(*) AS games
        FROM game_fp
        GROUP BY team
        ORDER BY team
    """, [season, week]).fetchall()

    return [
        {
            "x": row[1], "y": row[2],
            "label": row[0],
            "sublabel": f"{row[3]}G",
            "image_url": team_logo_url(row[0]),
            "team": row[0],
        }
        for row in rows
    ]


def _build_wr_volume_efficiency(db, season, week):
    rows = db.execute("""
        SELECT
            player_display_name,
            team,
            ROUND(AVG(target_share) * 100, 1)     AS x,
            ROUND(AVG(fantasy_points_ppr), 1)      AS y,
            COUNT(DISTINCT game_id)                AS games,
            MAX(headshot_url)                      AS headshot_url
        FROM player_stats_weekly
        WHERE season = ? AND week <= ? AND position = 'WR' AND season_type = 'REG'
        GROUP BY player_id, player_display_name, team
        HAVING COUNT(DISTINCT game_id) >= ?
        ORDER BY y DESC
    """, [season, week, MIN_GAMES]).fetchall()

    return [
        {
            "x": row[2], "y": row[3],
            "label": row[0], "sublabel": row[1],
            "image_url": row[5],
            "games": row[4],
        }
        for row in rows if row[2] is not None
    ]


def _build_rb_opportunity_efficiency(db, season, week):
    rows = db.execute("""
        SELECT
            player_display_name,
            team,
            ROUND(AVG(carries + targets), 1)       AS x,
            ROUND(AVG(fantasy_points_ppr), 1)       AS y,
            COUNT(DISTINCT game_id)                 AS games,
            MAX(headshot_url)                       AS headshot_url
        FROM player_stats_weekly
        WHERE season = ? AND week <= ? AND position = 'RB' AND season_type = 'REG'
        GROUP BY player_id, player_display_name, team
        HAVING COUNT(DISTINCT game_id) >= ?
        ORDER BY y DESC
    """, [season, week, MIN_GAMES]).fetchall()

    return [
        {
            "x": row[2], "y": row[3],
            "label": row[0], "sublabel": row[1],
            "image_url": row[5],
            "games": row[4],
        }
        for row in rows if row[2] is not None
    ]


def _build_qb_volume_efficiency(db, season, week):
    rows = db.execute("""
        SELECT
            player_display_name,
            team,
            ROUND(AVG(attempts), 1)                AS x,
            ROUND(AVG(passing_epa), 2)             AS y,
            COUNT(DISTINCT game_id)                AS games,
            MAX(headshot_url)                      AS headshot_url
        FROM player_stats_weekly
        WHERE season = ? AND week <= ? AND position = 'QB' AND season_type = 'REG'
        GROUP BY player_id, player_display_name, team
        HAVING COUNT(DISTINCT game_id) >= ?
        ORDER BY y DESC
    """, [season, week, MIN_GAMES]).fetchall()

    return [
        {
            "x": row[2], "y": row[3],
            "label": row[0], "sublabel": row[1],
            "image_url": row[5],
            "games": row[4],
        }
        for row in rows if row[2] is not None
    ]


def _build_te_volume_efficiency(db, season, week):
    rows = db.execute("""
        SELECT
            player_display_name,
            team,
            ROUND(AVG(target_share) * 100, 1)     AS x,
            ROUND(AVG(fantasy_points_ppr), 1)      AS y,
            COUNT(DISTINCT game_id)                AS games,
            MAX(headshot_url)                      AS headshot_url
        FROM player_stats_weekly
        WHERE season = ? AND week <= ? AND position = 'TE' AND season_type = 'REG'
        GROUP BY player_id, player_display_name, team
        HAVING COUNT(DISTINCT game_id) >= ?
        ORDER BY y DESC
    """, [season, week, MIN_GAMES]).fetchall()

    return [
        {
            "x": row[2], "y": row[3],
            "label": row[0], "sublabel": row[1],
            "image_url": row[5],
            "games": row[4],
        }
        for row in rows if row[2] is not None
    ]
