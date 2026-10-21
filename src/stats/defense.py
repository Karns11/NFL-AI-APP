"""
Defense vs. Position (DvP) stats.

Queries player_stats_weekly to compute how many fantasy points each
NFL defense allows per game to each position (QB/RB/WR/TE), with:
  - Season-to-date average and rank (1=most generous, 32=most stingy)
  - Last-N-game average (default 4)
  - Home / Away split
  - Sample size (games played)

All queries run directly against DuckDB — no new tables needed.
Results are cached via src/stats/cache.py.
"""

import logging
from pathlib import Path

import duckdb

from src.stats import cache as stats_cache

logger = logging.getLogger(__name__)

DB_PATH = Path("data/db/nfl.duckdb")

POSITIONS = ("QB", "RB", "WR", "TE")

# Sleeper/nflreadpy team abbreviation → ESPN CDN slug
# ESPN uses lowercase 2–3 letter codes; a few differ from nflreadpy abbrs.
TEAM_LOGO_SLUG: dict[str, str] = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf",
    "CAR": "car", "CHI": "chi", "CIN": "cin", "CLE": "cle",
    "DAL": "dal", "DEN": "den", "DET": "det", "GB":  "gb",
    "HOU": "hou", "IND": "ind", "JAX": "jax", "KC":  "kc",
    "LAC": "lac", "LAR": "lar", "LV":  "lv",  "MIA": "mia",
    "MIN": "min", "NE":  "ne",  "NO":  "no",  "NYG": "nyg",
    "NYJ": "nyj", "PHI": "phi", "PIT": "pit", "SEA": "sea",
    "SF":  "sf",  "TB":  "tb",  "TEN": "ten", "WSH": "wsh",
    # Legacy abbreviations that may appear in older data
    "JAC": "jax", "OAK": "lv", "STL": "lar", "SD": "lac",
}

ESPN_LOGO_BASE = "https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"


def team_logo_url(team: str) -> str:
    slug = TEAM_LOGO_SLUG.get(team, team.lower())
    return ESPN_LOGO_BASE.format(slug=slug)


def _get_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def get_dvp_table(
    position: str,
    season: int,
    week: int,
    recent_n: int = 4,
) -> list[dict]:
    """
    Return a list of dicts, one per team (32 entries), ranked by season-to-date
    FP allowed per game to the given position through the given week.

    Each dict has:
      team, games, fp_per_game, rank,
      recent_fp_per_game (last recent_n games),
      home_fp_per_game, away_fp_per_game,
      logo_url
    """
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")

    is_current = True  # caller can pass False for past weeks if needed
    cached = stats_cache.get("dvp", is_current, position, season, week, recent_n)
    if cached is not None:
        return cached

    try:
        db = _get_db()
        result = _query_dvp(db, position, season, week, recent_n)
        db.close()
    except Exception as e:
        logger.error(f"DvP query failed for {position} s{season}w{week}: {e}")
        return []

    stats_cache.set("dvp", result, position, season, week, recent_n)
    return result


def _query_dvp(
    db: duckdb.DuckDBPyConnection,
    position: str,
    season: int,
    week: int,
    recent_n: int,
) -> list[dict]:
    # ── Season-to-date FP allowed per team ───────────────────────────────────
    season_rows = db.execute("""
        SELECT
            p.opponent_team                                  AS team,
            COUNT(DISTINCT p.game_id)                        AS games,
            SUM(p.fantasy_points_ppr)                        AS fp_total,
            SUM(p.fantasy_points_ppr) / COUNT(DISTINCT p.game_id) AS fp_per_game
        FROM player_stats_weekly p
        WHERE p.season = ?
          AND p.week   <= ?
          AND p.position = ?
          AND p.season_type = 'REG'
          AND p.opponent_team IS NOT NULL
        GROUP BY p.opponent_team
        ORDER BY fp_per_game DESC
    """, [season, week, position]).fetchall()

    # Build ranked lookup
    season_data: dict[str, dict] = {}
    for rank, (team, games, fp_total, fp_pg) in enumerate(season_rows, start=1):
        season_data[team] = {
            "team": team,
            "games": int(games),
            "fp_per_game": round(float(fp_pg), 1),
            "rank": rank,
            "logo_url": team_logo_url(team),
        }

    # ── Recent N games ────────────────────────────────────────────────────────
    recent_rows = db.execute("""
        WITH ranked AS (
            SELECT
                p.opponent_team AS team,
                p.game_id,
                p.week,
                SUM(p.fantasy_points_ppr) OVER (PARTITION BY p.opponent_team, p.game_id) AS game_fp,
                ROW_NUMBER() OVER (PARTITION BY p.opponent_team ORDER BY p.season DESC, p.week DESC) AS rn
            FROM player_stats_weekly p
            WHERE p.season = ?
              AND p.week   <= ?
              AND p.position = ?
              AND p.season_type = 'REG'
              AND p.opponent_team IS NOT NULL
        ),
        recent_games AS (
            SELECT team, game_id, game_fp
            FROM ranked
            WHERE rn <= ?
            GROUP BY team, game_id, game_fp
        )
        SELECT team, AVG(game_fp) AS recent_fp_per_game
        FROM recent_games
        GROUP BY team
    """, [season, week, position, recent_n]).fetchall()

    recent_data = {team: round(float(fp), 1) for team, fp in recent_rows}

    # ── Home / Away split ─────────────────────────────────────────────────────
    split_rows = db.execute("""
        SELECT
            p.opponent_team AS team,
            s.home_team,
            SUM(p.fantasy_points_ppr)      AS fp_total,
            COUNT(DISTINCT p.game_id)      AS games
        FROM player_stats_weekly p
        JOIN schedules s ON s.game_id = p.game_id
        WHERE p.season = ?
          AND p.week   <= ?
          AND p.position = ?
          AND p.season_type = 'REG'
          AND p.opponent_team IS NOT NULL
        GROUP BY p.opponent_team, s.home_team
    """, [season, week, position]).fetchall()

    home_fp: dict[str, list] = {}
    away_fp: dict[str, list] = {}
    for team, home_team, fp_total, games in split_rows:
        # When the opponent_team is the home_team, the offense played AT that defense (away game for offense)
        # i.e. home_team IS the defense → this is a home game for the defense
        bucket = home_fp if team == home_team else away_fp
        bucket.setdefault(team, [0.0, 0])
        bucket[team][0] += float(fp_total)
        bucket[team][1] += int(games)

    def _avg(d: dict, team: str) -> float | None:
        if team not in d or d[team][1] == 0:
            return None
        return round(d[team][0] / d[team][1], 1)

    # ── Merge ─────────────────────────────────────────────────────────────────
    result = []
    for team, row in season_data.items():
        row["recent_fp_per_game"] = recent_data.get(team)
        row["home_fp_per_game"]   = _avg(home_fp, team)
        row["away_fp_per_game"]   = _avg(away_fp, team)
        result.append(row)

    # Sort by rank
    result.sort(key=lambda r: r["rank"])
    return result


def get_team_dvp_rank(
    team: str,
    position: str,
    season: int,
    week: int,
) -> int | None:
    """
    Return the 1–32 rank of team's defense against position through the given week.
    1 = most FP allowed (easiest matchup), 32 = fewest allowed (toughest).
    Returns None if data unavailable.
    """
    table = get_dvp_table(position, season, week)
    for row in table:
        if row["team"] == team:
            return row["rank"]
    return None
