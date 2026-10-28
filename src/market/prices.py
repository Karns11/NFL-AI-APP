"""
Player Market — price computation and market data.

Price formula (VOR-based):
  recent_vor = ewm(fp_ppr, alpha=0.40) - REPLACEMENT_FP[position]
  price      = clamp(100 + recent_vor * 5, low=1, high=200)

This isolates positional replacement value so prices are comparable across
positions: a $100 player is replacement-level at their spot, $150 is elite,
$50 is a bench stash.

Strategy agents (surfaced in UI as "market bids"):
  MOMENTUM   — EWM of recent FP (drives price)
  FAIR VALUE — single-week FP contribution vs replacement
  CONSISTENCY — stability grade from CV of weekly scores
"""

import logging
import statistics
from collections import defaultdict
from pathlib import Path

import duckdb

from src.stats import cache as stats_cache

logger = logging.getLogger(__name__)

DB_PATH = Path("data/db/nfl.duckdb")

REPLACEMENT_FP: dict[str, float] = {
    "QB": 18.0,
    "RB": 11.0,
    "WR": 10.0,
    "TE":  8.0,
}
EWM_ALPHA   = 0.40
PRICE_SCALE = 5.0   # VOR × scale = price delta from $100


def _get_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def _fp_to_price(vor: float) -> float:
    return max(1.0, min(200.0, round(100.0 + vor * PRICE_SCALE, 2)))


def _compute_ewm(fp_values: list[float], alpha: float = EWM_ALPHA) -> float:
    """Exponentially weighted mean of a list; returns final value."""
    if not fp_values:
        return 0.0
    result = fp_values[0]
    for v in fp_values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


def _consistency_grade(fps: list[float]) -> str:
    """Return A+/A/B/C/D based on coefficient of variation."""
    if len(fps) < 3:
        return "—"
    mean = statistics.mean(fps)
    if mean <= 0:
        return "D"
    cv = statistics.stdev(fps) / mean
    if cv < 0.30:
        return "A+"
    if cv < 0.50:
        return "A"
    if cv < 0.70:
        return "B"
    if cv < 1.00:
        return "C"
    return "D"


def _make_ticker(player_name: str) -> str:
    """Generate a short ticker from the player's last name (up to 4 chars)."""
    parts = player_name.split()
    last = parts[-1] if parts else player_name
    last = last.replace("-", "").replace("'", "")
    return last[:4].upper()


# ── Market overview ───────────────────────────────────────────────────────────

def get_market_overview(season: int, week: int, position: str = "ALL") -> dict:
    """
    Returns all active players with price, week-over-week change, and stats.
    Cached per (season, week, position).
    """
    cached = stats_cache.get("market_overview", True, season, week, position)
    if cached is not None:
        return cached

    pos_clause = (
        "AND position IN ('QB','RB','WR','TE')"
        if position == "ALL"
        else f"AND position = '{position}'"
    )

    try:
        db = _get_db()
        rows = db.execute(f"""
            SELECT
                player_id,
                player_display_name,
                position,
                team,
                headshot_url,
                week,
                COALESCE(fantasy_points_ppr, 0.0) AS fp
            FROM player_stats_weekly
            WHERE season = ?
              AND season_type = 'REG'
              {pos_clause}
            ORDER BY player_id, week
        """, [season]).fetchall()
        db.close()
    except Exception as e:
        logger.error(f"Market overview query failed: {e}")
        return {"players": [], "gainers": [], "losers": []}

    # Group by player
    player_weeks: dict[str, list[tuple[int, float]]] = defaultdict(list)
    player_meta: dict[str, dict] = {}
    for pid, name, pos, team, headshot, wk, fp in rows:
        player_weeks[pid].append((wk, fp))
        player_meta[pid] = {
            "player_id": pid,
            "name": name,
            "position": pos,
            "team": team,
            "headshot_url": headshot,
        }

    players = []
    for pid, wk_fps in player_weeks.items():
        if len(wk_fps) < 2:
            continue
        meta = player_meta[pid]
        pos = meta["position"]
        repl = REPLACEMENT_FP.get(pos, 10.0)

        fps = [fp for _, fp in sorted(wk_fps)]
        season_avg = round(sum(fps) / len(fps), 1)
        grade = _consistency_grade(fps)

        # Current price (EWM through all games)
        price = _fp_to_price(_compute_ewm(fps) - repl)
        # Previous price (EWM through all games except last)
        price_prev = _fp_to_price(_compute_ewm(fps[:-1]) - repl)

        change = round(price - price_prev, 2)
        change_pct = round(change / price_prev * 100 if price_prev > 0 else 0.0, 1)

        # Season high/low (compute EWM progressively)
        wk_prices = [_fp_to_price(_compute_ewm(fps[:i]) - repl) for i in range(1, len(fps) + 1)]
        season_high = round(max(wk_prices), 2)
        season_low  = round(min(wk_prices), 2)

        players.append({
            **meta,
            "ticker":       _make_ticker(meta["name"]),
            "price":        price,
            "change":       change,
            "change_pct":   change_pct,
            "season_avg_fp": season_avg,
            "games":        len(fps),
            "season_high":  season_high,
            "season_low":   season_low,
            "consistency":  grade,
        })

    players.sort(key=lambda p: p["price"], reverse=True)

    gainers = sorted(
        [p for p in players if p["change"] > 0],
        key=lambda p: p["change_pct"], reverse=True,
    )[:5]
    losers = sorted(
        [p for p in players if p["change"] < 0],
        key=lambda p: p["change_pct"],
    )[:5]

    result = {"players": players, "gainers": gainers, "losers": losers}
    stats_cache.set("market_overview", result, season, week, position)
    return result


# ── Player price history ──────────────────────────────────────────────────────

def get_player_price_history(player_id: str, season: int, num_seasons: int = 3) -> dict:
    """
    Multi-season weekly price history for a single player.
    Returns series + current-season summary stats.
    """
    cached = stats_cache.get("market_player", True, player_id, season, num_seasons)
    if cached is not None:
        return cached

    start_season = season - num_seasons + 1

    try:
        db = _get_db()
        rows = db.execute("""
            SELECT
                season,
                week,
                COALESCE(fantasy_points_ppr, 0.0) AS fp,
                opponent_team,
                player_display_name,
                position,
                team,
                headshot_url
            FROM player_stats_weekly
            WHERE player_id = ?
              AND season >= ?
              AND season_type = 'REG'
            ORDER BY season, week
        """, [player_id, start_season]).fetchall()
        db.close()
    except Exception as e:
        logger.error(f"Player price history query failed: {e}")
        return {}

    if not rows:
        return {}

    # Metadata from most recent row
    _, _, _, _, name, pos, team, headshot = rows[-1]
    repl = REPLACEMENT_FP.get(pos, 10.0)

    # Group by season
    season_weeks: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
    for szn, wk, fp, opp, _, _, _, _ in rows:
        season_weeks[szn].append((wk, fp, opp or ""))

    # Build cumulative EWM price series per season
    series: list[dict] = []
    for szn in sorted(season_weeks.keys()):
        week_data = sorted(season_weeks[szn])
        fps_so_far: list[float] = []
        for wk, fp, opp in week_data:
            fps_so_far.append(fp)
            price = _fp_to_price(_compute_ewm(fps_so_far) - repl)
            series.append({
                "season": szn,
                "week":   wk,
                "fp":     round(fp, 1),
                "price":  price,
                "opp":    opp,
            })

    if not series:
        return {}

    # Current-season stats
    curr_series = [s for s in series if s["season"] == season]
    curr_fps    = [s["fp"] for s in curr_series]
    curr_prices = [s["price"] for s in curr_series]

    current_price = curr_series[-1]["price"] if curr_series else series[-1]["price"]
    prev_price    = curr_series[-2]["price"] if len(curr_series) >= 2 else current_price
    change        = round(current_price - prev_price, 2)
    change_pct    = round(change / prev_price * 100 if prev_price > 0 else 0.0, 1)

    # Strategy agent bids (for the UI "market bids" panel)
    if curr_fps:
        ewm_val  = _compute_ewm(curr_fps)
        momentum_bid  = _fp_to_price(ewm_val - repl)
        last_fp_bid   = _fp_to_price(curr_fps[-1] - repl)
        avg_fp_bid    = _fp_to_price((sum(curr_fps) / len(curr_fps)) - repl)
    else:
        momentum_bid = last_fp_bid = avg_fp_bid = 100.0

    result = {
        "player_id":    player_id,
        "name":         name,
        "position":     pos,
        "team":         team,
        "headshot_url": headshot,
        "ticker":       _make_ticker(name),
        "series":       series,
        "current_price":    current_price,
        "price_change":     change,
        "price_change_pct": change_pct,
        "season_avg_fp":    round(sum(curr_fps) / len(curr_fps), 1) if curr_fps else None,
        "season_high":      round(max(curr_prices), 2) if curr_prices else None,
        "season_low":       round(min(curr_prices), 2) if curr_prices else None,
        "games":            len(curr_fps),
        "consistency":      _consistency_grade(curr_fps),
        "agent_bids": {
            "MOMENTUM":   round(momentum_bid, 2),
            "LAST_WEEK":  round(last_fp_bid, 2),
            "SEASON_AVG": round(avg_fp_bid, 2),
        },
    }

    stats_cache.set("market_player", result, player_id, season, num_seasons)
    return result
