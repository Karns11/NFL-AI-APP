"""
Sleeper API client — free, no auth required.

Fetches all leagues for a user, rosters, matchups, and player info.
Rate limit: 1000 req/min (we'll never come close).
"""

import logging
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)

BASE = "https://api.sleeper.app/v1"


def _get(path: str) -> dict | list:
    resp = requests.get(f"{BASE}{path}", timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── User ─────────────────────────────────────────────────────────────────────

def get_user(username: str) -> dict:
    """Returns user info including user_id."""
    return _get(f"/user/{username}")


# ── Leagues ──────────────────────────────────────────────────────────────────

def get_user_leagues(user_id: str, season: int) -> list[dict]:
    """All NFL leagues for a user in a given season."""
    return _get(f"/user/{user_id}/leagues/nfl/{season}")


def get_league(league_id: str) -> dict:
    return _get(f"/league/{league_id}")


def get_league_users(league_id: str) -> list[dict]:
    """All users in a league (includes display_name, user_id)."""
    return _get(f"/league/{league_id}/users")


def get_league_rosters(league_id: str) -> list[dict]:
    """All rosters in a league."""
    return _get(f"/league/{league_id}/rosters")


def get_matchups(league_id: str, week: int) -> list[dict]:
    """Weekly matchup data including starters and points."""
    return _get(f"/league/{league_id}/matchups/{week}")


# ── Players ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_all_players() -> dict:
    """
    Full Sleeper player directory: {sleeper_id: player_dict}.
    Cached in memory — call once per session (~12K players, ~5MB).
    """
    logger.info("  Fetching Sleeper player directory...")
    return _get("/players/nfl")


def resolve_player_ids(sleeper_ids: list[str]) -> dict[str, dict]:
    """
    Resolve a list of Sleeper player IDs to player info dicts.
    Returns {sleeper_id: {name, position, team}}.
    """
    all_players = get_all_players()
    result = {}
    for sid in sleeper_ids:
        p = all_players.get(str(sid), {})
        result[str(sid)] = {
            "name": p.get("full_name") or p.get("first_name", "") + " " + p.get("last_name", ""),
            "position": p.get("position", ""),
            "team": p.get("team") or p.get("fantasy_positions", ["?"])[0] if not p.get("team") else p.get("team", ""),
            "status": p.get("status", "Active"),
            "injury_status": p.get("injury_status"),
        }
    return result


# ── Scoring helpers ───────────────────────────────────────────────────────────

def get_scoring_format(league: dict) -> str:
    """Return 'PPR', 'Half PPR', or 'Standard' based on league settings."""
    rec = league.get("scoring_settings", {}).get("rec", 0)
    if rec >= 1.0:
        return "PPR"
    elif rec >= 0.5:
        return "Half PPR"
    return "Standard"


def get_roster_positions(league: dict) -> list[str]:
    """Return the starting roster slots (e.g. ['QB','RB','RB','WR','WR','TE','FLEX','K','DEF'])."""
    return league.get("roster_positions", [])


# ── High-level helpers ────────────────────────────────────────────────────────

def get_my_league_data(username: str, season: int) -> list[dict]:
    """
    Fetch all leagues for a user and enrich each with:
      - my_roster_id, my_starters, my_bench
      - opponent_roster_id, opponent_starters
      - scoring_format, roster_positions
      - all player names resolved

    Returns list of enriched league dicts, one per league.
    """
    user = get_user(username)
    user_id = user["user_id"]
    leagues = get_user_leagues(user_id, season)

    result = []
    for league in leagues:
        league_id = league["league_id"]
        try:
            enriched = _enrich_league(league, user_id)
            result.append(enriched)
        except Exception as e:
            logger.warning(f"  Failed to enrich league {league.get('name')}: {e}")

    return result


def _enrich_league(league: dict, user_id: str) -> dict:
    """Add roster, matchup, and player resolution to a league dict."""
    league_id = league["league_id"]

    rosters = get_league_rosters(league_id)
    users = get_league_users(league_id)

    # Map owner_id -> display_name
    user_display = {u["user_id"]: u.get("display_name", "Unknown") for u in users}

    # Find my roster
    my_roster = next((r for r in rosters if r.get("owner_id") == user_id), None)
    if my_roster is None:
        raise ValueError(f"User {user_id} not found in league {league_id}")

    my_roster_id = my_roster["roster_id"]
    my_players = my_roster.get("players") or []
    my_starters_ids = my_roster.get("starters") or []
    my_reserve_ids = my_roster.get("reserve") or []
    my_bench_ids = [
        p for p in my_players
        if p not in my_starters_ids and p not in my_reserve_ids
    ]

    # Collect all player IDs we need to resolve
    all_ids = set(my_players)
    for r in rosters:
        all_ids.update(r.get("starters") or [])

    player_info = resolve_player_ids(list(all_ids))

    # Build owner_id -> roster lookup
    roster_by_owner = {r.get("owner_id"): r for r in rosters}
    roster_by_id = {r["roster_id"]: r for r in rosters}

    league["_user_display"] = user_display
    league["_player_info"] = player_info
    league["_my_roster_id"] = my_roster_id
    league["_my_starters_ids"] = my_starters_ids
    league["_my_bench_ids"] = my_bench_ids
    league["_my_reserve_ids"] = my_reserve_ids
    league["_roster_by_id"] = roster_by_id
    league["_scoring_format"] = get_scoring_format(league)
    league["_roster_positions"] = get_roster_positions(league)

    return league


def get_my_matchup(league: dict, week: int) -> tuple[dict | None, dict | None]:
    """
    Returns (my_matchup, opponent_matchup) for the given week.
    Each matchup dict has starters, players, points.
    """
    league_id = league["league_id"]
    my_roster_id = league.get("_my_roster_id")
    matchups = get_matchups(league_id, week)

    my_matchup = next((m for m in matchups if m["roster_id"] == my_roster_id), None)
    if my_matchup is None:
        return None, None

    opp_matchup = next(
        (m for m in matchups
         if m["matchup_id"] == my_matchup["matchup_id"]
         and m["roster_id"] != my_roster_id),
        None,
    )
    return my_matchup, opp_matchup
