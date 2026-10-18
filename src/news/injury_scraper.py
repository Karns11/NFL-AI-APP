"""
Scrapes NFL.com official injury reports and stores them in SQLite.

NFL.com publishes injury designations (O/P/Q/D/IR) typically
Wednesday through Saturday each week of the season.

Falls back to nflreadpy.load_injuries() for historical weeks
where the live page is no longer available.
"""

import logging
import re
from datetime import datetime, timezone

import requests

from .db import get_conn, init_db

logger = logging.getLogger(__name__)

_NFL_INJURY_URL = "https://www.nfl.com/injuries"

# Maps designation text on the page to short codes
_STATUS_MAP = {
    "out": "Out",
    "doubtful": "D",
    "questionable": "Q",
    "probable": "P",
    "ir": "IR",
    "injured reserve": "IR",
    "pup": "PUP",
    "dnp": "DNP",        # practice only — not game designation
    "limited": "Limited",
    "full": "Full",
}

# Maps short practice codes from the page
_PRACTICE_MAP = {
    "did not participate": "DNP",
    "dnp": "DNP",
    "limited participation": "Limited",
    "limited": "Limited",
    "full participation": "Full",
    "full": "Full",
}


def _normalize_status(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    return _STATUS_MAP.get(key, raw.strip().title())


def scrape_nfl_injuries(week: int, season: int) -> list[dict]:
    """
    Scrape NFL.com injury report page for a given week/season.

    Returns list of dicts with keys:
        player_name, team, position, injury, status, practice
    """
    url = f"https://www.nfl.com/injuries/league/{season}/reg{week}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"NFL.com injury fetch failed: {e}")
        return []

    return _parse_nfl_injury_html(resp.text)


def _parse_nfl_injury_html(html: str) -> list[dict]:
    """
    Parse NFL.com injury page HTML.

    NFL.com renders injury tables like:
      <table class="d3-o-table">
        <thead>...</thead>
        <tbody>
          <tr> <td>Player Name</td> <td>Pos</td> <td>Injury</td>
               <td>Wed</td> <td>Thu</td> <td>Fri</td> <td>Status</td> </tr>
    """
    from html.parser import HTMLParser

    records: list[dict] = []
    current_team: str = ""

    class InjuryParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_table = False
            self.in_row = False
            self.cell_idx = 0
            self.current_row: list[str] = []
            self.current_team = ""

        def handle_starttag(self, tag, attrs):
            attr_dict = dict(attrs)
            if tag == "h2" and "d3-o-section-title" in attr_dict.get("class", ""):
                self._team_next = True
            if tag == "table":
                self.in_table = True
            if tag == "tr" and self.in_table:
                self.in_row = True
                self.current_row = []
                self.cell_idx = 0
            if tag in ("td", "th") and self.in_row:
                pass

        def handle_endtag(self, tag):
            if tag == "table":
                self.in_table = False
                self.in_row = False
            if tag == "tr" and self.in_row:
                self.in_row = False
                row = self.current_row
                if len(row) >= 4 and row[0] and row[0] not in ("Player", "Name"):
                    records.append({
                        "player_name": row[0].strip(),
                        "position": row[1].strip() if len(row) > 1 else None,
                        "injury": row[2].strip() if len(row) > 2 else None,
                        "practice": _normalize_status(row[-2].strip()) if len(row) > 3 else None,
                        "status": _normalize_status(row[-1].strip()) if len(row) > 1 else None,
                        "team": self.current_team,
                    })

        def handle_data(self, data):
            text = data.strip()
            if not text:
                return
            # Team names appear as section headers before their table
            if hasattr(self, "_team_next") and self._team_next:
                self.current_team = text
                self._team_next = False
                return
            if self.in_row:
                self.current_row.append(text)

    parser = InjuryParser()
    parser.feed(html)
    return records


def fetch_injuries_nflreadpy(week: int, season: int) -> list[dict]:
    """
    Fallback: use nflreadpy injury data for historical weeks.
    Returns same dict structure as scrape_nfl_injuries().
    """
    try:
        import nflreadpy as nfl
        df = nfl.load_injuries([season])
        if df is None or len(df) == 0:
            return []

        week_df = df.filter(__import__("polars").col("week") == week) if hasattr(df, "filter") else df[df["week"] == week]

        records = []
        for row in week_df.iter_rows(named=True) if hasattr(week_df, "iter_rows") else week_df.to_dict("records"):
            records.append({
                "player_name": row.get("full_name") or row.get("player_name", ""),
                "team": row.get("team", ""),
                "position": row.get("position", ""),
                "injury": row.get("report_primary_injury") or row.get("injury_type", ""),
                "status": _normalize_status(row.get("report_status") or row.get("game_status", "")),
                "practice": row.get("practice_status", ""),
            })
        return records
    except Exception as e:
        logger.warning(f"nflreadpy injury fallback failed: {e}")
        return []


def save_injuries(records: list[dict], week: int, season: int) -> int:
    """Upsert injury records into SQLite. Returns count inserted/updated."""
    if not records:
        return 0

    init_db()
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    with get_conn() as conn:
        for r in records:
            try:
                conn.execute(
                    """
                    INSERT INTO injury_reports
                        (scraped_at, week, season, player_name, team, position, injury, status, practice)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(season, week, player_name, team) DO UPDATE SET
                        scraped_at = excluded.scraped_at,
                        position   = excluded.position,
                        injury     = excluded.injury,
                        status     = excluded.status,
                        practice   = excluded.practice
                    """,
                    (
                        now, week, season,
                        r.get("player_name", ""),
                        r.get("team", ""),
                        r.get("position"),
                        r.get("injury"),
                        r.get("status"),
                        r.get("practice"),
                    ),
                )
                inserted += 1
            except Exception as e:
                logger.debug(f"Injury insert error ({r.get('player_name')}): {e}")

    return inserted


def get_player_injury(player_name: str, week: int, season: int) -> dict | None:
    """
    Look up stored injury status for a player.
    Returns the most recent row or None.
    """
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT * FROM injury_reports
            WHERE season = ? AND week = ?
              AND lower(player_name) LIKE lower(?)
            ORDER BY scraped_at DESC LIMIT 1
            """,
            (season, week, f"%{player_name}%"),
        ).fetchone()
    return dict(row) if row else None


def get_team_injuries(team: str, week: int, season: int, exclude_name: str | None = None) -> list[dict]:
    """
    Return all injury records for a team in a given week.
    Optionally exclude one player by name (the player being summarized).
    """
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM injury_reports
            WHERE season = ? AND week = ?
              AND lower(team) LIKE lower(?)
            ORDER BY player_name
            """,
            (season, week, f"%{team}%"),
        ).fetchall()

    results = [dict(r) for r in rows]
    if exclude_name:
        exclude_lower = exclude_name.lower()
        results = [r for r in results if exclude_lower not in r.get("player_name", "").lower()]
    return results


def run_scrape(week: int, season: int, use_nflreadpy_fallback: bool = True) -> int:
    """
    Full scrape pipeline: nflreadpy first (clean team abbreviations), then
    supplement with NFL.com for any additional records.
    Returns total records saved.
    """
    logger.info(f"Scraping injuries: Season {season} Week {week}")

    # nflreadpy primary — returns proper team abbreviations (ARI, ATL, etc.)
    records = fetch_injuries_nflreadpy(week, season)
    if records:
        logger.info(f"  nflreadpy: {len(records)} injury records")
    else:
        # Fall back to NFL.com scrape
        logger.info("  nflreadpy empty — trying NFL.com scrape")
        records = scrape_nfl_injuries(week, season)
        if records:
            logger.info(f"  NFL.com: {len(records)} injury records")

    # Clear stale data for this week before saving so bad team names don't persist
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM injury_reports WHERE season = ? AND week = ?",
            (season, week),
        )

    n = save_injuries(records, week, season)
    logger.info(f"  Saved {n} injury records")
    return n
