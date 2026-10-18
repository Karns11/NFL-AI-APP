"""
Fetches NFL news from RSS feeds and stores items in SQLite.

Sources:
  - NFL.com       https://www.nfl.com/rss/rsslanding?searchString=player
  - ESPN NFL      https://www.espn.com/espn/rss/nfl/news
  - ProFootballTalk  https://profootballtalk.nbcsports.com/feed/
  - Rotoworld     https://www.rotoworld.com/rss/feed/nfl/

Each item is stored in the news_items table. Player name tagging is
done by matching known player names against the title/summary text.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

from .db import get_conn, init_db

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    "nfl.com": "https://www.nfl.com/rss/rsslanding?searchString=player",
    "espn": "https://www.espn.com/espn/rss/nfl/news",
    "pft": "https://profootballtalk.nbcsports.com/feed/",
    "rotoworld": "https://www.rotoworld.com/rss/feed/nfl/",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# XML namespaces used by some feeds
_NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "media": "http://search.yahoo.com/mrss/",
}


def _strip_html(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"<[^>]+>", " ", text).strip()


def _parse_feed(source: str, xml_text: str) -> list[dict]:
    """Parse RSS XML text into list of item dicts."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"RSS parse error ({source}): {e}")
        return items

    channel = root.find("channel")
    if channel is None:
        channel = root  # Atom feeds

    for item in channel.findall("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        desc = _strip_html(item.findtext("description", ""))

        # Some feeds use content:encoded for full text
        content_el = item.find("content:encoded", _NS)
        content = _strip_html(content_el.text if content_el is not None else None)

        summary = content or desc or ""
        # Truncate to first 500 chars for storage
        if len(summary) > 500:
            summary = summary[:497] + "..."

        if not title:
            continue

        items.append({
            "source": source,
            "title": title,
            "url": link,
            "published": pub_date,
            "summary": summary,
        })

    return items


def fetch_feed(source: str, url: str) -> list[dict]:
    """Fetch one RSS feed and return parsed items."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return _parse_feed(source, resp.text)
    except requests.RequestException as e:
        logger.warning(f"RSS fetch failed ({source}): {e}")
        return []


def _tag_player(item: dict, known_players: list[str]) -> str | None:
    """
    Try to associate a news item with a specific player name.
    Checks title and summary against known player names.
    Returns first match or None.
    """
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    for name in known_players:
        # Match on last name at minimum
        parts = name.lower().split()
        last = parts[-1] if parts else ""
        if len(last) > 3 and last in text:
            # Confirm with first name or initial if available
            if len(parts) >= 2:
                first = parts[0]
                if first[0] in text and last in text:
                    return name
            else:
                return name
    return None


def save_items(items: list[dict], player_name: str | None = None) -> int:
    """Upsert news items into SQLite. Returns count inserted."""
    if not items:
        return 0

    init_db()
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    with get_conn() as conn:
        for item in items:
            try:
                conn.execute(
                    """
                    INSERT INTO news_items
                        (fetched_at, source, player_name, team, title, summary, url, published)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, url) DO UPDATE SET
                        fetched_at  = excluded.fetched_at,
                        title       = excluded.title,
                        summary     = excluded.summary,
                        player_name = COALESCE(excluded.player_name, news_items.player_name)
                    """,
                    (
                        now,
                        item.get("source", ""),
                        player_name or item.get("player_name"),
                        item.get("team"),
                        item.get("title", ""),
                        item.get("summary"),
                        item.get("url"),
                        item.get("published"),
                    ),
                )
                inserted += 1
            except Exception as e:
                logger.debug(f"News insert error: {e}")

    return inserted


def get_player_news(player_name: str, max_items: int = 5, hours_back: int = 168) -> list[dict]:
    """
    Retrieve recent news items for a player from SQLite.
    hours_back: look back this many hours (default 168 = 7 days).
    """
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT source, title, summary, url, published, fetched_at
            FROM news_items
            WHERE lower(player_name) LIKE lower(?)
              AND fetched_at >= datetime('now', ? || ' hours')
            ORDER BY fetched_at DESC
            LIMIT ?
            """,
            (f"%{player_name}%", f"-{hours_back}", max_items),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_all_feeds() -> int:
    """
    Fetch all configured RSS feeds and store in DB.
    Returns total items stored.
    """
    total = 0
    for source, url in RSS_FEEDS.items():
        items = fetch_feed(source, url)
        n = save_items(items)
        logger.info(f"  {source}: {len(items)} fetched, {n} stored")
        total += n
    return total


def fetch_player_news(player_name: str, team: str | None = None) -> int:
    """
    Fetch all feeds and tag items matching a specific player.
    Returns count of tagged items stored.
    """
    all_items: list[dict] = []
    for source, url in RSS_FEEDS.items():
        all_items.extend(fetch_feed(source, url))

    # Filter items that mention this player
    name_parts = player_name.lower().split()
    last_name = name_parts[-1] if name_parts else ""
    first_initial = name_parts[0][0] if name_parts else ""

    matched = []
    for item in all_items:
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        if last_name and len(last_name) > 3 and last_name in text:
            matched.append(item)

    return save_items(matched, player_name=player_name)
