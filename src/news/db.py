"""
SQLite database setup and helpers for Phase 4 app state:
injury reports, news items, and LLM summaries.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "db" / "app.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS injury_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                scraped_at  TEXT NOT NULL,
                week        INTEGER NOT NULL,
                season      INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team        TEXT NOT NULL,
                position    TEXT,
                injury      TEXT,
                status      TEXT,   -- O / P / Q / D / IR / Out
                practice    TEXT,   -- DNP / Limited / Full
                UNIQUE(season, week, player_name, team)
            );

            CREATE TABLE IF NOT EXISTS news_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                fetched_at  TEXT NOT NULL,
                source      TEXT NOT NULL,   -- nfl.com, espn, pft, rotoworld
                player_name TEXT,
                team        TEXT,
                title       TEXT NOT NULL,
                summary     TEXT,
                url         TEXT,
                published   TEXT,
                UNIQUE(source, url)
            );

            CREATE TABLE IF NOT EXISTS news_summaries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                summarized_at   TEXT NOT NULL,
                player_name     TEXT NOT NULL,
                week            INTEGER NOT NULL,
                season          INTEGER NOT NULL,
                injury_status   TEXT,
                summary_text    TEXT NOT NULL,
                UNIQUE(player_name, week, season)
            );
        """)
