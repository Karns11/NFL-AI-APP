"""
SQLite database setup and helpers for app state:
injury reports, news items, LLM summaries, and user accounts.
"""

import sqlite3
from datetime import datetime, timezone
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

            CREATE TABLE IF NOT EXISTS users (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at       TEXT NOT NULL,
                email            TEXT NOT NULL UNIQUE,
                password_hash    TEXT NOT NULL,
                sleeper_username TEXT,
                display_name     TEXT
            );

            CREATE TABLE IF NOT EXISTS ai_usage (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          INTEGER NOT NULL,
                used_on          TEXT NOT NULL,
                generation_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(user_id, used_on)
            );
        """)


# ── User helpers ─────────────────────────────────────────────────────────────

def create_user(email: str, password_hash: str, display_name: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (created_at, email, password_hash, display_name) VALUES (?,?,?,?)",
            (now, email.lower().strip(), password_hash, display_name),
        )
        return cur.lastrowid


def get_user_by_email(email: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE lower(email) = lower(?)", (email.strip(),)
        ).fetchone()


def get_user_by_id(user_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def update_sleeper_username(user_id: int, sleeper_username: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET sleeper_username = ? WHERE id = ?",
            (sleeper_username.strip(), user_id),
        )


def get_all_users() -> list:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users").fetchall()


# ── AI usage helpers ──────────────────────────────────────────────────────────

def _today_et() -> str:
    """Return today's date string in ET (UTC-5) as YYYY-MM-DD."""
    from datetime import timedelta
    now_et = datetime.now(timezone.utc) - timedelta(hours=5)
    return now_et.date().isoformat()


def get_ai_usage_today(user_id: int) -> int:
    """Return the number of AI generations used today (ET) for this user."""
    today = _today_et()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT generation_count FROM ai_usage WHERE user_id = ? AND used_on = ?",
            (user_id, today),
        ).fetchone()
    return row["generation_count"] if row else 0


def increment_ai_usage(user_id: int) -> int:
    """Increment today's AI generation count for this user. Returns new count."""
    today = _today_et()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ai_usage (user_id, used_on, generation_count)
            VALUES (?, ?, 1)
            ON CONFLICT(user_id, used_on) DO UPDATE SET
                generation_count = generation_count + 1
            """,
            (user_id, today),
        )
        row = conn.execute(
            "SELECT generation_count FROM ai_usage WHERE user_id = ? AND used_on = ?",
            (user_id, today),
        ).fetchone()
    return row["generation_count"] if row else 1
