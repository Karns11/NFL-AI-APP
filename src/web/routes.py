"""
Flask route handlers for the NFL Fantasy AI web app.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, render_template

from src.web import report_cache

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


def _fmt_age(iso_str: str | None) -> str:
    """Return a human-readable age string from an ISO timestamp."""
    if not iso_str:
        return "never"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        minutes = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        if minutes < 60:
            return f"{int(minutes)}m ago"
        elif minutes < 1440:
            return f"{int(minutes // 60)}h ago"
        else:
            return f"{int(minutes // 1440)}d ago"
    except Exception:
        return "unknown"


def get_data_freshness() -> dict:
    """Query all data stores and return their latest update timestamps."""
    freshness = {
        "nfl_stats":    {"label": "NFL Stats",       "when": "never", "detail": ""},
        "injuries":     {"label": "Injury Reports",  "when": "never", "detail": ""},
        "news":         {"label": "News Feed",        "when": "never", "detail": ""},
        "llm_summaries":{"label": "AI Summaries",    "when": "never", "detail": ""},
        "report":       {"label": "Report",          "when": "never", "detail": ""},
    }

    # NFL stats — DuckDB
    try:
        from src.ingestion.load_data import get_db
        db = get_db()
        row = db.execute(
            """SELECT season, MAX(week) FROM player_stats_weekly
               WHERE season = (SELECT MAX(season) FROM player_stats_weekly)
               GROUP BY season"""
        ).fetchone()
        if row and row[0]:
            freshness["nfl_stats"]["detail"] = f"Through {row[0]} Wk {row[1]}"
            freshness["nfl_stats"]["when"] = f"{row[0]} Wk {row[1]}"
    except Exception:
        pass

    # Injuries, news, LLM summaries — SQLite
    try:
        from src.news.db import get_conn, init_db
        init_db()
        with get_conn() as conn:
            r = conn.execute("SELECT MAX(scraped_at) FROM injury_reports").fetchone()
            freshness["injuries"]["when"] = _fmt_age(r[0] if r else None)

            r = conn.execute("SELECT MAX(fetched_at) FROM news_items").fetchone()
            freshness["news"]["when"] = _fmt_age(r[0] if r else None)

            r = conn.execute("SELECT MAX(summarized_at) FROM news_summaries").fetchone()
            freshness["llm_summaries"]["when"] = _fmt_age(r[0] if r else None)
    except Exception:
        pass

    # Report cache
    cached = report_cache.get()
    if cached:
        freshness["report"]["when"] = _fmt_age(cached.generated_at)
        freshness["report"]["detail"] = f"Wk {cached.week} · {cached.season}"

    return freshness


@bp.route("/")
def index():
    cached = report_cache.get()
    age = report_cache.age_minutes()

    if age is not None:
        if age < 60:
            age_str = f"{int(age)}m ago"
        elif age < 1440:
            age_str = f"{int(age // 60)}h ago"
        else:
            age_str = f"{int(age // 1440)}d ago"
    else:
        age_str = None

    freshness = get_data_freshness()
    return render_template("index.html", cached=cached, age_str=age_str, freshness=freshness)


@bp.route("/api/refresh", methods=["POST"])
def refresh():
    """Trigger a data refresh (no LLM) and return updated report JSON."""
    from src.web.pipeline import run_pipeline
    try:
        run_pipeline(use_llm=False)
        return jsonify({"status": "ok", "message": "Report refreshed"})
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/refresh-full", methods=["POST"])
def refresh_full():
    """Trigger a full refresh including LLM summaries."""
    from src.web.pipeline import run_pipeline
    try:
        run_pipeline(use_llm=True)
        return jsonify({"status": "ok", "message": "Full report refreshed"})
    except Exception as e:
        logger.error(f"Full refresh failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
