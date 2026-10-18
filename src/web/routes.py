"""
Flask route handlers for the NFL Fantasy AI web app.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, render_template

from src.web import report_cache

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


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

    return render_template("index.html", cached=cached, age_str=age_str)


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
