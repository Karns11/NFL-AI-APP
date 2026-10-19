"""
Flask route handlers for the NFL Fantasy AI web app.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, redirect, render_template, request, url_for, flash
from flask_login import login_required, login_user, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash

from src.web import report_cache
from src.news.db import (
    init_db, create_user, get_user_by_email,
    update_sleeper_username, get_all_users,
)

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


# ── Auth routes ───────────────────────────────────────────────────────────────

@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        row = get_user_by_email(email)
        if row and check_password_hash(row["password_hash"], password):
            from app import User
            user = User(row)
            login_user(user, remember=True)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main.index"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        display_name = request.form.get("display_name", "").strip()

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("register.html")

        if get_user_by_email(email):
            flash("An account with that email already exists.", "error")
            return render_template("register.html")

        password_hash = generate_password_hash(password)
        try:
            user_id = create_user(email, password_hash, display_name or email.split("@")[0])
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            flash("Registration failed. Please try again.", "error")
            return render_template("register.html")

        row = get_user_by_email(email)
        from app import User
        login_user(User(row), remember=True)
        flash("Account created! Enter your Sleeper username to get started.", "success")
        return redirect(url_for("main.settings"))

    return render_template("register.html")


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.login"))


@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        sleeper_username = request.form.get("sleeper_username", "").strip()
        if not sleeper_username:
            flash("Please enter a Sleeper username.", "error")
            return render_template("settings.html")

        # Validate the Sleeper username exists before saving
        try:
            from src.fantasy.sleeper import get_my_league_data
            from src.predictions.predict import _CURRENT_SEASON
            leagues = get_my_league_data(sleeper_username, _CURRENT_SEASON)
            if not leagues:
                flash(f"No Sleeper leagues found for '{sleeper_username}'. Check the username and try again.", "error")
                return render_template("settings.html")
        except Exception as e:
            flash(f"Could not verify Sleeper username: {e}", "error")
            return render_template("settings.html")

        update_sleeper_username(current_user.id, sleeper_username)
        flash(f"Sleeper username set to '{sleeper_username}'. Generating your report...", "success")
        return redirect(url_for("main.generate_report"))

    return render_template("settings.html")


@bp.route("/generate")
@login_required
def generate_report():
    """Trigger pipeline for the current user, then redirect to dashboard."""
    sleeper_username = current_user.sleeper_username
    if not sleeper_username:
        return redirect(url_for("main.settings"))

    try:
        from src.web.pipeline import run_pipeline
        run_pipeline(username=sleeper_username, use_llm=True)
    except Exception as e:
        logger.error(f"Report generation failed for {sleeper_username}: {e}")
        flash(f"Report generation failed: {e}", "error")

    return redirect(url_for("main.index"))


# ── Main dashboard ────────────────────────────────────────────────────────────

def _fmt_age(iso_str: str | None) -> str:
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
    freshness = {
        "nfl_stats":     {"label": "NFL Stats",      "when": "never", "detail": ""},
        "injuries":      {"label": "Injury Reports", "when": "never", "detail": ""},
        "news":          {"label": "News Feed",       "when": "never", "detail": ""},
        "llm_summaries": {"label": "AI Summaries",   "when": "never", "detail": ""},
        "report":        {"label": "Report",         "when": "never", "detail": ""},
    }

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

    try:
        from src.news.db import get_conn
        with get_conn() as conn:
            r = conn.execute("SELECT MAX(scraped_at) FROM injury_reports").fetchone()
            freshness["injuries"]["when"] = _fmt_age(r[0] if r else None)
            r = conn.execute("SELECT MAX(fetched_at) FROM news_items").fetchone()
            freshness["news"]["when"] = _fmt_age(r[0] if r else None)
            r = conn.execute("SELECT MAX(summarized_at) FROM news_summaries").fetchone()
            freshness["llm_summaries"]["when"] = _fmt_age(r[0] if r else None)
    except Exception:
        pass

    sleeper_username = getattr(current_user, "sleeper_username", None)
    if sleeper_username:
        cached = report_cache.get(sleeper_username)
        if cached:
            freshness["report"]["when"] = _fmt_age(cached.generated_at)
            freshness["report"]["detail"] = f"Wk {cached.week} · {cached.season}"

    return freshness


@bp.route("/")
@login_required
def index():
    if not current_user.sleeper_username:
        return redirect(url_for("main.settings"))

    sleeper_username = current_user.sleeper_username
    cached = report_cache.get(sleeper_username)
    age = report_cache.age_minutes(sleeper_username)

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


# ── API endpoints ─────────────────────────────────────────────────────────────

@bp.route("/api/refresh", methods=["POST"])
@login_required
def refresh():
    sleeper_username = current_user.sleeper_username
    if not sleeper_username:
        return jsonify({"status": "error", "message": "No Sleeper username set"}), 400
    try:
        from src.web.pipeline import run_pipeline
        run_pipeline(username=sleeper_username, use_llm=False)
        return jsonify({"status": "ok", "message": "Report refreshed"})
    except Exception as e:
        logger.error(f"Refresh failed for {sleeper_username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/refresh-full", methods=["POST"])
@login_required
def refresh_full():
    sleeper_username = current_user.sleeper_username
    if not sleeper_username:
        return jsonify({"status": "error", "message": "No Sleeper username set"}), 400
    try:
        from src.web.pipeline import run_pipeline
        run_pipeline(username=sleeper_username, use_llm=True)
        return jsonify({"status": "ok", "message": "Full report refreshed"})
    except Exception as e:
        logger.error(f"Full refresh failed for {sleeper_username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
