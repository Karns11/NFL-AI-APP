"""
LineupIQ — Local Web App

Run:
  python app.py              # starts at http://localhost:8080
  python app.py --no-browser # don't auto-open browser
  python app.py --port 9000  # use a different port
  python app.py --generate   # generate fresh reports for all users on startup
"""

import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask
from flask_login import LoginManager
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.web.routes import bp
from src.news.db import init_db, get_user_by_id

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src.web.pipeline").setLevel(logging.INFO)
logging.getLogger("src.predictions.weekly").setLevel(logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

app = Flask(__name__, template_folder="src/web/templates")
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# ── Flask-Login ───────────────────────────────────────────────────────────────

login_manager = LoginManager()
login_manager.login_view = "main.login"
login_manager.login_message = "Please log in to view your report."
login_manager.init_app(app)


class User:
    """Lightweight user wrapper for Flask-Login."""
    def __init__(self, row):
        self.id = row["id"]
        self.email = row["email"]
        self.password_hash = row["password_hash"]
        self.sleeper_username = row["sleeper_username"]
        self.display_name = row["display_name"] or row["email"].split("@")[0]

    # Flask-Login required interface
    @property
    def is_authenticated(self): return True
    @property
    def is_active(self): return True
    @property
    def is_anonymous(self): return False
    def get_id(self): return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    row = get_user_by_id(int(user_id))
    return User(row) if row else None


app.register_blueprint(bp)

# ── Scheduled jobs ────────────────────────────────────────────────────────────

def _scheduled_tuesday_stats():
    """Tuesday 8 AM — pull latest NFL stats after all weekend + MNF games finish."""
    from src.predictions.predict import _CURRENT_SEASON
    from src.ingestion.load_data import refresh_current_season
    from src.web.pipeline import run_pipeline_for_all_users
    log = logging.getLogger("src.web.pipeline")
    log.info(f"Tuesday stats refresh: pulling {_CURRENT_SEASON} data...")
    counts = refresh_current_season(_CURRENT_SEASON)
    log.info(f"  Stats refreshed: {counts}")
    from src.stats import cache as stats_cache
    stats_cache.invalidate_all()
    run_pipeline_for_all_users(use_llm=False)
    log.info("Tuesday stats refresh complete.")


def _scheduled_injury_refresh():
    """Wed/Thu/Fri 9 AM — re-scrape injuries and update projections (no LLM)."""
    from src.web.pipeline import run_pipeline_for_all_users
    logging.getLogger("src.web.pipeline").info("Scheduled injury refresh starting...")
    run_pipeline_for_all_users(use_llm=False, refresh_injuries=True)


def _scheduled_full_report():
    """Friday 6 AM — full report generation including LLM summaries."""
    from src.web.pipeline import run_pipeline_for_all_users
    logging.getLogger("src.web.pipeline").info("Scheduled full report generation starting...")
    run_pipeline_for_all_users(use_llm=True, refresh_injuries=True)


def start_scheduler():
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        _scheduled_tuesday_stats,
        CronTrigger(day_of_week="tue", hour=8, minute=0),
        id="tuesday_stats", name="Tuesday stats refresh", replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_injury_refresh,
        CronTrigger(day_of_week="wed,thu,fri,sat", hour=9, minute=0),
        id="injury_refresh", name="Injury refresh", replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_full_report,
        CronTrigger(day_of_week="fri", hour=6, minute=0),
        id="full_report", name="Full report generation", replace_existing=True,
    )

    scheduler.start()
    return scheduler


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LineupIQ Web App")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--generate", action="store_true",
                        help="Generate fresh reports for all users on startup")
    args = parser.parse_args()

    init_db()

    if args.generate:
        print("  Generating reports for all users on startup...")
        from src.web.pipeline import run_pipeline_for_all_users
        run_pipeline_for_all_users(use_llm=True, refresh_injuries=True)

    scheduler = start_scheduler()

    url = f"http://localhost:{args.port}"
    print(f"\n  LineupIQ running at {url}")
    print(f"  Scheduled: stats Tue 8am, injury refresh Wed/Thu/Fri 9am, full report Fri 6am")
    print(f"  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
