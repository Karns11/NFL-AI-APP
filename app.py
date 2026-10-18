"""
NFL Fantasy AI — Local Web App

Run:
  python app.py              # starts at http://localhost:5000
  python app.py --no-browser # don't auto-open browser
  python app.py --port 8080  # use a different port
"""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.web.routes import bp
from src.web import report_cache

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src.web.pipeline").setLevel(logging.INFO)
logging.getLogger("src.predictions.weekly").setLevel(logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

app = Flask(__name__, template_folder="src/web/templates")
app.register_blueprint(bp)


def _scheduled_injury_refresh():
    """Wed/Thu/Fri 9 AM — re-scrape injuries and update projections (no LLM)."""
    from src.web.pipeline import run_pipeline
    logging.getLogger("src.web.pipeline").info("Scheduled injury refresh starting...")
    run_pipeline(use_llm=False, refresh_injuries=True)


def _scheduled_full_report():
    """Friday 6 AM — full report generation including LLM summaries."""
    from src.web.pipeline import run_pipeline
    logging.getLogger("src.web.pipeline").info("Scheduled full report generation starting...")
    run_pipeline(use_llm=True, refresh_injuries=True)


def start_scheduler():
    scheduler = BackgroundScheduler()

    # Injury + projection refresh: Wed, Thu, Fri at 9 AM local time
    scheduler.add_job(
        _scheduled_injury_refresh,
        CronTrigger(day_of_week="wed,thu,fri", hour=9, minute=0),
        id="injury_refresh",
        name="Injury refresh",
        replace_existing=True,
    )

    # Full report with LLM: Friday at 6 AM local time
    scheduler.add_job(
        _scheduled_full_report,
        CronTrigger(day_of_week="fri", hour=6, minute=0),
        id="full_report",
        name="Full report generation",
        replace_existing=True,
    )

    scheduler.start()
    return scheduler


def main():
    parser = argparse.ArgumentParser(description="NFL Fantasy AI Web App")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--generate", action="store_true",
                        help="Generate a fresh report on startup (with LLM)")
    args = parser.parse_args()

    # Optionally generate report on startup
    if args.generate:
        print("  Generating report on startup...")
        from src.web.pipeline import run_pipeline
        run_pipeline(use_llm=True, refresh_injuries=True)

    # Start background scheduler
    scheduler = start_scheduler()

    url = f"http://localhost:{args.port}"
    print(f"\n  NFL Fantasy AI running at {url}")

    cached = report_cache.get()
    if cached:
        age = report_cache.age_minutes()
        age_str = f"{int(age)}m ago" if age is not None else "unknown"
        print(f"  Serving cached report (Week {cached.week}, updated {age_str})")
        print(f"  Use 'Refresh' in the browser or run with --generate to update.")
    else:
        print(f"  No cached report found.")
        print(f"  Click 'Generate Report' in the browser, or restart with --generate.")

    print(f"  Scheduled: injury refresh Wed/Thu/Fri 9am, full report Fri 6am")
    print(f"  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
