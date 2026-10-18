"""
NFL Fantasy AI — Unified Weekly Report

Single command that combines Phase 4 (injury/news/LLM) + Phase 5 (Sleeper).

For every league you're in:
  - My starters: projection + SHAP drivers + injury status + LLM summary
  - My bench: projection + injury status
  - Opponent starters: projection + injury status
  - Start/sit alerts
  - Matchup projection

Run:
  python run_weekly.py                        # auto-detects week, uses default username
  python run_weekly.py --username Karns11
  python run_weekly.py --week 6
  python run_weekly.py --no-llm               # skip Haiku calls (faster)
  python run_weekly.py --refresh              # re-scrape injury + news first
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.news.db import init_db
from src.news.injury_scraper import run_scrape
from src.news.rss_fetcher import fetch_all_feeds
from src.fantasy.sleeper import get_my_league_data
from src.predictions.predict import build_feature_df, _CURRENT_SEASON, get_current_week
from src.predictions.weekly import build_full_league_report, print_weekly_league_report

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# Show our logger at INFO for progress messages
logging.getLogger("src.predictions.weekly").setLevel(logging.INFO)
logging.getLogger("src.llm.summarize").setLevel(logging.INFO)
logging.getLogger("src.fantasy.sleeper").setLevel(logging.INFO)

DEFAULT_USERNAME = "Karns11"


def main():
    parser = argparse.ArgumentParser(description="NFL Fantasy AI — Unified Weekly Report")
    parser.add_argument("--username", "-u", default=DEFAULT_USERNAME, help="Sleeper username")
    parser.add_argument("--week", "-w", type=int, default=None, help="NFL week (default: auto-detect)")
    parser.add_argument("--season", "-s", type=int, default=_CURRENT_SEASON, help=f"Season (default: {_CURRENT_SEASON})")
    parser.add_argument("--no-llm", action="store_true", help="Skip Claude Haiku summaries (faster)")
    parser.add_argument("--refresh", action="store_true", help="Re-scrape injury reports + news before running")
    args = parser.parse_args()

    week = args.week or get_current_week(args.season)

    print("\n" + "="*60)
    print(f"  NFL FANTASY AI — WEEKLY REPORT")
    print(f"  {args.username}  |  Season {args.season}  |  Week {week}")
    print("="*60)

    init_db()

    # Optionally refresh injury + news data
    if args.refresh:
        print(f"\n  Scraping injury report (Week {week})...")
        n = run_scrape(week, args.season)
        print(f"  {n} injury records stored")
        print(f"  Fetching RSS news feeds...")
        n = fetch_all_feeds()
        print(f"  {n} news items stored")

    # Fetch Sleeper leagues
    print(f"\n  Fetching Sleeper leagues for {args.username}...")
    leagues = get_my_league_data(args.username, args.season)
    print(f"  Found {len(leagues)} leagues: {', '.join(l['name'] for l in leagues)}")

    # Load feature data once
    print(f"\n  Loading NFL feature data...")
    feature_df, schedules_df = build_feature_df()

    # Build + print report for each league
    bundle_cache = {}
    use_llm = not args.no_llm

    if use_llm:
        print(f"  Generating LLM summaries for starters (Claude Haiku)...")
    print()

    for league in leagues:
        report, starter_cards, bench_cards, opp_cards = build_full_league_report(
            league=league,
            week=week,
            season=args.season,
            feature_df=feature_df,
            schedules_df=schedules_df,
            bundle_cache=bundle_cache,
            use_llm=use_llm,
        )
        print_weekly_league_report(report, starter_cards, bench_cards, opp_cards)

    print("="*60)
    print(f"  Done.  Run with --refresh to pull fresh injury/news data.")
    print(f"         Run with --no-llm to skip Haiku API calls.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
