"""
Phase 4: Injury + News + LLM Summaries

This script:
1. Scrapes NFL.com injury report for the current week
2. Fetches latest NFL news from RSS feeds (NFL.com, ESPN, PFT, Rotoworld)
3. Generates Claude Haiku summaries for a given player list
4. Prints prediction cards for each player

Run: python run_phase4.py --week 8
     python run_phase4.py --week 8 --players "Ja'Marr Chase" "Justin Jefferson" "Saquon Barkley"
     python run_phase4.py --week 8 --no-llm          # skip API calls
     python run_phase4.py --week 8 --scrape-only     # only update DB, no predictions
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.news.db import init_db
from src.news.injury_scraper import run_scrape
from src.news.rss_fetcher import fetch_all_feeds
from src.predictions.predict import predict_and_display, build_feature_df, get_current_week

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default player list — edit to match your fantasy roster
DEFAULT_PLAYERS = [
    "Justin Jefferson",
    "Ja'Marr Chase",
    "Tyreek Hill",
    "CeeDee Lamb",
    "Saquon Barkley",
    "Christian McCaffrey",
    "Travis Kelce",
    "Sam LaPorta",
]

import datetime as _dt
_today = _dt.date.today()
CURRENT_SEASON = _today.year if _today.month >= 9 else _today.year - 1


def main():
    parser = argparse.ArgumentParser(description="NFL Fantasy AI — Phase 4 Weekly Pipeline")
    parser.add_argument("--week", "-w", type=int, default=None, help="NFL week number (default: auto-detect current week)")
    parser.add_argument("--season", "-s", type=int, default=CURRENT_SEASON, help="Season year")
    parser.add_argument(
        "--players", "-p", nargs="+", default=None,
        help="Player names to predict (space-separated, quoted). Defaults to DEFAULT_PLAYERS list."
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip Claude Haiku summaries")
    parser.add_argument("--no-scrape", action="store_true", help="Skip injury/news scraping (use cached data)")
    parser.add_argument("--scrape-only", action="store_true", help="Only scrape data, don't run predictions")

    args = parser.parse_args()
    week = args.week or get_current_week(args.season)
    players = args.players or DEFAULT_PLAYERS

    print("\n" + "="*60)
    print(f"  NFL FANTASY AI — PHASE 4")
    print(f"  Season {args.season}  |  Week {week}")
    print("="*60)

    # 1. Initialize DB
    init_db()

    # 2. Scrape fresh injury + news data
    if not args.no_scrape:
        print(f"\n[1/3] Scraping injury report (Week {week})...")
        n_injuries = run_scrape(week, args.season)
        print(f"  {n_injuries} injury records stored")

        print(f"\n[2/3] Fetching news from RSS feeds...")
        n_news = fetch_all_feeds()
        print(f"  {n_news} news items stored")
    else:
        print("\n[1/3] Skipping scrape (--no-scrape)")
        print("[2/3] Skipping news fetch (--no-scrape)")

    if args.scrape_only:
        print("\n  Data update complete (--scrape-only). Exiting.")
        return

    # 3. Run prediction cards
    print(f"\n[3/3] Generating prediction cards for {len(players)} players...")
    print(f"  Loading feature data (once for all players)...")
    feature_df, schedules_df = build_feature_df()

    use_llm = not args.no_llm
    results = []
    for player in players:
        result = predict_and_display(
            player_name=player,
            week=week,
            season=args.season,
            use_llm=use_llm,
            refresh_news=False,  # Already refreshed above
            feature_df=feature_df,
            schedules_df=schedules_df,
        )
        if result:
            results.append(result)

    # 4. Summary table
    if results:
        print("\n" + "="*60)
        print(f"  WEEK {week} SUMMARY")
        print("="*60)
        print(f"\n  {'Player':<22} {'Pos':<4} {'Floor':>6} {'Med':>6} {'Ceil':>6}  {'Status':<12}  Rec.")
        print(f"  {'-'*70}")
        for r in sorted(results, key=lambda x: x["predicted"], reverse=True):
            status = r.get("injury_status") or "Healthy"
            rec = r.get("recommendation", "")[:15]
            print(
                f"  {r['player_name']:<22} {r['position']:<4} "
                f"{r['floor']:>6.1f} {r['predicted']:>6.1f} {r['ceiling']:>6.1f}  "
                f"{status:<12}  {rec}"
            )

    print(f"\n  Phase 4 complete.")
    if use_llm:
        print(f"  LLM summaries were generated via Claude Haiku.")
        print(f"  Cached in data/db/app.db — re-run without --no-llm to refresh.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
