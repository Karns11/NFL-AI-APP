"""
Phase 5: Sleeper Integration + Personalized Reports

Fetches all your Sleeper leagues, runs model projections for every
player on your roster and your opponent's starters, and prints a
personalized report for each league.

Run: python run_phase5.py --week 7
     python run_phase5.py --week 7 --username Karns11
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import datetime as _dt

from src.fantasy.sleeper import get_my_league_data
from src.fantasy.report import build_league_report, print_league_report
from src.predictions.predict import build_feature_df, _CURRENT_SEASON, get_current_week

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_USERNAME = "Karns11"


def main():
    parser = argparse.ArgumentParser(description="LineupIQ — Phase 5 Personalized Report")
    parser.add_argument("--week", "-w", type=int, default=None, help="NFL week number (default: auto-detect current week)")
    parser.add_argument("--season", "-s", type=int, default=_CURRENT_SEASON, help=f"Season (default: {_CURRENT_SEASON})")
    parser.add_argument("--username", "-u", default=DEFAULT_USERNAME, help="Sleeper username")
    args = parser.parse_args()
    week = args.week or get_current_week(args.season)

    print("\n" + "="*60)
    print(f"  LINEUPIQ — PHASE 5")
    print(f"  Sleeper: {args.username}  |  Season {args.season}  |  Week {week}")
    print("="*60)

    # 1. Fetch all Sleeper leagues
    print(f"\n[1/3] Fetching Sleeper leagues for {args.username}...")
    leagues = get_my_league_data(args.username, args.season)
    print(f"  Found {len(leagues)} leagues:")
    for l in leagues:
        print(f"    • {l['name']} ({l.get('_scoring_format','?')})")

    # 2. Load feature data once for all predictions
    print(f"\n[2/3] Loading NFL feature data (once for all leagues)...")
    feature_df, schedules_df = build_feature_df()

    # 3. Build and print reports for each league
    print(f"\n[3/3] Generating personalized reports...\n")
    bundle_cache = {}

    for league in leagues:
        logger.info(f"  Building report: {league['name']}")
        report = build_league_report(
            league=league,
            week=week,
            season=args.season,
            feature_df=feature_df,
            schedules_df=schedules_df,
            bundle_cache=bundle_cache,
        )
        print_league_report(report)

    print("="*60)
    print(f"  Phase 5 complete — {len(leagues)} league report(s) generated.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
