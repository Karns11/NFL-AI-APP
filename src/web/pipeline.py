"""
Pipeline runner for the web app.

Wraps the existing weekly report engine and stores results in the
report cache so the web page can serve them instantly.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone

from src.news.db import init_db
from src.news.injury_scraper import run_scrape
from src.news.rss_fetcher import fetch_all_feeds
from src.fantasy.sleeper import get_my_league_data
from src.predictions.predict import build_feature_df, _CURRENT_SEASON, get_current_week
from src.predictions.weekly import build_full_league_report
from src.web import report_cache

logger = logging.getLogger(__name__)

DEFAULT_USERNAME = "Karns11"


def _serialize_card(card) -> dict:
    return {
        "sleeper_id": card.sleeper_id,
        "name": card.name,
        "position": card.position,
        "team": card.team,
        "is_starter": card.is_starter,
        "predicted": card.predicted,
        "floor": card.floor,
        "ceiling": card.ceiling,
        "shap_positive": card.shap_positive,
        "shap_negative": card.shap_negative,
        "injury_status": card.injury_status,
        "injury_type": getattr(card, "injury_type", None),
        "llm_summary": card.llm_summary,
        "opponent": card.opponent,
        "home_away": card.home_away,
        "start_sit_alert": None,  # filled in below
    }


def run_pipeline(
    username: str = DEFAULT_USERNAME,
    week: int | None = None,
    season: int = _CURRENT_SEASON,
    use_llm: bool = True,
    refresh_injuries: bool = False,
) -> None:
    """
    Run the full weekly report pipeline and store results in report_cache.
    Called by the scheduler and by the manual refresh endpoint.
    """
    logger.info(f"Pipeline starting — {username} week={week} llm={use_llm}")
    init_db()

    if refresh_injuries:
        logger.info("Scraping injuries...")
        run_scrape(week or get_current_week(season), season)
        fetch_all_feeds()

    resolved_week = week or get_current_week(season)
    leagues = get_my_league_data(username, season)
    feature_df, schedules_df = build_feature_df()

    bundle_cache = {}
    serialized_leagues = []

    # Build a lookup of existing LLM summaries from the prior cache so a
    # no-LLM refresh doesn't wipe out summaries that were already generated.
    prior_summaries: dict[str, str] = {}
    prior = report_cache.get()
    if prior and not use_llm:
        for prior_league in prior.leagues:
            for p in prior_league.get("my_roster", []):
                if p.get("llm_summary") and p.get("name"):
                    prior_summaries[p["name"].lower()] = p["llm_summary"]

    for league in leagues:
        report, starter_cards, bench_cards, opp_cards = build_full_league_report(
            league=league,
            week=resolved_week,
            season=season,
            feature_df=feature_df,
            schedules_df=schedules_df,
            bundle_cache=bundle_cache,
            use_llm=use_llm,
        )

        # Build a player-keyed set of start/sit alert names for badge rendering
        alert_names: set[str] = set()
        for alert in report.start_sit_alerts:
            # Alert format: "  ⚠  START PlayerName (pts) over ..."
            parts = alert.split("START ")
            if len(parts) > 1:
                name_part = parts[1].split(" (")[0].strip()
                alert_names.add(name_part.lower())

        # Merge all roster cards (starters + bench) into one unified list
        all_my_cards = []
        for card in starter_cards + bench_cards:
            d = _serialize_card(card)
            if card.name.lower() in alert_names:
                d["start_sit_alert"] = True
            # Preserve prior LLM summary if this run didn't generate one
            if not d["llm_summary"] and card.name.lower() in prior_summaries:
                d["llm_summary"] = prior_summaries[card.name.lower()]
            all_my_cards.append(d)

        all_my_cards.sort(key=lambda c: c["predicted"] or 0, reverse=True)

        opp_serialized = [_serialize_card(c) for c in opp_cards]
        opp_serialized.sort(key=lambda c: c["predicted"] or 0, reverse=True)

        opp_name = getattr(report, "_opp_name", "Opponent")

        serialized_leagues.append({
            "league_name": report.league_name,
            "league_id": report.league_id,
            "scoring_format": report.scoring_format,
            "week": report.week,
            "season": report.season,
            "my_projected_total": round(report.my_projected_total, 1),
            "opp_projected_total": round(report.opp_projected_total, 1),
            "opp_name": opp_name,
            "start_sit_alerts": report.start_sit_alerts,
            "my_roster": all_my_cards,
            "opp_starters": opp_serialized,
        })

    cached = report_cache.CachedReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        week=resolved_week,
        season=season,
        username=username,
        leagues=serialized_leagues,
    )
    report_cache.set(cached)
    logger.info(f"Pipeline complete — {len(leagues)} leagues cached")
