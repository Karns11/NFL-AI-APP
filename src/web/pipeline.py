"""
Pipeline runner for the web app.

Wraps the existing weekly report engine and stores results in the
report cache so the web page can serve them instantly.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone

from src.news.db import init_db, get_all_users
from src.news.injury_scraper import run_scrape
from src.news.rss_fetcher import fetch_all_feeds
from src.fantasy.sleeper import get_my_league_data, get_player_actual_scores, get_player_projected_scores, get_my_matchup, _scoring_key
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
        "start_sit_alert": None,       # filled in below
        "actual_points": None,         # filled in for past weeks
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
    current_week = get_current_week(season)
    is_past = resolved_week < current_week

    # LLM summaries only make sense for the current week
    if resolved_week != current_week:
        use_llm = False

    leagues = get_my_league_data(username, season)
    feature_df, schedules_df = build_feature_df()

    # Fetch Sleeper's own projections for this week (all players, single API call)
    sleeper_projs: dict[str, dict] = get_player_projected_scores(season, resolved_week)

    bundle_cache = {}
    serialized_leagues = []

    # Preserve existing LLM summaries across no-LLM refreshes of the current week
    prior_summaries: dict[str, str] = {}
    if not use_llm and not is_past:
        prior = report_cache.get(username, resolved_week, season)
        if prior:
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

        scoring_key = _scoring_key(league.get("_scoring_format", "PPR"))

        # Fetch actual scores for past weeks
        actual_scores: dict[str, float] = {}
        if is_past:
            actual_scores = get_player_actual_scores(report.league_id, resolved_week)

        # Compute Sleeper + actual team totals from raw matchup starter IDs so
        # K/DEF are included (they're hidden from player rows but count in totals).
        my_matchup, opp_matchup = get_my_matchup(league, resolved_week)

        def _sum_sleeper(starter_ids: list[str]) -> tuple[float, bool]:
            total, valid = 0.0, False
            for sid in starter_ids:
                val = (sleeper_projs.get(str(sid), {}) or {}).get(scoring_key)
                if val is not None:
                    total += float(val)
                    valid = True
            return total, valid

        def _sum_actual(starter_ids: list[str]) -> tuple[float, bool]:
            total, valid = 0.0, False
            for sid in starter_ids:
                pts = actual_scores.get(str(sid))
                if pts is not None:
                    total += pts
                    valid = True
            return total, valid

        my_starter_ids = my_matchup.get("starters", []) if my_matchup else league.get("_my_starters_ids", [])
        opp_starter_ids = opp_matchup.get("starters", []) if opp_matchup else []

        my_sleeper_total, my_sleeper_total_valid = _sum_sleeper(my_starter_ids)
        opp_sleeper_total, opp_sleeper_total_valid = _sum_sleeper(opp_starter_ids)
        my_actual_total, my_actual_total_valid = _sum_actual(my_starter_ids)
        opp_actual_total, opp_actual_total_valid = _sum_actual(opp_starter_ids)

        # Build a player-keyed set of start/sit alert names for badge rendering
        alert_names: set[str] = set()
        for alert in report.start_sit_alerts:
            parts = alert.split("START ")
            if len(parts) > 1:
                name_part = parts[1].split(" (")[0].strip()
                alert_names.add(name_part.lower())

        _SKIP_POS = {"K", "DEF", "DL", "LB", "DB", "IDP"}

        def _skip(card) -> bool:
            if not card.name:
                return True
            if card.name.strip().isdigit():       # unresolved numeric Sleeper ID
                return True
            if card.position in _SKIP_POS:        # kicker / defense / IDP
                return True
            if card.position in ("?", ""):        # unresolved position (e.g. team DEF "ATL")
                return True
            return False

        all_my_cards = []
        for card in starter_cards + bench_cards:
            if _skip(card):
                continue
            d = _serialize_card(card)
            if card.name.lower() in alert_names:
                d["start_sit_alert"] = True
            if not d["llm_summary"] and card.name.lower() in prior_summaries:
                d["llm_summary"] = prior_summaries[card.name.lower()]
            if actual_scores and card.sleeper_id:
                d["actual_points"] = actual_scores.get(str(card.sleeper_id))
            all_my_cards.append(d)

        all_my_cards.sort(key=lambda c: c["predicted"] or 0, reverse=True)

        opp_serialized = []
        for card in opp_cards:
            if _skip(card):
                continue
            d = _serialize_card(card)
            if actual_scores and card.sleeper_id:
                d["actual_points"] = actual_scores.get(str(card.sleeper_id))
            opp_serialized.append(d)
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
            "my_sleeper_projected_total": round(my_sleeper_total, 1) if my_sleeper_total_valid else None,
            "opp_sleeper_projected_total": round(opp_sleeper_total, 1) if opp_sleeper_total_valid else None,
            "my_actual_total": round(my_actual_total, 1) if my_actual_total_valid else None,
            "opp_actual_total": round(opp_actual_total, 1) if opp_actual_total_valid else None,
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
    report_cache.set(username, cached)
    logger.info(f"Pipeline complete — {len(leagues)} leagues cached for {username} wk{resolved_week}")


def run_pipeline_for_all_users(use_llm: bool = False, refresh_injuries: bool = False) -> None:
    """
    Run the pipeline for every registered user that has a Sleeper username set.
    Used by scheduled jobs so all users' reports stay current automatically.
    """
    init_db()
    users = [u for u in get_all_users() if u["sleeper_username"]]
    if not users:
        logger.info("No users with Sleeper usernames — skipping scheduled pipeline")
        return

    logger.info(f"Running pipeline for {len(users)} user(s)...")
    for user in users:
        sleeper_username = user["sleeper_username"]
        try:
            run_pipeline(
                username=sleeper_username,
                use_llm=use_llm,
                refresh_injuries=refresh_injuries,
            )
        except Exception as e:
            logger.error(f"Pipeline failed for {sleeper_username}: {e}")
