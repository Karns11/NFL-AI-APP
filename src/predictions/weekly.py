"""
Unified weekly report engine.

Combines Phase 4 (injury/news/LLM) + Phase 5 (Sleeper roster/matchup)
into a single per-league output for every player on your roster.

Output per league:
  - My starters: full card (projection + SHAP + injury + LLM summary)
  - My bench:    projection + injury status (no LLM)
  - Opp starters: projection + injury status (no LLM)
  - Start/sit alerts
  - Matchup projection
"""

import logging
import textwrap
from dataclasses import dataclass, field

import polars as pl

from src.news.injury_scraper import get_player_injury
from src.news.rss_fetcher import get_player_news
from src.llm.summarize import summarize_player, _fallback_summary
from src.fantasy.sleeper import get_my_matchup
from src.fantasy.report import _add_start_sit_alerts, LeagueReport, PlayerProjection, MODEL_POSITIONS

logger = logging.getLogger(__name__)


@dataclass
class FullPlayerCard:
    sleeper_id: str
    name: str
    position: str
    team: str
    is_starter: bool
    # Model output
    predicted: float | None = None
    floor: float | None = None
    ceiling: float | None = None
    shap_positive: list = field(default_factory=list)
    shap_negative: list = field(default_factory=list)
    # Injury + news
    injury_status: str | None = None
    injury_type: str | None = None
    llm_summary: str | None = None
    # Matchup context
    opponent: str | None = None
    home_away: str | None = None


def _project_full(
    player_info: dict,
    week: int,
    season: int,
    feature_df: pl.DataFrame,
    schedules_df: pl.DataFrame,
    bundle_cache: dict,
) -> tuple[float | None, float | None, float | None, list, list]:
    """Run model + SHAP. Returns (predicted, floor, ceiling, shap_pos, shap_neg)."""
    position = player_info.get("position", "")
    name = player_info.get("name", "")
    known_team = player_info.get("team")

    if position not in MODEL_POSITIONS or not name.strip():
        return None, None, None, [], []

    from src.predictions.predict import get_player_features, load_phase2_bundle
    from src.models.train import predict_player

    if position not in bundle_cache:
        try:
            bundle_cache[position] = load_phase2_bundle(position)
        except FileNotFoundError:
            return None, None, None, [], []

    bundle = bundle_cache[position]
    features, _, _ = get_player_features(
        name, week, season,
        feature_df=feature_df, schedules_df=schedules_df, team=known_team,
    )
    if features is None:
        return None, None, None, [], []

    try:
        pred = predict_player(bundle, features)
        return (
            pred["predicted"], pred["floor"], pred["ceiling"],
            pred["shap_top_positive"], pred["shap_top_negative"],
        )
    except Exception as e:
        logger.debug(f"Prediction failed for {name}: {e}")
        return None, None, None, [], []


def build_full_league_report(
    league: dict,
    week: int,
    season: int,
    feature_df: pl.DataFrame,
    schedules_df: pl.DataFrame,
    bundle_cache: dict,
    use_llm: bool = True,
) -> tuple[LeagueReport, list[FullPlayerCard], list[FullPlayerCard], list[FullPlayerCard]]:
    """
    Build full cards for my starters, lighter cards for bench + opponent.
    Returns (league_report, my_starter_cards, my_bench_cards, opp_cards).
    """
    player_info = league.get("_player_info", {})
    my_starters_ids = league.get("_my_starters_ids", [])
    my_bench_ids = league.get("_my_bench_ids", [])
    roster_by_id = league.get("_roster_by_id", {})
    user_display = league.get("_user_display", {})

    my_matchup, opp_matchup = get_my_matchup(league, week)
    opp_starters_ids = opp_matchup.get("starters", []) if opp_matchup else []

    opp_roster_id = opp_matchup["roster_id"] if opp_matchup else None
    opp_roster = roster_by_id.get(opp_roster_id, {}) if opp_roster_id else {}
    opp_owner_id = opp_roster.get("owner_id", "")
    opp_name = user_display.get(opp_owner_id, f"Roster {opp_roster_id}")

    report = LeagueReport(
        league_name=league["name"],
        league_id=league["league_id"],
        scoring_format=league.get("_scoring_format", "PPR"),
        week=week,
        season=season,
    )
    report._opp_name = opp_name

    starter_cards: list[FullPlayerCard] = []
    bench_cards: list[FullPlayerCard] = []
    opp_cards: list[FullPlayerCard] = []

    # ── My starters: full card ────────────────────────────────────────────────
    for sid in my_starters_ids:
        info = player_info.get(str(sid), {"name": str(sid), "position": "?", "team": "?"})
        name = info.get("name", "")
        position = info.get("position", "")
        team = info.get("team", "")

        predicted, floor, ceiling, shap_pos, shap_neg = _project_full(
            info, week, season, feature_df, schedules_df, bundle_cache
        )

        # Injury
        injury = get_player_injury(name, week, season)
        injury_status = injury.get("status") if injury else info.get("injury_status")
        injury_type = injury.get("injury") if injury else None

        # LLM summary (starters only)
        llm_summary = None
        if use_llm and position in MODEL_POSITIONS and name:
            news_items = get_player_news(name, max_items=4)
            llm_summary = summarize_player(
                player_name=name, week=week, season=season,
                position=position, team=team,
                injury_record=injury,
                news_items=news_items,
            )

        card = FullPlayerCard(
            sleeper_id=sid, name=name, position=position, team=team,
            is_starter=True,
            predicted=predicted, floor=floor, ceiling=ceiling,
            shap_positive=shap_pos, shap_negative=shap_neg,
            injury_status=injury_status, injury_type=injury_type,
            llm_summary=llm_summary,
        )
        starter_cards.append(card)

        # Add to league report for start/sit alert logic
        report.my_starters.append(PlayerProjection(
            sleeper_id=sid, name=name, position=position, team=team,
            is_starter=True, predicted=predicted, floor=floor, ceiling=ceiling,
            injury_status=injury_status,
        ))
        if predicted:
            report.my_projected_total += predicted

    # ── My bench: projection + injury only ────────────────────────────────────
    for sid in my_bench_ids:
        info = player_info.get(str(sid), {"name": str(sid), "position": "?", "team": "?"})
        name = info.get("name", "")
        position = info.get("position", "")
        team = info.get("team", "")

        predicted, floor, ceiling, _, _ = _project_full(
            info, week, season, feature_df, schedules_df, bundle_cache
        )

        injury = get_player_injury(name, week, season)
        injury_status = injury.get("status") if injury else info.get("injury_status")

        card = FullPlayerCard(
            sleeper_id=sid, name=name, position=position, team=team,
            is_starter=False,
            predicted=predicted, floor=floor, ceiling=ceiling,
            injury_status=injury_status,
        )
        bench_cards.append(card)

        report.my_bench.append(PlayerProjection(
            sleeper_id=sid, name=name, position=position, team=team,
            is_starter=False, predicted=predicted, floor=floor, ceiling=ceiling,
            injury_status=injury_status,
        ))

    # ── Opponent starters: projection + injury only ───────────────────────────
    for sid in opp_starters_ids:
        info = player_info.get(str(sid), {"name": str(sid), "position": "?", "team": "?"})
        name = info.get("name", "")
        position = info.get("position", "")
        team = info.get("team", "")

        predicted, floor, ceiling, _, _ = _project_full(
            info, week, season, feature_df, schedules_df, bundle_cache
        )

        injury = get_player_injury(name, week, season)
        injury_status = injury.get("status") if injury else info.get("injury_status")

        card = FullPlayerCard(
            sleeper_id=sid, name=name, position=position, team=team,
            is_starter=True,
            predicted=predicted, floor=floor, ceiling=ceiling,
            injury_status=injury_status,
        )
        opp_cards.append(card)

        report.opp_starters.append(PlayerProjection(
            sleeper_id=sid, name=name, position=position, team=team,
            is_starter=True, predicted=predicted, floor=floor, ceiling=ceiling,
            injury_status=injury_status,
        ))
        if predicted:
            report.opp_projected_total += predicted

    _add_start_sit_alerts(report)
    return report, starter_cards, bench_cards, opp_cards


# ── Display ───────────────────────────────────────────────────────────────────

def _shap_line(name: str, value: float) -> str:
    arrow = "▲" if value > 0 else "▼"
    label = (
        name.replace("fp_roll4_mean", "Recent form (4-game avg)")
             .replace("fp_ewm", "Recent form (EWM)")
             .replace("fp_roll8_mean", "Recent form (8-game avg)")
             .replace("target_share_roll4", "Target share (4-game)")
             .replace("target_share_ewm", "Target share (EWM)")
             .replace("snap_pct_roll3", "Snap% (3-game avg)")
             .replace("snap_pct_ewm", "Snap% (EWM)")
             .replace("team_implied_total", "Implied team total")
             .replace("spread_line", "Game spread")
             .replace("opp_fp_allowed_roll4", "Opp. defense (4-game)")
             .replace("air_yards_share_roll4", "Air yards share")
             .replace("rec_td_rate_roll8", "TD rate (recent)")
             .replace("yac_roll4", "YAC (recent)")
             .replace("team_pass_yards_roll4", "Team pass vol.")
             .replace("is_home", "Home/Away")
             .replace("rest_days", "Rest days")
             .replace("season_avg_fp", "Season avg FP")
             .replace("_", " ")
    )
    return f"    {arrow} {value:+.1f}  {label}"


def print_full_card(card: FullPlayerCard, indent: str = "  ") -> None:
    """Print a full player card inline within the league report."""
    status_str = f"  [{card.injury_status}]" if card.injury_status else ""
    proj_str = ""
    if card.predicted is not None:
        proj_str = f"  →  {card.floor:.1f} / {card.predicted:.1f} / {card.ceiling:.1f}  (floor/med/ceil)"

    print(f"{indent}{'─'*54}")
    print(f"{indent}{card.name.upper()}  |  {card.position}  |  {card.team}{status_str}{proj_str}")

    if card.shap_positive or card.shap_negative:
        drivers = (card.shap_positive[:3] + card.shap_negative[:2])
        for name, val in drivers:
            print(_shap_line(name, val))

    if card.llm_summary:
        for line in textwrap.wrap(card.llm_summary, width=52):
            print(f"{indent}  {line}")


def print_bench_row(card: FullPlayerCard, indent: str = "  ") -> None:
    """Print a compact one-line bench/opponent row."""
    status_str = f"  [{card.injury_status}]" if card.injury_status else ""
    if card.predicted is not None:
        proj = f"{card.floor:.1f} / {card.predicted:.1f} / {card.ceiling:.1f}"
    else:
        proj = "  —  /   —  /   —"
    print(f"{indent}  {card.name:<24} {card.position:<4} {card.team:<5}  {proj}{status_str}")


def print_weekly_league_report(
    report: LeagueReport,
    starter_cards: list[FullPlayerCard],
    bench_cards: list[FullPlayerCard],
    opp_cards: list[FullPlayerCard],
) -> None:
    width = 60
    opp_name = getattr(report, "_opp_name", "Opponent")

    print(f"\n{'='*width}")
    print(f"  {report.league_name.upper()}")
    print(f"  {report.scoring_format}  |  Week {report.week}  |  {report.season}")
    diff = report.my_projected_total - report.opp_projected_total
    edge = "▲ projected WIN" if diff > 0 else "▼ projected LOSS"
    print(f"  You ({report.my_projected_total:.1f}) vs {opp_name} ({report.opp_projected_total:.1f})  —  {edge} by {abs(diff):.1f} pts")
    print(f"{'='*width}")

    # ── My starters ──
    print(f"\n  MY STARTERS")
    for card in sorted(starter_cards, key=lambda c: c.predicted or 0, reverse=True):
        print_full_card(card)
    print()

    # ── Start/sit alerts ──
    if report.start_sit_alerts:
        print(f"  START/SIT ALERTS")
        for alert in report.start_sit_alerts:
            print(alert)
        print()

    # ── My bench ──
    bench_skill = [c for c in bench_cards if c.position in MODEL_POSITIONS]
    if bench_skill:
        print(f"  MY BENCH             {'Pos':<4} {'Team':<5}  {'Floor':>5} / {'Med':>5} / {'Ceil':>5}")
        print(f"  {'─'*54}")
        for card in sorted(bench_skill, key=lambda c: c.predicted or 0, reverse=True):
            print_bench_row(card)
        print()

    # ── Opponent starters ──
    if opp_cards:
        opp_skill = [c for c in opp_cards if c.position in MODEL_POSITIONS]
        print(f"  OPPONENT: {opp_name.upper()}  (proj {report.opp_projected_total:.1f} pts)")
        print(f"  {'─'*54}")
        if opp_skill:
            print(f"  {'Name':<26} {'Pos':<4} {'Team':<5}  {'Floor':>5} / {'Med':>5} / {'Ceil':>5}")
            for card in sorted(opp_skill, key=lambda c: c.predicted or 0, reverse=True):
                print_bench_row(card)
    print()
