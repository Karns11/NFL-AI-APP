"""
Personalized fantasy report generator.

For each league, produces:
  - My starters with projections
  - My bench with projections
  - Opponent's starters with projections
  - Start/sit recommendations (best available vs. current starters)
  - Projected matchup score
"""

import logging
from dataclasses import dataclass, field

import polars as pl

from .sleeper import get_my_matchup

logger = logging.getLogger(__name__)

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# Positions the model supports
MODEL_POSITIONS = {"QB", "RB", "WR", "TE"}


@dataclass
class PlayerProjection:
    sleeper_id: str
    name: str
    position: str
    team: str
    is_starter: bool
    predicted: float | None = None
    floor: float | None = None
    ceiling: float | None = None
    injury_status: str | None = None
    recommendation: str | None = None


@dataclass
class LeagueReport:
    league_name: str
    league_id: str
    scoring_format: str
    week: int
    season: int
    my_starters: list[PlayerProjection] = field(default_factory=list)
    my_bench: list[PlayerProjection] = field(default_factory=list)
    opp_starters: list[PlayerProjection] = field(default_factory=list)
    my_projected_total: float = 0.0
    opp_projected_total: float = 0.0
    start_sit_alerts: list[str] = field(default_factory=list)


def _project_player(
    player_info: dict,
    week: int,
    season: int,
    feature_df: pl.DataFrame,
    schedules_df: pl.DataFrame,
    bundle_cache: dict,
) -> tuple[float | None, float | None, float | None]:
    """
    Run model prediction for a player. Returns (predicted, floor, ceiling).
    Returns (None, None, None) for unsupported positions (K, DEF, etc).
    """
    position = player_info.get("position", "")
    name = player_info.get("name", "")
    known_team = player_info.get("team")

    if position not in MODEL_POSITIONS or not name or name.strip() == "":
        return None, None, None

    # Lazy import to avoid circular deps
    from src.predictions.predict import get_player_features
    from src.models.train import predict_player

    if position not in bundle_cache:
        try:
            from src.predictions.predict import load_phase2_bundle
            bundle_cache[position] = load_phase2_bundle(position)
        except FileNotFoundError:
            return None, None, None

    bundle = bundle_cache[position]

    features, pos_found, team = get_player_features(
        name, week, season,
        feature_df=feature_df,
        schedules_df=schedules_df,
        team=known_team,
    )
    if features is None:
        return None, None, None

    try:
        pred = predict_player(bundle, features)
        return pred["predicted"], pred["floor"], pred["ceiling"]
    except Exception as e:
        logger.debug(f"  Prediction failed for {name}: {e}")
        return None, None, None


def build_league_report(
    league: dict,
    week: int,
    season: int,
    feature_df: pl.DataFrame,
    schedules_df: pl.DataFrame,
    bundle_cache: dict,
) -> LeagueReport:
    """Build a full personalized report for one league."""
    report = LeagueReport(
        league_name=league["name"],
        league_id=league["league_id"],
        scoring_format=league.get("_scoring_format", "PPR"),
        week=week,
        season=season,
    )

    player_info = league.get("_player_info", {})
    my_starters_ids = league.get("_my_starters_ids", [])
    my_bench_ids = league.get("_my_bench_ids", [])
    roster_by_id = league.get("_roster_by_id", {})
    user_display = league.get("_user_display", {})

    # Get matchup
    my_matchup, opp_matchup = get_my_matchup(league, week)
    opp_starters_ids = opp_matchup.get("starters", []) if opp_matchup else []

    # Find opponent owner name
    opp_roster_id = opp_matchup["roster_id"] if opp_matchup else None
    opp_roster = roster_by_id.get(opp_roster_id, {}) if opp_roster_id else {}
    opp_owner_id = opp_roster.get("owner_id", "")
    opp_name = user_display.get(opp_owner_id, f"Roster {opp_roster_id}")
    report._opp_name = opp_name

    # Project my starters
    for sid in my_starters_ids:
        info = player_info.get(str(sid), {"name": sid, "position": "?", "team": "?"})
        predicted, floor, ceiling = _project_player(
            info, week, season, feature_df, schedules_df, bundle_cache
        )
        proj = PlayerProjection(
            sleeper_id=sid,
            name=info.get("name", sid),
            position=info.get("position", "?"),
            team=info.get("team", "?"),
            is_starter=True,
            predicted=predicted,
            floor=floor,
            ceiling=ceiling,
            injury_status=info.get("injury_status"),
        )
        report.my_starters.append(proj)
        if predicted is not None:
            report.my_projected_total += predicted

    # Project my bench
    for sid in my_bench_ids:
        info = player_info.get(str(sid), {"name": sid, "position": "?", "team": "?"})
        predicted, floor, ceiling = _project_player(
            info, week, season, feature_df, schedules_df, bundle_cache
        )
        proj = PlayerProjection(
            sleeper_id=sid,
            name=info.get("name", sid),
            position=info.get("position", "?"),
            team=info.get("team", "?"),
            is_starter=False,
            predicted=predicted,
            floor=floor,
            ceiling=ceiling,
            injury_status=info.get("injury_status"),
        )
        report.my_bench.append(proj)

    # Project opponent starters
    for sid in opp_starters_ids:
        info = player_info.get(str(sid), {"name": sid, "position": "?", "team": "?"})
        predicted, floor, ceiling = _project_player(
            info, week, season, feature_df, schedules_df, bundle_cache
        )
        proj = PlayerProjection(
            sleeper_id=sid,
            name=info.get("name", sid),
            position=info.get("position", "?"),
            team=info.get("team", "?"),
            is_starter=True,
            predicted=predicted,
            floor=floor,
            ceiling=ceiling,
        )
        report.opp_starters.append(proj)
        if predicted is not None:
            report.opp_projected_total += predicted

    # Start/sit alerts: bench player projected higher than a starter at same position
    _add_start_sit_alerts(report)

    return report


def _add_start_sit_alerts(report: LeagueReport) -> None:
    """Flag bench players projected significantly higher than a current starter."""
    starter_by_pos: dict[str, list[PlayerProjection]] = {}
    for p in report.my_starters:
        if p.position in MODEL_POSITIONS and p.predicted is not None:
            starter_by_pos.setdefault(p.position, []).append(p)

    for bench_player in report.my_bench:
        if bench_player.position not in MODEL_POSITIONS:
            continue
        if bench_player.predicted is None:
            continue

        starters = starter_by_pos.get(bench_player.position, [])
        for starter in starters:
            if starter.predicted is None:
                continue
            gap = bench_player.predicted - starter.predicted
            if gap >= 3.0:
                report.start_sit_alerts.append(
                    f"  ⚠  START {bench_player.name} ({bench_player.predicted:.1f}) "
                    f"over {starter.name} ({starter.predicted:.1f}) "
                    f"[+{gap:.1f} pts]"
                )


# ── Display ───────────────────────────────────────────────────────────────────

def print_league_report(report: LeagueReport) -> None:
    """Print the full personalized report for one league."""
    width = 60
    sep = "━" * width
    thin = "─" * width

    opp_name = getattr(report, "_opp_name", "Opponent")

    print(f"\n{'='*width}")
    print(f"  {report.league_name.upper()}")
    print(f"  {report.scoring_format}  |  Week {report.week}  |  Season {report.season}")
    print(f"{'='*width}")

    # ── My starters ──
    print(f"\n  MY STARTERS  (proj: {report.my_projected_total:.1f} pts)")
    print(f"  {thin}")
    _print_player_table(report.my_starters)

    # ── My bench ──
    bench_skill = [p for p in report.my_bench if p.position in MODEL_POSITIONS]
    if bench_skill:
        print(f"\n  MY BENCH")
        print(f"  {thin}")
        _print_player_table(bench_skill)

    # ── Opponent starters ──
    if report.opp_starters:
        print(f"\n  OPPONENT: {opp_name.upper()}  (proj: {report.opp_projected_total:.1f} pts)")
        print(f"  {thin}")
        _print_player_table(report.opp_starters)

    # ── Matchup projection ──
    print(f"\n  MATCHUP PROJECTION")
    print(f"  {thin}")
    me_str = f"You: {report.my_projected_total:.1f}"
    opp_str = f"{opp_name}: {report.opp_projected_total:.1f}"
    diff = report.my_projected_total - report.opp_projected_total
    edge = "▲ PROJECTED WIN" if diff > 0 else "▼ PROJECTED LOSS"
    print(f"  {me_str}  vs.  {opp_str}   ({edge}, {abs(diff):.1f} pt margin)")

    # ── Start/sit alerts ──
    if report.start_sit_alerts:
        print(f"\n  START/SIT ALERTS")
        print(f"  {thin}")
        for alert in report.start_sit_alerts:
            print(alert)

    print()


def _print_player_table(players: list[PlayerProjection]) -> None:
    """Print a compact player projection table."""
    print(f"  {'Name':<24} {'Pos':<4} {'Team':<5} {'Floor':>6} {'Med':>6} {'Ceil':>6}  Status")
    for p in sorted(players, key=lambda x: (x.predicted or 0), reverse=True):
        floor_str = f"{p.floor:.1f}" if p.floor is not None else "  —"
        med_str = f"{p.predicted:.1f}" if p.predicted is not None else "  —"
        ceil_str = f"{p.ceiling:.1f}" if p.ceiling is not None else "  —"
        status = p.injury_status or ""
        name = p.name[:23] if p.name else "Unknown"
        print(f"  {name:<24} {p.position:<4} {p.team:<5} {floor_str:>6} {med_str:>6} {ceil_str:>6}  {status}")
