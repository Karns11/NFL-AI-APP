"""
Claude integration for player and matchup AI summarization.

Player summaries: Claude Haiku 5.5 (fast, cheap, high-volume)
Matchup summaries: Claude Sonnet 5.5 (premium reasoning, 1 call/league/week)

Cost estimate: ~$0.033/week for 2 leagues + ~30 players

Requires ANTHROPIC_API_KEY in environment or .env file.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from ..news.db import get_conn, init_db

# Load .env from project root regardless of working directory
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

PLAYER_MODEL  = "claude-haiku-4-5-20251001"   # High-volume per-player summaries
MATCHUP_MODEL = "claude-sonnet-4-6"           # One per league per week, premium quality

PLAYER_MAX_TOKENS  = 300   # 3-4 rich sentences
MATCHUP_MAX_TOKENS = 500   # 5-6 paragraph-quality sentences


def _extract_text(message) -> str:
    """Extract text content from an API response, skipping any ThinkingBlocks."""
    for block in message.content:
        if hasattr(block, "text"):
            return block.text.strip()
    raise ValueError("No text block found in response")


def _build_prompt(
    player_name: str,
    week: int,
    season: int,
    position: str | None,
    team: str | None,
    injury_record: dict | None,
    news_items: list[dict],
    teammate_injuries: list[dict] | None = None,
    on_bye: bool = False,
    predicted: float | None = None,
    floor: float | None = None,
    ceiling: float | None = None,
    shap_positive: list | None = None,
    shap_negative: list | None = None,
) -> str:
    """Build the prompt for player summarization."""
    lines = [
        f"Player: {player_name}",
        f"Week: {week} of the {season} NFL season",
    ]
    if position:
        lines.append(f"Position: {position}")
    if team:
        lines.append(f"Team: {team}")

    if on_bye:
        lines.append(f"\nSchedule: {team} has a BYE WEEK this week — {player_name} will NOT play.")
        lines.append(
            "\nTask: Write 1 sentence noting this player is on bye this week and cannot be started. "
            "Be direct. Do NOT include headers — just the summary text."
        )
        return "\n".join(lines)

    if injury_record:
        lines.append(f"\nInjury Report:")
        if injury_record.get("status"):
            lines.append(f"  Game Status: {injury_record['status']}")
        if injury_record.get("injury"):
            lines.append(f"  Injury: {injury_record['injury']}")
        if injury_record.get("practice"):
            lines.append(f"  Practice: {injury_record['practice']}")
    else:
        lines.append("\nInjury Report: Not listed on injury report (healthy)")

    if teammate_injuries:
        lines.append("\nTeammate Injuries (same team this week):")
        for t in teammate_injuries:
            t_name = t.get("player_name", "Unknown")
            t_pos = t.get("position", "")
            t_status = t.get("status", "")
            t_injury = t.get("injury", "")
            detail = f"  - {t_name} ({t_pos}): {t_status}"
            if t_injury:
                detail += f" — {t_injury}"
            lines.append(detail)

    if predicted is not None:
        lines.append(f"\nModel Projection:")
        lines.append(f"  Expected points: {predicted:.1f}")
        if floor is not None and ceiling is not None:
            lines.append(f"  Range: Floor {floor:.1f} · Ceiling {ceiling:.1f}")
        if shap_positive:
            lines.append(f"  Key factors boosting projection:")
            for feat, val in (shap_positive or [])[:3]:
                feat_label = (feat
                    .replace("snap_pct_roll3", "Snap%")
                    .replace("snap_pct_ewm", "Snap%")
                    .replace("team_implied_total", "Game Total")
                    .replace("spread_line", "Spread")
                    .replace("opp_fp_allowed_roll4", "Opp DEF")
                    .replace("_", " "))
                lines.append(f"    ▲ {feat_label}: +{val:.1f}")
        if shap_negative:
            lines.append(f"  Key factors suppressing projection:")
            for feat, val in (shap_negative or [])[:2]:
                feat_label = (feat
                    .replace("snap_pct_roll3", "Snap%")
                    .replace("snap_pct_ewm", "Snap%")
                    .replace("team_implied_total", "Game Total")
                    .replace("spread_line", "Spread")
                    .replace("opp_fp_allowed_roll4", "Opp DEF")
                    .replace("_", " "))
                lines.append(f"    ▼ {feat_label}: -{abs(val):.1f}")

    if news_items:
        lines.append("\nRecent News:")
        for item in news_items[:4]:
            lines.append(f"  [{item.get('source', '?')}] {item.get('title', '')}")
            if item.get("summary"):
                snippet = item["summary"][:200]
                lines.append(f"    {snippet}")
    else:
        lines.append("\nRecent News: No recent news found")

    if predicted is None:
        lines.append(
            "\nTask: Write 2-3 sentences summarizing what a fantasy manager needs to know about this player "
            "this week. Focus on their current role, availability, and any relevant news or injury context. "
            "Do not reference projected points or model drivers — none are available for this player. "
            "Be direct and practical. Do NOT include headers — just the summary text."
        )
    else:
        lines.append(
            "\nTask: Write a concise premium fantasy football analysis of this player for the upcoming week "
            "using the available news, injury information, model projection, floor/ceiling range, and model drivers. "
            "Explain what the information means for the player's expected fantasy performance rather than simply repeating it. "
            "Identify the most important factors influencing his projection, including any meaningful positive or negative "
            "model drivers, changes in availability or role, matchup considerations, and relevant teammate situations. "
            "Distinguish between information that increases confidence in the projection and information that creates "
            "uncertainty or downside. "
            "Give the reader a clear bottom-line expectation for the player this week: what should a fantasy manager "
            "realistically expect, what could cause him to outperform or underperform, and how much confidence should "
            "they have in the projection. "
            "Use specific numbers and player names when useful. Be analytical and opinionated, but do not invent "
            "information or make generic fantasy-football statements. "
            "IMPORTANT: Only reference teammates, offensive schemes, or roster situations that are explicitly mentioned "
            "in the injury or news data provided above. Do NOT rely on your training data knowledge of historical rosters — "
            "rosters change every season and your knowledge may be outdated. If no teammate context is provided, do not speculate about it. "
            "Do NOT include headers — just the summary text."
        )

    return "\n".join(lines)


def summarize_player(
    player_name: str,
    week: int,
    season: int,
    position: str | None = None,
    team: str | None = None,
    injury_record: dict | None = None,
    news_items: list[dict] | None = None,
    teammate_injuries: list[dict] | None = None,
    on_bye: bool = False,
    predicted: float | None = None,
    floor: float | None = None,
    ceiling: float | None = None,
    shap_positive: list | None = None,
    shap_negative: list | None = None,
    force_refresh: bool = False,
) -> str:
    """
    Generate or retrieve a cached LLM summary for a player.
    Returns the summary text. Caches result in news_summaries table.
    """
    init_db()

    if not force_refresh:
        cached = _get_cached_summary(player_name, week, season)
        if cached:
            logger.debug(f"  Using cached summary for {player_name}")
            return cached

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM summary")
        return _fallback_summary(player_name, injury_record)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = _build_prompt(
            player_name, week, season, position, team,
            injury_record, news_items or [], teammate_injuries,
            on_bye=on_bye,
            predicted=predicted, floor=floor, ceiling=ceiling,
            shap_positive=shap_positive, shap_negative=shap_negative,
        )

        message = client.messages.create(
            model=PLAYER_MODEL,
            max_tokens=PLAYER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        summary_text = _extract_text(message)
        logger.info(f"  LLM summary generated for {player_name} (~{message.usage.input_tokens + message.usage.output_tokens} tokens)")

        _save_summary(player_name, week, season, injury_record, summary_text)
        return summary_text

    except Exception as e:
        logger.error(f"LLM summarization failed for {player_name}: {e}")
        return _fallback_summary(player_name, injury_record)


def summarize_matchup(
    league_name: str,
    scoring_format: str,
    week: int,
    season: int,
    my_projected_total: float,
    opp_projected_total: float,
    opp_name: str,
    my_starters: list[dict],
    opp_starters: list[dict],
    start_sit_alerts: list[str],
    my_sleeper_total: float | None = None,
    force_refresh: bool = False,
) -> str | None:
    """
    Generate or retrieve a cached AI narrative for a fantasy matchup.
    Cached in news_summaries table using '__matchup__{league_name}' as player_name.
    Returns None on failure.
    """
    init_db()
    cache_key = f"__matchup__{league_name}"

    if not force_refresh:
        cached = _get_cached_summary(cache_key, week, season)
        if cached:
            logger.debug(f"  Using cached matchup summary for {league_name}")
            return cached

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping matchup summary")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_matchup_prompt(
            league_name, scoring_format, week, season,
            my_projected_total, opp_projected_total, opp_name,
            my_starters, opp_starters, start_sit_alerts, my_sleeper_total,
        )
        message = client.messages.create(
            model=MATCHUP_MODEL,
            max_tokens=MATCHUP_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        summary_text = _extract_text(message)
        logger.info(f"  Matchup summary generated for {league_name} (~{message.usage.input_tokens + message.usage.output_tokens} tokens)")
        _save_summary(cache_key, week, season, None, summary_text)
        return summary_text

    except Exception as e:
        logger.error(f"Matchup summarization failed for {league_name}: {e}")
        return None


def _build_matchup_prompt(
    league_name: str,
    scoring_format: str,
    week: int,
    season: int,
    my_projected_total: float,
    opp_projected_total: float,
    opp_name: str,
    my_starters: list[dict],
    opp_starters: list[dict],
    start_sit_alerts: list[str],
    my_sleeper_total: float | None,
) -> str:
    lines = [
        f"Fantasy Football Matchup Analysis",
        f"League: {league_name} ({scoring_format})",
        f"Week {week}, {season} season",
        f"",
        f"Projected Score: You {my_projected_total:.1f} vs {opp_name} {opp_projected_total:.1f} "
        f"(margin: {my_projected_total - opp_projected_total:+.1f})",
    ]
    if my_sleeper_total is not None:
        lines.append(f"Sleeper Projection: You {my_sleeper_total:.1f} (vs model {my_projected_total:.1f})")

    def _fmt_player(p: dict) -> str:
        name = p.get("name", "?")
        pos = p.get("position", "?")
        team = p.get("team", "?")
        proj = p.get("predicted")
        floor_ = p.get("floor")
        ceil_ = p.get("ceiling")
        inj = p.get("injury_status")
        bye = p.get("on_bye", False)
        if bye:
            return f"  - {name} ({pos}/{team}): BYE WEEK — will not play"
        proj_str = f"{proj:.1f}" if proj is not None else "—"
        range_str = f" (floor {floor_:.1f}–ceil {ceil_:.1f})" if floor_ and ceil_ else ""
        inj_str = f" [{inj}]" if inj else ""
        # Flag high-variance players
        variance_flag = ""
        if floor_ and ceil_ and (ceil_ - floor_) > 10:
            variance_flag = " ⚡high-variance"
        summary = p.get("llm_summary") or ""
        detail = f"  - {name} ({pos}/{team}): {proj_str} pts{range_str}{inj_str}{variance_flag}"
        if summary:
            detail += f"\n    Analysis: {summary}"
        return detail

    lines += ["", "My Starters:"]
    for p in my_starters[:12]:
        lines.append(_fmt_player(p))

    lines += ["", f"{opp_name}'s Starters:"]
    for p in opp_starters[:12]:
        lines.append(_fmt_player(p))

    if start_sit_alerts:
        lines += ["", "Start/Sit Alerts (bench players outprojecting starters):"]
        for alert in start_sit_alerts:
            lines.append(f"  {alert}")

    lines += [
        "",
        "Task: Write a premium fantasy football matchup analysis based on all of the roster, projection, "
        "player-summary, matchup, injury, and variance information provided. "
        "Your job is to interpret the data and tell the story of how this matchup is likely to play out — "
        "not simply restate the numbers. "
        "Explain why the model favors one team, identify the 2–3 players or situations that are most likely "
        "to determine the outcome, and highlight any meaningful advantages or vulnerabilities on either roster. "
        "Pay particular attention to players whose projections, floor/ceiling ranges, injuries, usage, or matchup "
        "conditions create meaningful leverage or could cause the result to deviate from the projection. "
        "Discuss the most important opponent threats and explain what would have to happen for the projected "
        "outcome to change. "
        "Incorporate relevant player-level insights, but prioritize the information that actually matters to "
        "this matchup. "
        "If there is a meaningful start/sit or bench decision, call it out clearly and explain why. "
        "End with a confident prediction of the winner and the primary reason you believe they will win. "
        "Use specific player names and relevant numbers throughout. "
        "Be analytical, opinionated, and concise. Do not mechanically address every piece of available data, "
        "and do not make generic fantasy-football statements. "
        "Focus on the few insights that would actually help a fantasy manager make a better decision. "
        "Do not restate information already visible in the individual player summaries — synthesize and elevate. "
        "No headers — just the analysis text.",
    ]
    return "\n".join(lines)


def _fallback_summary(player_name: str, injury_record: dict | None) -> str:
    """Return a simple rule-based summary when LLM is unavailable."""
    if not injury_record:
        return f"{player_name} is not on the injury report. No availability concerns this week."

    status = injury_record.get("status", "Unknown")
    injury = injury_record.get("injury", "undisclosed")
    practice = injury_record.get("practice", "")

    if status in ("Out", "IR", "PUP"):
        return f"{player_name} is ruled OUT ({injury}). Do not start."
    elif status == "D":
        return f"{player_name} is listed as DOUBTFUL ({injury}). Have a backup ready."
    elif status == "Q":
        practice_note = f" ({practice} practice)" if practice else ""
        return f"{player_name} is QUESTIONABLE{practice_note} with a {injury} injury. Monitor status through game time."
    else:
        return f"{player_name} is listed as {status} with a {injury} concern. Expect to play."


def _get_cached_summary(player_name: str, week: int, season: int) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT summary_text FROM news_summaries
            WHERE lower(player_name) LIKE lower(?)
              AND week = ? AND season = ?
            ORDER BY summarized_at DESC LIMIT 1
            """,
            (f"%{player_name}%", week, season),
        ).fetchone()
    return row["summary_text"] if row else None


def _save_summary(
    player_name: str,
    week: int,
    season: int,
    injury_record: dict | None,
    summary_text: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    injury_status = injury_record.get("status") if injury_record else None

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO news_summaries
                (summarized_at, player_name, week, season, injury_status, summary_text)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_name, week, season) DO UPDATE SET
                summarized_at = excluded.summarized_at,
                injury_status = excluded.injury_status,
                summary_text  = excluded.summary_text
            """,
            (now, player_name, week, season, injury_status, summary_text),
        )
