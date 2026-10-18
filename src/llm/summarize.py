"""
Claude Haiku integration for player injury + news summarization.

Produces 2-3 sentence plain-English summaries combining:
  - Official injury designation (O/P/Q/D/IR)
  - Recent practice reports (DNP/Limited/Full)
  - Recent news headlines and snippets

Cost estimate: ~500 tokens/player × 50 players = 25K tokens ≈ $0.01/week

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

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 200  # 2-3 sentence summary


def _build_prompt(
    player_name: str,
    week: int,
    season: int,
    position: str | None,
    team: str | None,
    injury_record: dict | None,
    news_items: list[dict],
    teammate_injuries: list[dict] | None = None,
) -> str:
    """Build the prompt for Haiku summarization."""
    lines = [
        f"Player: {player_name}",
        f"Week: {week} of the {season} NFL season",
    ]
    if position:
        lines.append(f"Position: {position}")
    if team:
        lines.append(f"Team: {team}")

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

    if news_items:
        lines.append("\nRecent News:")
        for item in news_items[:4]:
            lines.append(f"  [{item.get('source', '?')}] {item.get('title', '')}")
            if item.get("summary"):
                snippet = item["summary"][:200]
                lines.append(f"    {snippet}")
    else:
        lines.append("\nRecent News: No recent news found")

    lines.append(
        "\nTask: Write a 2-3 sentence fantasy football injury and news summary for this player. "
        "Focus on: (1) current availability/status, (2) any injury concern and expected impact on production, "
        "(3) any relevant teammate injuries that elevate or reduce this player's role this week. "
        "Be concise, direct, and actionable. Do NOT include player/team/position headers — just the summary text."
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
    force_refresh: bool = False,
) -> str:
    """
    Generate or retrieve a cached LLM summary for a player.

    Returns the summary text. Caches result in news_summaries table.
    """
    init_db()

    # Check cache first
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

        prompt = _build_prompt(player_name, week, season, position, team, injury_record, news_items or [], teammate_injuries)

        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        summary_text = message.content[0].text.strip()
        logger.info(f"  LLM summary generated for {player_name} (~{message.usage.input_tokens + message.usage.output_tokens} tokens)")

        # Cache the result
        _save_summary(player_name, week, season, injury_record, summary_text)
        return summary_text

    except Exception as e:
        logger.error(f"LLM summarization failed for {player_name}: {e}")
        return _fallback_summary(player_name, injury_record)


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
