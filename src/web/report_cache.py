"""
Per-user, per-week report cache.

Each report is stored at:
  data/reports/{sleeper_username}_wk{week}_{season}.json

Also kept in an in-memory dict for fast access within a session.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/reports")

# In-memory cache: cache_key → dict
_cache: dict[str, dict] = {}


@dataclass
class CachedReport:
    generated_at: str
    week: int
    season: int
    username: str
    leagues: list[dict]


def _cache_key(sleeper_username: str, week: int, season: int) -> str:
    return f"{sleeper_username.lower()}_wk{week}_{season}"


def _path(sleeper_username: str, week: int, season: int) -> Path:
    return CACHE_DIR / f"{_cache_key(sleeper_username, week, season)}.json"


def get(sleeper_username: str, week: int, season: int) -> CachedReport | None:
    """Return cached report for a user+week, loading from disk if needed."""
    key = _cache_key(sleeper_username, week, season)
    if key in _cache:
        return CachedReport(**_cache[key])

    p = _path(sleeper_username, week, season)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            _cache[key] = data
            return CachedReport(**data)
        except Exception as e:
            logger.warning(f"Failed to load report cache {p.name}: {e}")

    return None


def set(sleeper_username: str, report: CachedReport) -> None:
    """Store a report in memory and on disk."""
    data = asdict(report)
    key = _cache_key(sleeper_username, report.week, report.season)
    _cache[key] = data

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _path(sleeper_username, report.week, report.season).write_text(
            json.dumps(data, indent=2, default=str)
        )
    except Exception as e:
        logger.warning(f"Failed to persist report cache for {sleeper_username} wk{report.week}: {e}")


def age_minutes(sleeper_username: str, week: int, season: int) -> float | None:
    """Return how many minutes ago the report for this week was generated."""
    cached = get(sleeper_username, week, season)
    if not cached:
        return None
    try:
        generated = datetime.fromisoformat(cached.generated_at)
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - generated).total_seconds() / 60
    except Exception:
        return None
