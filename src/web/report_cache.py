"""
Per-user report cache.

Each user's last-generated report is stored at:
  data/reports/{sleeper_username}.json

Also kept in an in-memory dict keyed by sleeper_username for fast access.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/reports")

# In-memory cache: sleeper_username → dict
_cache: dict[str, dict] = {}


@dataclass
class CachedReport:
    generated_at: str
    week: int
    season: int
    username: str
    leagues: list[dict]


def _path(sleeper_username: str) -> Path:
    return CACHE_DIR / f"{sleeper_username}.json"


def get(sleeper_username: str) -> CachedReport | None:
    """Return cached report for a user, loading from disk if needed."""
    key = sleeper_username.lower()
    if key in _cache:
        return CachedReport(**_cache[key])

    p = _path(sleeper_username)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            _cache[key] = data
            return CachedReport(**data)
        except Exception as e:
            logger.warning(f"Failed to load report cache for {sleeper_username}: {e}")

    return None


def set(sleeper_username: str, report: CachedReport) -> None:
    """Store a report for a user in memory and on disk."""
    data = asdict(report)
    key = sleeper_username.lower()
    _cache[key] = data

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _path(sleeper_username).write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.warning(f"Failed to persist report cache for {sleeper_username}: {e}")


def age_minutes(sleeper_username: str) -> float | None:
    """Return how many minutes ago the user's report was generated."""
    cached = get(sleeper_username)
    if not cached:
        return None
    try:
        generated = datetime.fromisoformat(cached.generated_at)
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - generated).total_seconds() / 60
    except Exception:
        return None
