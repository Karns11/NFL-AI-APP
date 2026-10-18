"""
In-memory + disk cache for the last-generated league reports.

Keeps a single snapshot of all league data so the web page loads
instantly without re-running the full pipeline on every request.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/report_cache.json")

# In-memory state
_cache: dict = {}


@dataclass
class CachedReport:
    generated_at: str          # ISO timestamp
    week: int
    season: int
    username: str
    leagues: list[dict]        # serialized league bundles


def get() -> CachedReport | None:
    """Return the in-memory cached report, or load from disk if available."""
    if _cache:
        return CachedReport(**_cache)

    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text())
            _cache.update(data)
            return CachedReport(**_cache)
        except Exception as e:
            logger.warning(f"Failed to load report cache from disk: {e}")

    return None


def set(report: CachedReport) -> None:
    """Store a new report snapshot in memory and on disk."""
    data = asdict(report)
    _cache.clear()
    _cache.update(data)

    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.warning(f"Failed to persist report cache: {e}")


def age_minutes() -> float | None:
    """Return how many minutes ago the cache was generated, or None."""
    cached = get()
    if not cached:
        return None
    try:
        generated = datetime.fromisoformat(cached.generated_at)
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - generated
        return delta.total_seconds() / 60
    except Exception:
        return None
