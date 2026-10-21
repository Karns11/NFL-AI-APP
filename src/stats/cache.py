"""
Simple TTL cache for stats queries.
Current-week data expires after 1 hour; past-week data is cached indefinitely.
"""

import time
from typing import Any

_cache: dict[str, tuple[float, Any]] = {}

TTL_CURRENT = 3600   # 1 hour for current-week data
TTL_PAST    = 0      # indefinite for past weeks (0 = never expire)


def _key(*parts) -> str:
    return "|".join(str(p) for p in parts)


def get(query_type: str, is_current_week: bool, *parts) -> Any | None:
    k = _key(query_type, *parts)
    entry = _cache.get(k)
    if entry is None:
        return None
    ts, data = entry
    if is_current_week and TTL_CURRENT > 0 and (time.time() - ts) > TTL_CURRENT:
        del _cache[k]
        return None
    return data


def set(query_type: str, data: Any, *parts) -> None:
    k = _key(query_type, *parts)
    _cache[k] = (time.time(), data)


def invalidate_all() -> None:
    """Call this after a stats refresh to flush all cached data."""
    _cache.clear()
