"""
Tests for src/stats/defense.py

Run with: python -m pytest tests/test_defense_stats.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Unit tests for helper functions (no DB required) ─────────────────────────

def test_team_logo_url_known_team():
    from src.stats.defense import team_logo_url
    url = team_logo_url("SF")
    assert "sf.png" in url
    assert "espncdn.com" in url


def test_team_logo_url_legacy_abbr():
    from src.stats.defense import team_logo_url
    # OAK → lv (Raiders moved to Las Vegas)
    url = team_logo_url("OAK")
    assert "lv.png" in url


def test_team_logo_url_unknown_falls_back_to_lower():
    from src.stats.defense import team_logo_url
    url = team_logo_url("XYZ")
    assert "xyz.png" in url


def test_all_32_teams_in_slug_map():
    from src.stats.defense import TEAM_LOGO_SLUG
    nfl_teams = {
        "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
        "DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA",
        "MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WSH"
    }
    assert nfl_teams.issubset(set(TEAM_LOGO_SLUG.keys()))


# ── Rank ordering tests ───────────────────────────────────────────────────────

def test_rank_1_has_highest_fp_per_game():
    """Rank 1 should be the team that allowed the MOST FP/game."""
    rows = [
        {"team": "A", "fp_per_game": 30.0, "rank": 1, "games": 5, "logo_url": ""},
        {"team": "B", "fp_per_game": 20.0, "rank": 2, "games": 5, "logo_url": ""},
        {"team": "C", "fp_per_game": 10.0, "rank": 3, "games": 5, "logo_url": ""},
    ]
    # Ranks should be ascending and fp_per_game should be descending
    fps = [r["fp_per_game"] for r in sorted(rows, key=lambda x: x["rank"])]
    assert fps == sorted(fps, reverse=True)


def test_rank_32_has_lowest_fp_per_game():
    """Rank 32 should be the toughest defense (fewest FP allowed)."""
    rows = [{"team": str(i), "fp_per_game": float(33 - i), "rank": i, "games": 4, "logo_url": ""} for i in range(1, 33)]
    rank_32 = next(r for r in rows if r["rank"] == 32)
    assert rank_32["fp_per_game"] == min(r["fp_per_game"] for r in rows)


# ── Cache tests ───────────────────────────────────────────────────────────────

def test_cache_set_and_get():
    from src.stats import cache as c
    c._cache.clear()
    c.set("dvp", [{"team": "KC"}], "WR", 2026, 7, 4)
    result = c.get("dvp", False, "WR", 2026, 7, 4)
    assert result == [{"team": "KC"}]


def test_cache_miss_returns_none():
    from src.stats import cache as c
    c._cache.clear()
    result = c.get("dvp", False, "WR", 2026, 99, 4)
    assert result is None


def test_cache_invalidate_all():
    from src.stats import cache as c
    c.set("dvp", [1, 2, 3], "RB", 2026, 7, 4)
    c.invalidate_all()
    assert len(c._cache) == 0


def test_cache_current_week_ttl_expired():
    """Past-TTL entries for current week should be treated as misses."""
    import time
    from src.stats import cache as c
    c._cache.clear()
    k = c._key("dvp", "WR", 2026, 7, 4)
    # Fake a timestamp that's 2 hours old
    c._cache[k] = (time.time() - 7200, [{"team": "BUF"}])
    result = c.get("dvp", True, "WR", 2026, 7, 4)  # is_current_week=True
    assert result is None


def test_cache_past_week_never_expires():
    """Past-week entries should never expire regardless of age."""
    import time
    from src.stats import cache as c
    c._cache.clear()
    k = c._key("dvp", "WR", 2025, 7, 4)
    c._cache[k] = (time.time() - 999999, [{"team": "BUF"}])
    result = c.get("dvp", False, "WR", 2025, 7, 4)  # is_current_week=False
    assert result == [{"team": "BUF"}]


# ── Position validation ───────────────────────────────────────────────────────

def test_invalid_position_raises():
    from src.stats.defense import get_dvp_table
    with pytest.raises(ValueError, match="position must be"):
        get_dvp_table("K", 2026, 7)


def test_valid_positions_accepted():
    from src.stats.defense import POSITIONS
    assert set(POSITIONS) == {"QB", "RB", "WR", "TE"}


# ── get_team_dvp_rank ─────────────────────────────────────────────────────────

def test_get_team_dvp_rank_returns_none_when_not_found():
    from src.stats.defense import get_team_dvp_rank
    with patch("src.stats.defense.get_dvp_table", return_value=[]):
        result = get_team_dvp_rank("KC", "WR", 2026, 7)
    assert result is None


def test_get_team_dvp_rank_returns_correct_rank():
    from src.stats.defense import get_team_dvp_rank
    mock_table = [
        {"team": "KC", "rank": 5, "fp_per_game": 25.0, "games": 6, "logo_url": ""},
        {"team": "SF", "rank": 1, "fp_per_game": 32.0, "games": 6, "logo_url": ""},
    ]
    with patch("src.stats.defense.get_dvp_table", return_value=mock_table):
        assert get_team_dvp_rank("KC", "WR", 2026, 7) == 5
        assert get_team_dvp_rank("SF", "WR", 2026, 7) == 1
