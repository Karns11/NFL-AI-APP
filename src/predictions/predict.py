"""
Unified prediction pipeline for a single player + week.

Combines:
  1. Frozen Phase 2 model (median, floor, ceiling, SHAP)
  2. Injury status from SQLite (scraped NFL.com)
  3. Recent news headlines (RSS feeds)
  4. Claude Haiku summary (cached)

Usage:
  python predict.py --player "Ja'Marr Chase" --week 8
  python predict.py --player "Justin Jefferson" --week 8 --season 2025
  python predict.py --player "Saquon Barkley" --week 8 --no-llm
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.load_data import (
    load_player_stats, load_snap_counts, load_schedules, load_player_id_map,
    build_fullname_to_gsis_map,
)
from src.features.engineer import build_features, drop_leaky_rows, get_feature_columns
from src.features.phase2 import build_phase2_features, get_phase2_feature_columns
from src.models.train import predict_player
from src.news.injury_scraper import get_player_injury, run_scrape
from src.news.rss_fetcher import get_player_news, fetch_player_news
from src.llm.summarize import summarize_player
from src.news.db import init_db

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Always include the current season dynamically
import datetime as _dt
_CURRENT_YEAR = _dt.date.today().year
# NFL season starts in September; before that, current year's season hasn't started
_CURRENT_SEASON = _CURRENT_YEAR if _dt.date.today().month >= 9 else _CURRENT_YEAR - 1
DEFAULT_SEASONS = list(range(2016, _CURRENT_SEASON + 1))


def get_current_week(season: int | None = None) -> int:
    """
    Detect the current NFL week based on today's date and the schedule.
    Returns the week whose games include or are closest after today.
    Falls back to week 1 if no schedule data is available.
    """
    if season is None:
        season = _CURRENT_SEASON
    try:
        import nflreadpy as nfl
        import polars as pl
        today_str = _dt.date.today().isoformat()
        sched = nfl.load_schedules([season])
        week_dates = (
            sched.filter(pl.col("game_type") == "REG")
            .group_by("week")
            .agg(pl.col("gameday").min().alias("first_game"))
            .sort("week")
        )
        # Current week = last week whose first game is on or before today
        past = week_dates.filter(pl.col("first_game") <= today_str)
        if len(past) == 0:
            return 1
        return past["week"].max()
    except Exception:
        return 1


def load_phase2_bundle(position: str):
    """Load the frozen Phase 2 model bundle."""
    path = MODELS_DIR / f"{position.lower()}_bundle_p2.joblib"
    if not path.exists():
        # Fall back to Phase 1
        path = MODELS_DIR / f"{position.lower()}_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No model bundle found for {position}. "
            f"Run run_phase2.py first."
        )
    return joblib.load(path)


def build_feature_df(seasons: list[int] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load and build the full feature DataFrame + schedules.
    Call once and reuse for multiple players (batch mode).
    Returns (feature_df, schedules_df).
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    logger.info("Loading data...")
    stats = load_player_stats(seasons)
    snaps = load_snap_counts(seasons)
    schedules = load_schedules(seasons)
    id_map = load_player_id_map()

    logger.info("Building features...")
    df = build_features(stats, schedules=schedules, snaps=snaps, id_map=id_map)
    df = build_phase2_features(df)
    df = drop_leaky_rows(df)
    return df, schedules


def get_player_features(
    player_name: str,
    week: int,
    season: int,
    seasons: list[int] | None = None,
    feature_df: pl.DataFrame | None = None,
    schedules_df: pl.DataFrame | None = None,
    team: str | None = None,
) -> tuple[dict | None, str | None, str | None]:
    """
    Build feature dict for a player for a specific week.

    Returns (features_dict, position, team) or (None, None, None) if not found.

    Pass pre-built feature_df + schedules_df to avoid reloading data for each player.
    """
    if feature_df is None or schedules_df is None:
        feature_df, schedules_df = build_feature_df(seasons)

    df = feature_df
    schedules = schedules_df

    name_lower = player_name.lower()
    parts = name_lower.split()

    # Strategy 1: gsis_id lookup via full name → bypasses all abbreviation issues
    # (handles "Bijan Robinson" vs "Brian Robinson" on same team perfectly)
    name_map = build_fullname_to_gsis_map()
    gsis_id = name_map.get(name_lower)
    if gsis_id:
        player_df = df.filter(pl.col("player_id") == gsis_id)

    # Strategy 2: player_display_name contains full name (nflreadpy stores this too)
    if not gsis_id or len(player_df) == 0:
        player_df = df.filter(
            pl.col("player_display_name").str.to_lowercase().str.contains(name_lower)
        )

    # Strategy 3: abbreviated name substring (e.g. "J.Williams")
    if len(player_df) == 0 and len(parts) >= 2:
        first_initial = parts[0][0]
        last = parts[-1]
        abbrev = f"{first_initial}.{last}"
        player_df = df.filter(
            pl.col("player_name").str.to_lowercase().str.contains(abbrev)
        )
        # Use team to disambiguate if multiple players match the abbreviation
        if team and player_df["player_id"].n_unique() > 1:
            team_filtered = player_df.filter(pl.col("team") == team)
            if len(team_filtered) > 0:
                player_df = team_filtered

    # Strategy 4: last name only (broadest fallback)
    if len(player_df) == 0 and len(parts) >= 2:
        last = parts[-1]
        player_df = df.filter(
            pl.col("player_name").str.to_lowercase().str.contains(last)
        )
        if team and player_df["player_id"].n_unique() > 1:
            team_filtered = player_df.filter(pl.col("team") == team)
            if len(team_filtered) > 0:
                player_df = team_filtered

    if len(player_df) == 0:
        return None, None, None

    # Filter to rows BEFORE the target week — prevents data leakage when
    # generating historical reports (e.g. Week 3 prediction must only use
    # stats from Weeks 1-2, not Weeks 4-6 which were available at query time).
    prior_df = player_df.filter(
        (pl.col("season") < season) |
        ((pl.col("season") == season) & (pl.col("week") < week))
    )
    if len(prior_df) == 0:
        return None, None, None

    # Get position and team from the most recent prior games
    recent = prior_df.sort(["season", "week"]).tail(5)
    position = recent["position"].tail(1).to_list()[0] if "position" in recent.columns else None
    team = recent["team"].tail(1).to_list()[0] if "team" in recent.columns else None
    player_id = recent["player_id"].tail(1).to_list()[0] if "player_id" in recent.columns else None

    if position is None:
        return None, None, None

    # Get features from the most recent row before the target week
    latest_row = prior_df.sort(["season", "week"]).tail(1)

    feature_cols = get_phase2_feature_columns(position, has_vegas=True)
    feature_cols = [f for f in feature_cols if f in latest_row.columns]

    features = {}
    for col in feature_cols:
        val = latest_row[col].to_list()[0]
        features[col] = val if val is not None and not (isinstance(val, float) and np.isnan(val)) else None

    # Override week/season context features with the requested week
    features["week"] = week
    features["season"] = season

    # Try to inject schedule features for the requested week/season
    sched_row = schedules.filter(
        (pl.col("season") == season) &
        (pl.col("week") == week) &
        (
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        )
    )
    if len(sched_row) > 0:
        s = sched_row.row(0, named=True)
        is_home = s.get("home_team") == team
        features["is_home"] = 1 if is_home else 0
        features["spread_line"] = s.get("spread_line")
        features["total_line"] = s.get("total_line")
        features["team_implied_total"] = (
            (s.get("total_line", 0) - abs(s.get("spread_line", 0))) / 2
            if s.get("total_line") and s.get("spread_line") is not None
            else None
        )
        opponent = s.get("away_team") if is_home else s.get("home_team")
        features["opponent"] = opponent  # informational only

    return features, position, team


def format_shap_line(name: str, value: float) -> str:
    """Format a SHAP driver line."""
    arrow = "▲" if value > 0 else "▼"
    # Make feature names human-readable
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
             .replace("is_home", "Home game")
             .replace("rest_days", "Rest days")
             .replace("season_avg_fp", "Season avg FP")
             .replace("_", " ")
    )
    return f"  {arrow} {value:+.1f}  {label}"


def build_recommendation(
    predicted: float,
    floor: float,
    ceiling: float,
    injury_status: str | None,
) -> str:
    """Generate a simple start/sit recommendation."""
    out_statuses = {"Out", "IR", "PUP", "DNP"}
    doubtful = {"D"}
    questionable = {"Q"}

    if injury_status in out_statuses:
        return "DO NOT START — Player is OUT"
    elif injury_status in doubtful:
        return "DOUBTFUL — Have a backup ready"
    elif injury_status in questionable:
        return "MONITOR — Start if active; have a backup"

    if predicted >= 20:
        return "STRONG START"
    elif predicted >= 15:
        return "START"
    elif predicted >= 10:
        return "FLEX CONSIDERATION"
    else:
        return "BENCH"


def predict_and_display(
    player_name: str,
    week: int,
    season: int,
    use_llm: bool = True,
    refresh_news: bool = False,
    seasons: list[int] | None = None,
    feature_df: pl.DataFrame | None = None,
    schedules_df: pl.DataFrame | None = None,
) -> dict | None:
    """
    Full prediction pipeline. Prints the player card and returns result dict.

    Pass pre-built feature_df + schedules_df (from build_feature_df()) to avoid
    reloading data when running multiple players in a batch.
    """
    init_db()

    # 1. Get model features
    print(f"\nLooking up {player_name}...")
    features, position, team = get_player_features(
        player_name, week, season, seasons,
        feature_df=feature_df, schedules_df=schedules_df,
    )

    if features is None:
        print(f"  Player '{player_name}' not found in historical data.")
        print("  Try using a more common version of the name (e.g., 'Ja'Marr Chase', 'Justin Jefferson').")
        return None

    # 2. Load frozen model bundle
    try:
        bundle = load_phase2_bundle(position)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    # 3. Run model prediction
    pred = predict_player(bundle, features)
    predicted = pred["predicted"]
    floor = pred["floor"]
    ceiling = pred["ceiling"]

    # 4. Injury status
    if refresh_news:
        print(f"  Fetching injury report...")
        run_scrape(week, season)

    injury = get_player_injury(player_name, week, season)
    injury_status = injury.get("status") if injury else None
    injury_type = injury.get("injury") if injury else None

    # 5. Recent news
    if refresh_news:
        print(f"  Fetching news...")
        fetch_player_news(player_name, team)

    news_items = get_player_news(player_name, max_items=4)

    # 6. LLM summary
    summary_text = None
    if use_llm:
        print(f"  Generating summary...")
        summary_text = summarize_player(
            player_name=player_name,
            week=week,
            season=season,
            position=position,
            team=team,
            injury_record=injury,
            news_items=news_items,
        )

    # 7. Recommendation
    recommendation = build_recommendation(predicted, floor, ceiling, injury_status)

    # 8. Determine opponent for display
    opponent = features.get("opponent", "?")
    home_away = "vs." if features.get("is_home") == 1 else "@"

    # ─── Print the prediction card ─────────────────────────────────────────
    width = 54
    sep = "━" * width

    print(f"\n{sep}")
    header = f"  {player_name.upper()}  |  {position}  |  {team}"
    if opponent and opponent != "?":
        header += f"  |  Week {week} {home_away} {opponent}"
    print(header)
    print(sep)

    print(f"\n  PROJECTION")
    print(f"    Median:   {predicted:>5.1f} pts")
    print(f"    Floor:    {floor:>5.1f} pts  (Q10)")
    print(f"    Ceiling:  {ceiling:>5.1f} pts  (Q90)")

    if pred["shap_top_positive"] or pred["shap_top_negative"]:
        print(f"\n  MODEL DRIVERS (SHAP)")
        for name, val in pred["shap_top_positive"][:3]:
            print(format_shap_line(name, val))
        for name, val in pred["shap_top_negative"][:2]:
            print(format_shap_line(name, val))

    print(f"\n  INJURY & NEWS")
    if injury_status:
        status_line = f"  Status: {injury_status}"
        if injury_type:
            status_line += f" ({injury_type})"
        print(status_line)
    else:
        print(f"  Status: Not on injury report")

    if summary_text:
        # Word-wrap summary at ~50 chars
        import textwrap
        for line in textwrap.wrap(summary_text, width=50):
            print(f"  {line}")
    elif news_items:
        print(f"  Recent news:")
        for item in news_items[:2]:
            title = item.get("title", "")
            if len(title) > 50:
                title = title[:47] + "..."
            print(f"    [{item.get('source', '?')}] {title}")
    else:
        print(f"  No recent news found.")

    print(f"\n  RECOMMENDATION:  {recommendation}")
    print(f"{sep}\n")

    return {
        "player_name": player_name,
        "position": position,
        "team": team,
        "week": week,
        "season": season,
        "predicted": predicted,
        "floor": floor,
        "ceiling": ceiling,
        "injury_status": injury_status,
        "recommendation": recommendation,
        "summary": summary_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NFL Fantasy AI — Player Prediction Card"
    )
    parser.add_argument("--player", "-p", required=True, help="Player name (e.g. 'Ja'Marr Chase')")
    parser.add_argument("--week", "-w", type=int, required=True, help="NFL week number")
    parser.add_argument("--season", "-s", type=int, default=_CURRENT_SEASON, help=f"Season year (default: {_CURRENT_SEASON})")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM summary (saves API calls)")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch injury + news (vs. use cached)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,  # suppress info logs for clean output
        format="%(levelname)s %(message)s",
    )
    # Show our logger at INFO so progress messages still appear
    logging.getLogger("src.predictions.predict").setLevel(logging.INFO)

    predict_and_display(
        player_name=args.player,
        week=args.week,
        season=args.season,
        use_llm=not args.no_llm,
        refresh_news=args.refresh,
    )


if __name__ == "__main__":
    main()
