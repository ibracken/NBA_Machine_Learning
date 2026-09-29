"""
Tonight's rolling features from box-score history
"""

import pandas as pd

from config import NBA_TEAM_COUNT, PROJECTION_START_MIN_TEAM_GAMES

ROLLING_COLS = ['Last3_FP_Avg', 'Last7_FP_Avg', 'Season_FP_Avg', 'Career_FP_Avg', 'Games_Played_Career',
                'Last7_MIN_Avg', 'Season_MIN_Avg', 'Career_MIN_Avg']


def latest_rows_as_of_tonight(box_scores):
    """
    Each player's most recent box-score row, with rolling features advanced to include that game.

    box-score-scraper computes every rolling column with shift(1), so a row describes the history
    *before* its game - exactly right for training. Serving from the latest row as-is therefore
    drops the most recent game from every average. Tonight's features must cover all games played.

    Season and last-N windows are recomputed from this season's games. Career windows span seasons
    not present here, so they are advanced incrementally from the latest row: Games_Played_Career
    counts the games before that row and Career_*_Avg is their mean.
    """
    games = box_scores.copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values(['PLAYER', 'GAME_DATE'])

    latest = games.groupby('PLAYER').tail(1).set_index('PLAYER')

    # Rolling windows reset each season; keep only games from each player's latest season
    if 'SEASON' in games.columns:
        games = games[games['SEASON'] == games['PLAYER'].map(latest['SEASON'])]
    by_player = games.groupby('PLAYER')

    latest['Season_MIN_Avg'] = by_player['MIN'].mean()
    latest['Season_FP_Avg'] = by_player['FP'].mean()
    latest['Last7_MIN_Avg'] = by_player['MIN'].apply(lambda s: s.tail(7).mean())
    latest['Last7_FP_Avg'] = by_player['FP'].apply(lambda s: s.tail(7).mean())
    latest['Last3_FP_Avg'] = by_player['FP'].apply(lambda s: s.tail(3).mean())

    prior_games = latest['Games_Played_Career'].fillna(0)
    for avg_col, value_col in [('Career_MIN_Avg', 'MIN'), ('Career_FP_Avg', 'FP')]:
        prior_total = latest[avg_col].fillna(0) * prior_games
        latest[avg_col] = (prior_total + latest[value_col]) / (prior_games + 1)
    latest['Games_Played_Career'] = prior_games + 1

    return latest.reset_index()


def season_of(date):
    """NBA season label ('2026-27') for a date; seasons start in October."""
    date = pd.Timestamp(date)
    start = date.year if date.month >= 10 else date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def projection_gate(box_scores, today):
    """
    (paused, reason). Paused until all NBA_TEAM_COUNT teams have played PROJECTION_START_MIN_TEAM_GAMES
    games of today's season before today. Mirrored in llm-analyst/slate.py - keep in sync.
    """
    season = season_of(today)
    if box_scores.empty:
        return True, f"no box scores for {season}"
    dates = pd.to_datetime(box_scores['GAME_DATE'])
    played = box_scores[(dates.apply(season_of) == season) & (dates.dt.date < pd.Timestamp(today).date())]
    team_games = played.groupby('TEAM_ABBREVIATION')['GAME_ID'].nunique()
    short = NBA_TEAM_COUNT - int((team_games >= PROJECTION_START_MIN_TEAM_GAMES).sum())
    if short > 0:
        return True, (f"{short} of {NBA_TEAM_COUNT} teams have fewer than {PROJECTION_START_MIN_TEAM_GAMES} "
                      f"{season} games")
    return False, ''
