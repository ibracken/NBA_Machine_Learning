"""
Tonight's slate: games, the DFF players in each, and recent box-score context
"""

import logging
import time

import pandas as pd

from config import (NBA_TEAM_COUNT, PROJECTION_START_MIN_TEAM_GAMES, UPSTREAM_POLL_SECONDS,
                    UPSTREAM_WAIT_SECONDS)
from s3_io import load_parquet

logger = logging.getLogger()

COMPLEX_PROJECTIONS = 'model_comparison/complex_position_overlap/minutes_projections.parquet'
FORMULA_C_PROJECTIONS = 'model_comparison/formula_c_baseline/minutes_projections.parquet'
HEAD_TO_HEAD_PROJECTIONS = 'model_comparison/llm_head_to_head/minutes_projections.parquet'


def load_box_scores():
    """Current-season box scores plus the previous season (for players yet to play this season)."""
    current = load_parquet('data/box_scores/current.parquet')
    if current.empty:
        return current, pd.DataFrame()
    current['GAME_DATE'] = pd.to_datetime(current['GAME_DATE'])
    prev_start = int(str(current['SEASON'].max()).split('-')[0]) - 1
    previous = load_parquet(f'data/box_scores/{prev_start}-{str(prev_start + 1)[2:]}.parquet')
    if not previous.empty:
        previous['GAME_DATE'] = pd.to_datetime(previous['GAME_DATE'])
    return current, previous


def season_of(date):
    """NBA season label ('2026-27') for a date; seasons start in October."""
    date = pd.Timestamp(date)
    start = date.year if date.month >= 10 else date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def projection_gate(current, today):
    """
    (paused, reason). Mirrors minutes-projection serving_features.projection_gate - keep in sync.
    In-house projections are paused until every team has PROJECTION_START_MIN_TEAM_GAMES games this season.
    """
    season = season_of(today)
    if current.empty:
        return True, f"no box scores for {season}"
    dates = pd.to_datetime(current['GAME_DATE'])
    played = current[(dates.apply(season_of) == season) & (dates.dt.date < pd.Timestamp(today).date())]
    team_games = played.groupby('TEAM_ABBREVIATION')['GAME_ID'].nunique()
    short = NBA_TEAM_COUNT - int((team_games >= PROJECTION_START_MIN_TEAM_GAMES).sum())
    if short > 0:
        return True, (f"{short} of {NBA_TEAM_COUNT} teams have fewer than {PROJECTION_START_MIN_TEAM_GAMES} "
                      f"{season} games")
    return False, ''


def latest_team(current, previous):
    """PLAYER -> team of his most recent game, preferring this season."""
    teams = {}
    for frame in (previous, current):  # current overwrites previous
        if not frame.empty:
            last = frame.sort_values('GAME_DATE').groupby('PLAYER').tail(1)
            teams.update(dict(zip(last['PLAYER'], last['TEAM_ABBREVIATION'])))
    return teams


def tonight(today):
    """
    Returns (slate, games):
      slate: DFF players for today with TEAM, OPPONENT, IS_HOME (TEAM 'UNKNOWN' when his last team
             does not play tonight - an offseason or deadline move the box scores cannot see yet)
      games: [{'game_id', 'home', 'away', 'players': {team: [player rows]}}]
    """
    dp = load_parquet('data/daily_predictions/current.parquet')
    if dp.empty:
        return pd.DataFrame(), []
    dp['GAME_DATE'] = pd.to_datetime(dp['GAME_DATE']).dt.date
    slate = dp[dp['GAME_DATE'] == today][['PLAYER', 'POSITION', 'SALARY', 'PPG_PROJECTION', 'STARTER_STATUS']].copy()
    slate = slate.drop_duplicates('PLAYER')

    schedule = load_parquet('data/schedule/current.parquet')
    if schedule.empty:
        logger.warning("No schedule file - cannot group the slate into games")
        return slate.assign(TEAM='UNKNOWN', OPPONENT=None, IS_HOME=None), []
    schedule['GAME_DATE'] = pd.to_datetime(schedule['GAME_DATE']).dt.date
    todays_games = schedule[schedule['GAME_DATE'] == today]
    opponent = dict(zip(todays_games['TEAM'], todays_games['OPPONENT']))
    is_home = dict(zip(todays_games['TEAM'], todays_games['IS_HOME']))

    current, previous = load_box_scores()
    teams = latest_team(current, previous)
    slate['TEAM'] = slate['PLAYER'].map(teams)
    slate.loc[~slate['TEAM'].isin(opponent.keys()), 'TEAM'] = 'UNKNOWN'
    slate['OPPONENT'] = slate['TEAM'].map(opponent)
    slate['IS_HOME'] = slate['TEAM'].map(is_home)
    unknown = (slate['TEAM'] == 'UNKNOWN').sum()
    if unknown:
        logger.warning(f"{unknown} slate players' last team does not play tonight: "
                       f"{slate.loc[slate['TEAM'] == 'UNKNOWN', 'PLAYER'].tolist()[:10]}")

    games = []
    for game_id, g in todays_games.groupby('GAME_ID'):
        homes, aways = g.loc[g['IS_HOME'] == 1, 'TEAM'], g.loc[g['IS_HOME'] == 0, 'TEAM']
        if len(g) != 2:
            logger.warning(f"Game {game_id} has {len(g)} team rows in the schedule; skipped")
            continue
        if len(homes) == 1 and len(aways) == 1:
            home, away = homes.iloc[0], aways.iloc[0]
        else:
            # Neutral-site games can arrive without a designated home; the pairing is what matters here
            away, home = sorted(g['TEAM'])
            logger.warning(f"Game {game_id} has no single designated home team; treating {away}@{home}")
        players = {t: slate[slate['TEAM'] == t].to_dict('records') for t in (away, home)}
        if any(players.values()):
            games.append({'game_id': game_id, 'home': home, 'away': away, 'players': players})
    return slate, games


def recent_minutes(current, n=5):
    """PLAYER -> dict of last-n minutes (oldest first), season average, games, last game date."""
    if current.empty:
        return {}
    ordered = current.sort_values('GAME_DATE')
    out = {}
    for player, g in ordered.groupby('PLAYER'):
        out[player] = {
            'last_minutes': [round(float(m), 1) for m in g['MIN'].tail(n)],
            'season_avg': round(float(g['MIN'].mean()), 1),
            'games': int(len(g)),
            'last_game': g['GAME_DATE'].max().date(),
        }
    return out


def todays_rows(key, today):
    df = load_parquet(key)
    if df.empty:
        return df
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    return df[df['DATE'] == today].copy()


def wait_for_todays_rows(key, today):
    """Poll until an upstream Lambda has written today's rows, up to UPSTREAM_WAIT_SECONDS."""
    waited = 0
    while True:
        rows = todays_rows(key, today)
        if not rows.empty or waited >= UPSTREAM_WAIT_SECONDS:
            return rows
        logger.info(f"Waiting for today's rows in {key} ({waited}s)")
        time.sleep(UPSTREAM_POLL_SECONDS)
        waited += UPSTREAM_POLL_SECONDS
