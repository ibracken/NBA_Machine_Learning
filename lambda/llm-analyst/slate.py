"""
Tonight's slate: games, the DFF players in each, and recent box-score context
"""

import logging
import time

import pandas as pd

from config import (NBA_TEAM_COUNT, OPEN_NEWS_ROTATION_MIN, PROJECTION_START_MIN_TEAM_GAMES, RETURN_MIN_MISSED,
                    UPSTREAM_POLL_SECONDS, UPSTREAM_WAIT_SECONDS)
from s3_io import load_parquet

try:
    from player_names import match_names
except ImportError:   # local run: the shared modules live in lambda/shared
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
    from player_names import match_names

logger = logging.getLogger()

# Our #1 projection since 2026-10-18. Rows cover every roster player for tonight's teams, box-score spelling in
# PLAYER and DFF's spelling in DFF_NAME (slate players only); slate_projections() lines them up with the slate.
LEARNED_PROJECTIONS = 'model_comparison/learned_minutes/minutes_projections.parquet'
FORMULA_C_PROJECTIONS = 'model_comparison/formula_c_baseline/minutes_projections.parquet'
# One AI-picked lineup per slate (research.ai_lineup); scored, never emailed
AI_LINEUP = 'model_comparison/ai_lineup/daily_lineups.parquet'


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
    # DFF and box scores spell some players differently ("bobby portis" / "bobby portis jr."); match against
    # the players whose last team plays tonight
    candidates = [p for p, t in teams.items() if t in opponent]
    mapping, _ = match_names(slate['PLAYER'], candidates, set(teams))
    slate['TEAM'] = slate['PLAYER'].map(lambda p: teams.get(mapping.get(p) or p))
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


def slate_projections(rows):
    """Learned projection rows renamed to DFF's spelling where we have it, so they join the DFF slate."""
    if rows.empty or 'DFF_NAME' not in rows.columns:
        return rows
    return rows.assign(BOX_NAME=rows['PLAYER'], PLAYER=rows['DFF_NAME'].fillna(rows['PLAYER']))


def open_news_teams(today, current, previous):
    """
    Teams with open injury news tonight, from the official report (data/injuries/report_statuses.parquet):
    a rotation player listed Questionable or Doubtful, or a rotation player returning from RETURN_MIN_MISSED+
    missed team games who is listed Questionable / Probable / Available. Returns ({team: [reasons]}, report_found).
    """
    report, waited = pd.DataFrame(), 0
    while True:
        report = load_parquet('data/injuries/report_statuses.parquet')
        fresh = (not report.empty and 'REPORT_DATE' in report.columns
                 and pd.to_datetime(report['REPORT_DATE']).dt.date.max() == today)
        if fresh or waited >= UPSTREAM_WAIT_SECONDS:
            break
        time.sleep(UPSTREAM_POLL_SECONDS)
        waited += UPSTREAM_POLL_SECONDS
    if not fresh:
        return {}, False

    def season_avgs(frame):
        if frame.empty:
            return pd.DataFrame(columns=['avg', 'gp'])
        return frame.groupby('PLAYER')['MIN'].agg(avg='mean', gp='size')
    cur, prev = season_avgs(current), season_avgs(previous)
    avg = prev['avg'].to_dict()
    avg.update(cur.loc[cur['gp'] >= 4, 'avg'].to_dict())   # this season once he has 4 games
    rotation = {p for p, a in avg.items() if a >= OPEN_NEWS_ROTATION_MIN}

    names, _ = match_names(report['PLAYER'], list(avg), set(avg))
    report = report.assign(BOX=report['PLAYER'].map(names))
    reasons = {}
    for r in report[report['BOX'].isin(rotation) & report['STATUS'].isin(['QUESTIONABLE', 'DOUBTFUL'])].itertuples():
        reasons.setdefault(r.TEAM, []).append(f"{r.PLAYER} {r.STATUS.lower()}")

    # Returning: his team has played RETURN_MIN_MISSED+ games since his last one this season, and he isn't Out.
    # This season only: current.parquet still holds last season until the first new box score lands.
    current = current[pd.to_datetime(current['GAME_DATE']).apply(season_of) == season_of(today)] if not current.empty else current
    if not current.empty:
        # Listed but not Out (Questionable / Probable / Available): how teams flag a player coming back. Requiring
        # this keeps traded and released players, whose old team keeps "missing" them, from counting.
        listed_back = set(report.loc[report['STATUS'].isin(['QUESTIONABLE', 'PROBABLE', 'AVAILABLE']), 'BOX'].dropna())
        games = current.drop_duplicates(['TEAM_ABBREVIATION', 'GAME_ID'])[['TEAM_ABBREVIATION', 'GAME_DATE']]
        last = current.sort_values('GAME_DATE').groupby('PLAYER').tail(1)
        for r in last[last['PLAYER'].isin(rotation & listed_back)].itertuples():
            team_games = games[games['TEAM_ABBREVIATION'] == r.TEAM_ABBREVIATION]
            missed = int((team_games['GAME_DATE'] > r.GAME_DATE).sum())
            if missed >= RETURN_MIN_MISSED:
                reasons.setdefault(r.TEAM_ABBREVIATION, []).append(f"{r.PLAYER} back after {missed} missed games?")
    return reasons, True


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
