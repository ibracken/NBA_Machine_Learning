"""
Post-game actuals: fills ACTUAL_MIN / ACTUAL_FP in every model's projections and lineups from the box scores.
Names are matched to box-score names per date (lambda/shared/player_names.py), so DFF-spelled lineups get their
actuals too. On a graded date, a projected player with no box-score row did not play: 0 minutes, 0 FP.

A date is graded only once it is before today (ET). Until 2026-10-25 any date with box-score rows counted, so on
nights with games before the main slate the AI lineup (written before grading) was graded 0 (Oct 23-24). A 0 is
regraded whenever the box scores show that player on that date, which also repairs those rows.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from player_names import match_names
from s3_utils import load_from_s3, save_to_s3

logger = logging.getLogger()

# Models still written nightly. Older folders (complex_position_overlap, fp_* lineups) are complete history.
PROJECTION_MODELS = ['learned_minutes', 'formula_c_baseline', 'learned_ai_adjusted']
LINEUP_MODELS = PROJECTION_MODELS + ['daily_fantasy_fuel_baseline', 'ai_lineup']   # ai_lineup: llm-analyst


def _fill(df, actuals, graded_dates, known, cols):
    """Fill `cols` (ACTUAL_MIN and/or ACTUAL_FP) for rows on graded dates that are still missing them."""
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    for c in cols:
        if c not in df.columns:
            df[c] = None
    played = set(zip(actuals['DATE'], actuals['PLAYER']))
    zero = pd.to_numeric(df[cols[-1]], errors='coerce').eq(0).to_numpy()
    zero_but_played = zero & np.array([(d, p) in played for d, p in zip(df['DATE'], df['PLAYER'])], dtype=bool)
    todo = df['DATE'].isin(graded_dates) & (df[cols[-1]].isna() | zero_but_played)
    for day, idx in df[todo].groupby('DATE').groups.items():
        box_day = actuals[actuals['DATE'] == day]
        mapping, _ = match_names(df.loc[idx, 'PLAYER'], box_day['PLAYER'], known)
        by_name = box_day.set_index('PLAYER')
        names = df.loc[idx, 'PLAYER'].map(mapping)
        for c in cols:
            df.loc[idx, c] = names.map(by_name[c]).fillna(0.0).values
    return df, int(todo.sum())


def update_actual_minutes(box_scores, target_date=None, today=None):
    """Fill actuals for every date before `today` (default: today in ET). target_date is unused."""
    if box_scores.empty:
        logger.warning("No box score data available")
        return 0
    box = box_scores.copy()
    box['DATE'] = pd.to_datetime(box['GAME_DATE']).dt.date
    actuals = box.groupby(['DATE', 'PLAYER'], as_index=False).agg(ACTUAL_MIN=('MIN', 'sum'), ACTUAL_FP=('FP', 'sum'))
    today = today or datetime.now(pytz.timezone('US/Eastern')).date()
    graded, known = {d for d in actuals['DATE'] if d < today}, set(actuals['PLAYER'])

    updated = 0
    for model in PROJECTION_MODELS:
        key = f'model_comparison/{model}/minutes_projections.parquet'
        df = load_from_s3(key)
        if not df.empty:
            df, n = _fill(df, actuals, graded, known, ['ACTUAL_MIN', 'ACTUAL_FP'])
            if n:
                save_to_s3(df, key)
                logger.info(f"{model}: actuals for {n} projection rows")
            updated += n
    for model in LINEUP_MODELS:
        key = f'model_comparison/{model}/daily_lineups.parquet'
        df = load_from_s3(key)
        if not df.empty:
            df, n = _fill(df, actuals, graded, known, ['ACTUAL_FP'])
            if n:
                save_to_s3(df, key)
                logger.info(f"{model}: actuals for {n} lineup rows")
    return updated
