"""
NBA Minutes Projection Lambda. Nightly lineups from three projections:
1. Learned minutes (our #1): lambda/shared/learned_minutes.py, trained nightly by supervised-learning
2. Formula C (control)
3. DailyFantasyFuel: DFF's own projections
For 1 and 2, FP = projected minutes x the player's season FP per minute. Lineups use the production optimizer with
DFF's salaries and positions.

The complex model was removed 2026-10-18. The barebones FP model is kept but not served (process_model.py,
lineup_optimizer.predict_fantasy_points).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

try:
    import learned_minutes as lm
except ImportError:   # local run: the shared modules live in lambda/shared
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
    import learned_minutes as lm
from actuals_updater import update_actual_minutes
from lineup_optimizer import optimize_lineup
from player_names import match_names
from s3_utils import load_from_s3, load_model_from_s3, save_to_s3, send_multi_model_notification
from serving_features import projection_gate

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

LEARNED_PATH = 'model_comparison/learned_minutes'
FORMULA_C_PATH = 'model_comparison/formula_c_baseline'
DFF_PATH = 'model_comparison/daily_fantasy_fuel_baseline'
LEARNED_MODEL_KEY = 'models/learned_minutes.pkl'
ABSENCE_FEATURES = ['OUT_MIN_FRESH', 'OUT_MIN_ONGOING', 'WOWY_MIN', 'WOWY_N']


def save_day(df, key, today, date_col='DATE'):
    """Replace today's rows in an accumulating S3 file."""
    existing = load_from_s3(key)
    if not existing.empty:
        existing[date_col] = pd.to_datetime(existing[date_col]).dt.date
        existing = existing[existing[date_col] != today]
        df = pd.concat([existing, df], ignore_index=True)
    save_to_s3(df, key)


def load_daily_preds():
    daily_preds = load_from_s3('data/daily_predictions/current.parquet')
    if not daily_preds.empty:
        daily_preds['GAME_DATE'] = pd.to_datetime(daily_preds['GAME_DATE']).dt.date
    return daily_preds


def tonights_injury_report(today):
    """Tonight's Out/Doubtful list (injury-scraper writes both as OUT). Empty if missing or not from today."""
    injuries = load_from_s3('data/injuries/current.parquet')
    empty = pd.DataFrame(columns=['D', 'PLAYER', 'STATUS'])
    if injuries.empty:
        logger.warning("No injury report: no teammates treated as out tonight")
        return empty
    report_dates = pd.to_datetime(injuries.get('REPORT_DATE', pd.Series(dtype=object))).dropna()
    if report_dates.empty or report_dates.max().date() != today:
        latest = report_dates.max().date() if not report_dates.empty else 'unknown'
        logger.warning(f"Injury report is from {latest}, not {today}: no teammates treated as out tonight")
        return empty
    return pd.DataFrame({'D': pd.Timestamp(today), 'PLAYER': injuries['PLAYER'], 'STATUS': injuries['STATUS']})


def tonight_rows(box_scores, today):
    """Roster rows for tonight's games with the learned model's features (lambda/shared/learned_minutes.py)."""
    schedule = load_from_s3('data/schedule/current.parquet')
    if schedule.empty:
        raise RuntimeError("data/schedule/current.parquet missing: can't tell which teams play tonight")
    games = schedule[pd.to_datetime(schedule['GAME_DATE']).dt.date == today]
    if games.empty:
        logger.warning(f"No scheduled games for {today}")
        return pd.DataFrame()
    tonight = games.assign(SEASON=str(box_scores['SEASON'].max()))[['GAME_DATE', 'GAME_ID', 'TEAM', 'SEASON']]
    rows, unmatched = lm.build(box_scores, tonights_injury_report(today), tonight)
    rows = rows[rows['D'] == pd.Timestamp(today)].copy()
    if unmatched:
        logger.info(f"Injury report names not on tonight's rosters (long-term or two-way): "
                    f"{sorted(n for names in unmatched.values() for n in names)}")
    logger.info(f"{len(rows)} roster rows for {rows['TEAM'].nunique()} teams; {int(rows['LISTED_OUT'].sum())} listed out")
    return rows


def add_learned_minutes(rows, model):
    """LEARNED_MIN for everyone not listed out, plus how much tonight's absences moved it (for the AI judge)."""
    rows = rows.copy()
    projectable = rows['S_MIN'].notna() & ~rows['LISTED_OUT']
    rows['LEARNED_MIN'] = 0.0
    rows['LEARNED_MIN_NO_ABSENCES'] = 0.0
    if projectable.any():
        rows.loc[projectable, 'LEARNED_MIN'] = lm.predict(model, rows[projectable])
        no_absences = rows[projectable].assign(**{c: 0.0 for c in ABSENCE_FEATURES})
        rows.loc[projectable, 'LEARNED_MIN_NO_ABSENCES'] = lm.predict(model, no_absences)
    rows['ABSENCE_ADJ_MIN'] = rows['LEARNED_MIN'] - rows['LEARNED_MIN_NO_ABSENCES']
    return rows


def slate_pool(rows, daily_preds, today, minutes_col, known_names):
    """
    Tonight's DFF slate joined to our projections. PLAYER stays DFF's spelling (the optimizer takes salaries from
    daily_preds by name); BOX_NAME is the box-score spelling, matched against tonight's rosters.
    """
    slate = daily_preds[daily_preds['GAME_DATE'] == today].drop_duplicates('PLAYER')
    if slate.empty or rows.empty:
        return pd.DataFrame()
    mapping, unmatched = match_names(slate['PLAYER'], rows['PLAYER'], known_names)
    if unmatched:
        logger.info(f"DFF slate players with no roster row tonight: {unmatched}")
    slate = slate.assign(BOX_NAME=slate['PLAYER'].map(mapping)).dropna(subset=['BOX_NAME'])
    pool = slate[['PLAYER', 'BOX_NAME', 'POSITION']].merge(
        rows.rename(columns={'PLAYER': 'BOX_NAME'})[['BOX_NAME', 'TEAM', minutes_col, 'S_RATE']], on='BOX_NAME')
    # Unrounded: rounding can flip near-ties in the optimizer (the email and saved files round for display)
    pool['PROJECTED_MIN'] = pool[minutes_col]
    pool['PROJECTED_FP'] = pool[minutes_col] * pool['S_RATE']
    return pool[pool['PROJECTED_FP'] > 0]


def build_lineup(label, path, pool, daily_preds, today):
    if pool.empty:
        logger.warning(f"{label}: empty pool for {today}")
        return pd.DataFrame()
    lineup = optimize_lineup(pool[['PLAYER', 'TEAM', 'POSITION', 'PROJECTED_MIN', 'PROJECTED_FP']], daily_preds, today)
    if lineup.empty:
        logger.warning(f"{label}: optimizer returned no lineup for {today}")
        return lineup
    save_day(lineup, f'{path}/daily_lineups.parquet', today)
    logger.info(f"{label} lineup: {lineup['PROJECTED_FP'].sum():.1f} projected FP")
    return lineup


def save_projections(rows, path, minutes_col, today, extra_cols=()):
    out = rows.assign(DATE=pd.Timestamp(today), PROJECTED_MIN=rows[minutes_col].round(1),
                      PROJECTED_FP=(rows[minutes_col] * rows['S_RATE']).round(1),
                      OUT_TONIGHT=rows['LISTED_OUT'], ACTUAL_MIN=None, ACTUAL_FP=None)
    cols = ['DATE', 'PLAYER', 'TEAM', 'PROJECTED_MIN', 'PROJECTED_FP', 'S_RATE', 'S_MIN', 'L7_MIN', 'GP',
            'TEAM_GAMES_MISSED', 'OUT_TONIGHT', *extra_cols, 'ACTUAL_MIN', 'ACTUAL_FP']
    save_day(out[cols], f'{path}/minutes_projections.parquet', today)


def update_daily_predictions_minutes(daily_preds, pool, today):
    """Write learned minutes and teams into today's DFF rows (read by the standalone lineup-optimizer)."""
    if daily_preds.empty or pool.empty:
        return
    by_name = pool.set_index('PLAYER')
    mask = (daily_preds['GAME_DATE'] == today) & daily_preds['PLAYER'].isin(by_name.index)
    daily_preds.loc[mask, 'PROJECTED_MIN'] = daily_preds.loc[mask, 'PLAYER'].map(by_name['PROJECTED_MIN'])
    daily_preds.loc[mask, 'TEAM_ABBREVIATION'] = daily_preds.loc[mask, 'PLAYER'].map(by_name['TEAM'])
    save_to_s3(daily_preds, 'data/daily_predictions/current.parquet')


def build_dff_lineup(daily_preds, today, team_mapping):
    """Optimize and save today's lineup from DFF's own FP projections."""
    logger.info("Generating lineup: DailyFantasyFuel Baseline")
    if daily_preds.empty:
        return pd.DataFrame()
    todays_dff = daily_preds[daily_preds['GAME_DATE'] == today].copy()
    logger.info(f"Found {len(todays_dff)} DailyFantasyFuel projections for {today}")
    # daily-predictions has no TEAM column
    team_mapping = team_mapping.drop_duplicates(subset='PLAYER', keep='last')
    todays_dff = todays_dff.merge(team_mapping, on='PLAYER', how='left')
    todays_dff['TEAM'] = todays_dff['TEAM'].fillna('UNKNOWN')
    todays_dff['PROJECTED_FP'] = todays_dff['PPG_PROJECTION']
    todays_dff['PROJECTED_MIN'] = 0  # DFF doesn't provide minutes
    lineup = optimize_lineup(todays_dff[['PLAYER', 'TEAM', 'POSITION', 'PROJECTED_MIN', 'PROJECTED_FP']],
                             daily_preds, today)
    if not lineup.empty:
        save_day(lineup, f'{DFF_PATH}/daily_lineups.parquet', today)
    return lineup


def run_paused(box_scores, today, reason):
    """Early-season pause: DFF lineup and email only; no in-house projections or lineups."""
    logger.warning(f"In-house projections paused for {today}: {reason}")
    daily_preds = load_daily_preds()
    latest = box_scores.sort_values('GAME_DATE').groupby('PLAYER').tail(1)
    team_mapping = latest[['PLAYER', 'TEAM_ABBREVIATION']].rename(columns={'TEAM_ABBREVIATION': 'TEAM'})
    lineup_dff = build_dff_lineup(daily_preds, today, team_mapping)
    send_multi_model_notification(
        {f"In-house projections paused ({reason})": pd.DataFrame(), "DailyFantasyFuel": lineup_dff}, today)
    return {'statusCode': 200, 'body': f'Projections paused ({reason}); DFF lineup {len(lineup_dff)} players'}


def check_model(box_scores):
    """
    Release check (AGENTS.md): load the published model in this environment and predict the last graded game day
    from stored box scores and that day's report in the injury history, as the nightly run would. No writes, no
    email. Normal range on 2025-26: MAE about 4-6 min. Without the report it reads far higher (10.8 on the season's
    last day, when 47% of roster players sat).
    """
    model = load_model_from_s3(LEARNED_MODEL_KEY)
    if model is None:
        raise RuntimeError(f"could not load {LEARNED_MODEL_KEY}")
    rows, _ = lm.build(box_scores, load_from_s3('data/injuries/history.parquet'))
    last = rows[(rows['D'] == rows['D'].max()) & rows['S_MIN'].notna() & ~rows['LISTED_OUT']]
    pred = lm.predict(model, last)
    mae = float(abs(pred - last['ACT_MIN']).mean())
    body = (f"{LEARNED_MODEL_KEY} loaded and predicted {len(last)} players not listed out for "
            f"{last['D'].max().date()}: MAE {mae:.2f} min (injury report {'found' if last['HAS_REPORT'].any() else 'MISSING'})")
    logger.info(body)
    return {'statusCode': 200, 'body': body}


def run_projections(box_scores, today):
    daily_preds = load_daily_preds()
    known = set(box_scores['PLAYER'])
    rows = tonight_rows(box_scores, today)
    lineups, failures = {}, []

    # 1. Learned minutes, our #1
    model = load_model_from_s3(LEARNED_MODEL_KEY)
    if model is None:
        failures.append(f"could not load {LEARNED_MODEL_KEY}")
        lineups["1. Learned minutes (#1) - MODEL FAILED TO LOAD"] = pd.DataFrame()
    elif not rows.empty:
        rows = add_learned_minutes(rows, model)
        pool = slate_pool(rows, daily_preds, today, 'LEARNED_MIN', known)
        # DFF's spelling, for llm-analyst, which works from the DFF slate
        rows['DFF_NAME'] = rows['PLAYER'].map(dict(zip(pool['BOX_NAME'], pool['PLAYER']))) if not pool.empty else None
        save_projections(rows, LEARNED_PATH, 'LEARNED_MIN', today,
                         extra_cols=['DFF_NAME', 'LEARNED_MIN_NO_ABSENCES', 'ABSENCE_ADJ_MIN', *ABSENCE_FEATURES])
        logger.info(f"Learned minutes: {len(pool)} slate players projected")
        lineups["1. Learned minutes (#1)"] = build_lineup("Learned minutes", LEARNED_PATH, pool, daily_preds, today)
        update_daily_predictions_minutes(daily_preds, pool, today)

    # 2. Formula C, the control
    if not rows.empty:
        fc_rows = rows.assign(FC_MIN=rows['FC_MIN'].where(~rows['LISTED_OUT'], 0.0))
        save_projections(fc_rows, FORMULA_C_PATH, 'FC_MIN', today)
        lineups["2. Formula C (control)"] = build_lineup(
            "Formula C", FORMULA_C_PATH, slate_pool(fc_rows, daily_preds, today, 'FC_MIN', known), daily_preds, today)

    # 3. DailyFantasyFuel
    team_mapping = (rows[['PLAYER', 'TEAM']] if not rows.empty else
                    box_scores[['PLAYER', 'TEAM_ABBREVIATION']].rename(columns={'TEAM_ABBREVIATION': 'TEAM'}))
    lineups["3. DailyFantasyFuel"] = build_dff_lineup(daily_preds, today, team_mapping)

    send_multi_model_notification(lineups, today)
    if failures:
        # Lineups that could be built were saved and emailed; fail loudly so the error alarm fires
        raise RuntimeError(f"minutes-projection for {today}: {failures}")
    return {'statusCode': 200, 'body': f"Lineups for {today}: " + ", ".join(
        f"{k} {len(v)}" for k, v in lineups.items())}


def lambda_handler(event, context):
    """
    Event parameters:
        action: 'project' (default), 'update_actuals' or 'check_model' (load + predict, no side effects)
        date: optional date override (YYYY-MM-DD)
    """
    event = event or {}
    action = event.get('action', 'project')
    today = (datetime.strptime(event['date'], '%Y-%m-%d').date() if event.get('date')
             else datetime.now(pytz.timezone('US/Eastern')).date())
    box_scores = load_from_s3('data/box_scores/current.parquet')

    if action == 'check_model':
        return check_model(box_scores)
    if action == 'update_actuals':
        return {'statusCode': 200, 'body': f'Updated {update_actual_minutes(box_scores)} actual records'}

    if box_scores.empty:
        raise RuntimeError("No box scores in data/box_scores/current.parquet")
    paused, reason = projection_gate(box_scores, today)
    update_actual_minutes(box_scores.copy())
    if paused:
        return run_paused(box_scores, today, reason)
    return run_projections(box_scores, today)


if __name__ == "__main__":
    print(lambda_handler({'date': sys.argv[1]} if len(sys.argv) > 1 else {}, None))
