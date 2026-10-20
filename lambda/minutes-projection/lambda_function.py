"""
NBA Minutes Projection Lambda. Nightly lineups from three projections:
1. Learned minutes (our #1): lambda/shared/learned_minutes.py, trained nightly by supervised-learning
2. Formula C (control)
3. DailyFantasyFuel: DFF's own projections
plus, after the AI adjustments judge (action adjusted_lineup, its own email about T-11): the learned model re-run
with the judge's verified status changes and its minutes adjustments applied, to test whether the judge helps. That
email also carries the AI's own lineup (llm-analyst research), and still goes out during the pause with just that.
For 1 and 2, FP = projected minutes x the player's season FP per minute. Lineups use the production optimizer with
DFF's salaries and positions.

The complex model was removed 2026-10-18. The barebones FP model is kept but not served (process_model.py,
lineup_optimizer.predict_fantasy_points).
"""

import json
import logging
import sys
import time
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
from config import BUCKET_NAME, SNS_TOPIC_ARN, s3_client, sns_client
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
ADJUSTED_PATH = 'model_comparison/learned_ai_adjusted'
AI_LINEUP_PATH = 'model_comparison/ai_lineup'   # written by llm-analyst research
ADJUSTMENTS_LOG = 'llm/adjustments/log.parquet'
JUDGE_WAIT_SECONDS, JUDGE_POLL_SECONDS = 300, 20
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


def tonight_rows(box_scores, today, status_overrides=None):
    """
    Roster rows for tonight's games with the learned model's features (lambda/shared/learned_minutes.py).
    status_overrides: {player name: True (out) / False (playing)} applied over the injury report before the
    absence features are computed, so teammates' minutes follow.
    """
    schedule = load_from_s3('data/schedule/current.parquet')
    if schedule.empty:
        raise RuntimeError("data/schedule/current.parquet missing: can't tell which teams play tonight")
    games = schedule[pd.to_datetime(schedule['GAME_DATE']).dt.date == today]
    if games.empty:
        logger.warning(f"No scheduled games for {today}")
        return pd.DataFrame()
    tonight = games.assign(SEASON=str(box_scores['SEASON'].max()))[['GAME_DATE', 'GAME_ID', 'TEAM', 'SEASON']]
    known = set(box_scores['PLAYER'])
    rows = lm.player_games(box_scores, tonight)
    rows, unmatched = lm.attach_listed_out(rows, tonights_injury_report(today), known)
    if status_overrides:
        night = rows['D'] == pd.Timestamp(today)
        mapping, missing = match_names(list(status_overrides), rows.loc[night, 'PLAYER'], known)
        for name, out in status_overrides.items():
            if mapping.get(name):
                rows.loc[night & (rows['PLAYER'] == mapping[name]), 'LISTED_OUT'] = out
        if missing:
            logger.warning(f"Status overrides with no roster row tonight: {missing}")
    rows = lm.absence_features(rows)
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


def load_json(key):
    try:
        return json.loads(s3_client.get_object(Bucket=BUCKET_NAME, Key=key)['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        return None


def build_adjusted_lineup(box_scores, today):
    """
    Learned model + the AI adjustments judge, for comparison with the learned lineup (never emailed):
    1. statuses the judge verified with a quote ("reported out" / "reported available") change the out list, and
       the model is re-run so it redistributes the minutes itself;
    2. the judge's valid +-25% minutes adjustments are applied on top.
    """
    marker, waited = f'llm/adjustments/conflicts/{today}.json', 0
    conflicts = load_json(marker)
    while conflicts is None and waited < JUDGE_WAIT_SECONDS:
        time.sleep(JUDGE_POLL_SECONDS)
        waited += JUDGE_POLL_SECONDS
        conflicts = load_json(marker)
    if conflicts is None:
        logger.warning(f"Adjustments judge didn't finish for {today}; no adjusted lineup")
        return {'statusCode': 200, 'body': 'no judge output'}
    verified = [c for c in conflicts.get('status_conflicts', [])
                if c.get('quote_verified') and c.get('reported_status') in ('out', 'available')]
    overrides = {c['player']: c['reported_status'] == 'out' for c in verified}

    log = load_from_s3(ADJUSTMENTS_LOG)
    pct = {}
    if not log.empty:
        log = log[(pd.to_datetime(log['DATE']).dt.date == today) & log['VALID'].astype(bool)]
        pct = log.groupby('PLAYER')['PCT_APPLIED'].mean().to_dict()
    else:
        log = pd.DataFrame(columns=['PLAYER', 'PCT_APPLIED', 'PRE_MIN', 'POST_MIN', 'QUOTE', 'SOURCE_URL'])

    model = load_model_from_s3(LEARNED_MODEL_KEY)
    if model is None:
        raise RuntimeError(f"could not load {LEARNED_MODEL_KEY}")
    known = set(box_scores['PLAYER'])
    rows = add_learned_minutes(tonight_rows(box_scores, today, overrides), model)
    mapping, _ = match_names(list(pct), rows['PLAYER'], known)
    rows['AI_PCT'] = rows['PLAYER'].map({mapping[n]: v for n, v in pct.items() if mapping.get(n)}).fillna(0.0)
    rows['LEARNED_MIN'] = rows['LEARNED_MIN'] * (1 + rows['AI_PCT'])
    rows['AI_STATUS_CHANGE'] = rows['PLAYER'].map(
        {m: ('out' if overrides[n] else 'available')
         for n, m in match_names(list(overrides), rows['PLAYER'], known)[0].items() if m})

    daily_preds = load_daily_preds()
    pool = slate_pool(rows, daily_preds, today, 'LEARNED_MIN', known)
    save_projections(rows, ADJUSTED_PATH, 'LEARNED_MIN', today, extra_cols=['AI_PCT', 'AI_STATUS_CHANGE'])
    lineup = build_lineup("Learned + AI adjustments", ADJUSTED_PATH, pool, daily_preds, today)
    send_adjusted_email(today, lineup, verified, log)
    body = (f"Adjusted lineup for {today}: {len(pct)} minutes adjustments, {len(overrides)} status changes, "
            f"{len(lineup)} players")
    logger.info(body)
    return {'statusCode': 200, 'body': body}


def send_adjusted_email(today, lineup, status_changes, adjustments):
    """The AI-adjusted lineup, what the judge changed and why, and which players differ from the learned lineup."""
    base = load_from_s3(f'{LEARNED_PATH}/daily_lineups.parquet')
    base_players = set(base.loc[pd.to_datetime(base['DATE']).dt.date == today, 'PLAYER']) if not base.empty else set()
    changes = len(status_changes) + len(adjustments)
    if not changes:
        subject = f"NBA AI-adjusted lineup ({today}): no changes - same as the learned lineup"
        lines = ["The judge found no news the model doesn't already account for. The AI-adjusted lineup is the "
                 "learned lineup."]
    else:
        different = set(lineup['PLAYER']) - base_players if not lineup.empty else set()
        subject = (f"NBA AI-adjusted lineup ({today}): {changes} judge changes, "
                   f"{len(different)} players differ from the learned lineup")
        lines = ["What the judge changed (each backed by a quote checked against its source):", ""]
        for c in status_changes:
            lines.append(f"- {c['player']}: reported {c['reported_status'].upper()} - \"{c.get('quote', '')}\" "
                         f"({c.get('source_url', '')})")
            lines.append("  The model was re-run with this status, so his teammates' minutes moved too.")
        for a in adjustments.to_dict('records'):
            lines.append(f"- {a['PLAYER']}: minutes {a['PCT_APPLIED']:+.0%} ({a.get('PRE_MIN', 0):.1f} -> "
                         f"{a.get('POST_MIN', 0):.1f}) - \"{a.get('QUOTE', '')}\" ({a.get('SOURCE_URL', '')})")
        lines.append("")
        if lineup.empty:
            lines.append("No lineup could be built.")
        else:
            lines += [f"Salary ${lineup['SALARY'].sum():,.0f} | Proj FP {lineup['PROJECTED_FP'].sum():.1f}",
                      "(* = not in the learned lineup)", "-" * 50]
            order = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4, 'G': 5, 'F': 6, 'UTIL': 7}
            for r in lineup.assign(o=lineup['SLOT'].map(order)).sort_values('o').itertuples():
                mark = '*' if r.PLAYER not in base_players else ' '
                lines.append(f"{mark}{r.SLOT:4} | {r.PLAYER:<20} | ${r.SALARY:<5,.0f} | {r.PROJECTED_FP:5.1f} FP | "
                             f"{r.PROJECTED_MIN:4.1f} Min")
    lines += ai_lineup_section(today)
    try:
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message='\n'.join(lines))
    except Exception as e:
        logger.error(f"AI-adjusted lineup email failed: {e}")


def ai_lineup_section(today):
    """The AI's own lineup (llm-analyst research), as email lines: picked from news and its own knowledge."""
    lines = ["", "=" * 20 + " The AI's own lineup " + "=" * 20,
             "Picked by the AI from tonight's research and its own knowledge. It never sees our projections or DFF's.",
             "No FP projection: the AI picks players, it doesn't project points."]
    ai = load_from_s3(f'{AI_LINEUP_PATH}/daily_lineups.parquet')
    ai = ai[pd.to_datetime(ai['DATE']).dt.date == today] if not ai.empty else ai
    if ai.empty:
        return lines + ["No AI lineup for tonight (research failed or hasn't finished; check llm-analyst logs)."]
    order = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4, 'G': 5, 'F': 6, 'UTIL': 7}
    lines += [f"Salary ${ai['SALARY'].sum():,.0f}", "-" * 50]
    for r in ai.assign(o=ai['SLOT'].map(order)).sort_values('o').itertuples():
        lines.append(f"{r.SLOT:4} | {r.PLAYER:<20} | {r.TEAM:<3} | ${r.SALARY:<5,.0f} | {r.REASON}")
    return lines


def send_paused_ai_email(today, reason):
    """During the early-season pause there is no judge or adjusted lineup; the AI's own lineup still goes out."""
    lines = [f"In-house projections are paused ({reason}), so there is no AI-adjusted lineup tonight."]
    lines += ai_lineup_section(today)
    try:
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Subject=f"NBA AI lineup ({today}): projections paused",
                           Message='\n'.join(lines))
    except Exception as e:
        logger.error(f"AI lineup email failed: {e}")


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
        action: 'project' (default), 'update_actuals', 'adjusted_lineup' (after the AI judge) or
                'check_model' (load + predict, no side effects)
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
    if action == 'adjusted_lineup':
        if paused:
            send_paused_ai_email(today, reason)
            return {'statusCode': 200, 'body': f'Projections paused ({reason}); AI lineup emailed'}
        return build_adjusted_lineup(box_scores, today)
    update_actual_minutes(box_scores.copy())
    if paused:
        return run_paused(box_scores, today, reason)
    return run_projections(box_scores, today)


if __name__ == "__main__":
    print(lambda_handler({'date': sys.argv[1]} if len(sys.argv) > 1 else {}, None))
