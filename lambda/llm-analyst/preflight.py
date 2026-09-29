"""
L1: pre-flight sanity check on tonight's projections, before lineups are used.

Deterministic rules run first and are the baseline: every example that motivated L1 (a regular
projected at 0, team totals near 290, a bench player projected like a starter, a whole-slate failure,
stale inputs) is a rule here. The LLM then reviews the same table and is asked only for what the rules
did not flag - its findings are logged separately so "catches a defect no rule catches" is measurable.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from claude_client import extract
from config import (ABOVE_ROLE_WARNING_MIN, MAX_INPUT_AGE_HOURS, MIN_SLATE_COVERAGE, PREFLIGHT_EFFORT,
                    PREFLIGHT_MODEL, TEAM_TOTAL_CRITICAL, TEAM_TOTAL_WARNING, ZEROED_REGULAR_LAST3_MIN)
from s3_io import last_modified, load_json, load_parquet, publish, save_json
from slate import (COMPLEX_PROJECTIONS, FORMULA_C_PROJECTIONS, HEAD_TO_HEAD_PROJECTIONS, load_box_scores,
                   projection_gate, recent_minutes, todays_rows, tonight, wait_for_todays_rows)

logger = logging.getLogger()

LINEUP_KEYS = [f'model_comparison/{m}/fp_{f}/daily_lineups.parquet'
               for m in ('complex_position_overlap', 'formula_c_baseline')
               for f in ('current', 'fp_per_min', 'barebones')]

REVIEW_SYSTEM = """You audit tonight's NBA minutes and fantasy-point projections before they are used to \
build DraftKings lineups. Automated rules have already run; their findings are listed. Find problems the \
rules did not catch: contradictions between a projection and the player's recent minutes, status, salary \
or the DFF projection; teams whose projections do not add up; anything that suggests an upstream input is \
wrong. Do not repeat a rule finding. Every finding must quote the specific numbers from the table that \
show the problem. Report nothing you cannot support from the table - an empty list is a normal answer. \
critical means a lineup built tonight would likely be wrong because of it; warning means worth a look."""

REVIEW_SCHEMA = {
    'type': 'object',
    'properties': {
        'findings': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'severity': {'type': 'string', 'enum': ['critical', 'warning']},
                    'subject': {'type': 'string'},
                    'issue': {'type': 'string'},
                    'evidence': {'type': 'string'},
                },
                'required': ['severity', 'subject', 'issue', 'evidence'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['findings'],
    'additionalProperties': False,
}


def finding(severity, rule, subject, issue):
    return {'source': 'rule', 'rule': rule, 'severity': severity, 'subject': subject, 'issue': issue}


def age_hours(ts):
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600 if ts is not None else None


def input_checks(today, current, schedule_today_teams):
    """Content-based staleness: box scores missing completed games, injury feed not refreshed."""
    out = []
    schedule = load_parquet('data/schedule/current.parquet')
    if schedule.empty:
        out.append(finding('warning', 'no_schedule', 'schedule',
                           "data/schedule/current.parquet missing: IS_HOME is blank, research/L3 have no games, "
                           "and the stale-box-score check is skipped"))
    if not current.empty and not schedule.empty:
        schedule['GAME_DATE'] = pd.to_datetime(schedule['GAME_DATE']).dt.date
        latest_box = current['GAME_DATE'].max().date()
        missing = schedule[(schedule['GAME_DATE'] > latest_box) & (schedule['GAME_DATE'] < today)]
        if not missing.empty:
            out.append(finding('critical', 'stale_box_scores', 'box scores',
                               f"latest box score is {latest_box} but {missing['GAME_ID'].nunique()} scheduled "
                               f"games since then ({missing['GAME_DATE'].min()} to {missing['GAME_DATE'].max()}) "
                               "are missing"))
    injuries_modified = last_modified('data/injuries/current.parquet')
    injury_age = age_hours(injuries_modified)
    if injury_age is None or injury_age > MAX_INPUT_AGE_HOURS:
        out.append(finding('critical', 'stale_injuries', 'injury feed',
                           f"data/injuries/current.parquet last written {injuries_modified} "
                           f"({injury_age:.0f}h ago)" if injury_age is not None else "injury file missing"))
    injuries = load_parquet('data/injuries/current.parquet')
    if 'REPORT_DATE' in injuries.columns and injuries['REPORT_DATE'].notna().any():
        report_date = pd.to_datetime(injuries['REPORT_DATE']).max().date()
        if report_date < today:
            out.append(finding('critical', 'stale_injury_report', 'injury feed',
                               f"newest injury report parsed is dated {report_date}, not today ({today})"))
    return out, injuries, injury_age


def projection_rules(slate, proj, formula_c, recent, injured):
    out = []
    slate_players = set(slate['PLAYER'])
    covered = slate_players & set(proj['PLAYER'])
    coverage = len(covered) / max(len(slate_players), 1)
    if coverage < MIN_SLATE_COVERAGE:
        missing = sorted(slate_players - covered, key=lambda p: -float(
            slate.loc[slate['PLAYER'] == p, 'SALARY'].fillna(0).iloc[0]))
        out.append(finding('critical', 'slate_coverage', 'slate',
                           f"only {len(covered)}/{len(slate_players)} DFF slate players have an in-house "
                           f"projection ({coverage:.0%}); highest-salary missing: {missing[:8]}"))

    if 'PROJECTED_FP_current' in proj.columns and len(proj):
        zero_fp = (proj[['PROJECTED_FP_current', 'PROJECTED_FP_fp_per_min', 'PROJECTED_FP_barebones']]
                   .fillna(0) <= 0).all(axis=1).mean()
        if zero_fp > 0.5:
            out.append(finding('critical', 'fp_models_zero', 'FP models',
                               f"{zero_fp:.0%} of projected players have 0 FP from all three FP models - "
                               "models likely failed to load"))

    for r in proj.itertuples():
        info = recent.get(r.PLAYER)
        if info is None:
            continue
        last3 = info['last_minutes'][-3:]
        last3_avg = sum(last3) / len(last3) if last3 else 0
        if r.PROJECTED_MIN <= 1 and last3_avg >= ZEROED_REGULAR_LAST3_MIN and r.PLAYER not in injured:
            out.append(finding('critical', 'zeroed_regular', r.PLAYER,
                               f"projected {r.PROJECTED_MIN:.1f} min but last 3 games {last3} "
                               "and not on the injury report"))
        gap = r.PROJECTED_MIN - info['season_avg']
        if gap >= ABOVE_ROLE_WARNING_MIN:
            out.append(finding('warning', 'above_role', r.PLAYER,
                               f"projected {r.PROJECTED_MIN:.1f} min vs season avg {info['season_avg']} "
                               f"(+{gap:.1f}); historically such projections ran ~6.7 min high"))

    totals = proj.groupby('TEAM')['PROJECTED_MIN'].sum()
    for team, total in totals.items():
        if total < TEAM_TOTAL_CRITICAL[0] or total > TEAM_TOTAL_CRITICAL[1]:
            out.append(finding('critical', 'team_total', team, f"slate players project to {total:.0f} team minutes"))
        elif total < TEAM_TOTAL_WARNING[0] or total > TEAM_TOTAL_WARNING[1]:
            out.append(finding('warning', 'team_total', team, f"slate players project to {total:.0f} team minutes"))
    return out, coverage, totals


def output_rules(today):
    out = []
    empty = []
    for key in LINEUP_KEYS:
        rows = todays_rows(key, today)
        if rows.empty:
            empty.append(key.split('/')[1] + '/' + key.split('/')[2])
    if len(empty) == len(LINEUP_KEYS):
        out.append(finding('critical', 'no_lineups', 'lineups', "no in-house lineup was built for today"))
    elif empty:
        out.append(finding('warning', 'missing_lineups', 'lineups', f"no lineup today for {empty}"))
    return out


def conflict_findings(today):
    conflicts = (load_json(f'llm/adjustments/conflicts/{today}.json') or {}).get('status_conflicts', [])
    out = []
    for c in conflicts:
        if not c.get('quote_verified'):
            continue
        out.append(finding('warning', 'reported_status_conflict', c['player'],
                           f"{c['judge_model']}: projected {c['projected_min']} min but reported "
                           f"{c['reported_status']}: \"{c['quote']}\" ({c['source_url']})"))
    return out


def review_table(slate, proj, formula_c, h2h, recent, injured):
    fc = dict(zip(formula_c['PLAYER'], formula_c['PROJECTED_MIN'])) if not formula_c.empty else {}
    llm = dict(zip(h2h['PLAYER'], h2h['PROJECTED_MIN'])) if not h2h.empty else {}
    proj_by = proj.set_index('PLAYER')
    lines = ["player | team vs opp | salary | DFF FP | our FP | our min | formula C min | LLM min | "
             "last 5 min | season avg (games) | injury report"]
    for s in slate.sort_values('SALARY', ascending=False).itertuples():
        p = proj_by.loc[s.PLAYER] if s.PLAYER in proj_by.index else None
        info = recent.get(s.PLAYER, {})
        our_fp = f"{float(p['PROJECTED_FP_current']):.1f}" if p is not None and 'PROJECTED_FP_current' in p else '-'
        our_min = f"{float(p['PROJECTED_MIN']):.1f}" if p is not None else 'NOT PROJECTED'
        lines.append(
            f"{s.PLAYER} | {s.TEAM} vs {s.OPPONENT if pd.notna(s.OPPONENT) else '?'} | "
            f"{int(s.SALARY) if pd.notna(s.SALARY) else '?'} | {s.PPG_PROJECTION} | {our_fp} | {our_min} | "
            f"{fc.get(s.PLAYER, '-')} | {llm.get(s.PLAYER, '-')} | {info.get('last_minutes', '-')} | "
            f"{info.get('season_avg', '-')} ({info.get('games', 0)}) | {'OUT' if s.PLAYER in injured else '-'}")
    return '\n'.join(lines)


def run_paused(today, slate, current, reason):
    """Early-season pause: no in-house projections to check. Input checks only; email only if critical."""
    findings, _, _ = input_checks(today, current, set(slate['TEAM']))
    if todays_rows(HEAD_TO_HEAD_PROJECTIONS, today).empty:
        findings.append(finding('warning', 'no_head_to_head', 'llm-analyst',
                                "no LLM head-to-head projections for today (L3)"))
    save_json({'date': today, 'paused': reason, 'rule_findings': findings, 'llm_findings': [],
               'llm_error': None}, f'llm/preflight/{today}.json')
    n_crit = sum(f['severity'] == 'critical' for f in findings)
    if n_crit:
        lines = [f"In-house projections paused ({reason}); only the DFF lineup was built.", ""]
        lines += [f"[{f['severity'].upper()}] {f['subject']}: {f['issue']}" for f in findings]
        publish(f"NBA preflight {today}: CRITICAL input problem ({n_crit} crit)", '\n'.join(lines))
    logger.info(f"Preflight {today} paused ({reason}): {len(findings)} findings, {n_crit} critical")
    return {'paused': reason, 'critical': n_crit, 'warning': len(findings) - n_crit}


def run(today):
    slate, _ = tonight(today)
    if slate.empty:
        logger.info(f"No DFF slate for {today}; preflight skipped")
        return {'skipped': 'no slate'}
    current, _ = load_box_scores()
    paused, reason = projection_gate(current, today)
    if paused:
        return run_paused(today, slate, current, reason)
    proj = wait_for_todays_rows(COMPLEX_PROJECTIONS, today)
    proj = proj[proj['PLAYER'].isin(slate['PLAYER'])] if not proj.empty else proj
    formula_c = todays_rows(FORMULA_C_PROJECTIONS, today)
    h2h = todays_rows(HEAD_TO_HEAD_PROJECTIONS, today)
    recent = recent_minutes(current)

    rule_findings, injuries, injury_age = input_checks(today, current, set(slate['TEAM']))
    injured = set(injuries['PLAYER']) if not injuries.empty else set()
    coverage, totals = 0.0, pd.Series(dtype=float)
    if proj.empty:
        rule_findings.append(finding('critical', 'no_projections', 'minutes-projection',
                                     "no complex_position_overlap projections for today"))
    else:
        found, coverage, totals = projection_rules(slate, proj, formula_c, recent, injured)
        rule_findings += found
    rule_findings += output_rules(today)
    rule_findings += conflict_findings(today)
    if h2h.empty:
        rule_findings.append(finding('warning', 'no_head_to_head', 'llm-analyst',
                                     "no LLM head-to-head projections for today (L3)"))

    rules_text = '\n'.join(f"- [{f['severity']}] {f['rule']} {f['subject']}: {f['issue']}"
                           for f in rule_findings) or '(none)'
    team_text = ', '.join(f"{t} {v:.0f}" for t, v in totals.items()) or '(none)'
    prompt = (f"Date: {today}. Injury file age: "
              f"{f'{injury_age:.1f}h' if injury_age is not None else 'missing'}. "
              f"Slate coverage: {coverage:.0%}.\n\nRule findings already reported:\n{rules_text}\n\n"
              f"Team projected minute totals (slate players only; ~190-274 is normal): {team_text}\n\n"
              f"Players:\n{review_table(slate, proj, formula_c, h2h, recent, injured)}")
    llm_findings, llm_error = [], None
    try:
        review = extract('preflight review', PREFLIGHT_MODEL, PREFLIGHT_EFFORT, REVIEW_SYSTEM, prompt,
                         REVIEW_SCHEMA)
        known = set(slate['PLAYER']) | set(slate['TEAM'])
        for f in review['findings']:
            llm_findings.append({'source': 'llm', 'rule': None, **f,
                                 'subject_in_table': f['subject'].strip().lower() in {k.lower() for k in known}})
    except Exception as e:
        # The rule findings still go out; the missing review is itself reported
        llm_error = str(e)
        logger.error(f"preflight LLM review failed: {e}")

    report = {'date': today, 'coverage': coverage, 'rule_findings': rule_findings,
              'llm_findings': llm_findings, 'llm_error': llm_error}
    save_json(report, f'llm/preflight/{today}.json')

    n_crit = sum(f['severity'] == 'critical' for f in rule_findings + llm_findings)
    n_warn = sum(f['severity'] == 'warning' for f in rule_findings + llm_findings)
    lines = [f"Preflight {today}: {n_crit} critical, {n_warn} warning", ""]
    for title, items in (("RULES", rule_findings), ("LLM REVIEW (not caught by rules)", llm_findings)):
        lines.append(f"== {title} ==")
        for f in sorted(items, key=lambda f: f['severity'] != 'critical'):
            evidence = f" | evidence: {f['evidence']}" if f.get('evidence') else ''
            lines.append(f"[{f['severity'].upper()}] {f['subject']}: {f['issue']}{evidence}")
        if not items:
            lines.append("(none)")
        lines.append("")
    if llm_error:
        lines.append(f"LLM review failed: {llm_error}")
    status = 'CRITICAL - check before using lineups' if n_crit else 'ok'
    publish(f"NBA preflight {today}: {status} ({n_crit} crit, {n_warn} warn)", '\n'.join(lines))
    return {'critical': n_crit, 'warning': n_warn, 'llm_findings': len(llm_findings), 'llm_error': llm_error}
