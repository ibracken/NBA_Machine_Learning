"""
L4: weekly post-mortem analyst.

Opus 5.5 looks for systematic patterns in projections vs actuals. It can only see the data through a
query tool that this Lambda executes and logs; every finding must cite the ids of queries it actually ran
and the row count behind them. Findings that cite no executed query are rejected, and claimed row counts
are checked against the log, so each finding can be re-run and falsified rather than believed.
"""

import json
import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from claude_client import LLMError, call, check_stop
from config import (POSTMORTEM_EFFORT, POSTMORTEM_MAX_TURNS, POSTMORTEM_MODEL, POSTMORTEM_TIME_LIMIT_SECONDS,
                    QUERY_RESULT_MAX_ROWS)
from s3_io import load_parquet, publish, save_json

logger = logging.getLogger()

MINUTES_MODELS = ['complex_position_overlap', 'formula_c_baseline', 'llm_head_to_head']
FP_MODELS = ['current', 'fp_per_min', 'barebones']
UNSAFE_FILTER_TOKENS = ('__', '@', 'import', 'lambda', 'eval', 'exec', 'open(', '`', 'getattr', 'globals')

SYSTEM = """You are the weekly post-mortem analyst for an NBA DraftKings projection pipeline. Find systematic \
patterns in how projections missed: which kinds of players are consistently over- or under-projected, \
whether bias is drifting, whether a model or rule has stopped earning its place, how the LLM head-to-head \
and the shadow-mode LLM adjustments are doing. Look at the last 7 days and compare with the season.

You can only see the data through run_query. Every finding must list the query_ids that show it and the \
number of rows behind it (rows_matched of the query it rests on); findings are checked against the query \
log and rejected if they cite nothing that was run or misstate the row count. Prefer few well-supported \
findings over many; say when a sample is too small to conclude anything. When done, reply with the report."""

TOOL = {
    'name': 'run_query',
    'description': ("Filter, group and aggregate one dataset. Returns query_id, rows_matched (rows after the "
                    "filter) and up to 50 result rows. With no group_by and no aggregations, returns raw rows."),
    'strict': True,
    'input_schema': {
        'type': 'object',
        'properties': {
            'dataset': {'type': 'string', 'enum': ['minutes', 'lineups', 'adjustments']},
            'filter': {'type': 'string', 'description': "pandas DataFrame.query expression, or '' for all rows"},
            'group_by': {'type': 'array', 'items': {'type': 'string'}},
            'aggregations': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'column': {'type': 'string'},
                        'func': {'type': 'string', 'enum': ['mean', 'median', 'sum', 'count', 'std', 'min', 'max']},
                    },
                    'required': ['column', 'func'],
                    'additionalProperties': False,
                },
            },
            'sort_by': {'type': 'string', 'description': "result column to sort by, or ''"},
            'descending': {'type': 'boolean'},
            'limit': {'type': 'integer'},
        },
        'required': ['dataset', 'filter', 'group_by', 'aggregations', 'sort_by', 'descending', 'limit'],
        'additionalProperties': False,
    },
}

REPORT_SCHEMA = {
    'type': 'object',
    'properties': {
        'summary': {'type': 'string'},
        'findings': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string'},
                    'evidence': {'type': 'string'},
                    'query_ids': {'type': 'array', 'items': {'type': 'string'}},
                    'rows_behind': {'type': 'integer'},
                    'recommendation': {'type': 'string'},
                    'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
                },
                'required': ['pattern', 'evidence', 'query_ids', 'rows_behind', 'recommendation', 'confidence'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['summary', 'findings'],
    'additionalProperties': False,
}


def build_datasets(today):
    week_start = today - timedelta(days=7)

    minutes = []
    for model in MINUTES_MODELS:
        df = load_parquet(f'model_comparison/{model}/minutes_projections.parquet')
        if df.empty:
            continue
        df = df.assign(MODEL=model)
        minutes.append(df)
    minutes = pd.concat(minutes, ignore_index=True) if minutes else pd.DataFrame()
    if not minutes.empty:
        minutes['DATE'] = pd.to_datetime(minutes['DATE'])
        minutes = minutes[minutes['DATE'].dt.date < today]
        keep = ['DATE', 'MODEL', 'PLAYER', 'TEAM', 'POSITION', 'PROJECTED_MIN', 'ACTUAL_MIN', 'CONFIDENCE',
                'PROJECTED_FP_current', 'PROJECTED_FP_fp_per_min', 'PROJECTED_FP_barebones', 'ACTUAL_FP']
        minutes = minutes[[c for c in keep if c in minutes.columns]]
        dff = load_parquet('data/daily_predictions/current.parquet')
        if not dff.empty:
            dff['DATE'] = pd.to_datetime(dff['GAME_DATE'])
            dff = dff[['DATE', 'PLAYER', 'SALARY', 'PPG_PROJECTION', 'STARTER_STATUS']].drop_duplicates(['DATE', 'PLAYER'])
            minutes = minutes.merge(dff.rename(columns={'PPG_PROJECTION': 'DFF_FP'}), on=['DATE', 'PLAYER'], how='left')
        minutes['MIN_ERR'] = minutes['PROJECTED_MIN'] - minutes['ACTUAL_MIN']
        minutes['ABS_MIN_ERR'] = minutes['MIN_ERR'].abs()
        minutes['DNP'] = minutes['ACTUAL_MIN'].isna() | (minutes['ACTUAL_MIN'] == 0)
        if 'PROJECTED_FP_current' in minutes.columns:
            minutes['FP_ERR_current'] = minutes['PROJECTED_FP_current'] - minutes['ACTUAL_FP']
        if 'DFF_FP' in minutes.columns:
            minutes['FP_ERR_dff'] = minutes['DFF_FP'] - minutes['ACTUAL_FP']
        minutes['LAST_7_DAYS'] = minutes['DATE'].dt.date >= week_start

    lineups = []
    for model in MINUTES_MODELS:
        for fp in FP_MODELS:
            df = load_parquet(f'model_comparison/{model}/fp_{fp}/daily_lineups.parquet')
            if not df.empty:
                lineups.append(df.assign(MINUTES_MODEL=model, FP_MODEL=fp))
    df = load_parquet('model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet')
    if not df.empty:
        lineups.append(df.assign(MINUTES_MODEL='dff', FP_MODEL='dff'))
    lineups = pd.concat(lineups, ignore_index=True) if lineups else pd.DataFrame()
    if not lineups.empty:
        lineups['DATE'] = pd.to_datetime(lineups['DATE'])
        lineups = lineups[lineups['DATE'].dt.date < today]
        lineups['FP_ERR'] = lineups['PROJECTED_FP'] - lineups['ACTUAL_FP']
        lineups['LAST_7_DAYS'] = lineups['DATE'].dt.date >= week_start

    adjustments = load_parquet('llm/adjustments/log.parquet')
    if not adjustments.empty:
        adjustments['DATE'] = pd.to_datetime(adjustments['DATE'])
        adjustments = adjustments[adjustments['DATE'].dt.date < today]
        box = load_parquet('data/box_scores/current.parquet')
        if not box.empty:
            box['DATE'] = pd.to_datetime(box['GAME_DATE'])
            adjustments = adjustments.merge(box[['DATE', 'PLAYER', 'MIN']].rename(columns={'MIN': 'ACTUAL_MIN'}),
                                            on=['DATE', 'PLAYER'], how='left')
            adjustments['PRE_ABS_ERR'] = (adjustments['PRE_MIN'] - adjustments['ACTUAL_MIN']).abs()
            adjustments['POST_ABS_ERR'] = (adjustments['POST_MIN'] - adjustments['ACTUAL_MIN']).abs()
        adjustments = adjustments.drop(columns=['SOURCE_TEXT'], errors='ignore')
        adjustments['LAST_7_DAYS'] = adjustments['DATE'].dt.date >= week_start

    return {'minutes': minutes, 'lineups': lineups, 'adjustments': adjustments}


def describe(datasets):
    parts = []
    for name, df in datasets.items():
        if df.empty:
            parts.append(f"{name}: empty")
            continue
        cols = ', '.join(f"{c} ({df[c].dtype})" for c in df.columns)
        parts.append(f"{name}: {len(df)} rows, {df['DATE'].min().date()} to {df['DATE'].max().date()}\n  {cols}")
    return '\n'.join(parts)


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else round(float(value), 3)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    return value


def run_query(spec, datasets, log):
    df = datasets.get(spec['dataset'])
    if df is None or df.empty:
        raise ValueError(f"dataset {spec['dataset']} is empty")
    expr = spec['filter'].strip()
    if expr:
        if any(tok in expr for tok in UNSAFE_FILTER_TOKENS):
            raise ValueError(f"filter contains a disallowed token {UNSAFE_FILTER_TOKENS}")
        df = df.query(expr)
    matched = len(df)

    columns = list(spec['group_by']) + [a['column'] for a in spec['aggregations']]
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise ValueError(f"unknown columns {unknown}")
    if spec['group_by'] or spec['aggregations']:
        if spec['group_by']:
            grouped = df.groupby(spec['group_by'], dropna=False)
            out = grouped.size().rename('rows').to_frame()
            for agg in spec['aggregations']:
                out[f"{agg['column']}_{agg['func']}"] = grouped[agg['column']].agg(agg['func'])
            out = out.reset_index()
        else:
            out = pd.DataFrame([{'rows': matched, **{f"{a['column']}_{a['func']}": df[a['column']].agg(a['func'])
                                                     for a in spec['aggregations']}}])
    else:
        out = df
    if spec['sort_by']:
        if spec['sort_by'] not in out.columns:
            raise ValueError(f"sort_by {spec['sort_by']} is not a result column: {list(out.columns)}")
        out = out.sort_values(spec['sort_by'], ascending=not spec['descending'])
    limit = min(max(int(spec['limit']), 1), QUERY_RESULT_MAX_ROWS)

    query_id = f"q{len(log) + 1}"
    records = [{k: _jsonable(v) for k, v in row.items()} for row in out.head(limit).to_dict('records')]
    log.append({'query_id': query_id, 'spec': spec, 'rows_matched': matched, 'result_rows': len(out)})
    return {'query_id': query_id, 'rows_matched': matched, 'result_rows': len(out), 'rows': records}


def verify(findings, log):
    by_id = {q['query_id']: q for q in log}
    verified, rejected = [], []
    for f in findings:
        cited = [q for q in f['query_ids'] if q in by_id]
        if not cited:
            rejected.append({**f, 'reject_reason': 'cites no executed query'})
            continue
        f = {**f, 'cited_queries': [by_id[q] for q in cited],
             'row_count_verified': any(by_id[q]['rows_matched'] == f['rows_behind'] for q in cited)}
        verified.append(f)
    return verified, rejected


def run(today):
    datasets = build_datasets(today)
    if all(df.empty for df in datasets.values()):
        return {'skipped': 'no data'}

    log = []
    messages = [{'role': 'user', 'content': (
        f"Today is {today}. Datasets (DATE is the slate date; *_ERR = projected - actual, so positive means "
        f"over-projected; LAST_7_DAYS marks the past week; DNP means no minutes recorded):\n{describe(datasets)}")}]
    start = time.time()
    warned = False
    for _ in range(POSTMORTEM_MAX_TURNS):
        response = call('postmortem', POSTMORTEM_MODEL, POSTMORTEM_EFFORT, SYSTEM, messages, 32000,
                        tools=[TOOL], output_schema=REPORT_SCHEMA, cache=True)
        messages.append({'role': 'assistant', 'content': response.content})
        if response.stop_reason != 'tool_use':
            check_stop('postmortem', response)
            text = next(b.text for b in response.content if b.type == 'text')
            report = json.loads(text)
            break

        out_of_time = time.time() - start > POSTMORTEM_TIME_LIMIT_SECONDS
        results = []
        for block in response.content:
            if block.type != 'tool_use':
                continue
            if out_of_time:
                results.append({'type': 'tool_result', 'tool_use_id': block.id, 'is_error': True,
                                'content': 'Time limit reached - no more queries. Write the report now.'})
                continue
            try:
                result = run_query(block.input, datasets, log)
                results.append({'type': 'tool_result', 'tool_use_id': block.id, 'content': json.dumps(result)})
            except Exception as e:
                results.append({'type': 'tool_result', 'tool_use_id': block.id, 'is_error': True,
                                'content': f"Query failed: {e}"})
        if out_of_time and not warned:
            results.append({'type': 'text', 'text': 'Time limit reached. Write the final report now using only '
                                                    'the queries already run.'})
            warned = True
        messages.append({'role': 'user', 'content': results})
    else:
        raise LLMError(f"postmortem did not finish within {POSTMORTEM_MAX_TURNS} turns")

    verified, rejected = verify(report['findings'], log)
    save_json({'date': today, 'summary': report['summary'], 'findings': verified, 'rejected': rejected,
               'query_log': log}, f'llm/postmortem/{today}.json')

    lines = [f"Weekly post-mortem ({today})", "", report['summary'], ""]
    for i, f in enumerate(verified, 1):
        check = 'row count verified' if f['row_count_verified'] else 'ROW COUNT DOES NOT MATCH ANY CITED QUERY'
        lines += [f"{i}. {f['pattern']} [{f['confidence']}]", f"   evidence: {f['evidence']}",
                  f"   recommendation: {f['recommendation']}",
                  f"   rows behind: {f['rows_behind']} ({check}); queries: {', '.join(f['query_ids'])}"]
        for q in f['cited_queries']:
            lines.append(f"     {q['query_id']}: {json.dumps(q['spec'])} -> {q['rows_matched']} rows")
        lines.append("")
    if rejected:
        lines.append(f"Rejected {len(rejected)} finding(s) that cited no executed query: "
                     f"{[r['pattern'] for r in rejected]}")
    lines.append(f"Full query log: s3://.../llm/postmortem/{today}.json ({len(log)} queries)")
    publish(f"NBA weekly post-mortem {today}: {len(verified)} findings", '\n'.join(lines))
    return {'findings': len(verified), 'rejected': len(rejected), 'queries': len(log)}
