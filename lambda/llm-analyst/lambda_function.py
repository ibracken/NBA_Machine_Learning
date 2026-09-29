"""
llm-analyst Lambda - the roadmap's LLM components (L1-L4).

Actions (event['action']), scheduled by game-scheduler relative to the slate:
  research     L3 + shared research: per-game web research, then LLM head-to-head minutes projections
  adjustments  L2: shadow-mode minutes adjustments from the research, judged by Opus 5.5
  preflight    L1: rule checks plus an LLM review of tonight's projections, emailed before lock
  postmortem   L4: weekly Opus 5.5 analysis of projections vs actuals, every finding tied to logged queries

Optional event['date'] (YYYY-MM-DD) overrides today's date for reruns.
"""

import json
import logging
from datetime import datetime

import pytz

import adjustments
import claude_client
import postmortem
import preflight
import research
from config import DAILY_BUDGET_USD
from s3_io import load_json, save_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

ACTIONS = {
    'research': research.run,
    'adjustments': adjustments.run,
    'preflight': preflight.run,
    'postmortem': postmortem.run,
}


def usage_key(today):
    return f'llm/usage/{today}.json'


def lambda_handler(event, context):
    event = event or {}
    action = event.get('action')
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}; expected one of {sorted(ACTIONS)}")
    today = (datetime.strptime(event['date'], '%Y-%m-%d').date() if event.get('date')
             else datetime.now(pytz.timezone('US/Eastern')).date())

    spent = load_json(usage_key(today)) or {'runs': []}
    spent_today = sum(r['cost_usd'] for r in spent['runs'])
    if spent_today >= DAILY_BUDGET_USD:
        raise RuntimeError(f"Daily LLM budget reached: ${spent_today:.2f} >= ${DAILY_BUDGET_USD:.2f}; "
                           f"{action} not run")

    claude_client.USAGE.clear()
    logger.info(f"llm-analyst {action} for {today} (spent today so far ${spent_today:.2f})")
    try:
        result = ACTIONS[action](today)
    finally:
        # Record spend even when the action fails partway
        calls = list(claude_client.USAGE)
        cost = sum(c['cost_usd'] or 0 for c in calls)
        if calls:
            spent = load_json(usage_key(today)) or {'runs': []}
            spent['runs'].append({'action': action, 'at': datetime.now(pytz.utc).isoformat(),
                                  'cost_usd': round(cost, 4),
                                  'web_search_requests': sum(c['web_search_requests'] for c in calls),
                                  'calls': calls})
            save_json(spent, usage_key(today))
        logger.info(f"llm-analyst {action}: {len(calls)} API calls, ${cost:.3f} (web searches billed separately)")

    return {'statusCode': 200, 'body': json.dumps({'action': action, 'date': str(today), **result}, default=str)}


if __name__ == '__main__':
    import os
    import sys

    from dotenv import load_dotenv  # local runs only; not in the Lambda image
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))
    print(lambda_handler({'action': sys.argv[1], **({'date': sys.argv[2]} if len(sys.argv) > 2 else {})}, None))
