"""
Nightly research (shared by L2 and L3) and L3: LLM head-to-head minutes projections.

Research: one web-search call per game gathers tonight's availability and minutes news, keeping the
exact snippets each statement rests on. L3 then projects minutes from that briefing plus public slate
facts only - no access to our projections, models, or engineered features.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from claude_client import LLMError, extract, research
from config import (HEAD_TO_HEAD_EFFORT, HEAD_TO_HEAD_MODEL, RESEARCH_EFFORT, RESEARCH_MAX_SEARCHES,
                    RESEARCH_MODEL, RESEARCH_PARALLEL_GAMES)
from s3_io import replace_date_rows, save_json
from slate import HEAD_TO_HEAD_PROJECTIONS, tonight

logger = logging.getLogger()

RESEARCH_SYSTEM = """You prepare a pre-game availability and playing-time briefing for an NBA daily fantasy model.

Search for news published for tonight's game or in the last 48 hours: official injury report statuses, \
game-time decisions, minutes restrictions, players returning from absences, rest or load management, \
expected starting lineups, and rotation changes announced by coaches or team beat writers.

Write one section per team. For each item name the player, state what was reported, and carry the \
source's own words when the meaning depends on the wording (a restriction, a coach's comment on \
workload). If no news was found for a listed player, say nothing about him rather than guessing. \
Distinguish confirmed facts from speculation, and say when a report is older than today."""

HEAD_TO_HEAD_SYSTEM = """You project minutes for tonight's NBA games the way a human handicapper would, from \
public information only: the research briefing provided and your own knowledge of the players and teams. \
Project every listed player. A player reported out gets 0 minutes. Minutes for a team's players who \
play normally should add up to about 240 across the whole roster, but only some of the roster is listed."""

HEAD_TO_HEAD_SCHEMA = {
    'type': 'object',
    'properties': {
        'players': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'player': {'type': 'string'},
                    'projected_minutes': {'type': 'number'},
                    'status': {'type': 'string', 'enum': ['playing', 'questionable', 'out']},
                    'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
                    'rationale': {'type': 'string'},
                },
                'required': ['player', 'projected_minutes', 'status', 'confidence', 'rationale'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['players'],
    'additionalProperties': False,
}


def _roster_lines(game):
    lines = []
    for team in (game['away'], game['home']):
        names = ', '.join(p['PLAYER'] for p in game['players'][team]) or '(none on the DFS slate)'
        lines.append(f"{team}: {names}")
    return '\n'.join(lines)


def research_game(today, game):
    label = f"research {game['away']}@{game['home']}"
    prompt = (f"Game: {game['away']} at {game['home']}, {today:%A %B %d, %Y}.\n"
              f"Players on tonight's DFS slate (names are lowercase and accent-free):\n{_roster_lines(game)}\n\n"
              "Write tonight's availability and playing-time briefing for both teams.")
    tools = [{'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': RESEARCH_MAX_SEARCHES}]
    text, citations = research(label, RESEARCH_MODEL, RESEARCH_EFFORT, RESEARCH_SYSTEM, prompt, tools)
    logger.info(f"{label}: {len(text)} chars, {len(citations)} cited snippets")
    return {'game_id': game['game_id'], 'away': game['away'], 'home': game['home'],
            'briefing': text, 'citations': citations}


def head_to_head_game(today, game, briefing):
    label = f"head_to_head {game['away']}@{game['home']}"
    rows = []
    for team in (game['away'], game['home']):
        opponent = game['home'] if team == game['away'] else game['away']
        for p in game['players'][team]:
            rows.append(f"- {p['PLAYER']} | {team} vs {opponent} | {p.get('POSITION') or '?'} | "
                        f"DraftKings salary ${int(p['SALARY']) if pd.notna(p.get('SALARY')) else '?'}")
    prompt = (f"Date: {today:%A %B %d, %Y}. Game: {game['away']} at {game['home']}.\n\n"
              f"Players to project (name | team vs opponent | position | salary):\n" + '\n'.join(rows) +
              f"\n\nResearch briefing:\n{briefing['briefing'] or '(no news found)'}")
    result = extract(label, HEAD_TO_HEAD_MODEL, HEAD_TO_HEAD_EFFORT, HEAD_TO_HEAD_SYSTEM, prompt,
                     HEAD_TO_HEAD_SCHEMA)
    return result['players']


def head_to_head_rows(today, game, projections):
    """Match the model's players back to the slate and bound the numbers."""
    by_name = {p['player'].strip().lower(): p for p in projections}
    out = []
    for team in (game['away'], game['home']):
        for p in game['players'][team]:
            proj = by_name.get(p['PLAYER'])
            if proj is None:
                logger.warning(f"head_to_head: no projection returned for {p['PLAYER']}")
                continue
            minutes = 0.0 if proj['status'] == 'out' else min(max(float(proj['projected_minutes']), 0.0), 48.0)
            out.append({
                'DATE': today, 'PLAYER': p['PLAYER'], 'TEAM': team, 'POSITION': p.get('POSITION') or 'UNKNOWN',
                'PROJECTED_MIN': round(minutes, 1), 'ACTUAL_MIN': None, 'ACTUAL_FP': None,
                'CONFIDENCE': proj['confidence'].upper(), 'LLM_STATUS': proj['status'],
                'LLM_RATIONALE': proj['rationale'], 'LLM_MODEL': HEAD_TO_HEAD_MODEL,
            })
    unknown = set(by_name) - {p['PLAYER'] for t in game['players'].values() for p in t}
    if unknown:
        logger.warning(f"head_to_head: ignored players not on the slate: {sorted(unknown)[:10]}")
    return out


def run(today):
    slate, games = tonight(today)
    if not games:
        logger.info(f"No slate games for {today}; nothing to research")
        return {'games': 0}

    def one(game):
        # A failed game must not discard the others; failures are raised after everything is saved
        try:
            briefing = research_game(today, game)
        except Exception as e:
            logger.error(f"research {game['away']}@{game['home']} failed: {e}")
            return None, [], f"{game['away']}@{game['home']} research: {e}"
        try:
            projections = head_to_head_game(today, game, briefing)
            return briefing, head_to_head_rows(today, game, projections), None
        except Exception as e:
            logger.error(f"head_to_head {game['away']}@{game['home']} failed: {e}")
            return briefing, [], f"{game['away']}@{game['home']} head_to_head: {e}"

    with ThreadPoolExecutor(max_workers=RESEARCH_PARALLEL_GAMES) as pool:
        results = list(pool.map(one, games))

    briefings = [b for b, _, _ in results if b is not None]
    save_json({'date': today, 'games': briefings}, f'llm/research/{today}.json')

    rows = [r for _, game_rows, _ in results for r in game_rows]
    if rows:
        replace_date_rows(pd.DataFrame(rows), HEAD_TO_HEAD_PROJECTIONS, today)
    logger.info(f"Research: {len(briefings)}/{len(games)} games; head-to-head projected "
                f"{len(rows)}/{len(slate)} slate players")

    errors = [err for _, _, err in results if err]
    if errors:
        raise LLMError(f"{len(errors)} of {len(games)} games failed (others saved): {errors}")
    return {'games': len(games), 'citations': sum(len(b['citations']) for b in briefings),
            'head_to_head_players': len(rows), 'slate_players': len(slate),
            'coverage': round(len(rows) / max(len(slate), 1), 3)}
