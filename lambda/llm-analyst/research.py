"""
Nightly research and the AI lineup.

Research: one web-search call per game with open injury news (slate.open_news_teams) gathers tonight's
availability and minutes news, keeping the exact snippets each statement rests on. Games with no open news on the
official report are skipped (Ian, 2026-10-20: cost). The adjustments judge and the AI lineup both read it.

AI lineup (an agent since 2026-10-25, Ian): one web-search call per slate (up to AI_LINEUP_MAX_SEARCHES) researches
the slate itself and writes its lineup, with the research briefings and the slate (names, teams, positions, salaries)
as input. It never sees our projections or DFF's from our data; it may read any site but is told to think for itself.
A cheap no-search call turns the write-up into JSON; code checks the picks (slate players, slots, salary cap) and
the JSON call retries once with the error. The write-up and its sources go to llm/ai_lineup/{date}.json. Scored on
realized FP against our lineup and DFF's. Before 2026-10-25 it was a single call with no search.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import pytz

import claude_client
from claude_client import LLMError, extract, research
from config import (AI_LINEUP_EFFORT, AI_LINEUP_MAX_SEARCHES, AI_LINEUP_MODEL, RESEARCH_EFFORT,
                    RESEARCH_MAX_SEARCHES, RESEARCH_MODEL, RESEARCH_PARALLEL_GAMES, SALARY_CAP)
from s3_io import replace_date_rows, save_json
from slate import AI_LINEUP, load_box_scores, open_news_teams, tonight

logger = logging.getLogger()

SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

RESEARCH_SYSTEM = """You prepare a pre-game availability and playing-time briefing for an NBA daily fantasy model.

Search for news published for tonight's game or in the last 48 hours: official injury report statuses, \
game-time decisions, minutes restrictions, players returning from absences, rest or load management, \
expected starting lineups, and rotation changes announced by coaches or team beat writers.

Write one section per team. Cover the listed slate players, and also any other rotation player whose status \
changes tonight's minutes (a starter out, a backup moving into the starting lineup). For each item name the \
player, state what was reported and when, and carry the source's own words when the meaning depends on the \
wording (a restriction, a coach's comment on workload). If no news was found for a listed player, say nothing \
about him rather than guessing. Distinguish confirmed facts from speculation, and say when a report is older \
than today."""

AI_LINEUP_SYSTEM = f"""You are a sharp daily fantasy player building one DraftKings NBA classic lineup for tonight. \
You have web search: do your own research before you pick, the way a serious player would.

Research tonight's injury news and late scratches, confirmed or expected starting lineups, minutes restrictions, \
players returning or changing roles, recent minutes and production, and matchups and pace. The briefings provided \
cover only games with open injury news; research the other games yourself.

Your own knowledge ends around mid-2026. Trades, signings, rookies and roles have changed since, so don't trust \
memory for who plays where or how much. The slate's team for each player is correct, and tonight's sources beat \
memory: when they conflict, go with the slate and the sources.

You may read any site, including other projections and lineup advice, as information. Think independently: don't \
copy another site's projections or lineup; form your own view of each player's minutes and production tonight.

Build for the most points per dollar across all eight slots: a player belongs only if his expected points beat \
what the same salary buys elsewhere. Don't add a minimum-salary player just to fit the cap, and take several \
players from one team only if each one stands on his own.

Rules: exactly 8 different players, one per slot - PG, SG, SF, PF, C, G (a PG or SG), F (an SF or PF), UTIL (any \
position). A player fits a slot only through a position listed for him on the slate. Total salary at most \
${SALARY_CAP:,}. Only players on tonight's slate are eligible. Never pick a player reported out.

Finish with your lineup, one line per slot, names spelled exactly as on the slate:
SLOT | player | salary | one-sentence reason"""

AI_LINEUP_PARSE_SYSTEM = """You copy a DraftKings lineup from an analyst's write-up into JSON: the eight slots, the \
players (spelled as on the slate) and the analyst's reasons. Don't change any pick. If you are told the lineup is \
invalid, fix it with as few changes as possible, choosing replacements the analyst's research supports, and say in \
that player's reason what was replaced and why."""

AI_LINEUP_SCHEMA = {
    'type': 'object',
    'properties': {
        'lineup': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'slot': {'type': 'string', 'enum': SLOTS},
                    'player': {'type': 'string'},
                    'reason': {'type': 'string'},
                },
                'required': ['slot', 'player', 'reason'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['lineup'],
    'additionalProperties': False,
}


def now_et():
    return datetime.now(pytz.timezone('US/Eastern'))


def _roster_lines(game):
    lines = []
    for team in (game['away'], game['home']):
        names = ', '.join(p['PLAYER'] for p in game['players'][team]) or '(none on the DFS slate)'
        lines.append(f"{team}: {names}")
    return '\n'.join(lines)


def research_game(today, game):
    label = f"research {game['away']}@{game['home']}"
    prompt = (f"Game: {game['away']} at {game['home']}, {today:%A %B %d, %Y}. "
              f"It is now {now_et():%I:%M %p} ET.\n"
              f"Players on tonight's DFS slate (names are lowercase and accent-free):\n{_roster_lines(game)}\n\n"
              "Write tonight's availability and playing-time briefing for both teams.")
    # Direct search, not dynamic filtering: with filtering (the default) the searches ran inside code execution
    # and the briefing came back with 0 citations, so the judge had no snippets to quote. Direct also cost less
    # (OKC@SAS 2026-10-20: 25 citations, $0.40 vs 0 citations, $0.56).
    tools = [{'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': RESEARCH_MAX_SEARCHES,
              'allowed_callers': ['direct']}]
    text, citations = research(label, RESEARCH_MODEL, RESEARCH_EFFORT, RESEARCH_SYSTEM, prompt, tools)
    logger.info(f"{label}: {len(text)} chars, {len(citations)} cited snippets")
    return {'game_id': game['game_id'], 'away': game['away'], 'home': game['home'],
            'briefing': text, 'citations': citations}


def eligible(position, slot):
    positions = {p.strip() for p in str(position).upper().split('/')}
    return {'G': bool(positions & {'PG', 'SG'}), 'F': bool(positions & {'SF', 'PF'}),
            'UTIL': True}.get(slot, slot in positions)


def check_lineup(picks, slate):
    """Returns (lineup rows, None) or (None, what is wrong)."""
    by_name = slate.drop_duplicates('PLAYER').set_index('PLAYER')
    names = [p['player'].strip().lower() for p in picks]
    problems = []
    if sorted(p['slot'] for p in picks) != sorted(SLOTS):
        problems.append(f"slots must be exactly {SLOTS}, got {[p['slot'] for p in picks]}")
    if len(set(names)) != len(names):
        problems.append("a player is picked twice")
    missing = [n for n in names if n not in by_name.index]
    if missing:
        problems.append(f"not on tonight's slate: {missing}")
    for p, n in zip(picks, names):
        if n in by_name.index and not eligible(by_name.loc[n, 'POSITION'], p['slot']):
            problems.append(f"{n} ({by_name.loc[n, 'POSITION']}) can't play {p['slot']}")
    salary = sum(float(by_name.loc[n, 'SALARY']) for n in names if n in by_name.index)
    if salary > SALARY_CAP:
        problems.append(f"salary ${salary:,.0f} is over the ${SALARY_CAP:,} cap")
    if problems:
        return None, '; '.join(problems)
    return [{'SLOT': p['slot'], 'PLAYER': n, 'TEAM': by_name.loc[n, 'TEAM'], 'POSITION': by_name.loc[n, 'POSITION'],
             'SALARY': float(by_name.loc[n, 'SALARY']), 'REASON': p['reason']} for p, n in zip(picks, names)], None


def ai_lineup(today, slate, briefings, skipped=()):
    """The agent researches and writes a lineup; a no-search call turns it into checked JSON (one retry)."""
    rows = '\n'.join(f"- {s.PLAYER} | {s.TEAM} vs {s.OPPONENT if pd.notna(s.OPPONENT) else '?'} | {s.POSITION} | "
                     f"${int(s.SALARY)}" for s in slate.dropna(subset=['SALARY', 'POSITION'])
                     .sort_values('SALARY', ascending=False).itertuples())
    news = '\n\n'.join(f"=== {b['away']} at {b['home']} ===\n{b['briefing'] or '(no news found)'}" for b in briefings)
    if skipped:
        news += ("\n\nNot researched (no Questionable, Doubtful or returning rotation player on the official injury "
                 "report): " + ', '.join(skipped))
    prompt = (f"Date: {today:%A %B %d, %Y}. It is now {now_et():%I:%M %p} ET.\n\n"
              f"Tonight's slate (name | team vs opponent | position | salary):\n{rows}\n\n"
              f"Research briefings:\n{news or '(none)'}")
    tools = [{'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': AI_LINEUP_MAX_SEARCHES,
              'allowed_callers': ['direct']}]
    before = len(claude_client.USAGE)
    writeup, citations = research("ai_lineup research", AI_LINEUP_MODEL, AI_LINEUP_EFFORT, AI_LINEUP_SYSTEM, prompt,
                                  tools)
    calls = claude_client.USAGE[before:]
    save_json({'date': today, 'writeup': writeup, 'citations': citations,
               'searches': sum(c['web_search_requests'] for c in calls),
               'cost_usd': round(sum(c['cost_usd'] or 0 for c in calls), 4)}, f'llm/ai_lineup/{today}.json')
    names = ', '.join(sorted(slate['PLAYER'].dropna().unique()))
    ask = f"Slate player names: {names}\n\nAnalyst's write-up:\n{writeup}"
    error = None
    for attempt in (1, 2):
        text = ask if error is None else f"{ask}\n\nThat lineup was invalid: {error}. Fix it."
        picks = extract(f"ai_lineup parse {attempt}", AI_LINEUP_MODEL, 'low', AI_LINEUP_PARSE_SYSTEM, text,
                        AI_LINEUP_SCHEMA)['lineup']
        lineup, error = check_lineup(picks, slate)
        if lineup:
            return pd.DataFrame(lineup).assign(DATE=today, PROJECTED_MIN=None, PROJECTED_FP=None, ACTUAL_FP=None,
                                               LLM_MODEL=AI_LINEUP_MODEL)
        logger.warning(f"AI lineup parse {attempt} invalid: {error}")
    raise LLMError(f"AI lineup invalid after 2 attempts: {error}")


def run(today):
    slate, games = tonight(today)
    if not games:
        logger.info(f"No slate games for {today}; nothing to research")
        return {'games': 0}
    open_news, report_found = open_news_teams(today, *load_box_scores())
    if report_found:
        skipped = [f"{g['away']}@{g['home']}" for g in games if not ({g['away'], g['home']} & set(open_news))]
        games_all, games = games, [g for g in games if {g['away'], g['home']} & set(open_news)]
        logger.info(f"Researching {len(games)} of {len(games_all)} games; open news: {open_news}; skipped {skipped}")
    else:
        skipped = []
        logger.warning(f"No injury report for {today} yet: researching every game")

    def one(game):
        # A failed game must not discard the others; failures are raised after everything is saved
        try:
            return research_game(today, game), None
        except Exception as e:
            logger.error(f"research {game['away']}@{game['home']} failed: {e}")
            return None, f"{game['away']}@{game['home']} research: {e}"

    with ThreadPoolExecutor(max_workers=RESEARCH_PARALLEL_GAMES) as pool:
        results = list(pool.map(one, games))
    briefings = [b for b, _ in results if b is not None]
    save_json({'date': today, 'games': briefings, 'skipped_no_open_news': skipped,
               'open_news': open_news if report_found else None}, f'llm/research/{today}.json')
    errors = [err for _, err in results if err]

    lineup = pd.DataFrame()
    try:
        lineup = ai_lineup(today, slate, briefings, skipped)
        replace_date_rows(lineup, AI_LINEUP, today)
        logger.info(f"AI lineup for {today}: {lineup['PLAYER'].tolist()} (${lineup['SALARY'].sum():,.0f})")
    except Exception as e:
        logger.error(f"AI lineup failed: {e}")
        errors.append(f"ai_lineup: {e}")

    if errors:
        raise LLMError(f"{len(errors)} failures (everything else saved): {errors}")
    return {'games': len(games), 'skipped': len(skipped), 'citations': sum(len(b['citations']) for b in briefings),
            'ai_lineup_salary': float(lineup['SALARY'].sum()) if not lineup.empty else None}
