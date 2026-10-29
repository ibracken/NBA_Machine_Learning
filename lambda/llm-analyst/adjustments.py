"""
Adjustments judge: news -> bounded minutes adjustments to the learned minutes model, in shadow mode.

The judge (Opus 5.5; config.ADJUSTMENT_JUDGES) reads tonight's research briefing and the exact snippets it cites,
plus what the model already did: which teammates it treats as out and how much each player's projection moved
for those absences (Ian, 2026-10-11: an uninformed judge stacks a second overreaction on top, the complex model's
old failure). Each proposed adjustment must name one snippet and copy a quote from it verbatim; the quote is
checked against the snippet, the result is kept within 0-48 minutes (no other size limit since 2026-10-29), and
the pre- and post-adjustment projections are logged. Nothing here changes a projection or a lineup.

Player calls (2026-10-20, logging only): for every Questionable or returning rotation player on the slate, the
judge also calls tonight as plays_normal / plays_limited / sits / unclear, with a quote checked the same way.
Logged to llm/player_calls/log.parquet with the model's chance he plays, to be scored against what happened
(mid-December) before any call is allowed to change a projection.
"""

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from claude_client import LLMError, extract
from config import ADJUSTMENT_JUDGES, MAX_PLAYER_MINUTES, MIN_QUOTE_CHARS, OPEN_NEWS_ROTATION_MIN
from s3_io import load_json, replace_date_rows, save_json
from slate import LEARNED_PROJECTIONS, load_box_scores, projection_gate, slate_projections, wait_for_todays_rows

logger = logging.getLogger()

ADJUSTMENTS_LOG = 'llm/adjustments/log.parquet'
PLAYER_CALLS_LOG = 'llm/player_calls/log.parquet'
CALLS = ['plays_normal', 'plays_limited', 'sits', 'unclear']

JUDGE_SYSTEM = f"""You review tonight's NBA news for a minutes-projection model and decide whether any \
player's projected minutes should move because of something a source reported that the model cannot know.

What the model already does: it is trained on four seasons of box scores and uses each player's season, \
last-7 and last-game minutes, games played and games missed, plus last season's minutes and FP per minute \
(weighted less as this season's games add up) and whether he changed teams. It reads tonight's official NBA injury report: \
every teammate listed Out or Doubtful is treated as out, and his minutes are redistributed according to how \
this player's minutes actually changed in earlier games this season when that teammate sat. A player's \
"absence adjustment" is how much that moved his projection tonight. His own Questionable / Probable / \
Available status is an input too: his projection is the chance he plays (learned from that status, his missed \
games and his history) times his minutes if he plays, so a Questionable or returning player's listed projection \
is already discounted for the chance he sits. The model has no other news.

So do not adjust for anything the model already counts: a teammate who is in the out list below, a player's \
general role or recent form, or anything else on the injury report. Adjust only for news the model cannot see: \
a stated minutes restriction, a return from injury with a stated limit, planned rest not on the report, a \
confirmed change to the starting lineup, a coach's stated rotation change, or a Questionable player confirmed \
in or out (which changes his teammates' minutes too). For a player whose absence adjustment is already large, \
adjust only if the news adds something beyond the absences the model counted. Vague language ("we'll see how \
he feels", "managing him") is not enough unless the source says what it means for tonight's minutes. When in \
doubt, propose nothing - an empty list is a normal answer.

For each adjustment:
- evidence_index is the number of the single item it rests on.
- quote is copied character for character from that item's text, at least {MIN_QUOTE_CHARS} characters long. \
It is checked against the source mechanically; a paraphrase is rejected.
- adjustment_pct is the fractional change to the listed projection (-0.2 means 20% fewer minutes, 2.0 means three \
times as many). There is no size limit: size it to what the evidence says. A bench player confirmed as tonight's \
starter in place of an absent starter can play several times his listed minutes; a stated restriction ("about 20 \
minutes") sets the number directly. The result is kept between 0 and {MAX_PLAYER_MINUTES} minutes. A player who \
will not play is a status conflict (below), not an adjustment, so the model can hand his minutes to teammates.

Separately, list status_conflicts: players whose listed projection contradicts a reported status for \
tonight (projected to play but reported out, or in the out list but reported available). Those are not \
adjustments; they are flagged for a human.

Finally, make one player_call for every player under "Players to call": what tonight's news says about him. \
plays_normal = expected to play his usual role; plays_limited = expected to play under a stated restriction \
or a reduced role; sits = reported out or not expected to play; unclear = no decisive news (then use \
evidence_index 0 and an empty quote). Every other call needs one evidence item and a quote copied from it, \
checked like the others. These calls are recorded and scored against what happens; they change nothing tonight, \
so call it as the evidence reads, not as the projection reads."""

JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'adjustments': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'player': {'type': 'string'},
                    'adjustment_pct': {'type': 'number'},
                    'evidence_index': {'type': 'integer'},
                    'quote': {'type': 'string'},
                    'reason': {'type': 'string'},
                    'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
                },
                'required': ['player', 'adjustment_pct', 'evidence_index', 'quote', 'reason', 'confidence'],
                'additionalProperties': False,
            },
        },
        'player_calls': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'player': {'type': 'string'},
                    'call': {'type': 'string', 'enum': CALLS},
                    'evidence_index': {'type': 'integer'},
                    'quote': {'type': 'string'},
                },
                'required': ['player', 'call', 'evidence_index', 'quote'],
                'additionalProperties': False,
            },
        },
        'status_conflicts': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'player': {'type': 'string'},
                    'reported_status': {'type': 'string', 'enum': ['out', 'available', 'questionable']},
                    'evidence_index': {'type': 'integer'},
                    'quote': {'type': 'string'},
                },
                'required': ['player', 'reported_status', 'evidence_index', 'quote'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['adjustments', 'status_conflicts', 'player_calls'],
    'additionalProperties': False,
}

_QUOTE_CHARS = str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"',
                              '–': '-', '—': '-', ' ': ' '})


def normalize(text):
    """Compare quotes on content, not typography: NFKC, straight quotes, plain dashes, whitespace, case."""
    text = unicodedata.normalize('NFKC', text or '').translate(_QUOTE_CHARS)
    return re.sub(r'\s+', ' ', text).strip().strip('"\'').strip().lower()


def quote_in_source(quote, cited_text):
    q = normalize(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in normalize(cited_text)


def validate(item, evidence, pre_minutes):
    """
    Check one proposed adjustment against the guardrails.
    Returns (row fields, reject_reason or None). Clamping is recorded, not rejected.
    """
    player = item['player'].strip().lower()
    idx = item['evidence_index']
    source = evidence[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(evidence) else None
    pct_raw = float(item['adjustment_pct'])
    pre = pre_minutes.get(player)
    pct = pct_raw
    if pre:   # only a game's worth of minutes is possible: 0..MAX_PLAYER_MINUTES
        pct = min(max(pre * (1 + pct_raw), 0.0), MAX_PLAYER_MINUTES) / pre - 1

    fields = {
        'PLAYER': player, 'PCT_RAW': pct_raw, 'PCT_APPLIED': pct, 'CLAMPED': pct != pct_raw,
        'QUOTE': item['quote'], 'REASON': item.get('reason', ''), 'CONFIDENCE': item.get('confidence', ''),
        'EVIDENCE_INDEX': idx, 'SOURCE_URL': source['url'] if source else None,
        'SOURCE_TITLE': source['title'] if source else None,
        'SOURCE_TEXT': source['cited_text'] if source else None,
        'PRE_MIN': pre, 'POST_MIN': round(pre * (1 + pct), 1) if pre is not None else None,
    }
    if pre is None:
        return fields, 'unknown_player'
    if source is None:
        return fields, 'bad_evidence_index'
    if not source['url']:
        return fields, 'no_source_url'
    if len(normalize(item['quote'])) < MIN_QUOTE_CHARS:
        return fields, 'quote_too_short'
    if not quote_in_source(item['quote'], source['cited_text']):
        return fields, 'quote_not_in_source'
    if pre <= 0:
        return fields, 'projected_zero'
    return fields, None


def evidence_block(evidence):
    return '\n'.join(f"[{i}] {e['title'] or ''} ({e['url']})\n    \"{e['cited_text']}\""
                     for i, e in enumerate(evidence, 1))


def judge_game(today, briefing, projections, judge_model, effort):
    teams = {briefing['away'], briefing['home']}
    game_proj = projections[projections['TEAM'].isin(teams)]
    if game_proj.empty:
        return [], [], []
    evidence = briefing['citations']
    out = game_proj[game_proj['OUT_TONIGHT'].fillna(False).astype(bool)]
    # Slate players only (DFF_NAME set); everyone else on the roster is context through the out list
    slate_rows = game_proj[~game_proj.index.isin(out.index) & game_proj['DFF_NAME'].notna()]
    pre_minutes = dict(zip(slate_rows['PLAYER'], slate_rows['PROJECTED_MIN'].astype(float)))
    pre_minutes.update({p: 0.0 for p in out['PLAYER']})
    label = f"judge[{judge_model}] {briefing['away']}@{briefing['home']}"
    players = '\n'.join(
        f"- {r.PLAYER} ({r.TEAM}): {float(r.PROJECTED_MIN):.1f} min projected | season avg {float(r.S_MIN):.1f} | "
        f"last-7 avg {float(r.L7_MIN):.1f} | absence adjustment {float(r.ABSENCE_ADJ_MIN):+.1f}"
        for r in slate_rows.itertuples())
    outs = '\n'.join(f"- {r.PLAYER} ({r.TEAM}, season avg {float(r.S_MIN):.1f} min)"
                     for r in out.sort_values('S_MIN', ascending=False).itertuples()) or '(none)'
    to_call = call_candidates(slate_rows)
    calls_list = '\n'.join(
        f"- {r.PLAYER} ({r.TEAM}): {str(r.STATUS).lower()} on the report, {int(r.TEAM_GAMES_MISSED)} team games missed "
        f"since his last game" for r in to_call.itertuples()) or '(none)'
    prompt = (f"Date: {today:%A %B %d, %Y}. Game: {briefing['away']} at {briefing['home']}.\n\n"
              f"Teammates the model already treats as out tonight (official injury report, Out or Doubtful):\n"
              f"{outs}\n\n"
              f"Current minutes projections:\n{players}\n\n"
              f"Players to call (Questionable, or back from missed games):\n{calls_list}\n\n"
              f"Evidence items (quote only from these):\n{evidence_block(evidence) or '(none)'}\n\n"
              f"Research briefing (context; quotes must come from the evidence items above):\n"
              f"{briefing['briefing'] or '(none)'}")
    result = extract(label, judge_model, effort, JUDGE_SYSTEM, prompt, JUDGE_SCHEMA)

    base = {'DATE': today, 'GAME_ID': briefing['game_id'], 'JUDGE_MODEL': judge_model}
    rows = []
    for item in result['adjustments']:
        fields, reject = validate(item, evidence, pre_minutes)
        team = game_proj.loc[game_proj['PLAYER'] == fields['PLAYER'], 'TEAM']
        rows.append({**base, 'TEAM': team.iloc[0] if not team.empty else None, **fields,
                     'VALID': reject is None, 'REJECT_REASON': reject})

    conflicts = []
    for c in result['status_conflicts']:
        idx = c['evidence_index']
        source = evidence[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(evidence) else None
        player = c['player'].strip().lower()
        conflicts.append({
            'judge_model': judge_model, 'player': player, 'reported_status': c['reported_status'],
            'projected_min': pre_minutes.get(player), 'quote': c['quote'],
            'source_url': source['url'] if source else None,
            'quote_verified': bool(source) and quote_in_source(c['quote'], source['cited_text']),
        })
    return rows, conflicts, player_call_rows(base, to_call, result['player_calls'], evidence)


def call_candidates(slate_rows):
    """Rotation slate players listed Questionable, or back from 1+ missed team games and not out."""
    if 'STATUS' not in slate_rows.columns:
        return slate_rows.iloc[0:0]
    rotation = slate_rows['S_MIN'].astype(float) >= OPEN_NEWS_ROTATION_MIN
    flagged = (slate_rows['STATUS'] == 'QUESTIONABLE') | (slate_rows['TEAM_GAMES_MISSED'].fillna(0) >= 1)
    return slate_rows[rotation & flagged]


def player_call_rows(base, to_call, calls, evidence):
    """One row per candidate (CALL 'no call' if the judge skipped him), plus any extra calls it made."""
    made = {c['player'].strip().lower(): c for c in calls}
    info = to_call.drop_duplicates('PLAYER').set_index('PLAYER')
    names = list(info.index) + [n for n in made if n not in info.index]

    def num(r, col):
        return float(r[col]) if r is not None and col in r.index and pd.notna(r[col]) else None
    rows = []
    for name in names:
        c = made.get(name)
        idx = c['evidence_index'] if c else None
        source = evidence[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(evidence) else None
        r = info.loc[name] if name in info.index else None
        rows.append({**base, 'PLAYER': name, 'CANDIDATE': r is not None,
                     'TEAM': r['TEAM'] if r is not None else None,
                     'STATUS': r['STATUS'] if r is not None else None,
                     'TEAM_GAMES_MISSED': num(r, 'TEAM_GAMES_MISSED'), 'S_MIN': num(r, 'S_MIN'),
                     'P_PLAY': num(r, 'P_PLAY'), 'MIN_IF_PLAY': num(r, 'MIN_IF_PLAY'),
                     'PROJECTED_MIN': num(r, 'PROJECTED_MIN'),
                     'CALL': c['call'] if c else 'no call', 'EVIDENCE_INDEX': idx,
                     'QUOTE': c['quote'] if c else None, 'SOURCE_URL': source['url'] if source else None,
                     'QUOTE_VERIFIED': bool(c) and bool(source) and quote_in_source(c['quote'], source['cited_text'])})
    return rows


def run(today):
    paused, reason = projection_gate(load_box_scores()[0], today)
    if paused:
        logger.warning(f"L2 skipped for {today}: in-house projections paused ({reason})")
        return {'skipped': f'projections paused ({reason})'}
    research = load_json(f'llm/research/{today}.json')
    if not research or not research.get('games'):
        logger.warning(f"No research for {today}; L2 skipped")
        return {'skipped': 'no research'}
    projections = slate_projections(wait_for_todays_rows(LEARNED_PROJECTIONS, today))
    if projections.empty:
        logger.warning(f"No complex projections for {today}; L2 skipped")
        return {'skipped': 'no projections'}

    tasks = [(b, model, effort) for b in research['games'] for model, effort in ADJUSTMENT_JUDGES.items()]

    def one(task):
        briefing, model, effort = task
        try:
            return judge_game(today, briefing, projections, model, effort) + (None,)
        except Exception as e:
            logger.error(f"judge[{model}] {briefing['away']}@{briefing['home']} failed: {e}")
            return [], [], [], f"{model} {briefing['away']}@{briefing['home']}: {e}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, tasks))

    rows = [r for game_rows, _, _, _ in results for r in game_rows]
    conflicts = [c for _, game_conflicts, _, _ in results for c in game_conflicts]
    calls = pd.DataFrame([c for _, _, game_calls, _ in results for c in game_calls])
    if not calls.empty:
        replace_date_rows(calls, PLAYER_CALLS_LOG, today)
        logger.info(f"Player calls for {today}: {calls['CALL'].value_counts().to_dict()}, "
                    f"quotes verified {int(calls['QUOTE_VERIFIED'].sum())}")
    columns = ['DATE', 'GAME_ID', 'JUDGE_MODEL', 'TEAM', 'PLAYER', 'PRE_MIN', 'POST_MIN', 'PCT_RAW',
               'PCT_APPLIED', 'CLAMPED', 'VALID', 'REJECT_REASON', 'CONFIDENCE', 'REASON', 'QUOTE',
               'EVIDENCE_INDEX', 'SOURCE_URL', 'SOURCE_TITLE', 'SOURCE_TEXT']
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        replace_date_rows(df, ADJUSTMENTS_LOG, today)
    save_json({'date': today, 'status_conflicts': conflicts}, f'llm/adjustments/conflicts/{today}.json')

    summary = {}
    for model in ADJUSTMENT_JUDGES:
        m = df[df['JUDGE_MODEL'] == model]
        summary[model] = {'proposed': int(len(m)), 'valid': int(m['VALID'].sum()),
                          'rejected': m.loc[~m['VALID'], 'REJECT_REASON'].value_counts().to_dict(),
                          'clamped': int(m['CLAMPED'].sum())}
    logger.info(f"L2 shadow adjustments for {today}: {summary}; status conflicts {len(conflicts)}")

    errors = [err for _, _, _, err in results if err]
    if errors:
        raise LLMError(f"{len(errors)} of {len(tasks)} judge calls failed (others saved): {errors}")
    return {'judges': summary, 'status_conflicts': len(conflicts),
            'player_calls': calls['CALL'].value_counts().to_dict() if not calls.empty else {}}
