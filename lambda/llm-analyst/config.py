"""
Configuration for the llm-analyst Lambda (roadmap L1-L4)
"""

BUCKET_NAME = 'nba-prediction-ibracken'
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:349928386418:lineup-optimizer-notifications'

OPUS = 'claude-opus-5-5'

# Every request opts into server-side refusal fallbacks: a classifier false positive retries on
# Anthropic's recommended model for that refusal category instead of failing the night.
FALLBACK_BETA = 'server-side-fallback-2026-07-01'

# Model and effort per task. Opus 5.5 defaults to effort "medium", so every call sets it explicitly.
RESEARCH_MODEL, RESEARCH_EFFORT = OPUS, 'medium'
AI_LINEUP_MODEL, AI_LINEUP_EFFORT = OPUS, 'medium'
PREFLIGHT_MODEL, PREFLIGHT_EFFORT = OPUS, 'medium'
POSTMORTEM_MODEL, POSTMORTEM_EFFORT = OPUS, 'high'
# L2 judge(s), model -> effort. One judge; a second model can be added here for an A/B on identical evidence
ADJUSTMENT_JUDGES = {OPUS: 'high'}

# Research only games with open injury news (Ian, 2026-10-20): a rotation player (season avg >= this, or last
# season's while he has < 4 games) listed Questionable/Doubtful, or listed but not Out after missing >= RETURN_MIN_MISSED
# team games. Without today's injury report every game is researched.
OPEN_NEWS_ROTATION_MIN = 15
RETURN_MIN_MISSED = 1   # 3 until 2026-10-20 (Ian): 53% -> 64% of 2025-26 games researched

# Web searches allowed per game in the nightly research call
RESEARCH_MAX_SEARCHES = 5
RESEARCH_PARALLEL_GAMES = 6
PAUSE_TURN_LIMIT = 5

# DraftKings classic salary cap (the AI lineup is checked against it)
SALARY_CAP = 50000

# L2 guardrails
MAX_ADJUSTMENT_PCT = 0.25   # an LLM may never move a projection more than 25% either way
MIN_QUOTE_CHARS = 20        # a quote shorter than this cannot carry a minutes claim

# Preflight thresholds, measured on the learned minutes model's 2025-26 backtest (Oct 30 - Jan 13, slate players
# only): team projected totals p1/p5/p95/p99 = 133/162/247/259 for the two-part model (2026-10-20; 132/162/243/254
# before). The old "above role" rule was dropped: learned
# projections 10+ min above the season average realized within 1.3 min (complex ones ran 6.7 min high).
TEAM_TOTAL_CRITICAL = (133, 259)
TEAM_TOTAL_WARNING = (162, 247)
ZEROED_REGULAR_LAST3_MIN = 20
MIN_SLATE_COVERAGE = 0.90
MAX_INPUT_AGE_HOURS = 24

# Early-season pause, mirrored from minutes-projection/config.py - keep in sync. While paused there are
# no in-house projections: research and L3 still run; L2 skips; L1 checks inputs only.
PROJECTION_START_MIN_TEAM_GAMES = 4
NBA_TEAM_COUNT = 30

# Upstream outputs may still be landing when an action starts; poll this long before giving up
UPSTREAM_WAIT_SECONDS = 300
UPSTREAM_POLL_SECONDS = 20

# Freak-overspend stops (Ian, 2026-10-20: stop a runaway, never normal progress; a normal night is ~$3-4).
# Both are checked before every API call (claude_client.call); a blocked call raises LLMError, so preflight still
# emails its rule findings and only the AI review is skipped.
# RUN_BUDGET_USD: one action (research, adjustments, preflight, postmortem) makes no further calls once its own
#   recorded cost (tokens + web searches) reaches this. Calls already in flight finish, so parallel research can
#   end a little above it.
# DAILY_BUDGET_USD: no further calls once the day's recorded cost reaches this. A big slate (15 games, most with
#   open news) can put research near $7, so a $7 cap would cut off preflight's review on exactly those nights.
# The monthly backstop is the spend limit Ian set in the Anthropic Console (Billing page, $220), enforced by the API.
RUN_BUDGET_USD = 6.5
DAILY_BUDGET_USD = 10.0
WEB_SEARCH_USD = 0.01   # $10 per 1,000 searches (platform.claude.com web search docs)

# $ per million tokens: input, output, 5-minute cache write, cache read
PRICING = {
    OPUS: (4.00, 20.00, 5.00, 0.20),
}

# L4 query tool limits
POSTMORTEM_MAX_TURNS = 30
POSTMORTEM_TIME_LIMIT_SECONDS = 660   # leaves room inside the 900s Lambda timeout for the final answer
QUERY_RESULT_MAX_ROWS = 50
