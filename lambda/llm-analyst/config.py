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
HEAD_TO_HEAD_MODEL, HEAD_TO_HEAD_EFFORT = OPUS, 'medium'
PREFLIGHT_MODEL, PREFLIGHT_EFFORT = OPUS, 'medium'
POSTMORTEM_MODEL, POSTMORTEM_EFFORT = OPUS, 'high'
# L2 judge(s), model -> effort. One judge; a second model can be added here for an A/B on identical evidence
ADJUSTMENT_JUDGES = {OPUS: 'high'}

# Web searches allowed per game in the nightly research call
RESEARCH_MAX_SEARCHES = 8
RESEARCH_PARALLEL_GAMES = 6
PAUSE_TURN_LIMIT = 5

# L2 guardrails
MAX_ADJUSTMENT_PCT = 0.25   # an LLM may never move a projection more than 25% either way
MIN_QUOTE_CHARS = 20        # a quote shorter than this cannot carry a minutes claim

# L1 thresholds, measured on 2025-26 fresh-input slates (Oct 25 - Jan 13; slate players only):
# team projected totals p1/p5/p95/p99 = 148/190/274/290 (actual team minutes are always >= 238);
# projections >= 10 min above the player's prior average (2.3% of rows) realized 6.7 min less.
TEAM_TOTAL_CRITICAL = (150, 290)
TEAM_TOTAL_WARNING = (190, 274)
ABOVE_ROLE_WARNING_MIN = 10
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

# Hard stop on runaway spend: an action refuses to start once today's recorded cost exceeds this
DAILY_BUDGET_USD = 25.0

# $ per million tokens: input, output, 5-minute cache write, cache read
PRICING = {
    OPUS: (4.00, 20.00, 5.00, 0.20),
}

# L4 query tool limits
POSTMORTEM_MAX_TURNS = 30
POSTMORTEM_TIME_LIMIT_SECONDS = 660   # leaves room inside the 900s Lambda timeout for the final answer
QUERY_RESULT_MAX_ROWS = 50
