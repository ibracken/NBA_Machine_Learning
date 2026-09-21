# Roadmap

Open work, ordered by expected value. Every item names how it gets measured — `scripts/evaluate_projections.py`
scores minutes and lineups by data-freshness regime, and 148 dates of stored projections-with-actuals
sit in `model_comparison/` for backtesting.

Standing rule for everything below: **a change that cannot beat its own absence gets switched off.**
That applies to LLM components exactly as it applies to features.

---

## LLM integration

The pipeline sees only what it scrapes. Every night there is public information that exists solely as
text — minutes restrictions, load management, game-time decisions, rotation changes announced in a
pregame presser — and no amount of box-score history recovers it. That gap is the case for an LLM.

Do **not** ask an LLM to emit numeric projections directly. It will lose to the GradientBoosting models
and replace a measurable component with an unmeasurable one. The value is text → structure, not
numbers → numbers.

### L1. Pre-flight sanity checker (start here)

Before lineups are written, pass the LLM tonight's projections plus supporting context and ask one
question: *which of these look wrong, and why?*

A model shown `rudy gobert: projected 0.0 min` beside `last 3 games: 29, 30, 33 min` flags it in one
pass. That exact contradiction ran unnoticed from 2026-01-14 to 2026-06-13 and cost 33.5% of all
absolute minutes error in that window. Also catches team totals near 290, 37-minute projections for
12-minute players, and whole-slate failures.

- Input: tonight's projections, each player's last 5 games, injury-file age, per-team minute totals
- Output: structured findings with severity; block the run on critical, log on warning
- Model: **Opus 5**, nightly. Pattern-noticing with structured output; must stay cheap.
- Success: catches ≥1 real defect per month that no existing alarm catches
- Cost: smallest item here, and it touches no projection — it only watches them

### L2. Beat-writer synthesis → bounded minutes adjustments

An LLM with web search reads tonight's beat coverage and emits structured adjustments:

```json
{"player": "victor wembanyama", "adjustment_pct": -0.25,
 "reason": "24-minute restriction, first game back",
 "source_url": "...", "confidence": "high"}
```

The failure mode to design against is not the model being dumb — it is the model being *fluent and
wrong*. A plausible chain of reasoning over an ambiguous quote ("coach mentioned managing minutes")
produces a confident adjustment that no downstream check would question. This is the same failure
that let the injury system run wrong for five months, so it gets the same answer: measure it.

Non-negotiable guardrails:

- **Bounded**: clamp to ±25%. An LLM must never zero a player or invent a starter.
- **Verbatim quote required**: the adjustment must carry the exact sentence it rests on, not a
  paraphrase. A paraphrase hides the reasoning step where the error happens; a quote makes a wrong
  adjustment auditable in seconds.
- **Sourced**: every adjustment carries a URL. No source, no adjustment.
- **Shadow mode first**: log adjustments for ≥30 slates *without applying them*. Score adjusted vs
  unadjusted before a single lineup is affected. If it does not beat its own absence, it never ships.
- **Logged**: write the pre- and post-adjustment projection so the effect is recoverable.

Model: **Opus 5** with `web_search_20260209` as the default. Judging whether an ambiguous quote means
what it appears to mean is the part most likely to fail, so run **Fable 5.1 as an A/B against Opus on
the same shadow-mode slates** rather than assuming either tier is sufficient. Compare adjustment
precision (share of adjustments a human would endorse), not just MAE — a tier that is wrong less
often on the judgment calls may justify 2× on a nightly job, and that is a measurable question, not
an architectural one.

### L3. LLM vs. the whole pipeline — head-to-head

Open question worth answering honestly: **does an LLM with web access beat three seasons of box scores
and a GradientBoosting stack?**

Design so the answer is trustworthy:

- Sample ~20 historical slates spanning December–March (avoid only-recent bias)
- The LLM gets: date, teams, DraftKings salaries, player names. **No** access to the projections,
  models, or engineered features — public information only, as a human handicapper would work
- It outputs projected minutes per player on the slate
- Score against `complex_position_overlap`, `formula_c_baseline`, and DFF on MAE, bias, and realized
  lineup FP
- Report where it wins and loses, not just the aggregate — the interesting result is almost certainly
  "loses on average, wins on players with news," which argues for L2 over replacement

Model: **Opus 5** for the slate projections themselves. Use **Fable 5.1** only if the experiment is
run as one autonomous long-horizon job that designs its own comparisons.

Prior worth stating up front so the result is not rationalized afterward: the LLM is expected to lose
on aggregate MAE and win on the subset where public news is decisive. If it wins outright, the
statistical pipeline needs rethinking, not defending.

### L4. Weekly post-mortem analyst

Feed a week of projections vs. actuals and ask for systematic patterns: which archetypes are
consistently over-projected, whether bias is drifting, whether a rule stopped earning its place.

Model: **Fable 5.1** — the one genuinely Fable-worthy task here. Open-ended multi-step analysis where
the reasoning is the product and the answer is not known in advance. Weekly cadence makes the 2×
price irrelevant.

Guard against the same fluency failure: require every claimed pattern to come with the query that
produced it and the row count behind it, so a confident-sounding finding can be re-run and falsified
rather than believed. An analysis that cannot be reproduced from its own stated method is noise
regardless of how well it reads.

---

## Model quality

### M1. Predict FP per minute, not FP

`MIN` is 69.9% of feature importance in the `current` model, but the model trains on **actual** minutes
and serves on **projected** minutes carrying ~5.0 MAE. It treats a noisy estimate as ground truth.

The `np.random.uniform(-8, 8)` noise in `lambda/supervised-learning/lambda_function.py` is a crude
patch: the magnitude is about right (std 4.6 ≈ the real 5.0 MAE) but the shape is wrong, the error is
assumed unbiased when it is not, and it is unseeded — so two training runs an hour apart differ by
~0.01 R², larger than most improvements worth detecting.

Retraining on *stored historical projections* is tempting and wrong: it teaches the model to correct
for bugs that have since been fixed (the stale-OUT zeros, the undamped redistribution).

Do this instead — the split the field uses, because per-minute production is stable while minutes are
volatile:

1. Target `FP_PER_MIN`, drop `MIN` from the feature set entirely
2. Serve `projected_FP = predicted_FP_PER_MIN × projected_MIN`
3. Minutes uncertainty now propagates as a clean multiplication that can carry error bars

`barebones` is already closest to this shape and posted the best R² (0.661 on 13 features vs 0.651 on
21). Seed the RNG regardless of which path is taken.

### M2. Shrink projections before optimizing

Lineups project ~262 FP and realize ~242 (+20). DFF projects 272 and realizes 265 (+7). Same optimizer,
3× the bias — so most of the gap is projection bias amplified by argmax, not argmax alone.

Argmax over noisy estimates is upward-biased no matter what, but the magnitude is controllable: shrink
each projection toward the positional mean in proportion to its uncertainty, so volatile players are
discounted more than stable ones. Optionally penalize the objective by variance
(`maximize proj_FP - λ·σ`), with λ tuned separately for cash (floor) and GPP (ceiling).

### M3. Model DNP risk explicitly

5% of players projected >10 minutes log zero. A late scratch burns a roster slot and the optimizer has
no way to price that risk. Attach P(plays) to each projection and use `P(plays) × projected_FP` in the
objective.

Likely better served by L1/L2 than by a classifier — a scratch is usually announced in text before it
is visible in data.

### M4. Cut dead features

The nine cluster dummies contribute ~0.4% of importance **combined**; `REST_DAYS` and `IS_HOME` are
~0.05% each. If clusters stay dead after M1, the `cluster-scraper` → `nba-clustering` branch of the
pipeline is two Lambdas and an ECR repo serving a rounding error. Measure, then decide.

### M5. Opponent and pace adjustment

`OPPONENT` was removed with a note to replace it with defensive rating; that never happened. Pace is the
largest driver of available possessions and is absent entirely. Unlike the cluster features, this is a
real missing signal — `cluster-scraper` already pulls `PACE` and `DEF_RATING`.

### M6. Team-minute constraint

Per-player bias tracks team projected total monotonically: 180 min → −2.22, 230 → +0.10, 292 → +5.62.
Naive rescaling to 240 makes MAE worse (5.011 → 5.399) because the DFF slate covers only ~10 of ~15
rotation players. The constraint is right; it needs a per-slate target derived from covered-player
count, not a constant.

---

## Operations

### O1. Retune `INJURY_ADJUSTMENT_WEIGHT` on fresh data

Set to 0.35 from the 2025-26 fresh-injury window (Nov 29 – Jan 19), validated on a Dec 26 – Jan 19
holdout; per-date optimum ranged 0.0–1.0 with median 0.40. Re-fit after ~30 slates of 2026-27 with a
working injury feed, which is the first clean sample this system has ever had.

### O2. Alarm on data staleness, not just Lambda errors

The CloudWatch alarms catch a Lambda that raises. They would not have caught the January–June failure,
because `injury-scraper` failed *inside* a handler that returned 200. An S3 object-age check on
`data/injuries/current.parquet` and `data/box_scores/current.parquet` closes that gap directly.

### O3. Pin transitive dependencies

Two of nine rebuilds failed on 2026-09-18/19 from unpinned transitive deps resolving to versions with
no Lambda-compatible wheel (Pillow via pdfplumber, scipy via scikit-learn). Remaining drift: `pandas`
2.1.3/2.1.4, `pyarrow` 14.0.1/14.0.2/20.0.0, unpinned `boto3` in three scrapers. A constraints file per
function freezes the resolved set; deferred deliberately, but the failure recurs on every rebuild.
