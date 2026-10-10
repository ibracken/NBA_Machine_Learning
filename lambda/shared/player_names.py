"""
Match player names across sources (box scores, DFF, the NBA injury report) to box-score names.

Box-score names are canonical: they come with NBA player IDs, and every model feature is built from them. Other
sources spell players differently ("bobby portis" / "bobby portis jr.", "gregory jackson" / "gg jackson",
"hansen yang" / "yang hansen"). Rather than keep an alias list, each name is matched only against the players who
could plausibly be meant (the caller passes them: tonight's rosters, or one team's):
  1. same key (letters only, accents and suffixes removed)
  2. same words in any order ("hansen yang" / "yang hansen")
  3. same last name and compatible first name: one a prefix of the other ("alex" / "alexandre") or initials
     ("gg" / "gregory")
  4. same last name and one clearly closest spelling (difflib ratio >= 0.85)
A rule is applied only when it picks exactly one candidate. Rules 2-4 apply only to names whose key matches no
player in `known` (every box-score name, if given): a known player missing from the candidates is not playing,
and must not be matched to a lookalike ("davion mitchell" -> "donovan mitchell" when Miami is off).
Unmatched names are returned so callers can log them.

Used by scripts/ and, through deploy.py's shared build context, by Lambdas whose Dockerfile copies it.
"""

import difflib
import re
from unidecode import unidecode

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
MIN_RATIO = 0.85


def words(name):
    """Lowercase ASCII words with punctuation and suffixes removed ("P.J. Washington Jr." -> ["pj", "washington"])."""
    tokens = re.sub(r"[^a-z\s-]", "", unidecode(str(name)).lower()).replace("-", " ").split()
    return [t for t in tokens if t not in SUFFIXES]


def key(name):
    return "".join(words(name))


def compatible_first(a, b):
    short, long_ = sorted((a, b), key=len)
    return long_.startswith(short) or (len(short) <= 2 and short[0] == long_[0])


def match_names(names, candidates, known=()):
    """Map each of `names` to one of `candidates` (box-score names), or None. Returns (mapping, unmatched)."""
    cand = {c: words(c) for c in set(candidates)}
    known_keys = {key(k) for k in known}
    by_key, by_set, by_last = {}, {}, {}
    for c, w in cand.items():
        if not w:
            continue
        by_key.setdefault("".join(w), set()).add(c)
        by_set.setdefault(frozenset(w), set()).add(c)
        by_last.setdefault(w[-1], set()).add(c)

    mapping = {}
    for n in set(names):
        w = words(n)
        found = None
        exact = by_key.get("".join(w), set()) if w else set()
        if len(exact) == 1:
            found = next(iter(exact))
        elif w and not exact and "".join(w) not in known_keys:
            same_last = by_last.get(w[-1], set())
            prefix = {c for c in same_last if len(w) > 1 and compatible_first(w[0], cand[c][0])}
            close = difflib.get_close_matches("".join(w), ["".join(cand[c]) for c in same_last], n=2, cutoff=MIN_RATIO)
            for hits in (by_set.get(frozenset(w), set()), prefix,
                         {c for c in same_last if "".join(cand[c]) in close} if len(close) == 1 else set()):
                if len(hits) == 1:
                    found = next(iter(hits))
                    break
        mapping[n] = found
    unmatched = sorted(n for n, m in mapping.items() if m is None)
    return mapping, unmatched
