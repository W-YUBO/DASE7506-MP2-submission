"""Shared feature extraction for MP2 selection method (v4).

Used by both training scripts and submission.py so that train and
inference features can never drift apart.

Feature layout per candidate: text (10) + group (10) + centered embedding (1024)

text (10):
  0 parse_ok            parse_answer succeeds
  1 log length
  2 log lines
  3 log #equals
  4 has \\boxed
  5 has ####
  6 log #digits
  7 has 'answer' keyword
  8 log #calc steps     lines containing '='
  9 qnum_overlap        fraction of distinct question numbers reused

group (10):
  0 vote_weight         share of candidates with same parsed answer
  1 is_majority
  2 is_unique
  3 n_distinct answers in group
  4 cos(emb, group centroid)
  5 max cos(emb, other candidate)   embedding-level agreement
  6 answer_dev          |answer - median(answers)| / (|median| + 1)
  7 answer_is_int       parsed answer is an integer
  8 is_last_number      parsed answer equals the last number in the text
  9 emb_norm            log L2 norm of the raw embedding

text v9 additions (indices 10-12): arithmetic self-check. Expressions like
"a op b = c" are extracted from the text and actually evaluated:
 10 log #checked        how many checkable equations were found
 11 frac_wrong          fraction of them that compute incorrectly
 12 has_wrong           at least one wrong equation
"""
import re
from fractions import Fraction

import numpy as np

from common import parse_answer

N_TEXT = 13
N_GROUP = 10
N_EMB = 1024
N_FEAT = N_TEXT + N_GROUP + N_EMB

_NUM_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
_CHAIN_RE = re.compile(
    r'((?:-?\d[\d,]*(?:\.\d+)?\s*[+\-*/×÷x]\s*)+-?\d[\d,]*(?:\.\d+)?)'
    r'\s*=\s*(-?\d[\d,]*(?:\.\d+)?)')
_SAFE_RE = re.compile(r'^[\d\.\+\-\*/\s]+$')


def _arith_check(text):
    """Evaluate 'expr = number' chains; returns (n_checked, n_wrong).

    Left side must be a pure arithmetic chain (digits and +-*/ only after
    normalization), evaluated with standard operator precedence via eval.
    """
    checked = wrong = 0
    for m in _CHAIN_RE.finditer(text):
        expr = (m.group(1).replace(',', '').replace('×', '*')
                .replace('÷', '/').replace('x', '*'))
        if not _SAFE_RE.match(expr):
            continue
        try:
            v = eval(expr)                      # noqa: S307 - sanitized above
            r = float(m.group(2).replace(',', ''))
        except (SyntaxError, ZeroDivisionError, ValueError, OverflowError):
            continue
        checked += 1
        if abs(v - r) > max(1e-6, abs(r) * 1e-9):
            wrong += 1
    return checked, wrong


def _numbers(text):
    return {m.group().replace(',', '') for m in _NUM_RE.finditer(text)}


def text_features(text, question):
    n = len(text)
    lines = text.split('\n')
    equals = text.count('=')
    digits = sum(ch.isdigit() for ch in text)
    low = text.lower()
    steps = sum(1 for ln in lines if '=' in ln)
    qnums = _numbers(question)
    overlap = (len(qnums & _numbers(text)) / len(qnums)) if qnums else 0.0
    checked, wrong = _arith_check(text)
    return [
        1.0 if parse_answer(text) is not None else 0.0,
        float(np.log1p(n)),
        float(np.log1p(len(lines))),
        float(np.log1p(equals)),
        1.0 if '\\boxed' in text else 0.0,
        1.0 if '####' in text else 0.0,
        float(np.log1p(digits)),
        1.0 if 'answer' in low else 0.0,
        float(np.log1p(steps)),
        float(overlap),
        float(np.log1p(checked)),
        (wrong / checked) if checked else 0.0,
        1.0 if wrong else 0.0,
    ]


def candidate_features(row, emb):
    """row: bank entry; emb: float32 [4, 1024]. Returns float32 [4, N_FEAT]."""
    answers = [parse_answer(c['text']) for c in row['candidates']]
    counts = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    n_distinct = len(counts)
    top = max(counts.values()) if counts else 0

    centroid = emb.mean(axis=0)
    cn = np.linalg.norm(centroid)
    norms = np.linalg.norm(emb, axis=1) + 1e-9
    embn = emb / norms[:, None]

    # numeric values of parsed answers, for deviation features
    vals = []
    for a in answers:
        try:
            vals.append(float(Fraction(a)) if a is not None else None)
        except (ValueError, ZeroDivisionError):
            vals.append(None)
    valid_vals = [v for v in vals if v is not None]
    median = float(np.median(valid_vals)) if valid_vals else 0.0

    out = np.zeros((4, N_FEAT), dtype=np.float32)
    for i, c in enumerate(row['candidates']):
        a = answers[i]
        share = counts.get(a, 0) / 4.0
        is_maj = 1.0 if (a is not None and counts[a] == top) else 0.0
        is_uniq = 1.0 if (a is not None and counts[a] == 1) else 0.0
        cos_c = float(emb[i] @ centroid / (norms[i] * cn + 1e-9))
        sims = [float(embn[i] @ embn[j]) for j in range(4) if j != i]
        max_sim = max(sims)
        v = vals[i]
        dev = abs(v - median) / (abs(median) + 1.0) if v is not None else 0.0
        is_int = 0.0
        if a is not None:
            try:
                is_int = 1.0 if Fraction(a).denominator == 1 else 0.0
            except (ValueError, ZeroDivisionError):
                is_int = 0.0
        last_nums = _numbers(c['text'].rsplit('\n', 3)[-1]) if a is not None else set()
        is_last = 1.0 if (a is not None and a.replace(',', '') in last_nums) else 0.0
        emb_norm = float(np.log1p(norms[i]))
        out[i, :N_TEXT] = text_features(c['text'], row['question'])
        out[i, N_TEXT:N_TEXT + N_GROUP] = [
            share, is_maj, is_uniq, float(n_distinct), cos_c, max_sim,
            float(dev), is_int, is_last, emb_norm,
        ]
        out[i, N_TEXT + N_GROUP:] = emb[i] - centroid
    return out
