"""Baseline statistics for DASE7506 MP2: label distribution + simple heuristics.

Run from mp2/:  python baseline_stats.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(CODE))

from common import read_jsonl, parse_answer  # noqa: E402


def majority_vote(cands):
    """Pick the candidate whose parsed answer is the majority; tie -> lowest index."""
    answers = [parse_answer(c['text']) for c in cands]
    valid = [(i, a) for i, a in enumerate(answers) if a is not None]
    if not valid:
        return 0
    counts = Counter(a for _, a in valid)
    best_count = max(counts.values())
    # lowest index among candidates whose answer has the max count
    for i, a in valid:
        if counts[a] == best_count:
            return i
    return 0


def main():
    for split in ('train', 'validation', 'test'):
        bank = read_jsonl(CODE / 'data' / 'candidates' / f'gsm8k_{split}.jsonl')
        gold = {row['id']: parse_answer(row['answer'])
                for row in read_jsonl(CODE / 'data' / f'gsm8k_{split}.jsonl')}

        n = len(bank)
        k_dist = Counter()          # number of correct candidates per group
        parse_fail = 0              # candidates whose answer cannot be parsed
        hit_random = 0.0            # expected accuracy of uniform random pick
        hit_index = [0, 0, 0, 0]    # accuracy of always picking index i
        hit_maj = 0                 # majority vote
        hit_longest = 0             # longest text
        hit_shortest = 0            # shortest text

        for row in bank:
            g = gold[row['id']]
            answers = [parse_answer(c['text']) for c in row['candidates']]
            correct = [a is not None and a == g for a in answers]
            k = sum(correct)
            k_dist[k] += 1
            parse_fail += sum(a is None for a in answers)
            hit_random += k / 4
            for i in range(4):
                hit_index[i] += correct[i]
            hit_maj += correct[majority_vote(row['candidates'])]
            lengths = [len(c['text']) for c in row['candidates']]
            hit_longest += correct[lengths.index(max(lengths))]
            hit_shortest += correct[lengths.index(min(lengths))]

        print(f'\n===== {split} ({n} questions) =====')
        print('correct-candidates-per-group distribution:')
        for k in range(5):
            cnt = k_dist.get(k, 0)
            print(f'  {k} correct: {cnt:5d}  ({100*cnt/n:5.1f}%)')
        print(f'candidate parse failures: {parse_fail}/{4*n} ({100*parse_fail/(4*n):.2f}%)')
        print('baselines (accuracy %):')
        print(f'  random uniform      : {100*hit_random/n:5.2f}')
        for i in range(4):
            print(f'  always pick index {i} : {100*hit_index[i]/n:5.2f}')
        print(f'  majority vote       : {100*hit_maj/n:5.2f}')
        print(f'  longest text        : {100*hit_longest/n:5.2f}')
        print(f'  shortest text       : {100*hit_shortest/n:5.2f}')


if __name__ == '__main__':
    main()
