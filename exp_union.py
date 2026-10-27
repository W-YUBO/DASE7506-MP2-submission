"""Experiment: greedy ensemble over the UNION of v8 pool (16 members,
1044-dim features) and v9 pool (7 members, 1047-dim features).

v4 layout = v9 layout with columns 10..12 (the 3 arithmetic features)
dropped, so both can be derived from the current sol_features output.
"""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(CODE))

from train_v2 import build, metrics, softmax_rows, scores  # noqa: E402

RUNS = CODE / 'runs'
DROP = [10, 11, 12]  # arithmetic columns in v9 layout


def load_members():
    X_va, Y_va, nc_va = build('validation')            # v9 features, raw
    X_va_v4 = np.delete(X_va, DROP, axis=2)

    members = {}
    with np.load(RUNS / 'model_bundle_v6' / 'scorer.npz') as f:  # v8 bundle
        mu4, sd4 = f['mu'], f['sd']
    Xv4 = (X_va_v4 - mu4) / sd4
    for p in sorted((RUNS / 'v5_parts').glob('seed_*.npz')):
        with np.load(p) as f:
            m = ('mlp', f['W1'], f['b1'], f['W2'], float(f['b2'][0]))
        members['v8:' + p.stem] = softmax_rows(scores(m, Xv4))

    with np.load(RUNS / 'model_bundle_v9' / 'scorer.npz') as f:
        mu9, sd9 = f['mu'], f['sd']
    Xv9 = (X_va - mu9) / sd9
    for p in sorted((RUNS / 'v9_parts').glob('seed_*.npz')):
        with np.load(p) as f:
            m = ('mlp', f['W1'], f['b1'], f['W2'], float(f['b2'][0]))
        members['v9:' + p.stem] = softmax_rows(scores(m, Xv9))
    return members, Y_va, nc_va


def main():
    members, Y_va, nc_va = load_members()
    n = list(members.values())[0].shape[0]
    chosen, pool = [], list(members)
    P = np.zeros((n, 4))
    best_acc, best_hard = -1.0, -1.0
    while pool:
        gains = []
        for cand in pool:
            Pn = (P * len(chosen) + members[cand]) / (len(chosen) + 1)
            acc, hacc, _ = metrics(Pn, Y_va, nc_va)
            gains.append((acc, hacc, cand, Pn))
        gains.sort(key=lambda g: (g[0], g[1]), reverse=True)
        acc, hacc, cand, Pn = gains[0]
        if acc <= best_acc and hacc <= best_hard:
            break
        chosen.append(cand)
        pool.remove(cand)
        P = Pn
        best_acc, best_hard = acc, hacc
        print(f'  + {cand}: val={acc:.2f} hard={hacc:.2f}')
    print(f'\nunion greedy {chosen}: val={best_acc:.2f} hard={best_hard:.2f}')


if __name__ == '__main__':
    main()
