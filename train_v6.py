"""v6: greedy ensemble subset selection over the 10 trained seeds.

Forward selection on validation: start from the strongest single seed,
repeatedly add the member that most improves validation accuracy
(softmax-probability averaging). No test data involved.
"""
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(CODE))

from train_v2 import build, metrics, softmax_rows, scores  # noqa: E402

RUNS = CODE / 'runs'
PARTS = RUNS / 'v5_parts'


def main():
    X_va, Y_va, nc_va = build('validation')
    X_tr, Y_tr, nc_tr = build('train')
    flat = X_tr.reshape(-1, X_tr.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    X_va = (X_va - mu) / sd
    members = {}
    for p in sorted(PARTS.glob('seed_*.npz')):
        with np.load(p) as f:
            m = ('mlp', f['W1'], f['b1'], f['W2'], float(f['b2'][0]))
        members[p.stem] = softmax_rows(scores(m, X_va))

    chosen, pool = [], list(members)
    P = np.zeros(X_va.shape[:2])
    best_acc, best_hard = -1, -1
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

    acc, hacc, hn = metrics(P, Y_va, nc_va)
    print(f'\nv6 greedy ensemble {chosen}: val={acc:.2f} hard={hacc:.2f} (n={hn})')

    W1s, b1s, W2s, b2s = [], [], [], []
    for name in chosen:
        with np.load(PARTS / f'{name}.npz') as f:
            W1s.append(f['W1']); b1s.append(f['b1'])
            W2s.append(f['W2']); b2s.append(float(f['b2'][0]))
    bundle = RUNS / 'model_bundle_v6'
    bundle.mkdir(exist_ok=True)
    np.savez(bundle / 'scorer.npz', kind=np.array(['mlp_ens']),
             W1=np.stack(W1s), b1=np.stack(b1s), W2=np.stack(W2s),
             b2=np.array(b2s), mu=mu, sd=sd)
    size = (bundle / 'scorer.npz').stat().st_size
    print(f'saved {bundle}/scorer.npz ({size/2**20:.3f} MiB)')
    (RUNS / 'ablation-v6.json').write_text(json.dumps(
        {'chosen': chosen, 'val': acc, 'hard': hacc}, default=float))


if __name__ == '__main__':
    main()
