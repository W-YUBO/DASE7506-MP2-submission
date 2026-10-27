"""v9: arithmetic self-check features + enlarged seed pool + greedy ensemble.

Usage:
  python train_v9.py train 0 1      # train seeds, save runs/v9_parts/seed_i.npz
  python train_v9.py combine        # greedy-select ensemble, save model_bundle_v9
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(CODE))

from train_v2 import build, metrics, train_mlp, softmax_rows, scores  # noqa: E402

RUNS = CODE / 'runs'
PARTS = RUNS / 'v9_parts'
CFG = dict(hidden=128, lr=0.02)


def load_data():
    X_tr, Y_tr, nc_tr = build('train')
    X_va, Y_va, nc_va = build('validation')
    flat = X_tr.reshape(-1, X_tr.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return (X_tr - mu) / sd, (X_va - mu) / sd, Y_tr, Y_va, nc_va, mu, sd


def cmd_train(seeds):
    print('building v9 features (with arithmetic self-check) ...')
    X_tr, X_va, Y_tr, Y_va, nc_va, mu, sd = load_data()
    w = np.ones(len(X_tr))
    PARTS.mkdir(exist_ok=True)
    for s in seeds:
        t = time.time()
        m = train_mlp(X_tr, Y_tr, w, seed=s, **CFG)
        acc, hacc, _ = metrics(scores(m, X_va), Y_va, nc_va)
        np.savez(PARTS / f'seed_{s}.npz', W1=m[1], b1=m[2], W2=m[3],
                 b2=np.array([m[4]]))
        print(f'  seed {s}: val={acc:5.2f} hard={hacc:5.2f}  [{time.time()-t:.0f}s]')


def cmd_combine():
    _, X_va, _, Y_va, nc_va, mu, sd = load_data()
    members = {}
    for p in sorted(PARTS.glob('seed_*.npz'), key=lambda p: int(p.stem.split('_')[1])):
        with np.load(p) as f:
            m = ('mlp', f['W1'], f['b1'], f['W2'], float(f['b2'][0]))
        members[p.stem] = softmax_rows(scores(m, X_va))

    chosen, pool = [], list(members)
    P = np.zeros(X_va.shape[:2])
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

    acc, hacc, hn = metrics(P, Y_va, nc_va)
    print(f'\nv9 greedy ensemble {chosen}: val={acc:.2f} hard={hacc:.2f} (n={hn})')

    W1s, b1s, W2s, b2s = [], [], [], []
    for name in chosen:
        with np.load(PARTS / f'{name}.npz') as f:
            W1s.append(f['W1']); b1s.append(f['b1'])
            W2s.append(f['W2']); b2s.append(float(f['b2'][0]))
    bundle = RUNS / 'model_bundle_v9'
    bundle.mkdir(exist_ok=True)
    np.savez(bundle / 'scorer.npz', kind=np.array(['mlp_ens']),
             W1=np.stack(W1s), b1=np.stack(b1s), W2=np.stack(W2s),
             b2=np.array(b2s), mu=mu, sd=sd)
    size = (bundle / 'scorer.npz').stat().st_size
    print(f'saved {bundle}/scorer.npz ({size/2**20:.3f} MiB)')
    (RUNS / 'ablation-v9.json').write_text(json.dumps(
        {'chosen': chosen, 'val': acc, 'hard': hacc}, default=float))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        cmd_train([int(x) for x in sys.argv[2:]])
    else:
        cmd_combine()
