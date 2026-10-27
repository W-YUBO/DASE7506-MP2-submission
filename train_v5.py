"""v5: 5-seed ensemble of the v4-winning config (h128, lr0.02, new features).

Usage:
  python train_v5.py train 0 1 2   # trains seeds, saves runs/v5_parts/seed_i.npz
  python train_v5.py combine       # evaluates ensemble, saves model_bundle_v5
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
PARTS = RUNS / 'v5_parts'
CFG = dict(hidden=128, lr=0.02)


def load_data():
    X_tr, Y_tr, nc_tr = build('train')
    X_va, Y_va, nc_va = build('validation')
    flat = X_tr.reshape(-1, X_tr.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return (X_tr - mu) / sd, (X_va - mu) / sd, Y_tr, Y_va, nc_va, mu, sd


def cmd_train(seeds):
    print('building features ...')
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
    X_tr, X_va, Y_tr, Y_va, nc_va, mu, sd = load_data()
    parts = sorted(PARTS.glob('seed_*.npz'))
    assert parts, 'no seed parts found'
    P = np.zeros(X_va.shape[:2])
    for p in parts:
        with np.load(p) as f:
            m = ('mlp', f['W1'], f['b1'], f['W2'], float(f['b2'][0]))
        P += softmax_rows(scores(m, X_va))
    P /= len(parts)
    acc, hacc, hn = metrics(P, Y_va, nc_va)
    print(f'v5 ensemble({len(parts)} seeds): val={acc:.2f} hard={hacc:.2f} (n={hn})')

    W1s, b1s, W2s, b2s = [], [], [], []
    for p in parts:
        with np.load(p) as f:
            W1s.append(f['W1']); b1s.append(f['b1'])
            W2s.append(f['W2']); b2s.append(float(f['b2'][0]))
    bundle = RUNS / 'model_bundle_v5'
    bundle.mkdir(exist_ok=True)
    np.savez(bundle / 'scorer.npz', kind=np.array(['mlp_ens']),
             W1=np.stack(W1s), b1=np.stack(b1s), W2=np.stack(W2s),
             b2=np.array(b2s), mu=mu, sd=sd)
    size = (bundle / 'scorer.npz').stat().st_size
    print(f'saved {bundle}/scorer.npz ({size/2**20:.3f} MiB)')
    (RUNS / 'ablation-v5.json').write_text(json.dumps(
        {'seeds': len(parts), 'val': acc, 'hard': hacc}, default=float))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        cmd_train([int(x) for x in sys.argv[2:]])
    else:
        cmd_combine()
