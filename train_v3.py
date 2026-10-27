"""v3: seed ensemble of the v2-winning config (MLP-64, group-softmax).

Trains N_SEEDS MLPs with different initializations, averages per-group
softmax probabilities, and evaluates the ensemble on validation.
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
N_SEEDS = 5


def ensemble_probs(models, X):
    P = np.zeros(X.shape[:2])
    for m in models:
        Z = scores(m, X)
        P += softmax_rows(Z)
    return P / len(models)


def main():
    t0 = time.time()
    print('building features ...')
    X_tr, Y_tr, nc_tr = build('train')
    X_va, Y_va, nc_va = build('validation')
    flat = X_tr.reshape(-1, X_tr.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    w = np.ones(len(X_tr))

    models = []
    for s in range(N_SEEDS):
        t = time.time()
        m = train_mlp(X_tr, Y_tr, w, seed=s)
        acc, hacc, _ = metrics(scores(m, X_va), Y_va, nc_va)
        print(f'  seed {s}: val={acc:5.2f} hard={hacc:5.2f}  [{time.time()-t:.0f}s]')
        models.append(m)

    P = ensemble_probs(models, X_va)
    acc, hacc, hn = metrics(P, Y_va, nc_va)
    print(f'\nv3 ensemble({N_SEEDS} seeds): val={acc:.2f} hard={hacc:.2f} (n={hn})')

    bundle = RUNS / 'model_bundle_v3'
    bundle.mkdir(exist_ok=True)
    np.savez(bundle / 'scorer.npz',
             kind=np.array(['mlp_ens']),
             W1=np.stack([m[1] for m in models]),
             b1=np.stack([m[2] for m in models]),
             W2=np.stack([m[3] for m in models]),
             b2=np.stack([m[4] for m in models]),
             mu=mu, sd=sd)
    size = (bundle / 'scorer.npz').stat().st_size
    print(f'saved {bundle}/scorer.npz ({size/2**20:.3f} MiB)')
    (RUNS / 'ablation-v3.json').write_text(json.dumps(
        {'seeds': N_SEEDS, 'val': acc, 'hard': hacc}, default=float))


if __name__ == '__main__':
    main()
