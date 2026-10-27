"""v7: group-centered embedding PCA denoising.

Fit PCA on train centered embeddings (feature cols 20:1044), keep top-k
components; model input becomes [20 handcrafted + k projected] dims.
PCA matrix P is a learned asset stored in the checkpoint and applied at
inference. Single seed per k; ensemble only if a win shows up.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(CODE))

from train_v2 import build, metrics, train_mlp, scores  # noqa: E402

RUNS = CODE / 'runs'
N_HAND = 20  # text (10) + group (10)


def apply_pca(X, P):
    return np.concatenate([X[:, :, :N_HAND], X[:, :, N_HAND:] @ P], axis=2)


def main():
    print('building features ...')
    X_tr, Y_tr, nc_tr = build('train')
    X_va, Y_va, nc_va = build('validation')

    emb_tr = X_tr[:, :, N_HAND:].reshape(-1, 1024).astype(np.float64)
    _, _, Vt = np.linalg.svd(emb_tr - emb_tr.mean(0), full_matrices=False)

    w = np.ones(len(X_tr))
    results = {}
    for k in (64, 128, 256):
        P = Vt[:k].T.astype(np.float32)
        Xt, Xv = apply_pca(X_tr, P), apply_pca(X_va, P)
        flat = Xt.reshape(-1, Xt.shape[2])
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
        t = time.time()
        m = train_mlp(Xt, Y_tr, w, hidden=128, lr=0.02, seed=0)
        acc, hacc, hn = metrics(scores(m, Xv), Y_va, nc_va)
        results[f'pca{k}'] = {'val': acc, 'hard': hacc}
        print(f'  pca{k}: val={acc:5.2f} hard={hacc:5.2f}  [{time.time()-t:.0f}s]')

    print('\nreference no-PCA h128 seed0: val=92.36 hard=75.83')
    (RUNS / 'ablation-v7.json').write_text(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
