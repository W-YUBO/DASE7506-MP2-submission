"""v4: new group features (answer_dev, answer_is_int, is_last_number,
emb_norm) + small hyperparameter search. Single seed per config.
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


def main():
    print('building v4 features ...')
    X_tr, Y_tr, nc_tr = build('train')
    X_va, Y_va, nc_va = build('validation')
    flat = X_tr.reshape(-1, X_tr.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    w = np.ones(len(X_tr))

    configs = [
        dict(hidden=64,  lr=0.01),
        dict(hidden=64,  lr=0.02),
        dict(hidden=128, lr=0.01),
        dict(hidden=128, lr=0.02),
    ]
    results = {}
    for cfg in configs:
        t = time.time()
        m = train_mlp(X_tr, Y_tr, w, seed=0, **cfg)
        acc, hacc, hn = metrics(scores(m, X_va), Y_va, nc_va)
        key = f"h{cfg['hidden']}_lr{cfg['lr']}"
        results[key] = {'val': acc, 'hard': hacc}
        print(f'  {key}: val={acc:5.2f} hard={hacc:5.2f}  [{time.time()-t:.0f}s]')

    best = max(results, key=lambda k: (results[k]['hard'], results[k]['val']))
    print(f'\nbest config: {best}')
    (RUNS / 'ablation-v4.json').write_text(json.dumps(
        {'results': results, 'best': best}, indent=2, default=float))


if __name__ == '__main__':
    main()
