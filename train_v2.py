"""v2 training: group-softmax objective + MLP + hard-sample weighting.

All variants train on train split, evaluate on validation. Selection metric
is hard-subset accuracy (groups with <=2 correct candidates, closest to the
test distribution), tie-broken by overall validation accuracy.

Variants:
  F1 LR, pointwise BCE            (continuity with v1, on v2 features)
  F2 LR, group-softmax loss
  F3 MLP(64), group-softmax loss
  F4 MLP(64), group-softmax + hard-group weight x2
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(CODE))

from common import read_jsonl, parse_answer  # noqa: E402
from sol_features import candidate_features, N_FEAT  # noqa: E402

RUNS = CODE / 'runs'


def build(split):
    bank = read_jsonl(CODE / 'data' / 'candidates' / f'gsm8k_{split}.jsonl')
    with np.load(CODE / 'data' / 'candidates' / f'gsm8k_{split}.npz') as f:
        emb = f['embeddings'].reshape(len(bank), 4, 1024).astype(np.float32)
    gold = {r['id']: parse_answer(r['answer'])
            for r in read_jsonl(CODE / 'data' / f'gsm8k_{split}.jsonl')}

    n = len(bank)
    X = np.zeros((n, 4, N_FEAT), np.float32)
    Y = np.zeros((n, 4), np.float32)
    for qi, row in enumerate(bank):
        X[qi] = candidate_features(row, emb[qi])
        g = gold[row['id']]
        for i, c in enumerate(row['candidates']):
            a = parse_answer(c['text'])
            Y[qi, i] = 1.0 if (a is not None and a == g) else 0.0
    return X, Y, Y.sum(axis=1).astype(np.int64)


def metrics(Z, Y, n_correct):
    """Z: group scores [n,4]."""
    pick = Z.argmax(axis=1)
    hit = Y[np.arange(len(pick)), pick]
    mask = n_correct <= 2
    return hit.mean() * 100, hit[mask].mean() * 100, int(mask.sum())


# ------------------------------------------------------------- objectives
def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def group_softmax_grad(Z, Y, w):
    """Loss per group: -log(sum of softmax probs of correct candidates).

    dL/dz_j = p_j * (1 - 1{j in C} / S),  S = sum_{i in C} p_i.
    Returns grad [n,4] and mean loss.
    """
    P = softmax_rows(Z)
    S = (P * Y).sum(axis=1)                      # [n]
    S = np.maximum(S, 1e-12)
    G = P * (1 - Y / S[:, None])
    G *= w[:, None]
    loss = (-np.log(S) * w).sum() / w.sum()
    return G / w.sum(), loss


def bce_grad(Z, Y, w):
    """Pointwise BCE on flattened candidates. Returns grad [n,4], loss."""
    P = 1 / (1 + np.exp(-Z))
    wg = np.repeat(w[:, None], 4, axis=1) / 4.0
    G = (P - Y) * wg
    loss = -(Y * np.log(P + 1e-12) + (1 - Y) * np.log(1 - P + 1e-12))
    return G / w.sum(), (loss * wg).sum() / w.sum()


# ------------------------------------------------------------- models
def _minibatches(n, bs, rng):
    idx = rng.permutation(n)
    return [idx[s:s + bs] for s in range(0, n, bs)]


def train_lr(X, Y, w, objective, epochs=60, lr=0.2, l2=1e-4, bs=512, seed=0):
    """Minibatch SGD with momentum on groups."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, X.shape[2])
    b = 0.0
    vW, vb = 0.0, 0.0
    grad_fn = group_softmax_grad if objective == 'softmax' else bce_grad
    for ep in range(epochs):
        for bi in _minibatches(len(X), bs, rng):
            Xb, Yb, wb = X[bi], Y[bi], w[bi]
            Z = np.einsum('nqf,f->nq', Xb, W) + b
            G, _ = grad_fn(Z, Yb, wb)
            G *= w[bi].sum() / len(bi)          # rescale to per-group mean
            gW = np.einsum('nqf,nq->f', Xb, G) + l2 * W
            gb = G.sum()
            vW = 0.9 * vW + gW
            vb = 0.9 * vb + gb
            W -= lr * vW
            b -= lr * vb
    return ('lr', W, b)


def train_mlp(X, Y, w, hidden=64, epochs=60, lr=0.01, l2=1e-4, bs=512, seed=0):
    """Minibatch SGD with momentum; group-softmax loss."""
    rng = np.random.default_rng(seed)
    F = X.shape[2]
    W1 = rng.normal(0, np.sqrt(2 / F), (F, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    W2 = rng.normal(0, np.sqrt(2 / hidden), hidden).astype(np.float32)
    b2 = 0.0
    m = {k: 0.0 for k in ('W1', 'b1', 'W2', 'b2')}   # momentum
    for ep in range(epochs):
        for bi in _minibatches(len(X), bs, rng):
            Xb, Yb, wb = X[bi], Y[bi], w[bi]
            H = np.maximum(Xb @ W1 + b1, 0)          # [n,4,h]
            Z = H @ W2 + b2                          # [n,4]
            G, _ = group_softmax_grad(Z, Yb, wb)     # [n,4]
            G *= w[bi].sum() / len(bi)
            gW2 = np.einsum('nqh,nq->h', H, G) + l2 * W2
            gb2 = G.sum()
            dH = (G[..., None] * W2) * (H > 0)       # [n,4,h]
            gW1 = np.einsum('nqf,nqh->fh', Xb, dH) + l2 * W1
            gb1 = dH.sum(axis=(0, 1))
            for k, g in (('W1', gW1), ('b1', gb1), ('W2', gW2), ('b2', gb2)):
                m[k] = 0.9 * m[k] + g
                if k == 'W1':
                    W1 -= lr * m[k]
                elif k == 'b1':
                    b1 -= lr * m[k]
                elif k == 'W2':
                    W2 -= lr * m[k]
                else:
                    b2 -= lr * m[k]
    return ('mlp', W1, b1, W2, b2)


def scores(model, X):
    if model[0] == 'lr':
        return np.einsum('nqf,f->nq', X, model[1]) + model[2]
    _, W1, b1, W2, b2 = model
    return np.maximum(X @ W1 + b1, 0) @ W2 + b2


def main():
    t0 = time.time()
    print('building v2 features ...')
    X_tr, Y_tr, nc_tr = build('train')
    X_va, Y_va, nc_va = build('validation')
    print(f'  done in {time.time()-t0:.1f}s')

    flat = X_tr.reshape(-1, N_FEAT)
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    w_easy = np.ones(len(X_tr))
    w_hard = np.where(nc_tr <= 2, 2.0, 1.0)

    variants = [
        ('F1 LR  pointwise',        train_lr,  dict(objective='bce'),      w_easy),
        ('F2 LR  group-softmax',    train_lr,  dict(objective='softmax'),  w_easy),
        ('F3 MLP group-softmax',    train_mlp, dict(),                     w_easy),
        ('F4 MLP + hard x2',        train_mlp, dict(),                     w_hard),
    ]
    results = {}
    trained = {}
    for name, fn, kw, w in variants:
        t = time.time()
        model = fn(X_tr, Y_tr, w, **kw)
        acc, hacc, hn = metrics(scores(model, X_va), Y_va, nc_va)
        results[name] = {'val': round(acc, 2), 'hard': round(hacc, 2), 'n_hard': hn}
        trained[name] = model
        print(f'{name:24s} val={acc:5.2f}  hard={hacc:5.2f} (n={hn})  [{time.time()-t:.0f}s]')

    best = max(results, key=lambda k: (results[k]['hard'], results[k]['val']))
    print(f'\nbest variant: {best}')
    model = trained[best]
    bundle = RUNS / 'model_bundle_v2'
    bundle.mkdir(exist_ok=True)
    if model[0] == 'lr':
        np.savez(bundle / 'scorer.npz', kind=np.array(['lr']), W1=model[1],
                 b1=np.array([model[2]]), mu=mu, sd=sd)
    else:
        np.savez(bundle / 'scorer.npz', kind=np.array(['mlp']), W1=model[1],
                 b1=model[2], W2=model[3], b2=np.array([model[4]]), mu=mu, sd=sd)
    size = (bundle / 'scorer.npz').stat().st_size
    print(f'saved {bundle}/scorer.npz ({size/2**20:.3f} MiB)')
    (RUNS / 'ablation-v2.json').write_text(json.dumps(
        {'results': results, 'best': best}, indent=2, default=float))


if __name__ == '__main__':
    main()
