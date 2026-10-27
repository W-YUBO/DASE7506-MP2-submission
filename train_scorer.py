"""Train a per-candidate correctness scorer and run ablations.

Features per candidate (question group g, candidate i):
  text  : parse_ok, log length, log lines, log #equals, has \\boxed, has ####,
          log #digits, has 'answer' keyword
  group : vote_weight (share of candidates with same parsed answer),
          is_majority, is_unique, n_distinct answers in group,
          cosine similarity to group embedding centroid
  emb   : group-centered 1024-d embedding (e_i - mean_g)

Model: logistic regression (pure numpy). Labels: parsed answer == gold.

Ablations (all trained on train, evaluated on validation):
  A text-only, B emb-only, C group-only, D all, E D + vote fusion,
  plus majority-vote reference. Also reports hard-subset accuracy
  (groups with <=2 correct candidates) which better matches test.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(CODE))

from common import read_jsonl, parse_answer  # noqa: E402

RUNS = CODE / 'runs'
N_TEXT = 8
N_GROUP = 5


# ---------------------------------------------------------------- features
def text_features(text):
    n = len(text)
    lines = text.count('\n') + 1
    equals = text.count('=')
    digits = sum(ch.isdigit() for ch in text)
    low = text.lower()
    return [
        1.0 if parse_answer(text) is not None else 0.0,
        np.log1p(n),
        np.log1p(lines),
        np.log1p(equals),
        1.0 if '\\boxed' in text else 0.0,
        1.0 if '####' in text else 0.0,
        np.log1p(digits),
        1.0 if 'answer' in low else 0.0,
    ]


def group_features(row, emb):
    answers = [parse_answer(c['text']) for c in row['candidates']]
    counts = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    n_distinct = len(counts)
    top = max(counts.values()) if counts else 0

    # cosine similarity to group centroid
    feats = []
    for i in range(4):
        a = answers[i]
        share = counts.get(a, 0) / 4.0
        is_maj = 1.0 if (a is not None and counts[a] == top) else 0.0
        is_uniq = 1.0 if (a is not None and counts[a] == 1) else 0.0
        centroid = emb.mean(axis=0)
        e = emb[i]
        cos = float(e @ centroid / (np.linalg.norm(e) * np.linalg.norm(centroid) + 1e-9))
        feats.append([share, is_maj, is_uniq, float(n_distinct), cos])
    return feats


def build(split):
    bank = read_jsonl(CODE / 'data' / 'candidates' / f'gsm8k_{split}.jsonl')
    with np.load(CODE / 'data' / 'candidates' / f'gsm8k_{split}.npz') as f:
        emb = f['embeddings'].reshape(len(bank), 4, 1024).astype(np.float32)
    gold = None
    gold_path = CODE / 'data' / f'gsm8k_{split}.jsonl'
    if gold_path.exists():
        gold = {r['id']: parse_answer(r['answer']) for r in read_jsonl(gold_path)}

    n = len(bank)
    X_text = np.zeros((n, 4, N_TEXT), np.float32)
    X_group = np.zeros((n, 4, N_GROUP), np.float32)
    X_emb = np.zeros((n, 4, 1024), np.float32)
    Y = np.zeros((n, 4), np.float32) if gold else None
    n_correct = np.zeros(n, np.int64) if gold else None

    for qi, row in enumerate(bank):
        for i, c in enumerate(row['candidates']):
            X_text[qi, i] = text_features(c['text'])
        X_group[qi] = group_features(row, emb[qi])
        X_emb[qi] = emb[qi] - emb[qi].mean(axis=0)  # group-centering
        if gold:
            g = gold[row['id']]
            for i, c in enumerate(row['candidates']):
                a = parse_answer(c['text'])
                Y[qi, i] = 1.0 if (a is not None and a == g) else 0.0
            n_correct[qi] = int(Y[qi].sum())
    return X_text, X_group, X_emb, Y, n_correct


# ---------------------------------------------------------------- model
def standardize(X_tr, X_va):
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-6
    return (X_tr - mu) / sd, (X_va - mu) / sd, mu, sd


def train_lr(X, y, epochs=300, lr=0.5, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, X.shape[1]).astype(np.float64)
    b = 0.0
    n = X.shape[0]
    for ep in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = (p - y) / n
        w -= lr * (X.T @ g + l2 * w)
        b -= lr * g.sum()
    return w, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def accuracy(P, Y):
    pick = P.reshape(-1, 4).argmax(axis=1)
    return float(Y[np.arange(len(pick)), pick].mean() * 100)


def hard_accuracy(P, Y, n_correct):
    mask = n_correct <= 2
    pick = P.reshape(-1, 4).argmax(axis=1)
    return float(Y[np.arange(len(pick)), pick][mask].mean() * 100), int(mask.sum())


def majority_pick(X_group):
    # vote_weight is column 0; argmax breaks ties toward lowest index
    return X_group[:, :, 0]


def main():
    t0 = time.time()
    print('building features (train/validation) ...')
    Xt_tr, Xg_tr, Xe_tr, Y_tr, nc_tr = build('train')
    Xt_va, Xg_va, Xe_va, Y_va, nc_va = build('validation')
    print(f'  done in {time.time()-t0:.1f}s; train candidates={Y_tr.size}')

    y_tr = Y_tr.reshape(-1)
    results = {}

    def run(name, build_xy):
        X_tr, X_va = build_xy()
        X_tr, X_va, mu, sd = standardize(X_tr, X_va)
        w, b = train_lr(X_tr.reshape(-1, X_tr.shape[-1]), y_tr)
        P = sigmoid(X_va.reshape(-1, X_va.shape[-1]) @ w + b)
        acc = accuracy(P, Y_va)
        hacc, hn = hard_accuracy(P, Y_va, nc_va)
        results[name] = (acc, hacc)
        print(f'{name:28s} val={acc:5.2f}  hard={hacc:5.2f} (n={hn})')
        return w, b, mu, sd, P

    # reference: pure majority vote
    Pv = majority_pick(Xg_va)
    acc = accuracy(Pv, Y_va)
    hacc, hn = hard_accuracy(Pv, Y_va, nc_va)
    results['majority-vote (ref)'] = (acc, hacc)
    print(f'{"majority-vote (ref)":28s} val={acc:5.2f}  hard={hacc:5.2f} (n={hn})')

    run('A text-only', lambda: (Xt_tr, Xt_va))
    run('B emb-only(centered)', lambda: (Xe_tr, Xe_va))
    run('C group-only', lambda: (Xg_tr, Xg_va))
    w, b, mu, sd, P_all = run('D text+group+emb',
                              lambda: (np.concatenate([Xt_tr, Xg_tr, Xe_tr], -1),
                                       np.concatenate([Xt_va, Xg_va, Xe_va], -1)))

    # E: fusion  score = prob + beta * vote_weight, beta tuned on validation
    vote = Xg_va[:, :, 0].reshape(-1)
    best = (None, -1)
    for beta in np.arange(0.0, 1.01, 0.05):
        a = accuracy(P_all + beta * vote, Y_va)
        if a > best[1]:
            best = (beta, a)
    P_fuse = P_all + best[0] * vote
    acc = accuracy(P_fuse, Y_va)
    hacc, hn = hard_accuracy(P_fuse, Y_va, nc_va)
    results[f'E fusion(beta={best[0]:.2f})'] = (acc, hacc)
    print(f'{"E fusion(beta=" + f"{best[0]:.2f})":28s} val={acc:5.2f}  hard={hacc:5.2f} (n={hn})')

    # save bundle for variant D + fusion
    RUNS.mkdir(exist_ok=True)
    bundle = RUNS / 'model_bundle'
    bundle.mkdir(exist_ok=True)
    np.savez(bundle / 'scorer.npz', w=w, b=np.array([b]), mu=mu, sd=sd,
             beta=np.array([best[0]]), n_text=np.array([N_TEXT]), n_group=np.array([N_GROUP]))
    size = (bundle / 'scorer.npz').stat().st_size
    print(f'\nsaved {bundle}/scorer.npz ({size/2**20:.3f} MiB)')
    (RUNS / 'ablation-v1.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
