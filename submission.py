"""Submission interface. Implement your own selection method here.

v10 (final): mixed ensemble of 3 MLP scorers chosen by greedy forward
selection on validation from a 23-member pool spanning two feature
generations:
  v9:seed_1  (1047-dim features, incl. arithmetic self-check)
  v8:seed_7, v8:seed_6  (1044-dim features, arithmetic columns dropped)

Each member is an MLP(128) trained with a group-softmax objective on the
train split. Per-group softmax probabilities are averaged; argmax wins.

Feature pipeline: sol_features.candidate_features (v9 layout, 1047 dims);
v4 members receive the same matrix with columns 10..12 dropped, which
exactly reproduces their training layout.

Checkpoint layout (runs/model_bundle_v10/scorer.npz):
  n_members, and per member i: m{i}_W1, m{i}_b1, m{i}_W2, m{i}_b2,
  m{i}_mu, m{i}_sd, m{i}_kind in {'v4','v9'}

If --checkpoint is missing, falls back to pure majority vote.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from common import parse_answer
from sol_features import candidate_features

DROP_V9_TO_V4 = [10, 11, 12]


def _majority_choice(row):
    answers = [parse_answer(c['text']) for c in row['candidates']]
    valid = [(i, a) for i, a in enumerate(answers) if a is not None]
    if not valid:
        return 0
    counts = Counter(a for _, a in valid)
    top = max(counts.values())
    return int(next(i for i, a in valid if counts[a] == top))


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    emb = np.asarray(arrays['embeddings'], dtype=np.float32)

    if checkpoint is None:
        return [_majority_choice(row) for row in bank]

    members = []
    with np.load(Path(checkpoint) / 'scorer.npz') as f:
        n_members = int(f['n_members'][0])
        for i in range(n_members):
            members.append({
                'W1': f[f'm{i}_W1'], 'b1': f[f'm{i}_b1'],
                'W2': f[f'm{i}_W2'], 'b2': float(f[f'm{i}_b2'][0]),
                'mu': f[f'm{i}_mu'], 'sd': f[f'm{i}_sd'],
                'kind': str(f[f'm{i}_kind'][0]),
            })

    choices = []
    for qi, row in enumerate(bank):
        X9 = candidate_features(row, emb[qi])
        P = np.zeros(4)
        for m in members:
            X = X9 if m['kind'] == 'v9' else np.delete(X9, DROP_V9_TO_V4, axis=1)
            X = (X - m['mu']) / m['sd']
            z = np.maximum(X @ m['W1'] + m['b1'], 0) @ m['W2'] + m['b2']
            z = z - z.max()
            e = np.exp(z)
            P += e / e.sum()
        choices.append(int(np.argmax(P)))
    return choices
