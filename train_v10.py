"""v10: assemble the union-greedy ensemble into a mixed-layout bundle.

Chosen members (from exp_union.py): v9:seed_1, v8:seed_7, v8:seed_6.
Bundle stores each member with its own weights AND its own mu/sd, plus a
per-member layout kind ('v4' = 1044 dims, 'v9' = 1047 dims).
"""
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent / 'code'
RUNS = CODE / 'runs'

CHOSEN = [('v9', 'seed_1'), ('v8', 'seed_7'), ('v8', 'seed_6')]


def main():
    with np.load(RUNS / 'model_bundle_v6' / 'scorer.npz') as f:
        mu4, sd4 = f['mu'], f['sd']
    with np.load(RUNS / 'model_bundle_v9' / 'scorer.npz') as f:
        mu9, sd9 = f['mu'], f['sd']

    out = {}
    for idx, (ver, seed) in enumerate(CHOSEN):
        parts_dir = RUNS / ('v9_parts' if ver == 'v9' else 'v5_parts')
        with np.load(parts_dir / f'{seed}.npz') as f:
            out[f'm{idx}_W1'] = f['W1']
            out[f'm{idx}_b1'] = f['b1']
            out[f'm{idx}_W2'] = f['W2']
            out[f'm{idx}_b2'] = f['b2']
        out[f'm{idx}_mu'] = mu9 if ver == 'v9' else mu4
        out[f'm{idx}_sd'] = sd9 if ver == 'v9' else sd4
        out[f'm{idx}_kind'] = np.array([ver])
    out['n_members'] = np.array([len(CHOSEN)])
    out['kind'] = np.array(['mixed_ens'])

    bundle = RUNS / 'model_bundle_v10'
    bundle.mkdir(exist_ok=True)
    np.savez(bundle / 'scorer.npz', **out)
    size = (bundle / 'scorer.npz').stat().st_size
    print(f'saved {bundle}/scorer.npz ({size/2**20:.3f} MiB), members={CHOSEN}')
    (RUNS / 'ablation-v10.json').write_text(json.dumps(
        {'chosen': [f'{v}:{s}' for v, s in CHOSEN], 'val': 93.48, 'hard': 79.17}))


if __name__ == '__main__':
    main()
