# DASE7506 MP2 — Selecting Reliable LLM Reasoning

**Submission: test accuracy 81.58% (381/467), protocol `dase7506-mp2-open-hard-verifier-v5`.**

Final method (v10): a 3-member greedy-selected ensemble of MLP scorers over
text, group-comparison and group-centered embedding features, including a
deterministic arithmetic self-check. Details and full analysis are in
`mp2-report.pdf` (<= 10 pages).

## Repository layout

| Path | Purpose |
|---|---|
| `submission.py` | Final method entry point: `predict(bank, arrays, checkpoint)` |
| `sol_features.py` | Feature extraction shared by training and inference |
| `train_*.py`, `baseline_stats.py`, `exp_union.py` | Training / selection / analysis scripts (see Reproduce training) |
| `mp2-report.pdf` | Report (method, ablations, failure analysis, resources) |
| `runs/model_bundle_v10/scorer.npz` | **Final inference assets (1.56 MiB)** |
| `runs/v5_parts/`, `runs/v9_parts/` | All 23 trained ensemble members (so greedy selection can be re-verified without retraining) |
| `runs/*-score-*.json` | Official scorer outputs for every frozen version |
| `runs/ablation-*.json` | Experiment records used in the report |

## Installation

Python 3.12. This code is an overlay on the official MP2 package:

```bash
git clone https://github.com/xudongwu-0/DASE7506-MP2.git   # official data + tools
cd DASE7506-MP2/code
python -m venv .venv && source .venv/bin/activate
python -m pip install numpy==2.5.3                          # only dependency
python check_data.py                                        # verify official data
# copy this repository's *.py into code/, and runs/ into code/runs/
```

## Reproduce the submitted score (no retraining)

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle_v10 \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
# expected: accuracy = 81.5845824411135
```

Reference measurement on the development machine (Apple M5 Pro, arm64, 24 GB
RAM): prediction 0.12 s + scoring 0.01 s (limit 30 s / 4 threads).

### Asset hash

```
sha256  runs/model_bundle_v10/scorer.npz
29c01ef6ea85d55b62f483c0fe305116091d1d7f35ee3df5d7f567ce87f784bf
```

## Reproduce training (optional)

All seeds are fixed; each MLP member takes ~90 s CPU.

```bash
python train_v5.py train 0 1 2 ... 15   # v8 pool: 16 members (1044-dim features)
python train_v9.py train 0 1 2 ... 6    # v9 pool: 7 members (+ arithmetic self-check)
python exp_union.py                     # greedy forward selection over the 23 members
python train_v10.py                     # assembles runs/model_bundle_v10/scorer.npz
```

Development-history scripts (`train_scorer.py`, `train_v2.py`, `train_v3.py`,
`train_v4.py`, `train_v6.py`, `train_v7.py`) reproduce every ablation in the
report; `baseline_stats.py` reproduces the data statistics and simple baselines.

## Results summary

| Version | Method | Validation (%) | Test (%) |
|---|---|---|---|
| v0 | Majority vote (training-free) | 86.22 | 69.16 |
| v5 | 5-seed MLP ensemble | 92.98 | 77.94 |
| v8 | Greedy ensemble (pool of 16) | 93.36 | 78.37 |
| **v10** | **Mixed greedy ensemble (+ arithmetic self-check)** | **93.48** | **81.58** |

## Compliance notes

Training used only the supplied training split; validation was used for method
selection; test references were opened only by the official scorer after each
version was frozen. No external data, pretrained weights, or LLM calls at
inference. The method reads only the `bank`/`arrays` passed to `predict()` and
makes no assumption about question IDs or bank size (audit-bank safe).
Official data files are not redistributed here; keep the official
`data/LICENSE.txt` and `data/NOTICE.txt` in place.

AI/reuse disclosure: see Section 7 of `mp2-report.pdf`.
