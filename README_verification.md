# Open-set user verification baselines (B0 / B1 / B2)

Repurposes the existing Apple Watch IMU age/gender pipeline (`apw_network.py`)
into **open-set user authentication** via metric learning. Each 15 s window is
mapped to a 128-d embedding; verification asks "is this window within distance τ
of the enrolled owner's reference vector?" — **no softmax-over-N classifier**, so
impostors never seen in training can be rejected.

## Models
- **B0** classical floor (hand-crafted features + cosine / one-class SVM).
- **B1** CNN embedding (APW_Net encoder + projection head), SupCon.
- **B2** GRU embedding, SupCon.
- **M1** SE-CNN: B1 + Squeeze-and-Excitation channel attention. Trained as
  **M1a = `--model m1 --loss supcon`** and **M1b = `--model m1 --loss arcface`**
  (ArcFace = additive angular-margin head, training-only, discarded at test → still
  open-set). SE is a backward-compatible `se=True` flag on `APW_Net` (B1/age-gender
  unchanged when off).
- **M2** SE-CNN on a **slow / fast(tremor) split**: each window is separated into
  SLOW (smooth voluntary motion, via a moving-average low-pass) and FAST (the
  leftover tremor/jitter); both are stacked (28+28=56 ch) and fed to the SE-CNN, so
  the model can lean on the always-present tremor. Train with `--model m2`
  (`--loss supcon` or `arcface`).

### Intensive training (M1) — example
```bash
# M1a: SE-CNN + SupCon            # M1b: SE-CNN + ArcFace
python train_verification.py --model m1 --loss supcon  --data_dirs dataset \
    --split_file split_ids.json --cache_dir cache_verification --device cuda
python train_verification.py --model m1 --loss arcface --data_dirs dataset \
    --split_file split_ids.json --cache_dir cache_verification --device cuda
# fair SE ablation: retrain B1 under the SAME regime, only SE differs
python train_verification.py --model b1 --loss supcon  --data_dirs dataset \
    --split_file split_ids.json --cache_dir cache_verification --device cuda
# evaluate (checkpoints are named <model>_<loss>.pt)
python eval_verification.py --model m1 --checkpoint checkpoints_verification/m1_arcface.pt \
    --data_dirs dataset --cache_dir cache_verification --agg_windows 8
```
Defaults are intensive (P=32, batch 256, 500 batches/epoch, 150 epochs, IMU
augmentation on, cosine LR). They fit ~8 GB (GTX 2080); raise `--subjects_per_batch`
/ batch on a bigger GPU. Disable augmentation with `--no_augment`. Bigger P = more
(and harder) in-batch negatives. Val logs EER **and** rank-1.

**Determinism:** the per-subject enroll/verify split now uses `hashlib`, so it's
reproducible **without** needing `PYTHONHASHSEED` set.

## Files (3 new, plus your originals)
| File | Role |
|------|------|
| `verification.py` | **Everything as a library**: data layer, enroll/verify partition, activity clustering, B0/B1/B2 models, SupCon loss, embedding extraction, eval harness. Run `python verification.py selftest` for a synthetic end-to-end check. |
| `train_verification.py` | Trains B1/B2 (identity-balanced P×K SupCon batches). |
| `eval_verification.py` | Evaluation CLI for all three baselines. |
| `apw_network.py`, `train_apw_biometrics.py` | Your original age/gender code (reused/unchanged). |

## Two splits
1. **Between people (`split_ids.json`):** train / val / test are disjoint
   identities. The model is trained only on train identities; test people are
   strangers (open-set).
2. **Within each test person (enroll / verify):** see below.

## Enroll / verify partition (block-interleaved, default 80 % enroll / 20 % verify)
For each test subject the session is cut into contiguous **blocks**
(`--block_seconds`, default 120 s → ~90 blocks over 3 hr). Blocks are randomly
assigned so ~80 % are **enroll** and ~20 % are **verify**, **scattered across the
whole session** (not one contiguous chunk):
- Scattering means **both enroll and verify see the full activity mix**
  (still/active/walking) — a single contiguous 20 % slice could miss an activity.
- **Overlapping windows are kept inside runs of same-role blocks** (the 50 %
  overlap is not dropped).
- Wherever an enroll block meets a verify block there is a role boundary; every
  window within a **guard gap** (`--gap_seconds`, default 15 s) of that boundary
  is dropped. This guarantees kept enroll and verify windows are ≥ 2·gap apart →
  **zero enroll/verify raw-sample overlap** (no leakage). Combined with the
  between-people split, there's no leakage on either axis.

Tune with `--enroll_ratio`, `--block_seconds`, `--gap_seconds`.

## Activity states (unsupervised, required output)
Per verify window we compute `[log1p(mean walking_act²), log1p(var ‖userAccel‖)]`
from the 0.5–2 Hz `walking_act` band, **winsorise + z-score** them (so sensor
artefacts don't collapse the clusters), fit `KMeans(k=3)` (seeded), and map
clusters → **still / active / walking**. EER is reported overall and per state.

## Usage
```bash
# Train (defaults write to ./checkpoints_verification/<model>.pt)
python train_verification.py --model b1 --data_dirs dataset \
    --split_file split_ids.json --cache_dir cache_verification --device cuda
python train_verification.py --model b2 --data_dirs dataset \
    --split_file split_ids.json --cache_dir cache_verification --device cuda

# Evaluate (caveat + EER + FAR/FRR@EER + per-activity EER + rank-1)
python eval_verification.py --model b0 --split test \
    --data_dirs dataset --cache_dir cache_verification             # cosine AND one-class SVM
python eval_verification.py --model b1 --split test \
    --checkpoint checkpoints_verification/b1.pt \
    --data_dirs dataset --cache_dir cache_verification

# Validate the whole pipeline on synthetic data (no dataset needed)
python verification.py selftest
```

### Multi-window aggregation (optional, `--agg_windows N`)
By default a verification *attempt* is a single 15 s window (`--agg_windows 1`).
Set `--agg_windows N` to average N temporally-consecutive windows into one attempt
(~`15 + 7.5*(N-1)` s), e.g. `--agg_windows 8` ≈ a 1-minute decision. Averaging
cancels per-window noise → lower EER and higher rank-1, and mirrors real
authentication (you don't decide on a single 15 s read). Applies to enroll and
verify (and impostor) attempts identically, for all of B0/B1/B2; no retraining
needed. Report EER vs N (1, 2, 4, 8) to show accuracy improving with observation
length.
```bash
python eval_verification.py --model b1 --checkpoint checkpoints_verification/b1.pt \
    --data_dirs dataset --cache_dir cache_verification --agg_windows 8
```

**Optional POOLED aggregation** (`--pooled --pooled_n N`): additionally reports a
clearly-labelled `[POOLED N]` block with its **own EER + rank-1**, averaging N
verify windows sampled across the WHOLE session (ignores contiguity, so N is
meaningful beyond a ~2-min run). Less realistic than contiguous aggregation —
large N inflates the numbers — so read it as an upper-ish bound. Self-contained:
delete `pool_verify` (verification.py) + the pooled block (eval_verification.py)
to remove it.

> **Tip:** delete `cache_verification/` after any preprocessing change — the cache
> is keyed by subject id and won't refresh itself. Rebuilding once makes all
> models share an identical cache (and identical activity labels).

## Notes
- **Scales** to ~3 hr × 1099 subjects: per-subject disk cache, memory-mapped
  window slicing, streamed channel stats, per-subject embedding extraction.
- **Deterministic & logged:** all seeds, split sizes, partition/activity RNG.
- **Not tuned:** correct, runnable baselines at sensible defaults.
- Within-session caveat printed in every eval header.
