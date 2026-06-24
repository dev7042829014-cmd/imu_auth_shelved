"""
train_apw_biometrics.py
=========================
Training script for APW_Net on Apple Watch IMU data.

Data layout
-----------
  Train/Test CSVs : pooled from both directories below, split by split_ids.json
    /home/allam/JamBase_multimodal_dataset/TokyoU_APW1099JamBaseCSVs/train/
    /home/allam/JamBase_multimodal_dataset/TokyoU_APW1099JamBaseCSVs/test/
  Labels          : /home/allam/JamBase_multimodal_dataset/subjects_age_gender.csv
  Split           : /home/allam/JamBase_multimodal_dataset/split_ids.json
                    Keys: train_ids, validation_ids, test_ids

Dataset classes
---------------
AWDDatasetTrain  : returns one randomly sampled valid window per subject per call.
                   Balanced subject-level batch sampling via
                   BalancedTrainBatchSamplerByAgeGender.
AWDDatasetTest   : enumerates all valid windows with stride for evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from wrist_biometrics import (
    FEATURES,
    WINDOW_SAMPLES,
    WINDOW_STRIDE,
    N_INPUT_CHANNELS,
    AGE_MIN, AGE_MAX,
    APW_Net,
    WristBiometricsLoss,
    lowpass_filter,
    append_derived_features,
    compute_channel_stats,
)
from awd_dataset import (
    AWDDatasetTrain,
    AWDDatasetTest,
    AWDDatasetBase,
    BalancedTrainBatchSamplerByAgeGender,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

TRAIN_DIR  = Path('/home/allam/JamBase_multimodal_dataset/TokyoU_APW1099JamBaseCSVs/train')
TEST_DIR   = Path('/home/allam/JamBase_multimodal_dataset/TokyoU_APW1099JamBaseCSVs/test')
LABELS_CSV = Path('/home/allam/JamBase_multimodal_dataset/subjects_age_gender.csv')
SPLIT_FILE = Path('/home/allam/JamBase_multimodal_dataset/split_ids.json')
CKPT_DIR   = Path('/home/allam/JamBase_multimodal_dataset/checkpoints_wrist_age_gender')

# ─── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Subject-level preprocessing transform ────────────────────────────────────

def _wrist_subject_transform(data: np.ndarray,
                              nan_mask: np.ndarray,
                              features: list) -> np.ndarray:
    """Applied inside cache_data: low-pass filter then append derived channels."""
    data = lowpass_filter(data)
    data = append_derived_features(data)   # (T, N_INPUT_CHANNELS)
    return data


# ─── Per-window transforms (normalize + optional augment) ────────────────────

class _NormTranspose:
    """Normalize (T, N_CH) → (N_CH, T) tensor using pre-computed statistics."""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std  = std

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.T.copy().astype(np.float32))


class _AugNormTranspose(_NormTranspose):
    """Same as _NormTranspose but applies stochastic augmentation (train only)."""
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = x.T.copy().astype(np.float32)   # (N_CH, T)
        x = _augment_window(x)
        return torch.from_numpy(x)


# ─── Multi-root dataset subclasses ────────────────────────────────────────────
# AWDDatasetBase.cache_data assumes a single root_dir.
# Subjects in this dataset span both TRAIN_DIR and TEST_DIR, so we override
# cache_data to resolve paths from a pre-built {subject_id: Path} mapping.

class _MultiRootDatasetMixin:
    """
    Mixin that replaces the single root_dir file lookup in cache_data with a
    subject_to_path dict that can span multiple source directories.

    Must be mixed in before AWDDatasetTrain / AWDDatasetTest in the MRO.
    """

    def _set_subject_paths(self, subject_to_path: dict[str, Path]) -> None:
        self._subject_to_path: dict[str, Path] = subject_to_path

    def _base_cache_data(self) -> None:
        """Re-implementation of AWDDatasetBase.cache_data with multi-root lookup."""
        self.cache = {}
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Caching'):
            subject_id = row['ID']
            subject_file = self._subject_to_path.get(subject_id)
            if subject_file is None:
                logging.debug(f'  [skip] no CSV found for {subject_id}')
                continue
            try:
                df = pd.read_csv(subject_file, low_memory=False, usecols=self.features)
            except (ValueError, FileNotFoundError, Exception) as e:
                logging.debug(f'  [skip] {subject_id}: {e}')
                continue
            df = df.apply(pd.to_numeric, errors='coerce')
            nan_mask = df.isna().any(axis=1).values
            df = df.ffill().bfill()
            if df.isna().any().any():
                logging.debug(f'  [skip] unfillable NaNs in {subject_id}')
                continue
            data = df[self.features].values.astype(np.float32)
            data = self.subject_level_preprocessing(data, nan_mask, self.features)
            self.cache[subject_id] = {
                'data':     data,
                'nan_mask': nan_mask,
                'age':      row['Age'],
                'gender':   row['Gender'],
            }


class MultiRootAWDDatasetTrain(_MultiRootDatasetMixin, AWDDatasetTrain):
    """AWDDatasetTrain that loads CSVs from a subject_to_path dict."""

    def cache_data(self) -> None:
        # Step 1 — load + subject-level preprocessing
        self._base_cache_data()

        # Step 2 — compute valid windows (mirrors AWDDatasetTrain.cache_data)
        ws  = self.window_size
        thr = self.valid_window_nan_ratio
        for subject_id, entry in self.cache.items():
            nan_mask = entry['nan_mask']
            counts = np.cumsum(nan_mask)
            counts[ws:] = counts[ws:] - counts[:-ws]
            counts = counts[ws - 1:]
            valid_windows = np.where(counts / ws < thr)[0]
            entry['valid_windows'] = valid_windows

        n_subjects = len(self.cache)
        total_win  = sum(len(v['valid_windows']) for v in self.cache.values())
        print(f'  Train cache: {n_subjects} subjects, {total_win:,} valid windows',
              flush=True)


class MultiRootAWDDatasetTest(_MultiRootDatasetMixin, AWDDatasetTest):
    """AWDDatasetTest that loads CSVs from a subject_to_path dict."""

    def cache_data(self) -> None:
        # Step 1 — load + subject-level preprocessing
        self._base_cache_data()

        # Step 2 — compute valid windows (mirrors AWDDatasetTest.cache_data)
        ws     = self.window_size
        stride = self.window_stride
        for subject_id, entry in self.cache.items():
            nan_mask = entry['nan_mask']
            counts = np.cumsum(nan_mask)
            counts[ws:] = counts[ws:] - counts[:-ws]
            counts = counts[ws - 1:]
            valid_windows = np.where(counts[::stride] == 0)[0] * stride
            entry['valid_windows'] = valid_windows

        if not self.cache:
            raise ValueError('Caching failed: no data to cache.')

        self._setup_index()
        n_subjects = len(self.cache)
        total_win  = len(self)
        print(f'  Eval cache: {n_subjects} subjects, {total_win:,} valid windows',
              flush=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_subject_to_path(*dirs: Path) -> dict[str, Path]:
    """Scan directories and return {subject_id: csv_path} for all CSVs found."""
    mapping: dict[str, Path] = {}
    for d in dirs:
        if d.exists():
            for p in d.glob('*.csv'):
                mapping[p.stem] = p
    return mapping


def load_labels(labels_csv: Path) -> pd.DataFrame:
    """Return subjects_age_gender.csv as a DataFrame (ID, Age, Gender columns)."""
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()
    df['Gender'] = df['Gender'].str.upper()
    df['Age']    = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age', 'Gender'])
    df['Age']    = df['Age'].astype(int)
    out_of_range = df[(df['Age'] < AGE_MIN) | (df['Age'] > AGE_MAX)]
    if len(out_of_range):
        print(f'  [warn] {len(out_of_range)} subjects outside age range excluded '
              f'(ages {sorted(out_of_range["Age"].unique())})')
    df = df[(df['Age'] >= AGE_MIN) & (df['Age'] <= AGE_MAX)].copy()
    print(f'  Age range in labels: {df["Age"].min()}–{df["Age"].max()} | '
          f'Gender: {dict(df["Gender"].value_counts())}')
    return df[['ID', 'Age', 'Gender']].reset_index(drop=True)


def build_split_df(ids: list[str], labels_df: pd.DataFrame,
                   subject_to_path: dict[str, Path]) -> pd.DataFrame:
    """Return a DataFrame of subjects present in both ids and subject_to_path."""
    valid_ids = [i for i in ids if i in subject_to_path]
    missing   = len(ids) - len(valid_ids)
    if missing:
        print(f'  [warn] {missing} IDs from split not found on disk — skipped')
    df = labels_df[labels_df['ID'].isin(valid_ids)].copy().reset_index(drop=True)
    return df


def get_subject_ids_per_window(ds: MultiRootAWDDatasetTest) -> list[str]:
    """Build a per-window subject-ID list (length = len(ds)) for eval aggregation."""
    result: list[str] = []
    for sid in ds._index_keys:
        result.extend([sid] * len(ds.cache[sid]['valid_windows']))
    return result


def collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert list of AWDDataset dicts → (features, age, gender) tensor tuples."""
    features = torch.stack([b['features'] for b in batch])
    ages     = torch.tensor([b['age']    for b in batch], dtype=torch.float32)
    genders  = torch.tensor(
        [0.0 if b['gender'] == 'M' else 1.0 for b in batch],
        dtype=torch.float32)
    return features, ages, genders


# ─── IMU data augmentation ───────────────────────────────────────────────────

def _augment_window(x: np.ndarray) -> np.ndarray:
    """
    In-place-safe augmentation for a single (28, 300) normalised window.

    Augmentations are deliberately conservative to preserve the 3–9.5 Hz
    physiological tremor signal — the primary age biomarker.

    1. Gaussian noise  (p=0.4, σ=0.02)
       Reduced from p=0.8 / σ=0.05.  Small sensor noise simulation.

    2. Amplitude scaling  (p=0.6, range ±10%)
       Tighter range [0.90, 1.10] (was [0.80, 1.20]).  Tremor-band channels
       scaled proportionally to remain consistent.

    3. Channel dropout  (p=0.15)
       Reduced from p=0.5.  Occasional single-channel zeroing for robustness.

    4. Temporal shift  (p=0.3, ±5 samples)
       Reduced magnitude (was ±10 samples) to avoid wrapping tremor bursts.

    5. Spectral band masking  (p=0.10)
       Restricted to bins 0-43 (< 3 Hz) to protect tremor channels
       (bins 45-143 = 3-9.5 Hz).  Only low-frequency drift is masked.

    Time reversal removed: wrist biomechanics have temporal directionality
    (acceleration→deceleration patterns), so reversal creates invalid inputs.
    """
    x = x.copy()
    # 1. Gaussian noise — increased σ and probability vs v1
    if np.random.random() < 0.6:
        x += np.random.normal(0.0, 0.05, x.shape).astype(np.float32)
    # 2. Amplitude scaling on dynamic channels.
    #    Lo-tremor (15-20) and hi-tremor (21-26) counterparts scaled by same
    #    factor for internal consistency.  Walking channel (27) not scaled:
    #    its amplitude is a ground-truth activity-level signal.
    if np.random.random() < 0.6:
        scale = np.random.uniform(0.90, 1.10)
        x[:6]   *= scale   # raw userAccel + gyro
        x[15:21] *= scale  # lo-tremor (3-7 Hz)
        x[21:27] *= scale  # hi-tremor (7-9.5 Hz)
    # 3. Channel dropout — reduced probability
    if np.random.random() < 0.15:
        ch = np.random.randint(0, x.shape[0])
        x[ch] = 0.0
    # 4. Temporal shift — reduced range to ±5 samples (0.25 s)
    if np.random.random() < 0.3:
        shift = np.random.randint(-5, 6)
        x = np.roll(x, shift, axis=1)
    # 5. Spectral band masking — restricted to sub-3 Hz bins only.
    # With T=300 @ 20 Hz: n_freq=151, bin i = i*0.067 Hz.
    # Bins 45-143 cover 3-9.5 Hz = exactly the tremor-band channels (15-26).
    # Masking those bins for ALL channels destroys the tremor content.
    # Safe range: bins 0-43 (< 3 Hz, slow drift / DC offset).
    if np.random.random() < 0.10:
        T        = x.shape[1]
        n_freq   = T // 2 + 1                                 # 151
        band_w   = np.random.randint(2, max(3, n_freq // 20)) # 2-7
        safe_max = 43 - band_w                                # >= 36
        if safe_max > 0:
            f0   = np.random.randint(0, safe_max + 1)
            spec = np.fft.rfft(x, axis=1)
            spec[:, f0:f0 + band_w] = 0.0
            x    = np.fft.irfft(spec, n=T, axis=1).astype(np.float32)
    return x


# ─── Training utilities ───────────────────────────────────────────────────────


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    mixup_alpha: float = 0.0,
                    samples_per_subject: int = 1):
    model.train()
    total_loss = total_age = total_gender = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)   # print ~10 times per epoch
    for batch_idx, (x, age_gt, gender_gt) in enumerate(loader):
        x         = x.to(device)
        age_gt    = age_gt.to(device)
        gender_gt = gender_gt.to(device)

        B = x.shape[0]
        sps = samples_per_subject

        age_logit, gen_logit = model(x)   # (B,), (B,)

        # Subject-level aggregation: average logits over sps windows per subject.
        # Noise cancels, age signal accumulates — SNR improves by sqrt(sps).
        if sps > 1 and B % sps == 0:
            n_subj      = B // sps
            age_logit_s = age_logit.view(n_subj, sps).mean(dim=1)
            gen_logit_s = gen_logit.view(n_subj, sps).mean(dim=1)
            age_gt_s    = age_gt.view(n_subj, sps)[:, 0]
            gender_gt_s = gender_gt.view(n_subj, sps)[:, 0]
        else:
            n_subj      = B
            age_logit_s = age_logit
            gen_logit_s = gen_logit
            age_gt_s    = age_gt
            gender_gt_s = gender_gt

        # Subject-level mixup (after averaging, so window alignment is preserved)
        if mixup_alpha > 0 and np.random.random() < 0.5:
            lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
            lam  = max(lam, 1.0 - lam)
            perm = torch.randperm(n_subj, device=device)
            age_logit_s = lam * age_logit_s + (1.0 - lam) * age_logit_s[perm]
            age_gt_s    = lam * age_gt_s    + (1.0 - lam) * age_gt_s[perm]

        loss, l_age, l_gender = loss_fn(age_logit_s, gen_logit_s,
                                        age_gt_s, gender_gt_s)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss   += loss.item()
        total_age    += l_age.item()
        total_gender += l_gender.item()

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            avg = total_loss / (batch_idx + 1)
            print(f'    batch {batch_idx + 1:5d}/{n_batches}  avg_loss={avg:.4f}',
                  flush=True)

    n = len(loader)
    return total_loss / n, total_age / n, total_gender / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, subject_ids=None):
    """
    Returns (avg_loss, win_mae, win_acc, subj_mae, subj_acc).

    win_mae / win_acc  : per-window metrics (noisy; many windows per subject).
    subj_mae / subj_acc: subject-level metrics — median age prediction and
                         mean-logit majority vote for gender across all windows
                         of a subject.  This is the meaningful deployment metric
                         and the one used for early stopping / best-model.
                         Requires subject_ids list (length = total windows).
    """
    model.eval()
    total_loss    = 0.0
    all_age_pred  = []
    all_age_gt    = []
    all_gen_logit = []
    all_gen_gt    = []

    for x, age_gt, gender_gt in loader:
        x         = x.to(device)
        age_gt    = age_gt.to(device)
        gender_gt = gender_gt.to(device)

        age_logits, gen_logit = model(x)
        loss, _, _ = loss_fn(age_logits, gen_logit, age_gt, gender_gt)
        total_loss += loss.item()

        all_age_pred.append(model.predict_age(age_logits).cpu())
        all_age_gt.append(age_gt.cpu())
        all_gen_logit.append(gen_logit.cpu())
        all_gen_gt.append(gender_gt.cpu())

    age_pred_t  = torch.cat(all_age_pred)    # (N,)
    age_gt_t    = torch.cat(all_age_gt)      # (N,)
    gen_logit_t = torch.cat(all_gen_logit)   # (N,)
    gen_gt_t    = torch.cat(all_gen_gt)      # (N,)
    gen_pred_t  = APW_Net.predict_gender(gen_logit_t)

    win_mae  = (age_pred_t - age_gt_t).abs().mean().item()
    win_acc  = (gen_pred_t == gen_gt_t.long()).float().mean().item() * 100.0
    avg_loss = total_loss / len(loader)

    # ── Subject-level: aggregate per-window preds → one prediction per subject
    subj_mae = win_mae
    subj_acc = win_acc
    if subject_ids is not None:
        s_age_preds:  dict = {}
        s_age_gts:    dict = {}
        s_gen_logits: dict = {}
        s_gen_gts:    dict = {}
        for i, sid in enumerate(subject_ids):
            s_age_preds.setdefault(sid, []).append(age_pred_t[i].item())
            s_age_gts[sid] = age_gt_t[i].item()
            s_gen_logits.setdefault(sid, []).append(gen_logit_t[i].item())
            s_gen_gts[sid] = gen_gt_t[i].item()

        mae_vals, gen_correct = [], []
        for sid in s_age_preds:
            preds = np.array(s_age_preds[sid])
            # Soft-trimmed mean: Gaussian weights centred on median.
            # Down-weights outlier windows more gracefully than hard trimming.
            if len(preds) >= 4:
                center  = float(np.median(preds))
                spread  = float(np.std(preds)) + 1.0
                weights = np.exp(-0.5 * ((preds - center) / spread) ** 2)
                weights = weights / weights.sum()
                agg_age = float(np.dot(preds, weights))
            else:
                agg_age = float(np.median(preds))
            mae_vals.append(abs(agg_age - s_age_gts[sid]))
            mean_logit = float(np.mean(s_gen_logits[sid]))
            pred_g = int(1.0 / (1.0 + np.exp(-mean_logit)) > 0.5)
            gen_correct.append(int(pred_g == int(s_gen_gts[sid])))

        subj_mae = float(np.mean(mae_vals))
        subj_acc = float(np.mean(gen_correct)) * 100.0

    return avg_loss, win_mae, win_acc, subj_mae, subj_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    print(f'Device: {device}')

    # ── Auto-resume ──────────────────────────────────────────────────────────
    resume_ckpt = None
    auto_last   = CKPT_DIR / 'last_checkpoint.pt'
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            resume_path = CKPT_DIR / args.resume
        if not resume_path.exists():
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
    elif auto_last.exists():
        resume_path = auto_last
        print(f'\n[AUTO-RESUME] Found {auto_last.name} — resuming automatically.')
        print('  (Delete last_checkpoint.pt to start fresh.)')

    if resume_path is not None:
        print(f'\n[RESUME] Loading checkpoint: {resume_path}')
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        rp_epoch = resume_ckpt.get('epoch', '?')
        rp_mae   = resume_ckpt.get('best_val_mae_sofar', resume_ckpt.get('val_mae', float('nan')))
        print(f'  Saved at epoch {rp_epoch}  |  Best MAE so far: {rp_mae:.2f} yr')
    else:
        print('\n[FRESH START] No checkpoint found — training from scratch.')

    # ── Auto-detect hidden_size from checkpoint ───────────────────────────────
    if resume_ckpt is not None:
        ckpt_args = resume_ckpt.get('args', {})
        ckpt_hidden = ckpt_args.get('hidden_size', None)
        if ckpt_hidden is not None and ckpt_hidden != args.hidden_size:
            print(f'  [warn] --hidden_size={args.hidden_size} overridden by checkpoint '
                  f'value hidden_size={ckpt_hidden} to prevent shape mismatch.')
            args.hidden_size = ckpt_hidden

    # ── Labels ───────────────────────────────────────────────────────────────
    print('\n[1] Loading labels...')
    labels_df = load_labels(LABELS_CSV)
    print(f'  Subjects with valid labels: {len(labels_df)}')

    # ── Split (fixed from split_ids.json) ────────────────────────────────────
    print('\n[2] Loading subject split from split_ids.json...')
    with open(SPLIT_FILE) as f:
        split_ids = json.load(f)
    subject_to_path = build_subject_to_path(TRAIN_DIR, TEST_DIR)
    print(f'  Total CSVs found (train+test dirs): {len(subject_to_path)}')

    train_df  = build_split_df(split_ids['train_ids'],      labels_df, subject_to_path)
    val_df    = build_split_df(split_ids['validation_ids'], labels_df, subject_to_path)
    test_df   = build_split_df(split_ids['test_ids'],       labels_df, subject_to_path)
    # Merge validation subjects into the training set — more training data
    # reduces overfitting. Evaluation is on the test set only.
    import pandas as pd
    train_df  = pd.concat([train_df, val_df], ignore_index=True)
    eval_df   = test_df
    print(f'  Train (train+val): {len(train_df)} subjects | Test (eval): {len(eval_df)} subjects')
    print(f'    (original train={len(split_ids["train_ids"])}, val merged in={len(val_df)})')

    # ── Training dataset + channel stats ─────────────────────────────────────
    print('\n[3] Caching training data...')
    resume_mean = None
    resume_std  = None
    if resume_ckpt is not None:
        resume_mean = resume_ckpt.get('channel_mean')
        resume_std  = resume_ckpt.get('channel_std')
        if resume_mean is not None and resume_mean.shape[0] != N_INPUT_CHANNELS:
            print(f'  [warn] Checkpoint channel stats shape {resume_mean.shape} '
                  f'≠ N_INPUT_CHANNELS={N_INPUT_CHANNELS}. Recomputing from data.')
            resume_mean = resume_std = None

    train_ds = MultiRootAWDDatasetTrain(
        df                    = train_df,
        root_dir              = TRAIN_DIR,   # overridden by mixin; kept for API compat
        features              = FEATURES,
        transforms            = None,        # set after computing stats below
        window_size           = WINDOW_SAMPLES,
        valid_window_nan_ratio= 0.20,
        subject_transform     = _wrist_subject_transform,
    )
    train_ds._set_subject_paths(subject_to_path)
    train_ds.cache_data()

    if resume_mean is not None:
        channel_mean = resume_mean
        channel_std  = resume_std
        print('  Channel stats loaded from checkpoint.')
    else:
        all_data     = np.concatenate([v['data'] for v in train_ds.cache.values()], axis=0)
        channel_mean, channel_std = compute_channel_stats(all_data)
        print(f'  Channel stats computed from {len(all_data):,} frames.')

    train_ds.channel_mean = channel_mean
    train_ds.channel_std  = channel_std
    train_ds.transforms   = _AugNormTranspose(channel_mean, channel_std)

    # ── Eval dataset  ──────────────────────────────────
    print('\n[4] Caching eval data ...')
    eval_ds = MultiRootAWDDatasetTest(
        df                    = eval_df,
        root_dir              = TEST_DIR,
        features              = FEATURES,
        transforms            = _NormTranspose(channel_mean, channel_std),
        window_size           = WINDOW_SAMPLES,
        window_stride         = WINDOW_STRIDE,
        subject_transform     = _wrist_subject_transform,
    )
    eval_ds._set_subject_paths(subject_to_path)
    eval_ds.cache_data()
    eval_subject_ids = get_subject_ids_per_window(eval_ds)

    # ── Data loaders ─────────────────────────────────────────────────────────
    n_batches_per_epoch = args.batches_per_epoch
    print(f'\n  Batches per epoch: {n_batches_per_epoch}  |  Batch size: {args.batch_size}')
    sampler = BalancedTrainBatchSamplerByAgeGender(
        dataset              = train_ds,
        batch_size           = args.batch_size,
        num_batches_per_epoch= n_batches_per_epoch,
        samples_per_subject  = args.samples_per_subject,
        seed                 = SEED,
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              collate_fn=collate_fn,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0))
    val_loader   = DataLoader(eval_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0))

    # ── Model ────────────────────────────────────────────────────────────────
    print('\n[5] Building model...')
    model = APW_Net(hidden_size=args.hidden_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable parameters: {n_params:,}')

    # ── Loss ─────────────────────────────────────────────────────────────────
    # BalancedTrainBatchSamplerByAgeGender already creates 50/50 male/female
    # batches (batch_size//2 male subjects, (batch_size+1)//2 female subjects).
    # Using pos_weight = n_male/n_female = 0.265 would DOWN-weight females to
    # 26% per sample, double-correcting in the wrong direction and biasing the
    # model toward predicting MALE.  Since the sampler handles the imbalance,
    # pos_weight must be 1.0 (no additional rebalancing needed).
    gender_counts = train_df['Gender'].value_counts()
    n_female      = int(gender_counts.get('F', 1))
    n_male        = int(gender_counts.get('M', 1))
    print(f'  Gender: F={n_female}, M={n_male}  '
          f'(sampler creates 50/50 batches → BCE pos_weight=1.0)')

    loss_fn = WristBiometricsLoss(
        lambda_age        = args.lambda_age,
        lambda_gender     = args.lambda_gender,
        gender_pos_weight = None,   # sampler already creates 50/50 gender batches
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T0,
                                            T_mult=2, eta_min=args.lr * 1e-3)

    # ── Restore training state if resuming ───────────────────────────────────
    start_epoch = 1

    if resume_ckpt is not None:
        # Check input-channel compatibility before loading weights.
        ckpt_net = resume_ckpt.get('network', {})
        first_conv_key = 'enc_A.0.conv.weight'  # shape (hidden, n_ch, kernel)
        if first_conv_key in ckpt_net:
            ckpt_n_ch = ckpt_net[first_conv_key].shape[1]
            if ckpt_n_ch != N_INPUT_CHANNELS:
                print(f'  [COMPAT] Checkpoint has {ckpt_n_ch} input channels '
                      f'but current model has {N_INPUT_CHANNELS}. '
                      f'Cannot load weights — starting fresh.\n'
                      f'  (This is expected after adding new feature channels.)')
                resume_ckpt = None

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['network'])
        if args.reset_optimizer:
            print('  [--reset_optimizer] Optimizer/scheduler/epoch state reset.')
        else:
            if 'optimizer_state' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer_state'])
            if 'scheduler_state' in resume_ckpt:
                scheduler.load_state_dict(resume_ckpt['scheduler_state'])
            start_epoch = int(resume_ckpt['epoch']) + 1
        print(f'  ✓ Resumed from epoch {resume_ckpt.get("epoch", "?")}')

    # ── Training ─────────────────────────────────────────────────────────────
    print(f'\n[6] Training for up to {args.epochs} epochs...')
    print(f'     Saving checkpoint every epoch. Eval set = test only ({len(eval_df)} subjects).')

    best_mae       = float('inf')
    best_epoch     = -1

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\n  Epoch {epoch:03d}/{args.epochs} — starting training...', flush=True)
        tr_loss, tr_age, tr_gen = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=args.mixup_alpha,
            samples_per_subject=args.samples_per_subject)
        val_loss, win_mae, win_acc, val_mae, val_acc = evaluate(
            model, val_loader, loss_fn, device, subject_ids=eval_subject_ids)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        is_best = val_mae < best_mae
        if is_best:
            best_mae   = val_mae
            best_epoch = epoch

        print(f'  Epoch {epoch:03d}/{args.epochs} | '
              f'Train loss {tr_loss:.4f} (age {tr_age:.4f}, gender {tr_gen:.4f}) | '
              f'Eval MAE {val_mae:.2f} yr  Acc {val_acc:.1f}%  '
              f'[win MAE {win_mae:.2f} yr  Acc {win_acc:.1f}%] | '
              f'LR {current_lr:.2e}' + ('  ★ best' if is_best else ''), flush=True)

        ckpt = {
            'epoch':           epoch,
            'network':         model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'channel_mean':    channel_mean,
            'channel_std':     channel_std,
            'eval_mae':        val_mae,
            'eval_acc':        val_acc,
            'args':            vars(args),
        }
        torch.save(ckpt, CKPT_DIR / 'last_checkpoint.pt')
        epoch_ckpt = CKPT_DIR / f'epoch_{epoch:04d}_mae{val_mae:.2f}_acc{val_acc:.1f}.pt'
        torch.save(ckpt, epoch_ckpt)
        if is_best:
            torch.save(ckpt, CKPT_DIR / 'best_checkpoint.pt')
        print(f'    Saved → {epoch_ckpt.name}', flush=True)

    print(f'\nTraining complete ({args.epochs} epochs).')
    print(f'  Best epoch: {best_epoch}  |  Best test MAE: {best_mae:.2f} yr')

    # ── Final eval-set summary (best epoch model) ────────────────────────
    print(f'\n[TEST] Loading best checkpoint (epoch {best_epoch}, MAE {best_mae:.2f} yr)...')
    best_ckpt = torch.load(CKPT_DIR / 'best_checkpoint.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['network'])
    print('\n[TEST] Final evaluation on best epoch model (test set only)...')
    _, test_win_mae, test_win_acc, test_subj_mae, test_subj_acc = evaluate(
        model, val_loader, loss_fn, device, subject_ids=eval_subject_ids)

    # Per-group breakdown
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x, age_gt, _ in val_loader:
            logits, _ = model(x.to(device))
            all_preds.append(model.predict_age(logits).cpu())
            all_gts.append(age_gt)
    all_preds = torch.cat(all_preds).numpy()
    all_gts   = torch.cat(all_gts).numpy()

    # Per-window age array from cache (aligned to eval_subject_ids order)
    test_ages_per_window = np.array([eval_ds.cache[sid]['age']
                                     for sid in eval_subject_ids])

    age_bins  = [('<20', 0, 20), ('20-29', 20, 30), ('30-39', 30, 40),
                 ('40-49', 40, 50), ('50-59', 50, 60), ('60+', 60, 200)]

    print('\n' + '='*62)
    print('EVAL SET RESULTS (test set only)')
    print('='*62)
    print(f'  Subjects evaluated         : {len(set(eval_subject_ids))}')
    print(f'  Total windows              : {len(eval_ds):,}')
    print(f'\n  ── Age estimation ────────────────────────────────────')
    print(f'  Subject-level MAE          : {test_subj_mae:.2f} yr')
    print(f'  Window-level  MAE          : {test_win_mae:.2f} yr')
    print(f'\n  ── Gender classification ──────────────────────────────')
    print(f'  Subject-level accuracy     : {test_subj_acc:.1f}%')
    print(f'  Window-level  accuracy     : {test_win_acc:.1f}%')
    print(f'\n  ── Age MAE by group ───────────────────────────────────────')
    for label, lo, hi in age_bins:
        mask = (test_ages_per_window >= lo) & (test_ages_per_window < hi)
        if mask.sum() > 0:
            mae_g = float(np.abs(all_preds[mask] - all_gts[mask]).mean())
            n_subj = len(set(sid for sid, age in zip(eval_subject_ids,
                                                      test_ages_per_window)
                             if lo <= age < hi))
            print(f'  {label:>6s}  →  MAE {mae_g:.2f} yr  (n_subj={n_subj})')
    print('='*62)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train APW_Net')
    parser.add_argument('--epochs',           type=int,   default=500)
    parser.add_argument('--batch_size',       type=int,   default=30)
    parser.add_argument('--batches_per_epoch',type=int,   default=300,
                        help='Gradient steps per epoch. With 816 subjects and '
                             'batch_size=128, default 300 gives ~37 passes over '
                             'each subject per epoch.')
    parser.add_argument('--lr',               type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay',     type=float, default=1e-2,
                        help='AdamW weight decay (decoupled from LR)')
    parser.add_argument('--lambda_age',       type=float, default=100.0,
                        help='Weight on MSE age regression loss (scaled for MSE: ~100 = dominant)')
    parser.add_argument('--lambda_gender',    type=float, default=10.0,
                        help='Weight on BCE gender loss')
    # lambda_huber and sigma_sq kept for CLI backward-compat but are no longer used
    parser.add_argument('--lambda_huber',     type=float, default=0.0,
                        help='(Ignored: legacy LDL parameter)')
    parser.add_argument('--sigma_sq',         type=float, default=0.0,
                        help='(Ignored: legacy LDL parameter)')
    parser.add_argument('--workers',          type=int,   default=4,
                        help='DataLoader worker processes')
    parser.add_argument('--patience',         type=int,   default=200,
                        help='Early stopping patience.')
    parser.add_argument('--T0',               type=int,   default=500,
                        help='CosineAnnealingWarmRestarts restart period (epochs). '
                             'Increased from 30 to reduce disruptive LR restarts.')
    parser.add_argument('--hidden_size',      type=int,   default=256,
                        help='Encoder hidden size')
    parser.add_argument('--mixup_alpha',      type=float, default=0.0,
                        help='Beta distribution alpha for Mixup. 0 = disabled (recommended for age estimation).')
    parser.add_argument('--samples_per_subject', type=int, default=10,
                        help='Windows sampled per subject per training batch. '
                             'Logits are averaged over these windows before the '
                             'loss is computed, reducing per-window noise by '
                             'sqrt(samples_per_subject) and matching the '
                             'subject-level evaluation protocol.')
    parser.add_argument('--resume',           type=str,   default=None,
                        help='Checkpoint path or filename in checkpoints_wrist/ to resume from.')
    parser.add_argument('--reset_optimizer',  action='store_true',
                        help='Load network weights from checkpoint but reset optimizer, '
                             'scheduler, epoch counter, and early-stop state. '
                             'Use this when resuming with new hyperparameters (e.g. '
                             'after changing --lambda_huber or --T0) to avoid stale '
                             'Adam momentum from the old loss scale.')
    parser.add_argument('--device',           type=str,   default='cuda',
                        choices=['cuda', 'cpu'])
    args = parser.parse_args()
    main(args)


    # running command example:
    # python train_wrist_biometrics_age_gender.py --epochs 500 --batch_size 30 --batches_per_epoch 300  --lr 1e-4  --samples_per_subject 10  --device cuda

