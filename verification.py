"""
verification.py
===============
Open-set user verification from Apple Watch IMU — the whole library in one file.

This repurposes the existing age/gender pipeline (apw_network.py) into user
AUTHENTICATION via metric learning: a 15 s window is mapped to a 128-d embedding
where same-person windows are close and different-person windows are far, so a
new window is verified by checking it is within distance tau of the enrolled
owner's reference vector. There is deliberately NO softmax-over-N-users
classifier, so impostors never seen in training can be rejected.

Contents (search for the ===== banners)
---------------------------------------
  1. Channels & constants        — channel names, magnetometer guard
  2. Splits & file discovery      — read split_ids.json, find CSVs
  3. Enroll / verify partition    — contiguous regions, random verify, guard gap
  4. Activity states              — unsupervised still / active / walking
  5. Data layer (VerificationData)— cache, channel stats, windows  (model-agnostic)
  6. B0 classical features        — hand-crafted stats + band powers
  7. Models                       — B1 CNN, B2 GRU, SupCon loss
  8. Embedding extraction         — build the per-subject embedding dict
  9. Evaluation harness           — EER, FAR/FRR, per-activity EER, rank-1, report
 10. Self-test                    — synthetic end-to-end check (python verification.py selftest)

Used by train_verification.py (trains B1/B2) and eval_verification.py (scores all).
Everything stochastic is seeded and logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from apw_network import (
    FEATURES,
    WINDOW_SAMPLES,
    WINDOW_STRIDE,
    N_INPUT_CHANNELS,
    SAMPLING_RATE,
    lowpass_filter,
    append_derived_features,
)

log = logging.getLogger("verification")

EMBED_DIM_DEFAULT = 128
# ACTIVITY_STATES = ("still", "active", "walking")


# ============================================================================
# 1. CHANNELS & CONSTANTS
# ============================================================================
DERIVED_NAMES: List[str] = [
    "accel_mag", "gyro_mag", "jerk_mag", "ang_jerk_mag",          # ch 11-14
    "tremor_lo_ua_x", "tremor_lo_ua_y", "tremor_lo_ua_z",          # ch 15-17
    "tremor_lo_gyro_x", "tremor_lo_gyro_y", "tremor_lo_gyro_z",    # ch 18-20
    "tremor_hi_ua_x", "tremor_hi_ua_y", "tremor_hi_ua_z",          # ch 21-23
    "tremor_hi_gyro_x", "tremor_hi_gyro_y", "tremor_hi_gyro_z",    # ch 24-26
    "walking_act",                                                  # ch 27
]
ALL_CHANNEL_NAMES: List[str] = list(FEATURES) + DERIVED_NAMES
assert len(ALL_CHANNEL_NAMES) == N_INPUT_CHANNELS, (
    "channel-name table out of sync with apw_network.append_derived_features")

# Privacy guard: the magnetometer / heading / location columns can leak location
_BANNED = ("magnetic", "heading", "location", "latitude", "longitude", "yaw")
for _name in ALL_CHANNEL_NAMES:
    assert not any(b in _name.lower() for b in _BANNED), f"banned channel: {_name}"


def channel_index(name: str) -> int:
    """Resolve a channel NAME to its index in the 28-channel derived array."""
    return ALL_CHANNEL_NAMES.index(name)


# Default dataset locations 
# can override on CLI
DEFAULT_DATA_DIRS = (
    "/media/sharma/CE9C1E919C1E7465/APW_data/baseline models/dataset"
)


# ============================================================================
# 2. SPLITS & FILE DISCOVERY
# ============================================================================

def build_subject_to_path(*dirs) -> Dict[str, Path]:
    """Scan directories → {subject_id (CSV file stem): path}."""
    mapping: Dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if d.exists():
            for p in sorted(d.glob("*.csv")):
                mapping[p.stem] = p
    return mapping


def load_split(split_file) -> Dict[str, List[str]]:
    """Read split_ids.json → {'train': [...], 'val': [...], 'test': [...]}."""
    with open(split_file) as f:
        raw = json.load(f)
    return {
        "train": list(raw["train_ids"]),
        "val": list(raw.get("validation_ids", raw.get("val_ids", []))),
        "test": list(raw["test_ids"]),
    }


# ============================================================================
# 3. ENROLL / VERIFY PARTITION
# ============================================================================

def valid_window_starts(nan_mask: np.ndarray,
                        ws: int = WINDOW_SAMPLES,
                        stride: int = WINDOW_STRIDE,
                        nan_ratio: float = 0.20) -> np.ndarray:
    """
    Start indices of windows (at the 50 %-overlap stride) whose NaN fraction is
    below `nan_ratio`. Uses a prefix-sum so it is O(T) not O(T*ws) — important at
    ~3 hr x 1099 subjects.
    """

    T = len(nan_mask)

    if T < ws:
        return np.empty(0, dtype=np.int64)
    cumsum = np.cumsum(nan_mask.astype(np.int64))
    nan_in_window = cumsum.copy()
    nan_in_window[ws:] = cumsum[ws:] - cumsum[:-ws]    # NaNs inside window starting at i
    nan_in_window = nan_in_window[ws - 1:]
    starts = np.arange(0, T - ws + 1, stride, dtype=np.int64)
    return starts[(nan_in_window[starts] / ws) < nan_ratio]


def partition_blocks(starts: np.ndarray,
                     total_len: int,
                     block_len: int,
                     enroll_ratio: float,
                     gap: int,
                     rng: np.random.Generator,
                     ws: int = WINDOW_SAMPLES,
                     ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Block-interleaved enroll/verify split with guard gaps and no raw overlap.

    How it works
    ------------
    * The session div into contiguous blocks of `block_len` = 2 min
      randomly assigned - `enroll_ratio`
    
      With a 3 hr session and 2 min blocks: ~90 blocks → ~72 enroll + ~18 verify (80-20 split)
      
      Returns (enroll_starts, verify_starts, n_dropped).

    Scattering ensures both enroll and verify see the full mix of still, active, walking, ect.
    """
    #edge case
    if len(starts) == 0:
        return starts.copy(), starts.copy(), 0

    n_blocks = max(2, int(np.ceil(total_len / block_len)))
    n_verify = int(round(n_blocks * (1.0 - enroll_ratio)))
    n_verify = min(max(n_verify, 1), n_blocks - 1)              # both sets non-empty

    verify_blocks = set(                                                                        # Randomly choose verification blocks
        int(b) 
        for b in rng.choice(
            n_blocks, 
            size=n_verify, 
            replace=False )) 

    role = np.array(
        [1 if b in verify_blocks else 0
          for b in range(n_blocks)])  # create block label


    boundaries = np.array(
        [(i + 1) * block_len 
        for i in range(n_blocks - 1)
        if role[i] != role[i + 1]], dtype=np.int64)

    win_start, win_end = starts, starts + ws
    win_role = role[np.clip(starts // block_len, min=0, max=n_blocks - 1)]

    # Drop any window whose [start, end) lies within `gap` of a role boundary.
    near_boundary = np.zeros(len(starts), dtype=bool)
    for b in boundaries:
        near_boundary |= (win_start < b + gap) & (win_end > b - gap)

    enroll = (win_role == 0) & ~near_boundary
    verify = (win_role == 1) & ~near_boundary
    return starts[enroll], starts[verify], int(near_boundary.sum())


# ============================================================================
# 4. ACTIVITY STATES  (unsupervised: still / active / walking)
# ============================================================================

# @dataclass
# class ActivityModel:
#     """
#     KMeans(k=3) over a 2-D per-window descriptor, plus the cluster->state mapping.
#     """
#     kmeans: object
#     mapping: Dict[int, str]
#     centers: np.ndarray
#     seed: int
#     clip_lo: np.ndarray
#     clip_hi: np.ndarray
#     feat_mean: np.ndarray
#     feat_std: np.ndarray

#     def _transform(self, feats: np.ndarray) -> np.ndarray:
#         f = np.clip(feats, self.clip_lo, self.clip_hi)
#         return (f - self.feat_mean) / self.feat_std

#     def label(self, feats: np.ndarray) -> np.ndarray:
#         if len(feats) == 0:
#             return np.empty(0, dtype=object)
#         clusters = self.kmeans.predict(self._transform(feats))
#         return np.array([self.mapping[int(c)] for c in clusters], dtype=object)


# def window_activity_features(data_unnorm: np.ndarray, starts: np.ndarray,
#                              ws: int = WINDOW_SAMPLES) -> np.ndarray:
#     """
#     Per-window 2-D activity descriptor (physical units, model-independent):

#         axis 0 = log1p( mean walking_act^2 )   — 0.5-2 Hz step-cadence energy
#         axis 1 = log1p( var ||userAccel|| )    — overall movement energy
#     """
#     wa, am = channel_index("walking_act"), channel_index("accel_mag")
#     if len(starts) == 0:
#         return np.empty((0, 2), dtype=np.float32)
#     feats = np.empty((len(starts), 2), dtype=np.float32)
#     for i, s in enumerate(starts):
#         w = data_unnorm[s:s + ws]
#         feats[i, 0] = np.log1p(np.mean(w[:, wa] ** 2))
#         feats[i, 1] = np.log1p(np.var(w[:, am]))
#     return feats


# def fit_activity_model(all_feats: np.ndarray, seed: int) -> ActivityModel:
#     """Cluster pooled features into still/active/walking by movement & cadence energy."""
#     from sklearn.cluster import KMeans
#     lo = np.percentile(all_feats, 1.0, axis=0)
#     hi = np.percentile(all_feats, 99.0, axis=0)
#     clipped = np.clip(all_feats, lo, hi)
#     mean = clipped.mean(axis=0)
#     std = clipped.std(axis=0) + 1e-8
#     feats_z = (clipped - mean) / std

#     km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(feats_z)
#     centers = km.cluster_centers_                       # (3, 2): [cadence, movement]
#     by_movement = np.argsort(centers[:, 1])            # ascending overall movement
#     still = int(by_movement[0])
#     moving = [int(by_movement[1]), int(by_movement[2])]
#     walking = moving[int(np.argmax(centers[moving, 0]))]   # more cadence energy = walking
#     active = moving[int(np.argmin(centers[moving, 0]))]
#     mapping = {still: "still", active: "active", walking: "walking"}
#     log.info("Activity KMeans centers (z-space cadence,movement): %s", centers.tolist())
#     log.info("Activity cluster->state mapping: %s", mapping)
#     return ActivityModel(km, mapping, centers, seed, lo, hi, mean, std)


# ============================================================================
# 5. DATA LAYER  (model-agnostic: B0/B1/B2 all consume this)
# ============================================================================

@dataclass
class SubjectPartition:
    subject_id: str
    enroll_starts: np.ndarray
    verify_starts: np.ndarray
    n_dropped: int


class VerificationData:
    """
    Streams each subject to a disk cache and serves normalised windows, the
    subject-disjoint split, the enroll/verify partition and activity labels
    """

    def __init__(self, cache_dir, split_file, data_dirs=DEFAULT_DATA_DIRS,
                 seed: int = 42, nan_ratio: float = 0.20,
                 enroll_ratio: float = 0.8, block_seconds: float = 120.0,
                 gap_seconds: float = 15.0):
        

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)

        self.nan_ratio = float(nan_ratio)

        self.enroll_ratio = float(enroll_ratio)

        # Seconds to Samples
        self.block_len = int(round(block_seconds * SAMPLING_RATE))
        self.gap = int(round(gap_seconds * SAMPLING_RATE))
        assert self.block_len > WINDOW_SAMPLES, "block_seconds too small for the window"

        #Load Train/Val/Test Split
        self.split = load_split(split_file)
        self.subject_to_path = build_subject_to_path(*data_dirs)


        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None

        #self.activity_model: Optional[ActivityModel] = None

        self._mmap: Dict[str, np.ndarray] = {}
        self._starts: Dict[str, np.ndarray] = {}

        log.info("Split: train=%d val=%d test=%d | CSVs on disk=%d | seed=%d",
                 len(self.split["train"]), len(self.split["val"]),
                 len(self.split["test"]), len(self.subject_to_path), self.seed)
        log.info("window=%d stride=%d enroll_ratio=%.2f block_len=%d gap=%d samples nan_ratio=%.2f",
                 WINDOW_SAMPLES, WINDOW_STRIDE, self.enroll_ratio,
                 self.block_len, self.gap, self.nan_ratio)

    # --- disk cache ---------------------------------------------------------
    def _data_path(self, sid):   return self.cache_dir / f"{sid}.npy"
    def _starts_path(self, sid): return self.cache_dir / f"{sid}.starts.npy"

    def prepare_cache(self, subject_ids, force: bool = False) -> List[str]:
        """Stream each CSV → 8 Hz low-pass → derived channels → disk; cache valid windows."""
        
        ok: List[str] = []
        
        for sid in subject_ids:
            dp, sp = self._data_path(sid), self._starts_path(sid)

            if dp.exists() and sp.exists() and not force:
                ok.append(sid)
                continue

            csv = self.subject_to_path.get(sid)
            if csv is None:
                log.warning("[skip] no CSV for %s", sid)
                continue
            try:
                df = pd.read_csv(csv, low_memory=False, usecols=FEATURES)
            except Exception as e:                                   
                log.warning("[skip] %s: %s", sid, e)
                continue


            df = df.apply(pd.to_numeric, errors="coerce")

            nan_mask = df.isna().any(axis=1).values
            df = df.ffill().bfill()

            if df.isna().any().any():
                log.warning("[skip] unfillable NaNs in %s", sid)
                continue


            data = df[FEATURES].values.astype(np.float32)    #Convert to NumPy  

            data = lowpass_filter(data)
            data = append_derived_features(data)                     # (T, 28)
            starts = valid_window_starts(nan_mask, nan_ratio=self.nan_ratio)

            if len(starts) == 0:
                log.warning("[skip] no valid windows for %s", sid)
                continue

            np.save(dp, data)
            np.save(sp, starts)

            ok.append(sid)
        log.info("Cached %d/%d subjects in %s", len(ok), len(subject_ids), self.cache_dir)
        return ok


    # Loads file as memory-mapped + dOES NOT load entire file into RAM.
    def _load(self, sid) -> np.ndarray:
        if sid not in self._mmap:
            # Bound the number of simultaneously-open memory-maps so we never hit
            # the OS "too many open files" limit (B0 touches train+test = >1000).
            if len(self._mmap) >= 64:
                self._mmap.clear()
            self._mmap[sid] = np.load(self._data_path(sid), mmap_mode="r")
        return self._mmap[sid]

    def starts(self, sid) -> np.ndarray:
        if sid not in self._starts:
            self._starts[sid] = np.load(self._starts_path(sid))
        return self._starts[sid]


    # --- channel statistics (train only) --------------------------
    def fit_channel_stats(self, train_ids) -> None:
        cached = [s for s in train_ids 
                  if self._data_path(s).exists()]
        
        # Running totals so we never hold all subjects in memory at once.
        n = 0
        ssum = np.zeros(N_INPUT_CHANNELS)          # sum of values, per channel
        ssq = np.zeros(N_INPUT_CHANNELS)           # sum of squares, per channel
        for sid in cached:
            d = np.asarray(self._load(sid), dtype=np.float64)
            ssum += d.sum(axis=0)
            ssq += (d * d).sum(axis=0)
            n += d.shape[0]

        if n == 0:
            raise RuntimeError("fit_channel_stats: no cached train data found.")
        mean = ssum / n
        std = np.sqrt(np.maximum(ssq / n - mean ** 2, 0.0)) + 1e-8
        self.channel_mean, self.channel_std = mean.astype(np.float32), std.astype(np.float32)
        log.info("Channel stats from %d subjects, %d frames.", len(cached), n)

    def set_channel_stats(self, mean, std) -> None:
        self.channel_mean = np.asarray(mean, dtype=np.float32)
        self.channel_std = np.asarray(std, dtype=np.float32)

    # --- window access ------------------------------------------------------
    def get_windows(self, sid, starts) -> np.ndarray:
        """Normalised windows (len(starts), 28, 300), channels-first."""
        assert self.channel_mean is not None, "set channel stats first"
        d = self._load(sid)
        out = np.empty((len(starts), N_INPUT_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
        for i, s in enumerate(starts):
            w = np.asarray(d[s:s + WINDOW_SAMPLES], dtype=np.float32)
            out[i] = ((w - self.channel_mean) / self.channel_std).T
        return out



    # --- enroll / verify ----------------------------------------------------
    def partition(self, sid) -> SubjectPartition:
        """Deterministic per-subject contiguous enroll/verify partition."""

        # Deterministic per-subject seed. Uses hashlib (NOT Python's built-in
        # hash(), which is randomised per process) so the split is reproducible
        # WITHOUT needing PYTHONHASHSEED set.
        digest = hashlib.md5(str(sid).encode()).hexdigest()
        sub_seed = (self.seed * 1_000_003 + (int(digest, 16) & 0xFFFFFFFF)) & 0xFFFFFFFF
        rng = np.random.default_rng(sub_seed)

        starts = self.starts(sid)
        total_len = int(self._load(sid).shape[0])
        enroll, verify, dropped = partition_blocks(
            starts, total_len, self.block_len, self.enroll_ratio, self.gap, rng)
        
        return SubjectPartition(sid, enroll, verify, dropped)

    def all_windows_index(self, subject_ids) -> List[Tuple[str, int]]:
        """Flat [(subject_id, start)] over all valid windows — for training."""
        return [(sid, int(s)) for sid in subject_ids for s in self.starts(sid)]




    # # --- activity -----------------------------------------------------------
    # def fit_activity(self, subject_ids) -> ActivityModel:
    #     feats = [window_activity_features(np.asarray(self._load(sid)),
    #                                       self.partition(sid).verify_starts)
    #              for sid in subject_ids]
    #     feats = np.concatenate(feats, 0) if feats else np.zeros((0, 2), np.float32)
    #     if len(feats) < 3:
    #         raise RuntimeError("fit_activity: not enough verify windows to cluster.")
    #     self.activity_model = fit_activity_model(feats, self.seed)
    #     labels = self.activity_model.label(feats)
    #     uniq, cnt = np.unique(labels, return_counts=True)
    #     log.info("Activity distribution (verify windows): %s",
    #              dict(zip(uniq.tolist(), cnt.tolist())))
    #     return self.activity_model

    # def activity_labels(self, sid, starts) -> np.ndarray:
    #     assert self.activity_model is not None, "call fit_activity first"
    #     feats = window_activity_features(np.asarray(self._load(sid)), starts)
    #     return self.activity_model.label(feats)


# ============================================================================
# 6. B0 — CLASSICAL HAND-CRAFTED FEATURES (no deep learning)
# ============================================================================
_B0_BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, 10.0)]
_B0_N_DYN = 6      # first 6 dynamic channels: userAccel XYZ + gyro XYZ


def classical_window_features(windows: np.ndarray,
                              fs: float = float(SAMPLING_RATE)) -> np.ndarray:
    """
    (N, C, T) normalised windows -> (N, C*8 + 6*4) hand-crafted features.

    Per channel: mean, std, skew, excess-kurtosis, P10, P90, min, max (8 stats),
    plus log band-power in 4 physiological bands for the 6 dynamic channels.
    These are the same statistics the deep _GlobalStatsEncoder / _BandPowerEncoder
    compute — B0 is the "do we even need deep learning?" floor.
    """
    x = windows.astype(np.float64)
    mu, sd = x.mean(-1), x.std(-1) + 1e-8
    z = (x - mu[..., None]) / sd[..., None]
    skew, kurt = (z ** 3).mean(-1), (z ** 4).mean(-1) - 3.0
    p10, p90 = np.percentile(x, 10, -1), np.percentile(x, 90, -1)
    stats = np.concatenate([mu, sd, skew, kurt, p10, p90, x.min(-1), x.max(-1)], -1)

    spec = np.abs(np.fft.rfft(x[:, :_B0_N_DYN, :], axis=-1)) ** 2
    freqs = np.fft.rfftfreq(windows.shape[-1], d=1.0 / fs)
    bands = [np.log1p(spec[:, :, (freqs >= lo) & (freqs < hi)].sum(-1))
             for lo, hi in _B0_BANDS]
    return np.concatenate([stats, np.concatenate(bands, -1)], -1).astype(np.float32)


# ============================================================================
# 7. MODELS — B1 (CNN), B2 (GRU), SupCon loss
# ============================================================================

def _make_cnn(hidden_size, embed_dim, se: bool = False):
    import torch.nn as nn
    import torch.nn.functional as F
    from apw_network import APW_Net

    class CNNEmbedding(APW_Net):
        """APW_Net encoder with age/gender heads removed and a 128-d L2 head added.
        se=True adds Squeeze-and-Excitation channel attention in every conv block (M1)."""

        def __init__(self):
            super().__init__(n_channels=N_INPUT_CHANNELS, hidden_size=hidden_size, se=se)

            del self.age_head
            del self.gender_head

            self.embed_dim = embed_dim
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_size, embed_dim))

        def forward(self, x):
            # three temporal scales from the dilated residual stack
            a = self.enc_A(x)
            b = self.enc_B(a)
            c = self.enc_C(b)
            # concatenate all branches: 3 temporal pools + spectral + stats + band-power
            branches = torch.cat([
                self.pool_fine(a),
                self.pool_mid(b),
                self.pool_coarse(c),
                self.spectral(x),
                self.stats_enc(x),
                self.band_enc(x),
            ], dim=-1)
            fused = self.fusion(branches)
            return F.normalize(self.proj(fused), dim=-1)

    import torch
    return CNNEmbedding()


def _make_gru(hidden_size, embed_dim):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GRUEmbedding(nn.Module):
        """Bi-GRU encoder + mean/last pooling → same 128-d L2 embedding as B1."""

        def __init__(self):
            super().__init__()
            self.embed_dim = embed_dim
            self.gru = nn.GRU(N_INPUT_CHANNELS, hidden_size, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=0.1)
            
            self.proj = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, embed_dim))

        def forward(self, x):
            x = x.transpose(1, 2)                       # (B, C, T) -> (B, T, C)
            out, _ = self.gru(x)                        # (B, T, 2H)
            pooled = torch.cat([out.mean(1), out[:, -1, :]], dim=-1)
            return F.normalize(self.proj(pooled), dim=-1)

    return GRUEmbedding()


def _make_m2(hidden_size, embed_dim, smooth_kernel: int = 11, se: bool = True):
    """
    M2 = "slow / fast (tremor) split" model.

    Idea (plain words): every 15 s window is separated into
      * SLOW  = the smooth, voluntary motion (low frequency), and
      * FAST  = what's left over = the tremor / jitter (high frequency).
    Both streams are stacked (28 + 28 = 56 channels) and fed to the SAME SE-CNN
    encoder used by M1. The FAST stream is the always-present, activity-independent
    identity signal we want the model to lean on.

    The split is done with a simple moving-average low-pass (easy to read and to
    change later): slow = local average over `smooth_kernel` samples; fast = x - slow.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from apw_network import APW_Net

    def moving_average(x, k):
        # Average each sample with its k neighbours (centred). Reflect-pad the ends
        # so the output is the same length as the input.
        pad = k // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")
        return F.avg_pool1d(x_padded, kernel_size=k, stride=1)

    class M2Embedding(APW_Net):
        """SE-CNN over the stacked (slow ‖ fast) streams → 128-d L2 embedding."""

        def __init__(self):
            # 56 input channels = 28 slow + 28 fast (the two streams stacked).
            super().__init__(n_channels=2 * N_INPUT_CHANNELS,
                             hidden_size=hidden_size, se=se)
            del self.age_head
            del self.gender_head
            self.embed_dim = embed_dim
            self.smooth_kernel = smooth_kernel
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_size, embed_dim))

        def split_slow_fast(self, x):
            slow = moving_average(x, self.smooth_kernel)   # voluntary motion (low freq)
            fast = x - slow                                # tremor / jitter (high freq)
            return torch.cat([slow, fast], dim=1)          # (B, 28, T) -> (B, 56, T)

        def forward(self, x):
            x = self.split_slow_fast(x)
            # Same multi-branch fusion as M1, now over the 56-channel input.
            a = self.enc_A(x)
            b = self.enc_B(a)
            c = self.enc_C(b)
            branches = torch.cat([
                self.pool_fine(a),
                self.pool_mid(b),
                self.pool_coarse(c),
                self.spectral(x),
                self.stats_enc(x),
                self.band_enc(x),
            ], dim=-1)
            fused = self.fusion(branches)
            return F.normalize(self.proj(fused), dim=-1)

    return M2Embedding()


def build_model(name: str, hidden_size: int = 256, embed_dim: int = EMBED_DIM_DEFAULT):
    """Create one of the deep embedding models by name."""
    name = name.lower()
    if name == "b1":                       # CNN, no SE
        return _make_cnn(hidden_size, embed_dim, se=False)
    if name == "m1":                       # SE-CNN (M1a=supcon / M1b=arcface)
        return _make_cnn(hidden_size, embed_dim, se=True)
    if name == "m2":                       # SE-CNN over slow/fast(tremor) split
        return _make_m2(hidden_size, embed_dim)
    if name == "b2":                       # bidirectional GRU
        return _make_gru(max(128, hidden_size // 2), embed_dim)
    raise ValueError(f"build_model: unknown deep model '{name}' (use b1/m1/m2/b2).")


class ArcFaceHead:
    """
    ArcFace additive-angular-margin head (Deng et al., 2019) — TRAINING ONLY.

    Holds a learnable class-centre per TRAIN identity. During training it adds an
    angular margin `m` to the true class before a scaled softmax cross-entropy,
    forcing each identity's embeddings to sit well inside their own angular region
    with a clear gap to neighbours -> embeddings that separate even unseen people.

    At TEST time this head is DISCARDED: verification still uses the embedding +
    cosine distance, so the system stays open-set. (This is a thin wrapper; the
    nn.Module is built lazily so the B0 path stays torch-free.)
    """
    def __init__(self, embed_dim: int, n_classes: int, s: float = 30.0, m: float = 0.5):
        import torch.nn as nn
        self.module = _ArcFaceModule(embed_dim, n_classes, s, m)


def _ArcFaceModule(embed_dim, n_classes, s, m):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _ArcFace(nn.Module):
        """
        Additive COSINE margin (CosFace / AM-Softmax). We subtract the margin from
        the true class's cosine directly — no acos — which is numerically stable and
        avoids the gradient blow-up (near cos=±1) that collapses plain ArcFace from
        scratch. Same "angular margin" idea, just the robust variant.
            logit_true   = s * (cos - m)
            logit_others = s *  cos
        """
        def __init__(self):
            super().__init__()
            self.s, self.m = s, m
            self.W = nn.Parameter(torch.empty(n_classes, embed_dim))
            nn.init.xavier_uniform_(self.W)

        def forward(self, emb, labels):           # emb already L2-normalised (B, D)
            cos = emb @ F.normalize(self.W, dim=1).t()        # (B, n_classes), in [-1, 1]
            onehot = F.one_hot(labels, num_classes=self.W.shape[0]).float()
            logits = self.s * (cos - self.m * onehot)         # subtract margin on true class
            return F.cross_entropy(logits, labels)

    return _ArcFace()


def supcon_loss(embeddings, labels, temperature: float = 0.1):
    """
    Supervised-contrastive loss over L2-normalised embeddings
    Batches must be identity-balanced (P subjects x K windows) so positives exist
    """

    import torch

    device = embeddings.device
    B = embeddings.shape[0]

    sim = embeddings @ embeddings.t() / temperature             # already l2norm so this is Cosine Similarity/ temp for stronger separation
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()       # numerical stability trivk so that numbers dont explode later on

    labels = labels.view(-1, 1)
    same = (labels == labels.t()).float().to(device)        # positive-pair matrix
    eye = torch.eye(B, device=device)
    positives = same - eye                                     # exclude self
    not_self = 1.0 - eye                                    # everyone except self

    exp_sim = torch.exp(sim) * not_self         #e^(similarity) os Large similarities dominate
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)                #softmax probability in log-space
    n_pos = positives.sum(1)            # Counts positives per anchor
    valid = n_pos > 0
    if valid.sum() == 0:
        return embeddings.sum() * 0.0
    
    mean_log_prob_pos = (positives * log_prob).sum(1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


# ============================================================================
# 8. EMBEDDING EXTRACTION  (per subject: enroll, verify, verify activity)
# ============================================================================
# emb_by_subject[sid] = {'enroll': (Ne,D), 'verify': (Nv,D), 'verify_activity': (Nv,)}

def _batched(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def _aggregate(emb: np.ndarray, starts: np.ndarray, n: int,
               activity: Optional[np.ndarray] = None):
    """
    Multi-window aggregation: turn N consecutive windows into ONE attempt
    """
    if n <= 1 or len(emb) == 0:
        return (emb, activity) if activity is not None else emb

    groups, cur = [], [0]
    for i in range(1, len(starts)):
        if (starts[i] - starts[i - 1] == WINDOW_STRIDE) and (len(cur) < n):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    agg_emb = np.stack([emb[g].mean(axis=0) for g in groups]).astype(emb.dtype)
    if activity is None:
        return agg_emb
    
    agg_act = []
    for g in groups:
        vals, counts = np.unique(activity[g], return_counts=True)
        agg_act.append(vals[int(np.argmax(counts))])
    return agg_emb, np.array(agg_act, dtype=object)


def compute_deep_embeddings(model, data: VerificationData, sids, device,
                            batch_size: int = 256, agg_windows: int = 1) -> Dict[str, dict]:
    """B1/B2: forward enroll + verify windows through the network."""
    import torch
    model.eval()

    out: Dict[str, dict] = {}

    with torch.no_grad():

        for sid in sids:

            part = data.partition(sid)
            entry = {}
            embs = {}

            for role, starts in (("enroll", part.enroll_starts),
                                 ("verify", part.verify_starts)):
                
                chunks = [model(torch.from_numpy(data.get_windows(sid, cs)).to(device)
                                ).cpu().numpy()
                          for cs in _batched(starts, batch_size)]
                
                embs[role] = (np.concatenate(chunks, 0) if chunks
                              else np.zeros((0, getattr(model, "embed_dim", 128)), np.float32))
                
            # act = data.activity_labels(sid, part.verify_starts)   # ASA disabled

            entry["enroll"] = _aggregate(embs["enroll"], part.enroll_starts, agg_windows)
            entry["verify"] = _aggregate(embs["verify"], part.verify_starts, agg_windows)
            out[sid] = entry

    return out


def fit_b0_scaler(data: VerificationData, train_ids, max_windows: int = 50_000, seed: int = 42):
    """StandardScaler for B0 features, fit on windows sampled from TRAIN subjects."""
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    feats, n = [], 0
    for sid in train_ids:
        starts = data.starts(sid)
        if len(starts) == 0:
            continue

        take = min(len(starts), max(8, max_windows // max(1, len(train_ids))))

        sel = np.sort(rng.choice(len(starts), size=take, replace=False))

        feats.append(classical_window_features(data.get_windows(sid, starts[sel])))
        n += take
        if n >= max_windows:
            break

    F = np.concatenate(feats, 0)
    log.info("B0 scaler fit on %d windows (%d features).", len(F), F.shape[1])
    return StandardScaler().fit(F)


def compute_b0_embeddings(data: VerificationData, sids, scaler,
                          agg_windows: int = 1) -> Dict[str, dict]:
    """B0: standardised hand-crafted features act as the embeddings."""
    out: Dict[str, dict] = {}
    for sid in sids:
        part = data.partition(sid)
        feats = {}
        for role, starts in (("enroll", part.enroll_starts), ("verify", part.verify_starts)):
            if len(starts) == 0:
                feats[role] = np.zeros((0, scaler.mean_.shape[0]), np.float32)
            else:
                feats[role] = scaler.transform(
                    classical_window_features(data.get_windows(sid, starts))).astype(np.float32)
        # act = data.activity_labels(sid, part.verify_starts)   # ASA disabled
        entry = {"enroll": _aggregate(feats["enroll"], part.enroll_starts, agg_windows)}
        entry["verify"] = _aggregate(feats["verify"], part.verify_starts, agg_windows)
        out[sid] = entry
    return out


# ============================================================================
# 9. EVALUATION HARNESS  (one harness, all three models)
# ============================================================================

CAVEAT = (
    "Subjects were recorded in a single session, so our evaluation does not\n"
    "  capture cross-session variability (watch re-placement, day-to-day\n"
    "  behavioral drift). Reported EER should be read as a WITHIN-SESSION LOWER\n"
    "  BOUND; cross-session robustness is left for future work.")


def compute_eer(genuine, impostor):
    """
    Equal Error Rate
    """
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    if len(genuine) == 0 or len(impostor) == 0:
        return (float("nan"),) * 4
    g, im = np.sort(genuine), np.sort(impostor)
    ts = np.unique(np.concatenate([genuine, impostor,
                                   [min(g[0], im[0]) - 1, max(g[-1], im[-1]) + 1]]))
    frr = np.searchsorted(g, ts, "left") / len(g)             # genuine rejected
    far = (len(im) - np.searchsorted(im, ts, "left")) / len(im)   # impostor accepted
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ts[i]), float(far[i]), float(frr[i])


class CosineScorer:
    """Reference = L2-normalised mean enroll embedding; score = cosine similarity."""
    name = "cosine"

    @staticmethod
    def _unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    def fit(self, enroll):
        return self._unit(enroll.mean(0))

    def score(self, ref, attempts):
        return self._unit(attempts) @ ref


class OCSVMScorer:
    """Per-user one-class SVM (RBF); score = signed decision function."""
    name = "ocsvm"

    def __init__(self, nu: float = 0.1, gamma: str = "scale"):
        self.nu, self.gamma = nu, gamma

    def fit(self, enroll):
        from sklearn.svm import OneClassSVM
        return OneClassSVM(nu=self.nu, gamma=self.gamma).fit(enroll)

    def score(self, model, attempts):
        if len(attempts) == 0:
            return np.empty(0, dtype=float)
        return model.decision_function(attempts)


@dataclass
class VerificationResult:
    scorer: str
    n_subjects: int
    n_genuine: int
    n_impostor: int
    eer: float
    threshold: float
    far_at_eer: float
    frr_at_eer: float
    # per_activity: Dict[str, dict]      # ASA disabled
    rank1: float
    rank1_n: int
    seed: int


def run_verification(emb_by_subject, scorer, seed: int = 42,
                     impostors_per_user: int = 2000,
                     rank1_max_attempts: int = 4000) -> VerificationResult:
    """For each subject: genuine = own verify windows; impostors = sampled from
    other subjects' verify windows. Reports EER overall + per activity + rank-1."""
    rng = np.random.default_rng(seed)


    # Skip subjects we can't enroll or test (empty enroll/verify set) — e.g. a
    # short/sparse recording where the random verify segment + guard gap left no
    # valid windows on one side.
    sids = [s for s in sorted(emb_by_subject)
            if len(emb_by_subject[s]["enroll"]) > 0 and len(emb_by_subject[s]["verify"]) > 0]
    n_skipped = len(emb_by_subject) - len(sids)
    if n_skipped:
        log.warning("run_verification: skipped %d subject(s) with empty enroll/verify.", n_skipped)
    models = {s: scorer.fit(emb_by_subject[s]["enroll"]) for s in sids}

    v_emb = {s: emb_by_subject[s]["verify"] for s in sids}
    # v_act = {s: np.asarray(emb_by_subject[s]["verify_activity"]) for s in sids}   # ASA disabled
    all_emb = np.concatenate([v_emb[s] for s in sids], 0)
    all_owner = np.concatenate([[s] * len(v_emb[s]) for s in sids]).astype(object)
    # all_act = np.concatenate([v_act[s] for s in sids]).astype(object)             # ASA disabled

    scores = []        # similarity score of every attempt
    is_gen = []        # 1 = genuine attempt, 0 = impostor attempt
    for sid in sids:
        # genuine: this subject's own verify windows vs their own reference
        genuine_scores = scorer.score(models[sid], v_emb[sid])
        scores += genuine_scores.tolist()
        is_gen += [1] * len(genuine_scores)

        # impostor: a random sample of OTHER subjects' verify windows vs this reference
        other_idx = np.where(all_owner != sid)[0]
        if len(other_idx) == 0:
            continue
        n_take = min(impostors_per_user, len(other_idx))
        sel = rng.choice(other_idx, size=n_take, replace=False)
        impostor_scores = scorer.score(models[sid], all_emb[sel])
        scores += impostor_scores.tolist()
        is_gen += [0] * len(impostor_scores)

    scores = np.asarray(scores)
    is_gen = np.asarray(is_gen)
    # acts = np.asarray(acts, dtype=object)                                          # ASA disabled
    gmask = is_gen == 1
    eer, thr, far, frr = compute_eer(scores[gmask], scores[~gmask])

    # per_activity = {}
    # for st in ACTIVITY_STATES:
    #     sm = acts == st
    #     e, t, fa, fr = compute_eer(scores[sm & gmask], scores[sm & ~gmask])
    #     per_activity[st] = {"eer": e, "far_at_eer": fa, "frr_at_eer": fr,
    #                         "n_genuine": int((sm & gmask).sum()),
    #                         "n_impostor": int((sm & ~gmask).sum())}

    rank1, rank1_n = _rank1(emb_by_subject, models, scorer, sids, rng, rank1_max_attempts)
    return VerificationResult(scorer.name, len(sids), int(gmask.sum()), int((~gmask).sum()),
                              eer, thr, far, frr, rank1, rank1_n, seed)


def _rank1(emb_by_subject, models, scorer, sids, rng, max_attempts):
    """Closed-set check: is the true owner's reference the nearest among all users?"""
    attempts, owners = [], []
    for sid in sids:
        v = emb_by_subject[sid]["verify"]
        attempts.append(v)
        owners += [sid] * len(v)
    attempts = np.concatenate(attempts, 0)
    owners = np.asarray(owners, dtype=object)
    if len(attempts) == 0:
        return float("nan"), 0
    if len(attempts) > max_attempts:
        sel = rng.choice(len(attempts), size=max_attempts, replace=False)
        attempts, owners = attempts[sel], owners[sel]
    S = np.column_stack([scorer.score(models[s], attempts) for s in sids])
    pred = np.array(sids, dtype=object)[np.argmax(S, axis=1)]
    return float((pred == owners).mean()), int(len(attempts))


def format_report(model_name, results, config) -> str:
    bar = "=" * 72
    out = [bar, f"OPEN-SET USER VERIFICATION — model: {model_name.upper()}", bar,
           "EVALUATION CAVEAT:", "  " + CAVEAT, bar, "Configuration:"]
    out += [f"  {k:22s}: {config[k]}" for k in sorted(config)]
    out.append(bar)
    for r in results:
        out.append(f"\n[ scorer = {r.scorer} ]")
        out.append(f"  Test subjects (gallery)    : {r.n_subjects}")
        out.append(f"  Genuine / impostor attempts: {r.n_genuine:,} / {r.n_impostor:,}")
        if r.eer == r.eer:
            out.append(f"  EER                        : {r.eer*100:6.2f} %")
            out.append(f"  FAR @ EER threshold        : {r.far_at_eer*100:6.2f} %")
            out.append(f"  FRR @ EER threshold        : {r.frr_at_eer*100:6.2f} %")
            out.append(f"  EER threshold (tau)        : {r.threshold:.4f}")
        rank = f"{r.rank1*100:6.2f} % (n={r.rank1_n:,})" if r.rank1 == r.rank1 else "n/a"
        out.append(f"  Rank-1 identification acc. : {rank}")
        #out.append("  EER by activity state (unsupervised, 0.5-2 Hz band):")
        
        # for st in ACTIVITY_STATES:
        #     pa = r.per_activity.get(st, {})
        #     e = pa.get("eer", float("nan"))
        #     es = f"{e*100:6.2f} %" if e == e else "   n/a "
        #     out.append(f"    {st:8s} EER {es}  (genuine={pa.get('n_genuine',0):,}, "
        #                f"impostor={pa.get('n_impostor',0):,})")
    out.append("\n" + bar)
    return "\n".join(out)


def result_to_dict(r: VerificationResult) -> dict:
    return r.__dict__.copy()


# ===========================================================================
# POOLED AGGREGATION (optional, labelled separately) — easy to remove
# ---------------------------------------------------------------------------
# Pools N verify windows sampled across the WHOLE session per subject, ignoring
# temporal contiguity. Unlike the contiguous _aggregate (capped at one ~2-min
# run), here N stays meaningful up to a subject's total verify count. It is a
# LESS realistic protocol (large N inflates metrics), so it is reported on its
# own with its own rank-1. Delete this function + the pooled block in
# eval_verification.py to disable.
def pool_verify(emb_by_subject: Dict[str, dict], n: int, seed: int = 42) -> Dict[str, dict]:
    rng = np.random.default_rng(seed)
    out: Dict[str, dict] = {}
    for sid, e in emb_by_subject.items():
        v = e["verify"]
        if n <= 1 or len(v) == 0:
            out[sid] = {"enroll": e["enroll"], "verify": v}
            continue
        idx = rng.permutation(len(v))                       # shuffle → pool across session
        groups = [idx[i:i + n] for i in range(0, len(idx), n)]
        pooled = np.stack([v[g].mean(axis=0) for g in groups]).astype(v.dtype)
        out[sid] = {"enroll": e["enroll"], "verify": pooled}
    return out
# ===========================================================================


# ============================================================================
# 10. SELF-TEST  
# ============================================================================