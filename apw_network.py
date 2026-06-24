"""
apw_network.py
===================
Apple Watch inertial sensor → age estimation and gender classification
using a dilated 1-D residual CNN with multi-scale spectral features.

Dataset characteristics
-----------------------
  • 3-hour continuous recordings per subject
  • Dominant activity: sitting  (phone use, reading, head/hand movements)
  • Minority activity: free walking
  • Subject pool: majority female  (sampler creates 50/50 batches)
  • Age range: 7–82 years

Input channels — 11 raw + 17 derived = 28 total
-----------------------------------------------
  Raw (11):
    motionUserAccelerationX/Y/Z  (G)    — gravity-removed wrist dynamics
    motionRotationRateX/Y/Z      (rad/s) — wrist rotation / tremor
    motionGravityX/Y/Z           (G)    — habitual resting wrist posture
    motionRoll, motionPitch      (rad)  — absolute wrist angles
  Derived (17):
    ‖userAccel‖, ‖gyro‖                 — orientation-invariant magnitudes
    ‖jerk‖, ‖ang_jerk‖                  — movement roughness (motor control)
    tremor_lo XYZ×2  (3–7 Hz)          — adult physiological tremor waveform
    tremor_hi XYZ×2  (7–9.5 Hz)        — child/young-adult tremor waveform
    walking_act      (0.5–2 Hz ‖a‖)    — step-frequency activity indicator

Age head: direct regression
---------------------------
  Scalar logit → sigmoid → [AGE_MIN, AGE_MAX].  Loss = SmoothL1 on
  normalised [0,1] age.  Matches the evaluation metric (MAE) exactly.

Window: 300 samples = 15 s @ 20 Hz, 50 % overlap (stride 150)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ─── Age LDL configuration ────────────────────────────────────────────────────

AGE_MIN      : int   = 7
AGE_MAX      : int   = 82
AGE_BINS     : int   = AGE_MAX - AGE_MIN + 1   # 76
AGE_SIGMA_SQ : float = 2.0                     # Gaussian variance for soft GT

# Bin values tensor — built once, moved to device on demand.
_AGE_BIN_VALUES = torch.arange(AGE_MIN, AGE_MAX + 1, dtype=torch.float32)  # (76,)

# ─── Feature configuration ────────────────────────────────────────────────────

# 10 raw channels selected for sitting-dominant, mixed-activity recordings.
# See module docstring for full selection rationale.
FEATURES = [
    # index 0-2 : gravity-removed wrist dynamics
    'motionUserAccelerationX(G)',
    'motionUserAccelerationY(G)',
    'motionUserAccelerationZ(G)',
    # index 3-5 : wrist rotation / micro-tremor
    'motionRotationRateX(rad/s)',
    'motionRotationRateY(rad/s)',
    'motionRotationRateZ(rad/s)',
    # index 6-8 : gravity direction = habitual resting posture
    'motionGravityX(G)',
    'motionGravityY(G)',
    'motionGravityZ(G)',
    # index 9-10: stable absolute wrist angles (CoreMotion sensor fusion)
    'motionRoll(rad)',
    'motionPitch(rad)',
    # motionYaw excluded — drifts over 3-hour sessions
]

_USERACCEL_IDX = [0, 1, 2]   # motionUserAcceleration columns
_GYRO_IDX      = [3, 4, 5]   # motionRotationRate columns
# gravity (6-8) and angles (9-10) have no magnitude appended:
#   gravity magnitude ≈ 1 G always (uninformative); angles are scalars.

SAMPLING_RATE    : int = 20
WINDOW_SECONDS   : int = 15
WINDOW_SAMPLES   : int = SAMPLING_RATE * WINDOW_SECONDS   # 300  (15 s @ 20 Hz)
WINDOW_STRIDE    : int = WINDOW_SAMPLES // 2               # 150  (50 % overlap)
N_INPUT_CHANNELS : int = len(FEATURES) + 17                # 28  (11 raw + 4 magnitudes + 6 lo-tremor + 6 hi-tremor + 1 walking)


# ─── Preprocessing ────────────────────────────────────────────────────────────

def lowpass_filter(data: np.ndarray,
                   cutoff_hz: float = 8.0,
                   fs: float        = float(SAMPLING_RATE),
                   order: int       = 4) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth low-pass filter applied per channel.

    Cutoff 8 Hz preserves all relevant wrist micro-dynamics (tremor, small
    gestures) while suppressing sensor quantisation noise. Nyquist = 10 Hz
    at 20 Hz sampling.  Applied only to dynamic channels (userAccel, gyro);
    gravity and angle channels are already smooth by CoreMotion fusion, but
    filtering them is harmless and handled uniformly.

    data : (T, C) → (T, C)
    """
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * fs
    sos = butter(order, cutoff_hz / nyq, btype='low', analog=False, output='sos')
    return sosfiltfilt(sos, data, axis=0).astype(data.dtype)


def bandpass_filter(data: np.ndarray,
                    low_hz:  float,
                    high_hz: float,
                    fs:      float = float(SAMPLING_RATE),
                    order:   int   = 4) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth band-pass filter applied per channel.

    Used to isolate physiological tremor (3–7 Hz) from the wrist IMU signal.
    The tremor band is the single most age-discriminative feature in wrist IMU:
      • Essential tremor prevalence: ~5 % at age 40, ~40 % at age 80.
      • Even sub-clinical physiological tremor amplitude increases with age.
    Bandpass output retains temporal structure so the CNN can learn waveform
    morphology (amplitude modulation, burst patterns) in addition to power.

    Note: must be called AFTER lowpass_filter so both filters operate on the
    same signal path and edge effects do not compound.

    data : (T, C) → (T, C)
    """
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * fs
    lo  = low_hz  / nyq
    hi  = high_hz / nyq
    # Clamp to valid range to avoid corner-case errors on short sequences
    lo = max(lo, 1e-4)
    hi = min(hi, 1.0 - 1e-4)
    sos = butter(order, [lo, hi], btype='band', analog=False, output='sos')
    return sosfiltfilt(sos, data, axis=0).astype(data.dtype)


def append_magnitudes(data: np.ndarray) -> np.ndarray:
    """
    Append orientation-invariant L2 magnitudes of userAcceleration and gyro.

    ‖userAccel‖ captures total wrist movement energy regardless of wrist angle.
    ‖gyro‖      captures total rotational speed regardless of axis orientation.
    Gravity magnitude is always ≈1 G and is therefore not appended.

    data : (T, 10) → (T, 12)
    """
    ua_mag   = np.linalg.norm(data[:, _USERACCEL_IDX], axis=1, keepdims=True)  # (T,1)
    gyro_mag = np.linalg.norm(data[:, _GYRO_IDX],      axis=1, keepdims=True)  # (T,1)
    return np.concatenate([data, ua_mag, gyro_mag], axis=1)


def append_derived_features(data: np.ndarray) -> np.ndarray:
    """
    Append 17 derived channels to the 11 raw sensor channels: (T, 11) → (T, 28).

    Standard derived (4 channels)  — ch 11-14
    ─────────────────────────────
    1. ‖userAccel‖       — orientation-invariant wrist movement energy.
    2. ‖gyro‖            — orientation-invariant rotational speed.
    3. ‖jerk‖            — ‖d(userAccel)/dt‖ × fs  (G/s).
    4. ‖angular_jerk‖    — ‖d(gyro)/dt‖ × fs.

    Lo-tremor channels (6 channels)  — ch 15-20
    ────────────────────────────────
    5–7.  tremor_lo_userAccel XYZ  — bandpass 3–7 Hz (adult physiological tremor).
    8–10. tremor_lo_gyro XYZ       — bandpass 3–7 Hz.
    Essential tremor prevalence ≈ 5 % at 40 yr, 40 % at 80 yr.
    Amplitude grows monotonically with age even sub-clinically.

    Hi-tremor channels (6 channels)  — ch 21-26
    ────────────────────────────────
    11–13. tremor_hi_userAccel XYZ — bandpass 7–9.5 Hz.
    14–16. tremor_hi_gyro XYZ      — bandpass 7–9.5 Hz.
    In children/young adults (age 7–30) physiological tremor peaks at
    8–12 Hz, not 3–7 Hz.  Without this band the model cannot distinguish
    a 10-year-old from a 35-year-old via tremor features alone.
    Upper limit 9.5 Hz (< Nyquist=10 Hz for 20 Hz sampling).

    Walking activity indicator (1 channel)  — ch 27
    ────────────────────────────────────────
    17. walking_act — bandpass 0.5–2 Hz of ‖userAccel‖ magnitude.
    The step-frequency band (cadence ≈ 1–2 Hz) is large during walking
    and near-zero during sitting.  Giving this as an explicit channel lets
    the model condition its tremor-band interpretation on activity state,
    reducing confusion between walking harmonics (1–3 Hz) and resting tremor.

    All bandpass filters applied AFTER the upstream 8 Hz lowpass so both
    filters operate on the same signal path.
    Jerk uses finite difference (8 Hz lowpass suppresses noise amplification).
    """
    ua_mag       = np.linalg.norm(data[:, _USERACCEL_IDX], axis=1, keepdims=True)
    gyro_mag     = np.linalg.norm(data[:, _GYRO_IDX],      axis=1, keepdims=True)
    # Finite-difference derivative × sampling rate → physical jerk units
    accel_diff   = np.diff(data[:, _USERACCEL_IDX], axis=0,
                           prepend=data[:1, _USERACCEL_IDX]) * float(SAMPLING_RATE)
    gyro_diff    = np.diff(data[:, _GYRO_IDX], axis=0,
                           prepend=data[:1, _GYRO_IDX])    * float(SAMPLING_RATE)
    jerk_mag     = np.linalg.norm(accel_diff, axis=1, keepdims=True)
    ang_jerk_mag = np.linalg.norm(gyro_diff,  axis=1, keepdims=True)

    # Lo-tremor (3–7 Hz): adult physiological tremor
    tremor_lo_ua   = bandpass_filter(data[:, _USERACCEL_IDX], 3.0, 7.0)   # (T, 3)
    tremor_lo_gyro = bandpass_filter(data[:, _GYRO_IDX],      3.0, 7.0)   # (T, 3)

    # Hi-tremor (7–9.5 Hz): child/young-adult tremor (below 20 Hz Nyquist)
    tremor_hi_ua   = bandpass_filter(data[:, _USERACCEL_IDX], 7.0, 9.5)   # (T, 3)
    tremor_hi_gyro = bandpass_filter(data[:, _GYRO_IDX],      7.0, 9.5)   # (T, 3)

    # Walking indicator: step-frequency band of ‖userAccel‖
    walking_act = bandpass_filter(ua_mag, 0.5, 2.0)                        # (T, 1)

    return np.concatenate(
        [data, ua_mag, gyro_mag, jerk_mag, ang_jerk_mag,
         tremor_lo_ua, tremor_lo_gyro,
         tremor_hi_ua, tremor_hi_gyro,
         walking_act], axis=1)  # (T, 28)


def compute_channel_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from a (T, C) array.
    Fit ONCE on the concatenated training corpus; reuse on val/test/inference.
    """
    mean = data.mean(axis=0)
    std  = data.std(axis=0) + 1e-8
    return mean, std


def preprocess(data:         np.ndarray,
               channel_mean: Optional[np.ndarray] = None,
               channel_std:  Optional[np.ndarray] = None,
               filter_data:  bool                 = True,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for one recording.

    Steps
    -----
    1. Low-pass filter at 8 Hz  (zero-phase, per channel)
    2. Append ‖accel‖ and ‖gyro‖ magnitude channels  →  8 ch
    3. Z-score normalisation  (per channel; fit on training set)
    4. Sliding-window segmentation  (50 % overlap, 100-sample windows)
    5. Transpose to channels-first  →  (N, 8, 100) for Conv1d

    Parameters
    ----------
    data         : (T, 10) float32 — raw columns in FEATURES order, at 20 Hz
    channel_mean : (12,) training-set mean — computed inline if None (training only)
    channel_std  : (12,) training-set std  — computed inline if None (training only)
    filter_data  : set False if data has already been filtered upstream

    Returns
    -------
    windows      : (N, 12, 100) float32 — ready for APW_Net
    channel_mean : (12,) — persist this from the training run
    channel_std  : (12,) — persist this from the training run
    """
    assert data.ndim == 2 and data.shape[1] == len(FEATURES), (
        f"Expected (T, {len(FEATURES)}) input, got {data.shape}")

    if filter_data:
        data = lowpass_filter(data)

    data = append_derived_features(data)  # (T, 28)

    if channel_mean is None or channel_std is None:
        channel_mean, channel_std = compute_channel_stats(data)

    data = (data - channel_mean) / channel_std  # z-score

    # Sliding-window segmentation
    T      = data.shape[0]
    starts = range(0, T - WINDOW_SAMPLES + 1, WINDOW_STRIDE)
    if not starts:
        raise ValueError(
            f"Recording too short: need ≥ {WINDOW_SAMPLES} samples, got {T}.")

    windows = np.stack([data[s:s + WINDOW_SAMPLES] for s in starts])  # (N,100,12)
    windows = windows.transpose(0, 2, 1).astype(np.float32)           # (N,12,100)

    return windows, channel_mean, channel_std


# ─── Label Distribution Learning utilities ────────────────────────────────────

def make_age_soft_labels(ages_gt: np.ndarray,
                         sigma_sq: float = AGE_SIGMA_SQ) -> torch.Tensor:
    """
    Convert integer GT ages to Gaussian soft-label distributions.

    For age μ the soft label over bins {18…56} is:
        p_k ∝ exp( -(k - μ)² / (2 σ²) ),  normalised to sum = 1.

    σ²=2 gives a spread of ≈ ±2 years — reflects realistic label uncertainty
    (self-reported age, rounding) without over-smoothing the distribution.

    Parameters
    ----------
    ages_gt  : (N,) integer or float GT ages
    sigma_sq : Gaussian variance  (default 2)

    Returns
    -------
    soft_labels : (N, AGE_BINS=39) float32 probability vectors
    """
    bins = _AGE_BIN_VALUES.numpy()                              # (39,)
    mu   = np.asarray(ages_gt, dtype=np.float32)[:, None]      # (N,1)
    log_p = -0.5 * (bins[None, :] - mu) ** 2 / sigma_sq        # (N,39)
    log_p -= log_p.max(axis=1, keepdims=True)                   # numerical stability
    p = np.exp(log_p)
    p /= p.sum(axis=1, keepdims=True)
    return torch.from_numpy(p.astype(np.float32))               # (N,39)


def expected_age(age_logits: torch.Tensor) -> torch.Tensor:
    """
    Predicted age as the expected value of the softmax distribution.

        Ê[age] = Σ_k  k · softmax(logits)_k

    Parameters
    ----------
    age_logits : (B, AGE_BINS=39)

    Returns
    -------
    age_pred : (B,) float32 in [AGE_MIN, AGE_MAX] after clamping
    """
    bins  = _AGE_BIN_VALUES.to(age_logits.device)   # (39,)
    probs = F.softmax(age_logits, dim=-1)             # (B,39)
    return (probs * bins).sum(dim=-1)                 # (B,)


# ─── Network building blocks ──────────────────────────────────────────────────

def _add_residual_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Crop/pad y to match x's temporal dimension and channel count, then add."""
    s1 = (y.shape[-1] - x.shape[-1]) // 2
    y  = y[..., s1:s1 + x.shape[-1]]
    if x.shape[1] > y.shape[1]:
        pad = torch.zeros(
            y.shape[0], x.shape[1] - y.shape[1], y.shape[-1],
            dtype=y.dtype, device=y.device)
        y = torch.cat([y, pad], dim=1)
    return x + y


class _SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., 2018) for 1-D features.

    Squeeze: global-average-pool each channel over time -> one scalar per channel.
    Excitation: a tiny bottleneck MLP (channels -> channels/r -> channels) with a
    sigmoid produces a 0..1 gate per channel. Multiply each channel by its gate.

    Effect: the block learns to amplify informative channels and suppress
    uninformative ones, per input. Cheap, and shown to improve wearable-auth CNNs
    (Mekruksavanich & Jitpattanakul, 2024).
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T)
        s = x.mean(dim=-1)                       # squeeze  -> (B, C)
        w = self.fc(s).unsqueeze(-1)             # excite   -> (B, C, 1)
        return x * w


class _ResBlock1D(nn.Module):
    """
    Dilated 1-D residual block: Conv → GroupNorm → GELU (+ optional SE + skip).

    n_groups=8 balances normalisation quality with the moderate hidden_size=256.
    GELU replaces LeakyReLU for smoother gradient flow in the regression head.
    se=True inserts Squeeze-and-Excitation channel attention before the skip add
    (off by default, so the original age/gender APW_Net is byte-for-byte unchanged).
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int = 1, n_groups: int = 8, residue: bool = True,
                 se: bool = False):
        super().__init__()
        self.residue = residue
        # Symmetric "same" padding: keeps temporal dimension unchanged.
        # For odd kernels: padding = (kernel - 1) * dilation // 2
        same_pad = (kernel - 1) * dilation // 2
        self.conv    = nn.Conv1d(in_ch, out_ch, kernel,
                                 dilation=dilation, padding=same_pad, bias=True)
        self.norm    = nn.GroupNorm(n_groups, out_ch)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(0.1)
        self.se      = _SEBlock1D(out_ch) if se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.norm(self.conv(x))))
        if self.se is not None:
            y = self.se(y)
        if self.residue:
            y = _add_residual_1d(y, x)
        return y


# ─── Attention pooling ────────────────────────────────────────────────────────

class _AttentionPool1D(nn.Module):
    """
    Temporal attention pooling: learns a scalar weight per time step so the
    model can focus on the most discriminative parts of the signal instead of
    averaging all time steps equally.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T)
        x_t = x.permute(0, 2, 1)                        # (B, T, C)
        w   = torch.softmax(self.attn(x_t), dim=1)      # (B, T, 1)
        return (x_t * w).sum(dim=1)                      # (B, C)


class _SpectralEncoder(nn.Module):
    """
    Lightweight 1-D CNN on the one-sided FFT magnitude spectrum.

    Physiological tremor (4–8 Hz) and activity cadence (< 2 Hz) are the
    key age-discriminating signals in wrist IMU data.  These live in the
    frequency domain and are hard to extract implicitly by a time-domain CNN.

    With WINDOW_SECONDS=10 and SAMPLING_RATE=20:
      • frequency resolution = 0.1 Hz
      • usable range = 0–10 Hz  (Nyquist)
      • tremor band = bins 40–80 (4–8 Hz)

    The spectrum is L1-normalised per channel before convolution so the
    model learns spectral *shape* (relative power) rather than absolute
    power, which varies with watch tightness and wrist size.
    """
    def __init__(self, n_channels: int, out_size: int):
        super().__init__()
        n_grp = 8 if out_size % 8 == 0 else 4
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, out_size, kernel_size=5, padding=2),
            nn.GroupNorm(n_grp, out_size),
            nn.GELU(),
            nn.Conv1d(out_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(n_grp, out_size),
            nn.GELU(),
        )
        self.pool = _AttentionPool1D(out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T) — time domain (z-scored)
        spec = torch.fft.rfft(x, dim=-1).abs()                # (B, C, T//2+1)
        spec = spec / (spec.sum(dim=-1, keepdim=True) + 1e-8) # L1 normalise
        return self.pool(self.net(spec))                       # (B, out_size)


# ─── Global statistics encoder ───────────────────────────────────────────────

class _GlobalStatsEncoder(nn.Module):
    """
    Per-window global statistics → compact embedding.

    Computes 8 descriptive statistics per channel: mean, std, skewness,
    excess kurtosis, P10, P90, min, max.  For 21 channels → 168-dim stat
    vector, projected through a small MLP.

    Motivation: the time-domain CNN learns local waveform patterns but may
    not explicitly encode the full statistical distribution of each channel.
    For age estimation from seated wrist IMU, these statistics are directly
    meaningful:
      - P10/P90 of tremor-band channels (ch 15–20): capture the asymmetric
        heavy-tailed distribution of tremor bursts (weak P10, large P90
        indicates intermittent tremor typical of early-stage essential tremor).
      - Kurtosis of gyro (ch 3–5): impulsive high-kurtosis signal in tremor
        grows strongly after age 60.
      - P90 of ‖jerk‖ (ch 13): captures peak movement roughness during phone
        interaction; increases monotonically with age due to motor decline.
      - Std of ‖userAccel‖ (ch 11): higher in active/younger subjects.

    Input:  (B, C, T)  — z-scored windows
    Output: (B, out_size)
    """
    def __init__(self, n_channels: int, out_size: int = 64):
        super().__init__()
        n_stats = 8   # mean, std, skewness, excess-kurtosis, P10, P90, min, max
        in_size = n_channels * n_stats   # e.g. 28 × 8 = 224
        self.mlp = nn.Sequential(
            nn.Linear(in_size, out_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_size * 2, out_size),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T)
        mu   = x.mean(dim=-1)                                  # (B, C)
        std  = x.std(dim=-1, correction=0).clamp(min=1e-8)    # (B, C)
        z    = (x - mu.unsqueeze(-1)) / std.unsqueeze(-1)      # (B, C, T)
        skew = (z ** 3).mean(dim=-1)                           # (B, C)
        kurt = (z ** 4).mean(dim=-1) - 3.0                     # (B, C) excess
        # Percentiles: capture asymmetric/heavy-tailed distributions better
        # than min/max alone (which are sensitive to outliers).
        p10  = torch.quantile(x, 0.10, dim=-1)                 # (B, C)
        p90  = torch.quantile(x, 0.90, dim=-1)                 # (B, C)
        xmin = x.amin(dim=-1)                                  # (B, C)
        xmax = x.amax(dim=-1)                                  # (B, C)
        stats = torch.cat([mu, std, skew, kurt, p10, p90, xmin, xmax], dim=-1)  # (B, C*8)
        return self.mlp(stats)                                  # (B, out_size)


# ─── Band power encoder ───────────────────────────────────────────────────────

class _BandPowerEncoder(nn.Module):
    """
    Explicit physiological band-power encoder for seated wrist IMU.

    The SpectralEncoder processes the full spectrum with L1-normalisation,
    which encodes spectral *shape* but destroys absolute power information.
    For age estimation, the *absolute* power in the tremor band (relative to
    other bands) is critical: it grows monotonically with age even without
    full-blown essential tremor.

    This encoder computes log-power in 4 physiologically-motivated bands for
    the 6 dynamic channels (userAccel XYZ + gyro XYZ):

        Band 0:  0–1 Hz  — postural drift, very slow voluntary movement
        Band 1:  1–3 Hz  — voluntary phone gestures (scroll, tap initiation)
        Band 2:  3–7 Hz  — physiological tremor (the primary age biomarker)
        Band 3:  7–10 Hz — mechanical noise / fast micro-tremor

    Band ratios (e.g. band2/band1) implicitly encode the tremor-to-movement
    ratio; log-transform compresses the wide dynamic range across subjects.

    With T=300 @ 20 Hz: frequency resolution = 0.0667 Hz, 151 FFT bins.

    Input:  (B, C_total, T) — uses channels [:6] (first dynamic channels)
    Output: (B, out_size)
    """
    _BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, 10.0)]

    def __init__(self, n_dyn_channels: int = 6, out_size: int = 32,
                 window_samples: int = WINDOW_SAMPLES,
                 fs: float = float(SAMPLING_RATE)):
        super().__init__()
        n_freq = window_samples // 2 + 1                    # 151 for T=300
        freqs  = torch.linspace(0.0, fs / 2.0, n_freq)     # (151,) Hz
        masks  = []
        for lo, hi in self._BANDS:
            masks.append(((freqs >= lo) & (freqs < hi)).float())
        self.register_buffer('band_masks', torch.stack(masks, dim=0))  # (4, 151)

        n_features = n_dyn_channels * len(self._BANDS)      # 6 × 4 = 24
        self.mlp = nn.Sequential(
            nn.Linear(n_features, out_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_size * 2, out_size),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C_total, T) — only use first 6 (dynamic) channels
        x_dyn = x[:, :6, :]                                 # (B, 6, T)
        # Power spectrum
        pwr   = torch.fft.rfft(x_dyn, dim=-1).abs().pow(2) # (B, 6, F)
        # Band power: dot each channel's spectrum with each band mask
        # band_masks: (4, F) → result: (B, 6, 4)
        bp    = torch.einsum('bcf,nf->bcn', pwr, self.band_masks)  # (B, 6, 4)
        bp    = torch.log1p(bp)                             # log-compress
        bp    = bp.flatten(1)                               # (B, 24)
        return self.mlp(bp)                                 # (B, out_size)


# ─── APW_Net ───────────────────────────────────────────────────────

class APW_Net(nn.Module):
    """
    Dual-head dilated 1-D residual CNN with spectral branch.

    Changes from v2
    ---------------
    • Spectral branch: lightweight Conv1D on FFT magnitude appended to the
      temporal encoder output.  Explicitly encodes tremor (4–8 Hz) and
      activity cadence signals that are the primary age discriminators.
    • Feature fusion: (temporal ‖ spectral) → Linear → shared representation.
    • Input window doubled to 10 s (200 samples) for better frequency
      resolution (0.1 Hz) and more tremor cycles per window.

    Shared temporal encoder — 10 dilated residual blocks (unchanged)
    ----------------------------------------------------------------
    Kernels   : [1, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    Dilations : [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]
    Receptive field ≈ 249 samples > WINDOW_SAMPLES=200 ✓

    Input  : (B, 13, 200)
    Output : age_logits (B, AGE_MAX−AGE_MIN), gender_logit (B,)
    """

    _KERNELS   = [1,  3,  3,  3,  3,  3,  3,  3,  3,  3]
    _DILATIONS = [1,  1,  2,  2,  4,  4,  8,  8, 16, 16]

    def __init__(self,
                 n_channels:  int = N_INPUT_CHANNELS,
                 hidden_size: int = 256,
                 embed_size:  int = 128,
                 n_groups:    int = 8,
                 se:          bool = False):
        super().__init__()
        spec_size = hidden_size // 2   # 128 for hidden_size=256

        # ── Temporal encoder — split into 3 sub-encoders for C++ speed ───────
        # Using nn.Sequential per segment lets TorchScript fuse ops; a single
        # ModuleList + Python for-loop is ~20% slower on GPU due to interpreter
        # overhead per block call.
        # Segment A: blocks 0-3  RF≈11 samples (0.55 s) — fine tremor cycles
        # Segment B: blocks 4-6  RF≈43 samples (2.15 s) — gesture dynamics
        # Segment C: blocks 7-9  RF≈249 samples (12.5 s) — full-window context
        all_blocks = []
        for i, (k, d) in enumerate(zip(self._KERNELS, self._DILATIONS)):
            in_ch   = n_channels if i == 0 else hidden_size
            residue = i > 0
            all_blocks.append(_ResBlock1D(in_ch, hidden_size, k, d, n_groups, residue, se=se))
        self.enc_A = nn.Sequential(*all_blocks[0:4])   # blocks 0-3
        self.enc_B = nn.Sequential(*all_blocks[4:7])   # blocks 4-6
        self.enc_C = nn.Sequential(*all_blocks[7:10])  # blocks 7-9

        # ── Pyramid attention pools (one per encoder segment) ────────────────
        self.pool_fine   = _AttentionPool1D(hidden_size)
        self.pool_mid    = _AttentionPool1D(hidden_size)
        self.pool_coarse = _AttentionPool1D(hidden_size)

        # ── Spectral encoder ────────────────────────────────────────────────
        self.spectral = _SpectralEncoder(n_channels, spec_size)

        # ── Global statistics encoder ────────────────────────────────────────
        # Captures per-channel moments + percentiles (8 stats × 21 channels = 168-dim).
        # Percentiles better capture heavy-tailed tremor burst distributions.
        _stats_size = 64
        self.stats_enc = _GlobalStatsEncoder(n_channels, out_size=_stats_size)

        # ── Band-power encoder ──────────────────────────────────────────────
        # Explicit log-power in 4 physiological bands for the 6 dynamic channels.
        # Absolute tremor-band power (3–7 Hz) is the primary age biomarker and
        # is destroyed by the L1-normalisation inside SpectralEncoder.
        _band_size = 32
        self.band_enc = _BandPowerEncoder(n_dyn_channels=6, out_size=_band_size)

        # ── Fusion: 3 temporal scales + spectral + stats + band-power → hidden ─
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3 + spec_size + _stats_size + _band_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        # ── Age regression head: scalar logit → sigmoid-scaled age ──────────
        # Output: AGE_MIN + (AGE_MAX - AGE_MIN) * sigmoid(logit)
        # Sigmoid ensures age is always in a bounded range with smooth gradients
        # at the extremes (no clamp discontinuity).
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(embed_size, 1),
        )

        # ── Gender head — scalar logit (0=male, 1=female) ─────────────────
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(embed_size, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-scale feature pyramid via 3 C++-compiled sub-encoders
        feat_a = self.enc_A(x)                                           # (B, H, T)
        feat_b = self.enc_B(feat_a)                                      # (B, H, T)
        feat_c = self.enc_C(feat_b)                                      # (B, H, T)
        p_fine   = self.pool_fine(feat_a)                                # (B, H)
        p_mid    = self.pool_mid(feat_b)                                 # (B, H)
        p_coarse = self.pool_coarse(feat_c)                              # (B, H)
        spec_feat  = self.spectral(x)                                    # (B, spec_size)
        stats_feat = self.stats_enc(x)                                   # (B, 64)
        band_feat  = self.band_enc(x)                                    # (B, 32)
        combined   = torch.cat(
            [p_fine, p_mid, p_coarse, spec_feat, stats_feat, band_feat], dim=-1)
        feat         = self.fusion(combined)                             # (B, hidden)
        age_logit    = self.age_head(feat).squeeze(-1)                   # (B,)
        gender_logit = self.gender_head(feat).squeeze(-1)                # (B,)
        return age_logit, gender_logit

    def predict_age(self, age_logit: torch.Tensor) -> torch.Tensor:
        """Sigmoid-scaled age: AGE_MIN + (AGE_MAX - AGE_MIN) * sigmoid(logit)."""
        return (AGE_MIN + float(AGE_MAX - AGE_MIN) * torch.sigmoid(age_logit))

    @staticmethod
    def predict_gender(gender_logit: torch.Tensor,
                       threshold: float = 0.5) -> torch.Tensor:
        """Returns (B,) int64: 0 = male, 1 = female."""
        return (torch.sigmoid(gender_logit) > threshold).long()


# ─── Loss function ────────────────────────────────────────────────────────────

class WristBiometricsLoss(nn.Module):
    """
    Multi-task regression loss:

        L_total = λ_age · L_huber  +  λ_gender · L_bce

    Age — SmoothL1 (Huber) on sigmoid-normalised [0, 1] age
    --------------------------------------------------------
    The age head outputs a scalar logit; predicted age =
        AGE_MIN + (AGE_MAX - AGE_MIN) * sigmoid(logit)
    Both prediction and target are normalised to [0, 1] before the loss so
    λ_age is scale-invariant.  At init (MAE≈18 yr): l_huber ≈ 0.22.
    beta = 3 yr / range means quadratic below 3 yr error, linear above.

    Why regression instead of LDL (KL + softmax)?
    The LDL loss can collapse to predicting the mode of the training age
    distribution (minimising KL without learning per-subject features).
    Direct Huber gives a gradient always proportional to the prediction error,
    preventing this collapse and matching the evaluation metric (MAE) exactly.

    Gender — BCEWithLogitsLoss.
    """

    def __init__(self,
                 lambda_age:        float = 1.0,
                 lambda_gender:     float = 1.0,
                 # kept for API backward compat (ignored)
                 lambda_huber:      float = 0.0,
                 sigma_sq:          float = 0.0,
                 gender_pos_weight: Optional[float] = None):
        super().__init__()
        self.lambda_age    = lambda_age
        self.lambda_gender = lambda_gender
        # MSE on normalised [0,1] age: gradient scales with error magnitude,
        # so large prediction errors (extremes: children, elderly) receive
        # proportionally stronger gradients than near-mean predictions.
        # This directly combats regression-to-mean collapse.
        self.age_mse = nn.MSELoss()
        pw = torch.tensor([gender_pos_weight]) if gender_pos_weight is not None else None
        self.bce_gender = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self,
                age_logit:     torch.Tensor,   # (B,) scalar from regression head
                gender_logit:  torch.Tensor,   # (B,)
                age_target:    torch.Tensor,   # (B,) integer/float GT ages
                gender_target: torch.Tensor,   # (B,) float 0.0=male / 1.0=female
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        age_range  = float(AGE_MAX - AGE_MIN)

        # Age prediction in [0, 1] — sigmoid is already applied in predict_age;
        # here we apply it directly to the raw logit for the loss gradient.
        age_pred_n = torch.sigmoid(age_logit)                 # (B,)  in [0, 1]
        age_tgt_n  = (age_target.float() - AGE_MIN) / age_range  # (B,)  in [0, 1]
        l_age      = self.age_mse(age_pred_n, age_tgt_n)

        if self.bce_gender.pos_weight is not None:
            self.bce_gender.pos_weight = self.bce_gender.pos_weight.to(
                gender_logit.device)
        l_gender = self.bce_gender(gender_logit, gender_target.float())
        total    = self.lambda_age * l_age + self.lambda_gender * l_gender
        return total, l_age, l_gender


# ─── Inference wrapper ────────────────────────────────────────────────────────

class WristBiometrics:
    """
    Ready-to-use inference wrapper mirroring the IDreveal API.

    Parameters
    ----------
    device       : 'cpu' or 'cuda'
    weights_file : path to .pt checkpoint with keys:
                   'network'      — state_dict
                   'channel_mean' — (12,) numpy array  (optional)
                   'channel_std'  — (12,) numpy array  (optional)

    Usage
    -----
    >>> model = WristBiometrics(device='cpu', weights_file='wrist_model.pt')
    >>> ages, genders = model(raw_data, channel_mean, channel_std)
    """

    def __init__(self,
                 device:       str           = 'cuda' if torch.cuda.is_available() else 'cpu',
                 weights_file: Optional[str] = None):
        self.device = device
        self.network = APW_Net().to(device)
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std:  Optional[np.ndarray] = None

        if weights_file is not None:
            ckpt = torch.load(weights_file, map_location=device)
            self.network.load_state_dict(ckpt['network'])
            self.channel_mean = ckpt.get('channel_mean', None)
            self.channel_std  = ckpt.get('channel_std',  None)

        self.network.eval()

    def __call__(self,
                 raw_data:     np.ndarray,
                 channel_mean: Optional[np.ndarray] = None,
                 channel_std:  Optional[np.ndarray] = None,
                 batch_size:   int                  = 256,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        raw_data     : (T, 10) float32 — columns in FEATURES order, at 20 Hz
        channel_mean : (12,) training-set mean (overrides stored checkpoint value)
        channel_std  : (12,) training-set std  (overrides stored checkpoint value)
        batch_size   : windows per forward pass

        Returns
        -------
        ages    : (N,) float32 — predicted age per 5-s window, clamped [18, 56]
        genders : (N,) int64  — 0 = male, 1 = female per window
        """
        mean = channel_mean if channel_mean is not None else self.channel_mean
        std  = channel_std  if channel_std  is not None else self.channel_std

        windows, _, _ = preprocess(raw_data, mean, std)   # (N, 12, 100)

        all_ages, all_genders = [], []
        splits = max(1, int(np.ceil(len(windows) / batch_size)))
        with torch.no_grad():
            for batch in np.array_split(windows, splits):
                x                     = torch.from_numpy(batch).to(self.device)
                age_logits, gen_logit = self.network(x)
                all_ages.append(
                    self.network.predict_age(age_logits).cpu().numpy())
                all_genders.append(
                    APW_Net.predict_gender(gen_logit).cpu().numpy())

        return np.concatenate(all_ages), np.concatenate(all_genders)
