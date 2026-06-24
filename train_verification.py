"""
train_verification.py
=====================
Train a metric-learning embedding network (B1 = CNN, B2 = GRU) for open-set user
verification, using SUBJECT IDENTITY as the label and supervised-contrastive loss.

Only B1/B2 need training; B0 (classical) is training-free — evaluate it directly
with eval_verification.py. Subject identities are read from split_ids.json; only
the TRAIN identities are used here (val/test people are never seen).

Example
-------
  python train_verification.py --model b1 \
      --data_dirs dataset --split_file split_ids.json \
      --cache_dir cache_verification --epochs 50 \
      --subjects_per_batch 16 --windows_per_subject 8 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from verification import (
    VerificationData, DEFAULT_DATA_DIRS, EMBED_DIM_DEFAULT,
    build_model, supcon_loss, compute_deep_embeddings,
    CosineScorer, run_verification, ArcFaceHead,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- IMU augmentation (ported from the supervisors' age/gender training) ------
# Conservative, tremor-preserving augmentation on a normalised (28, 300) window.
# Doubles as extra contrastive "views" for SupCon and improves generalisation.
def _augment_window(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    if np.random.random() < 0.6:                      # 1. small Gaussian sensor noise
        x += np.random.normal(0.0, 0.05, x.shape).astype(np.float32)
    if np.random.random() < 0.6:                      # 2. amplitude scaling (dynamic + tremor chans)
        scale = np.random.uniform(0.90, 1.10)
        x[:6] *= scale; x[15:21] *= scale; x[21:27] *= scale
    if np.random.random() < 0.15:                     # 3. channel dropout
        x[np.random.randint(0, x.shape[0])] = 0.0
    if np.random.random() < 0.3:                      # 4. small temporal shift (±5 samples)
        x = np.roll(x, np.random.randint(-5, 6), axis=1)
    if np.random.random() < 0.10:                     # 5. sub-3 Hz band masking (protects tremor)
        T = x.shape[1]; band_w = np.random.randint(2, max(3, (T // 2 + 1) // 20))
        safe_max = 43 - band_w
        if safe_max > 0:
            f0 = np.random.randint(0, safe_max + 1)
            spec = np.fft.rfft(x, axis=1); spec[:, f0:f0 + band_w] = 0.0
            x = np.fft.irfft(spec, n=T, axis=1).astype(np.float32)
    return x


# --- identity-labelled window dataset + balanced P×K sampler ----------------

class WindowDataset(Dataset):
    """Each item: (window (28,300) tensor, identity label int)."""

    def __init__(self, data: VerificationData, subject_ids: List[str], augment: bool = False):
        self.data = data
        self.augment = augment
        self.index: List[Tuple[str, int]] = data.all_windows_index(subject_ids)
        identities = sorted({sid for sid, _ in self.index})
        self.label_of: Dict[str, int] = {s: i for i, s in enumerate(identities)}
        self.n_classes = len(identities)
        self.rows_by_label: Dict[int, List[int]] = {i: [] for i in range(self.n_classes)}
        for row, (sid, _) in enumerate(self.index):
            self.rows_by_label[self.label_of[sid]].append(row)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, row: int):
        sid, start = self.index[row]
        w = self.data.get_windows(sid, np.array([start], dtype=np.int64))[0]
        if self.augment:
            w = _augment_window(w)
        return torch.from_numpy(w), self.label_of[sid]


class IdentityBalancedSampler(Sampler):
    """Each batch = P subjects × K windows, so SupCon always has positive pairs."""

    def __init__(self, ds: WindowDataset, subjects_per_batch, windows_per_subject,
                 num_batches, seed=42):
        self.ds = ds
        self.P, self.K = subjects_per_batch, windows_per_subject
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed)
        self.labels = [l for l, rows in ds.rows_by_label.items() if rows]

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for lab in self.rng.choice(self.labels, size=min(self.P, len(self.labels)),
                                       replace=False):
                rows = self.ds.rows_by_label[lab]
                sel = self.rng.choice(rows, size=self.K, replace=len(rows) < self.K)
                batch += [int(r) for r in sel]
            yield batch


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch], dtype=torch.long))


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("train")
    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    log.info("Device %s | model %s | seed %d", device, args.model, args.seed)
    # Fixed-size inputs (B,28,300) → let cuDNN pick the fastest conv algorithms.
    torch.backends.cudnn.benchmark = True

    data = VerificationData(args.cache_dir, args.split_file,
                            [Path(d) for d in args.data_dirs], seed=args.seed,
                            enroll_ratio=args.enroll_ratio,
                            block_seconds=args.block_seconds, gap_seconds=args.gap_seconds)

    train_ids = data.prepare_cache(data.split["train"])
    if not train_ids:
        raise RuntimeError("No train subjects cached — check --data_dirs / --split_file.")
    data.fit_channel_stats(train_ids)

    ds = WindowDataset(data, train_ids, augment=args.augment)
    log.info("Dataset: %d identities, %d windows. (augment=%s, loss=%s)",
             ds.n_classes, len(ds), args.augment, args.loss)
    data._mmap.clear()                                   # workers reopen lazily
    sampler = IdentityBalancedSampler(ds, args.subjects_per_batch,
                                      args.windows_per_subject,
                                      args.batches_per_epoch, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.workers > 0))

    model = build_model(args.model, hidden_size=args.hidden_size,
                        embed_dim=args.embed_dim).to(device)
    params = list(model.parameters())

    # ArcFace adds a TRAIN-ONLY classification head over the train identities.
    # Its parameters are optimised alongside the encoder, then discarded at test.
    arcface = None
    if args.loss == "arcface":
        arcface = ArcFaceHead(args.embed_dim, ds.n_classes,
                              s=args.arc_s, m=args.arc_m).module.to(device)
        params += list(arcface.parameters())

    log.info("Trainable parameters: %d", sum(p.numel() for p in params if p.requires_grad))

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    # --out defaults to a model+loss-specific name so runs never overwrite each other.
    ckpt_path = Path(args.out or f"./checkpoints_verification/{args.model}_{args.loss}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_eer = float("inf")

    # Mixed precision (fp16) on GPU: big speedup + less VRAM. The heavy conv/matmul
    # run in fp16 inside autocast; the loss math is done in fp32 (cast emb below)
    # for numerical stability. Disable with --no_amp.
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    log.info("Mixed precision (AMP): %s | workers: %d", use_amp, args.workers)

    log.info("Training %d epochs (%d batches/epoch, batch=%d)...", args.epochs,
             args.batches_per_epoch, args.subjects_per_batch * args.windows_per_subject)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if arcface is not None:
            arcface.train()
        total = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                emb = model(x)
            emb = emb.float()                       # loss in fp32 for stability
            if arcface is not None:
                loss = arcface(emb, y)
            else:
                loss = supcon_loss(emb, y, args.temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)              # unscale before grad clipping
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()
        msg = (f"Epoch {epoch:03d}/{args.epochs} | {args.loss} {total/max(1,len(loader)):.4f} "
               f"| LR {optimizer.param_groups[0]['lr']:.2e}")

        val_eer = float("nan")
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            try:
                val_ids = data.prepare_cache(data.split["val"])
                emb = compute_deep_embeddings(model, data, val_ids, device, args.batch_size,
                                              agg_windows=args.agg_windows)
                res = run_verification(emb, CosineScorer(), seed=args.seed,
                                       impostors_per_user=args.impostors_per_user)
                val_eer = res.eer
                msg += f" | val EER {val_eer*100:.2f}% rank1 {res.rank1*100:.1f}%"
            except Exception as e:                                   # noqa: BLE001
                log.warning("val EER skipped: %s", e)

        is_best = val_eer == val_eer and val_eer < best_eer
        if is_best:
            best_eer = val_eer; msg += "  ★ best"
        log.info(msg)

        ckpt = {"model": args.model, "loss": args.loss, "network": model.state_dict(),
                "channel_mean": data.channel_mean, "channel_std": data.channel_std,
                "hidden_size": args.hidden_size, "embed_dim": args.embed_dim,
                "epoch": epoch, "val_eer": val_eer, "args": vars(args)}
        torch.save(ckpt, ckpt_path)
        if is_best:
            torch.save(ckpt, ckpt_path.with_suffix(".best.pt"))

    log.info("Done. Best val EER: %.2f%% → %s",
             best_eer * 100 if best_eer < float("inf") else float("nan"), ckpt_path)


def build_argparser():
    p = argparse.ArgumentParser(description="Train verification embedding (B1/B2/M1)")
    p.add_argument("--model", choices=["b1", "b2", "m1", "m2"], default="m1",
                   help="b1=CNN, b2=GRU, m1=SE-CNN, m2=SE-CNN on slow/fast(tremor) split")
    p.add_argument("--loss", choices=["supcon", "arcface"], default="supcon",
                   help="supcon (M1a) or arcface (M1b)")
    p.add_argument("--data_dirs", nargs="+", default=list(DEFAULT_DATA_DIRS))
    p.add_argument("--split_file", default="split_ids.json")
    p.add_argument("--cache_dir", default="./cache_verification")
    p.add_argument("--out", default=None,
                   help="checkpoint path; default ./checkpoints_verification/<model>_<loss>.pt")
    # ── Intensive training defaults (tuned for ~8 GB GTX 2080; scale up on a server) ──
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--subjects_per_batch", type=int, default=32, help="P in P×K batch (more = harder negatives)")
    p.add_argument("--windows_per_subject", type=int, default=8, help="K in P×K batch")
    p.add_argument("--batches_per_epoch", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--arc_s", type=float, default=30.0, help="ArcFace scale")
    p.add_argument("--arc_m", type=float, default=0.35, help="cosine (CosFace/AM-Softmax) margin")
    p.add_argument("--augment", action="store_true", default=True,
                   help="IMU augmentation during training (on by default)")
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM_DEFAULT)
    p.add_argument("--enroll_ratio", type=float, default=0.8,
                   help="fraction of 2-min blocks assigned to enroll (rest = verify)")
    p.add_argument("--block_seconds", type=float, default=120.0,
                   help="block size for the interleaved enroll/verify split")
    p.add_argument("--gap_seconds", type=float, default=15.0,
                   help="guard gap dropped around each enroll<->verify boundary")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader workers (parallel window loading; raise/lower to your CPU)")
    p.add_argument("--amp", action="store_true",
                   help="enable fp16 mixed precision. OFF by default: this FFT-heavy "
                        "model overflows in fp16 (NaN loss) unless the FFT branches are "
                        "forced to fp32. Leave off unless you've verified stability.")
    p.add_argument("--batch_size", type=int, default=256, help="inference batch for val eval")
    p.add_argument("--eval_every", type=int, default=10, help="0 disables val EER")
    p.add_argument("--agg_windows", type=int, default=1,
                   help="multi-window aggregation for val EER (1 = single-window)")
    p.add_argument("--impostors_per_user", type=int, default=2000)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
