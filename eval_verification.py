"""
eval_verification.py
====================
Shared evaluation CLI for all three baselines. Builds embeddings for the chosen
model, runs the ONE verification harness in verification.py, and prints EER,
FAR/FRR@EER, per-activity-state EER and rank-1, with the within-session caveat.

  B0 — no checkpoint; runs cosine-to-enrolled-mean AND a per-user one-class SVM.
  B1/B2 — load a checkpoint from train_verification.py; cosine scorer.

Examples
--------
  python eval_verification.py --model b0 --split test \
      --data_dirs dataset --cache_dir cache_verification

  python eval_verification.py --model b1 --split test \
      --checkpoint checkpoints_verification/b1.pt \
      --data_dirs dataset --cache_dir cache_verification
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from verification import (
    VerificationData, DEFAULT_DATA_DIRS, build_model,
    compute_deep_embeddings, compute_b0_embeddings, fit_b0_scaler,
    CosineScorer, OCSVMScorer, run_verification, format_report, result_to_dict,
)


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("eval")

    data = VerificationData(args.cache_dir, args.split_file,
                            [Path(d) for d in args.data_dirs], seed=args.seed,
                            enroll_ratio=args.enroll_ratio,
                            block_seconds=args.block_seconds, gap_seconds=args.gap_seconds)

    eval_ids = data.prepare_cache(data.split[args.split])
    if len(eval_ids) < 2:
        raise RuntimeError(f"Need >=2 cached {args.split} subjects; got {len(eval_ids)}.")
    log.info("Evaluating %d %s subjects.", len(eval_ids), args.split)

    config = {"model": args.model, "split": args.split, "seed": args.seed,
              "n_subjects": len(eval_ids), "enroll_ratio": args.enroll_ratio,
              "block_seconds": args.block_seconds, "gap_seconds": args.gap_seconds,
              "agg_windows": args.agg_windows,
              "impostors_per_user": args.impostors_per_user}

    results = []
    scorers_used = []          # remember which scorers ran (for the optional pooled block)
    pooled_src = None          # single-window embeddings, captured only if --pooled
    if args.model == "b0":
        train_ids = data.prepare_cache(data.split["train"])   # stats + scaler from TRAIN only
        data.fit_channel_stats(train_ids)
        # data.fit_activity(eval_ids)   # ASA disabled
        scaler = fit_b0_scaler(data, train_ids, seed=args.seed)
        emb = compute_b0_embeddings(data, eval_ids, scaler, agg_windows=args.agg_windows)
        scorers_used = [CosineScorer(), OCSVMScorer(nu=args.ocsvm_nu)]
        for scorer in scorers_used:
            log.info("scorer: %s", scorer.name)
            results.append(run_verification(emb, scorer, args.seed,
                                            args.impostors_per_user, args.rank1_max_attempts))
        if args.pooled:
            pooled_src = compute_b0_embeddings(data, eval_ids, scaler, agg_windows=1)
    else:
        import torch
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for B1/B2/M1.")
        device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                              else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_model(ckpt.get("model", args.model),
                            hidden_size=ckpt.get("hidden_size", args.hidden_size),
                            embed_dim=ckpt.get("embed_dim", 128)).to(device)
        model.load_state_dict(ckpt["network"])
        data.set_channel_stats(ckpt["channel_mean"], ckpt["channel_std"])
        # data.fit_activity(eval_ids)   # ASA disabled
        emb = compute_deep_embeddings(model, data, eval_ids, device, args.batch_size,
                                      agg_windows=args.agg_windows)
        config["checkpoint"] = args.checkpoint
        scorers_used = [CosineScorer()] + ([OCSVMScorer(nu=args.ocsvm_nu)] if args.ocsvm else [])
        for scorer in scorers_used:
            results.append(run_verification(emb, scorer, args.seed,
                                            args.impostors_per_user, args.rank1_max_attempts))
        if args.pooled:
            pooled_src = compute_deep_embeddings(model, data, eval_ids, device,
                                                 args.batch_size, agg_windows=1)

    print("\n" + format_report(args.model, results, config))

    # ======================================================================
    # POOLED AGGREGATION (optional; comment out this whole block to disable)
    # ----------------------------------------------------------------------
    # Pools --pooled_n verify windows sampled across the WHOLE session per
    # subject (ignores temporal contiguity), reported SEPARATELY with its own
    # EER and rank-1. Less realistic than contiguous aggregation; large N
    # inflates the numbers, so read it as an upper-ish bound, not the headline.
    pooled_results = []
    if args.pooled and pooled_src is not None:
        from verification import pool_verify
        pooled = pool_verify(pooled_src, args.pooled_n, seed=args.seed)
        for scorer in scorers_used:
            pooled_results.append(run_verification(pooled, scorer, args.seed,
                                                   args.impostors_per_user, args.rank1_max_attempts))
        pooled_cfg = {**config, "agg_mode": "POOLED", "pooled_n": args.pooled_n}
        print("\n" + format_report(f"{args.model} [POOLED N={args.pooled_n}]",
                                   pooled_results, pooled_cfg))
    # ======================================================================

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"config": config,
                       "results": [result_to_dict(r) for r in results],
                       "pooled_results": [result_to_dict(r) for r in pooled_results]},
                      f, indent=2, default=str)
        log.info("Wrote results → %s", args.out_json)


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate B0/B1/B2 verification baselines")
    p.add_argument("--model", choices=["b0", "b1", "b2", "m1", "m2"], required=True)
    p.add_argument("--checkpoint", default=None, help="required for b1/b2/m1/m2")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--data_dirs", nargs="+", default=list(DEFAULT_DATA_DIRS))
    p.add_argument("--split_file", default="split_ids.json")
    p.add_argument("--cache_dir", default="./cache_verification")
    p.add_argument("--out_json", default=None)
    p.add_argument("--enroll_ratio", type=float, default=0.8,
                   help="fraction of 2-min blocks assigned to enroll (rest = verify)")
    p.add_argument("--block_seconds", type=float, default=120.0,
                   help="block size for the interleaved enroll/verify split")
    p.add_argument("--gap_seconds", type=float, default=15.0,
                   help="guard gap dropped around each enroll<->verify boundary")
    p.add_argument("--agg_windows", type=int, default=1,
                   help="multi-window aggregation: average N consecutive windows per "
                        "attempt (1 = single-window, current behaviour)")
    p.add_argument("--pooled", action="store_true",
                   help="ALSO report a labelled POOLED aggregation (averages "
                        "--pooled_n verify windows sampled across the whole session, "
                        "ignoring contiguity) with its own EER + rank-1")
    p.add_argument("--pooled_n", type=int, default=64,
                   help="windows per pooled attempt (used only with --pooled)")
    p.add_argument("--impostors_per_user", type=int, default=2000)
    p.add_argument("--rank1_max_attempts", type=int, default=4000)
    p.add_argument("--ocsvm", action="store_true", help="also run OC-SVM for b1/b2")
    p.add_argument("--ocsvm_nu", type=float, default=0.1)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
