"""
train.py
--------
Entry point for training the probe placement RL agent.

Usage:
    python train.py                        # sanity check (fast preset)
    python train.py --preset fast          # quick experiment
    python train.py --preset full          # full training run
    python train.py --preset full --run-name my_experiment
    python train.py --resume checkpoints/probe_placement_full_best.pt
"""

import argparse
import sys
import torch
import os
from src.config.train_config import get_config, describe
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train probe placement RL agent")
    parser.add_argument(
        "--preset", type=str, default="fast",
        choices=["debug", "fast", "full"],
        help="Training preset (default: fast)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this run (default: preset name + timestamp)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (default: auto-detect)"
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for logs (default: logs/)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for checkpoints (default: checkpoints/)"
    )
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed (default: None)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"[train.py] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[train.py] VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
    else:
        device = "cpu"
        print("[train.py] No GPU detected — running on CPU")


    # ── Config ───────────────────────────────────────────────────────────────
    cfg = get_config(args.preset)
    cfg.device          = device
    cfg.use_tensorboard = not args.no_tensorboard
    cfg.log_dir         = args.log_dir
    cfg.checkpoint_dir  = args.checkpoint_dir

    if args.run_name is not None:
        cfg.run_name = args.run_name

    if args.resume is not None:
        cfg.resume_from = args.resume

    if args.seed is not None:
        cfg.seed = args.seed

    # ── Print config ─────────────────────────────────────────────────────────
    print()
    describe(cfg)
    print()

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(cfg)
    trainer.train()

    print()
    print(f"[train.py] Done. Logs: {args.log_dir}/{cfg.run_name}.jsonl")
    print(f"[train.py] Best checkpoint: {args.checkpoint_dir}/{cfg.run_name}_best.pt")


if __name__ == "__main__":
    main()