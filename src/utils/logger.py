"""
logger.py
---------
Lightweight structured logger for training metrics.

Writes to:
  - stdout (human-readable, filtered by log_interval)
  - a JSON Lines file (one record per iteration, for analysis)
  - TensorBoard (optional, if tensorboard is installed)

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  Logger(run_name, log_dir, use_tensorboard)
    .log(iteration, env_steps, ppo_stats, curriculum_info,
         episode_metrics)
    .close()
"""

import json
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

from src.training.ppo import PPOStats


class Logger:
    """
    Structured logger for training metrics.

    Parameters
    ----------
    run_name : str
        Unique name for this training run.
    log_dir : str
        Directory for log files.
    use_tensorboard : bool
        If True, attempt to log to TensorBoard.
    """

    def __init__(
        self,
        run_name:        str,
        log_dir:         str  = "logs",
        use_tensorboard: bool = False,
    ):
        self.run_name = run_name
        self.log_dir  = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # JSON Lines file — one dict per iteration
        self._jsonl_path = os.path.join(log_dir, f"{run_name}.jsonl")
        self._jsonl_file = open(self._jsonl_path, "a")

        # TensorBoard (optional)
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tb", run_name)
                self._tb_writer = SummaryWriter(tb_dir)
            except ImportError:
                print("[Logger] TensorBoard not available — skipping.")

        self._start_time = time.time()

    def log(
        self,
        iteration:        int,
        env_steps:        int,
        ppo_stats:        PPOStats,
        curriculum_info:  dict,
        episode_metrics:  dict,
    ) -> None:
        """
        Log one training iteration.

        Parameters
        ----------
        iteration : int
        env_steps : int
            Total environment steps taken so far.
        ppo_stats : PPOStats
            From PPO.update().
        curriculum_info : dict
            From CurriculumScheduler.step().
        episode_metrics : dict
            Keys: mean_reward, mean_ep_len, mean_coverage,
                  mean_blind_rate, n_episodes.
        """
        elapsed = time.time() - self._start_time

        record = {
            "iteration":   iteration,
            "env_steps":   env_steps,
            "elapsed_s":   round(elapsed, 1),
            # PPO stats
            "total_loss":    round(ppo_stats.total_loss,    4),
            "policy_loss":   round(ppo_stats.policy_loss,   4),
            "value_loss":    round(ppo_stats.value_loss,    4),
            "entropy":       round(ppo_stats.entropy,       4),
            "approx_kl":     round(ppo_stats.approx_kl,    6),
            "clip_fraction": round(ppo_stats.clip_fraction, 4),
            "explained_var": round(ppo_stats.explained_var, 4),
            "n_updates":     ppo_stats.n_updates,
            # Curriculum
            **{f"curr_{k}": v for k, v in curriculum_info.items()},
            # Episode metrics
            **{f"ep_{k}": v for k, v in episode_metrics.items()},
        }

        # JSON Lines
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

        # TensorBoard — grouped into sections using tag/name format
        if self._tb_writer is not None:
            # ── Episode metrics ──────────────────────────────────────────
            ep = episode_metrics
            self._tb_writer.add_scalar("episode/mean_reward",      ep.get("mean_reward",      0.0), iteration)
            self._tb_writer.add_scalar("episode/mean_weighted_coverage",    ep.get("mean_coverage",    0.0), iteration)
            self._tb_writer.add_scalar("episode/mean_blind_rate",  ep.get("mean_blind_rate",  0.0), iteration)
            self._tb_writer.add_scalar("episode/mean_probe_count", ep.get("mean_probe_count", 0.0), iteration)
            self._tb_writer.add_scalar("episode/survival_rate",    ep.get("survival_rate",    0.0), iteration)
            self._tb_writer.add_scalar("episode/mean_ep_len",      ep.get("mean_ep_len",      0.0), iteration)
            self._tb_writer.add_scalar("episode/n_episodes",       ep.get("n_episodes",       0),   iteration)
            self._tb_writer.add_scalar("episode/mean_num_nodes",     ep.get("mean_num_nodes",     0.0), iteration)
            self._tb_writer.add_scalar("episode/mean_num_probeable", ep.get("mean_num_probeable", 0.0), iteration)
            self._tb_writer.add_scalar("episode/mean_detection_rate",ep.get("mean_detection_rate", 1.0), iteration)
            # ── PPO losses ───────────────────────────────────────────────
            self._tb_writer.add_scalar("ppo/total_loss",    ppo_stats.total_loss,    iteration)
            self._tb_writer.add_scalar("ppo/policy_loss",   ppo_stats.policy_loss,   iteration)
            self._tb_writer.add_scalar("ppo/value_loss",    ppo_stats.value_loss,    iteration)
            self._tb_writer.add_scalar("ppo/entropy",       ppo_stats.entropy,       iteration)
            # ── PPO diagnostics ──────────────────────────────────────────
            self._tb_writer.add_scalar("ppo/approx_kl",     ppo_stats.approx_kl,     iteration)
            self._tb_writer.add_scalar("ppo/clip_fraction", ppo_stats.clip_fraction, iteration)
            self._tb_writer.add_scalar("ppo/explained_var", ppo_stats.explained_var, iteration)
            # ── Curriculum ───────────────────────────────────────────────
            ci = curriculum_info
            self._tb_writer.add_scalar("curriculum/K",             ci.get("K",            50),           iteration)
            self._tb_writer.add_scalar("curriculum/stage",         ci.get("stage",        0),            iteration)
            self._tb_writer.add_scalar("curriculum/learning_rate", ci.get("lr",           3e-4),         iteration)
            self._tb_writer.add_scalar("curriculum/entropy_coef",  ci.get("entropy_coef", 0.05),         iteration)
            self._tb_writer.add_scalar("curriculum/promoted",      float(ci.get("promoted", False)),     iteration)
            # ── Meta ─────────────────────────────────────────────────────
            self._tb_writer.add_scalar("meta/env_steps", env_steps, iteration)
            self._tb_writer.add_scalar("meta/elapsed_s", elapsed,   iteration)

    def print_summary(
        self,
        iteration:       int,
        total_iters:     int,
        env_steps:       int,
        ppo_stats:       PPOStats,
        curriculum_info: dict,
        episode_metrics: dict,
    ) -> None:
        """Print a human-readable summary to stdout."""
        elapsed = time.time() - self._start_time
        pct     = 100 * iteration / max(total_iters, 1)

        print(
            f"[{iteration:5d}/{total_iters}] ({pct:5.1f}%) "
            f"steps={env_steps:7,d} | "
            f"R={episode_metrics.get('mean_reward', 0.0):7.3f} | "
            f"Wcov={episode_metrics.get('mean_coverage', 0.0):.3f} | "
            f"blind={episode_metrics.get('mean_blind_rate', 0.0):.3f} | "
            f"probes={episode_metrics.get('mean_probe_count', 0.0):.1f}"
            f"/{episode_metrics.get('mean_num_probeable', 0.0):.0f}"
            f"(n={episode_metrics.get('mean_num_nodes', 0.0):.0f}) | "
            f"surv={episode_metrics.get('survival_rate', 0.0):.2f} | "
            f"L={ppo_stats.total_loss:7.4f} "
            f"(pol={ppo_stats.policy_loss:.4f} "
            f"val={ppo_stats.value_loss:.4f} "
            f"ent={ppo_stats.entropy:.3f}) | "
            f"kl={ppo_stats.approx_kl:.5f} "
            f"clip={ppo_stats.clip_fraction:.3f} | "
            f"K={curriculum_info.get('K', '?')} "
            f"stage={curriculum_info.get('stage', 0)} | "
            f"t={elapsed:.0f}s"
            f"det={episode_metrics.get('mean_detection_rate', 1.0):.3f} | "
        )

    def close(self) -> None:
        """Flush and close all log files."""
        self._jsonl_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()