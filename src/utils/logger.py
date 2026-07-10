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
import math
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
            From EpisodeTracker.aggregate(). Keys include: mean_reward,
            mean_ep_len, mean_coverage (binary), mean_coverage_frac,
            mean_blind_rate, mean_detection_rate, mean_blind_viol_rate,
            mean_n_violations, mean_probe_count, survival_rate,
            mean_num_nodes, mean_num_probeable, n_episodes.
            Violation-ratio metrics may be NaN when no violations occurred.
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

        # Sanitise non-finite values (NaN/Inf) → None so the JSONL stays
        # valid JSON (NaN is not part of the JSON spec and breaks strict parsers).
        record = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                  for k, v in record.items()}

        # JSON Lines
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

        # TensorBoard — grouped into sections using tag/name format
        if self._tb_writer is not None:
            def _tb(tag, val, step=iteration):
                # skip None / NaN / Inf so TB curves don't break
                if val is None:
                    return
                if isinstance(val, float) and not math.isfinite(val):
                    return
                self._tb_writer.add_scalar(tag, val, step)

            # ── Episode metrics ──────────────────────────────────────────
            ep = episode_metrics
            _tb("episode/mean_reward",       ep.get("mean_reward"))
            _tb("episode/coverage_any",      ep.get("mean_coverage"))       # binary: ≥1 SLO covered
            _tb("episode/coverage_frac",     ep.get("mean_coverage_frac"))  # |covered|/|coverable|
            _tb("episode/blind_rate",        ep.get("mean_blind_rate"))     # steps with ≥1 blind viol
            _tb("episode/detection_rate",    ep.get("mean_detection_rate"))    # covered/total violations
            _tb("episode/blind_violation_rate", ep.get("mean_blind_viol_rate"))# blind/total violations
            _tb("episode/mean_n_violations", ep.get("mean_n_violations"))
            _tb("episode/mean_probe_count",  ep.get("mean_probe_count"))
            _tb("episode/survival_rate",     ep.get("survival_rate"))
            _tb("episode/mean_ep_len",       ep.get("mean_ep_len"))
            _tb("episode/n_episodes",        ep.get("n_episodes"))
            _tb("episode/mean_num_nodes",     ep.get("mean_num_nodes"))
            _tb("episode/mean_num_probeable", ep.get("mean_num_probeable"))
            # ── PPO losses ───────────────────────────────────────────────
            _tb("ppo/total_loss",    ppo_stats.total_loss)
            _tb("ppo/policy_loss",   ppo_stats.policy_loss)
            _tb("ppo/value_loss",    ppo_stats.value_loss)
            _tb("ppo/entropy",       ppo_stats.entropy)
            # ── PPO diagnostics ──────────────────────────────────────────
            _tb("ppo/approx_kl",     ppo_stats.approx_kl)
            _tb("ppo/clip_fraction", ppo_stats.clip_fraction)
            _tb("ppo/explained_var", ppo_stats.explained_var)
            # ── Curriculum ───────────────────────────────────────────────
            ci = curriculum_info
            _tb("curriculum/K",             ci.get("K"))
            _tb("curriculum/stage",         ci.get("stage"))
            _tb("curriculum/learning_rate", ci.get("lr"))
            _tb("curriculum/entropy_coef",  ci.get("entropy_coef"))
            _tb("curriculum/promoted",      float(ci.get("promoted", False)))
            # ── Meta ─────────────────────────────────────────────────────
            _tb("meta/env_steps", env_steps)
            _tb("meta/elapsed_s", elapsed)

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
        em      = episode_metrics

        def _f(x):  # nan-safe short formatter for violation ratios
            return "  n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"

        print(
            f"[{iteration:5d}/{total_iters}] ({pct:5.1f}%) "
            f"steps={env_steps:7,d} | "
            f"R={em.get('mean_reward', 0.0):7.3f} | "
            f"covF={em.get('mean_coverage_frac', 0.0):.3f} | "
            f"det={_f(em.get('mean_detection_rate'))} | "
            f"blindV={_f(em.get('mean_blind_viol_rate'))} | "
            f"probes={em.get('mean_probe_count', 0.0):.1f}"
            f"/{em.get('mean_num_probeable', 0.0):.0f}"
            f"(n={em.get('mean_num_nodes', 0.0):.0f}) | "
            f"surv={em.get('survival_rate', 0.0):.2f} | "
            f"L={ppo_stats.total_loss:7.4f} "
            f"(pol={ppo_stats.policy_loss:.4f} "
            f"val={ppo_stats.value_loss:.4f} "
            f"ent={ppo_stats.entropy:.3f}) | "
            f"kl={ppo_stats.approx_kl:.5f} "
            f"clip={ppo_stats.clip_fraction:.3f} | "
            f"K={curriculum_info.get('K', '?')} "
            f"stage={curriculum_info.get('stage', 0)} | "
            f"t={elapsed:.0f}s"
        )

    def close(self) -> None:
        """Flush and close all log files."""
        self._jsonl_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()