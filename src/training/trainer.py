"""
trainer.py
----------
Outer training loop for the probe placement PPO agent.

Orchestrates:
  env ← ProbeEnv
  policy ← PolicyNetwork (GNN + actor-critic)
  buffer ← RolloutBuffer
  ppo ← PPO
  curriculum ← CurriculumScheduler
  logger ← Logger
  checkpointer ← Checkpointer

─────────────────────────────────────────────────────────────────
Training loop (per iteration)
─────────────────────────────────────────────────────────────────
  1. Collect rollout_len transitions:
       while not buffer.is_full:
           action, log_prob, value = policy.act(obs, ep)
           obs, reward, done = env.step(action)
           buffer.add(...)
           if done: obs = env.reset()

  2. Bootstrap: last_value = V(s_T)  (0 if terminal)

  3. compute_gae(last_value)

  4. stats = ppo.update(buffer)

  5. buffer.reset()

  6. curriculum.step(mean_reward, iteration)

  7. logger.log(...)  every log_interval

  8. checkpointer.save(...)  every save_interval

  9. evaluate(...)  every eval_interval

─────────────────────────────────────────────────────────────────
Episode tracking
─────────────────────────────────────────────────────────────────
The trainer tracks per-episode statistics across rollout boundaries:
  - episode reward sum
  - episode length
  - coverage rate (fraction of timesteps with full SLO coverage)
  - blind violation rate

These are aggregated over complete episodes in the rollout and
reported as mean_reward, mean_ep_len, mean_coverage, mean_blind_rate.

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  TrainerConfig           — all training hyperparameters
  Trainer(config)
    .train()              — run the full training loop
    .collect_rollout()    — single rollout (used internally + by tests)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.simulator.env.reward import RewardConfig
from src.models.gnn_encoder import GNNConfig
from src.models.policy_network import PolicyConfig, PolicyNetwork
from src.training.rollout_buffer import RolloutBuffer
from src.training.ppo import PPO, PPOConfig, PPOStats
from src.training.curriculum import CurriculumScheduler, CurriculumConfig
from src.utils.logger import Logger
from src.utils.checkpointing import Checkpointer


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    All hyperparameters for one training run.

    Attributes
    ----------
    run_name : str
        Unique identifier for this run (used in log/checkpoint filenames).
    total_iterations : int
        Number of collect+update cycles.
    rollout_len : int
        Transitions collected per iteration.
    log_interval : int
        Print stdout summary every N iterations.
    eval_interval : int
        Run evaluation every N iterations (0 = never).
    save_interval : int
        Save checkpoint every N iterations (0 = never).
    device : str
        "cpu" or "cuda".
    seed : int
        Global random seed.
    log_dir : str
        Directory for log files.
    checkpoint_dir : str
        Directory for checkpoint files.
    use_tensorboard : bool
        Whether to log to TensorBoard.
    resume_from : str or None
        Path to checkpoint to resume from.

    Sub-configs (nested):
        env_config, gnn_config, policy_config, ppo_config, curriculum_config
    """
    run_name:          str  = "probe_placement"
    total_iterations:  int  = 500
    rollout_len:       int  = 512
    log_interval:      int  = 10
    eval_interval:     int  = 50
    save_interval:     int  = 50
    device:            str  = "cpu"
    seed:              int  = 42
    log_dir:           str  = "logs"
    checkpoint_dir:    str  = "checkpoints"
    use_tensorboard:   bool = False
    resume_from:       Optional[str] = None
    debug_print:       bool = False  # print per-step trace during collect_rollout

    # Sub-configs
    env_config:        ProbeEnvConfig  = field(default_factory=ProbeEnvConfig)
    gnn_config:        GNNConfig       = field(default_factory=GNNConfig)
    policy_config:     PolicyConfig    = field(default_factory=PolicyConfig)
    ppo_config:        PPOConfig       = field(default_factory=PPOConfig)
    curriculum_config: CurriculumConfig = field(default_factory=CurriculumConfig)

    def __post_init__(self):
        assert self.total_iterations > 0
        assert self.rollout_len      > 0
        assert self.log_interval     > 0
        assert self.ppo_config.batch_size <= self.rollout_len, \
            "batch_size must be <= rollout_len"


# ---------------------------------------------------------------------------
# EpisodeTracker  — lightweight per-episode stat accumulator
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """Accumulates per-episode statistics across rollout boundaries."""

    def __init__(self):
        self.reset_current()
        self.completed: List[dict] = []

    def reset_current(self):
        self._reward        = 0.0
        self._length        = 0
        self._coverage      = 0.0   # fraction of steps with >= 1 SLO covered
        self._blind         = 0.0   # fraction of steps with >= 1 blind violation
        self._probe_sum     = 0.0   # sum of probe counts per step
        self._survived      = False # did this episode reach T without exceeding K
        self._num_nodes     = 0     # present nodes in this episode graph
        self._num_probeable = 0     # probeable nodes in this episode graph

    def step(self, reward: float, info: dict) -> None:
        self._reward     += reward
        self._length     += 1
        rb = info.get("reward_breakdown", None)
        if rb is not None:
            self._coverage += 1.0 if len(rb.covered_slos) > 0 else 0.0
            self._blind    += 1.0 if len(rb.blind_slos)   > 0 else 0.0
        probe_set = info.get("probe_set", set())
        self._probe_sum += len(probe_set)

    def end_episode(self, survived: bool = False) -> None:
        L = max(self._length, 1)
        self.completed.append({
            "reward":        self._reward,
            "length":        self._length,
            "coverage":      self._coverage / L,
            "blind_rate":    self._blind    / L,
            "probe_count":   self._probe_sum / L,
            "survived":      float(survived),
            "num_nodes":     self._num_nodes,
            "num_probeable": self._num_probeable,
        })
        self.reset_current()

    def aggregate(self) -> dict:
        """Return mean stats over all completed episodes since last call."""
        if not self.completed:
            return {
                "mean_reward":        0.0,
                "mean_ep_len":        0.0,
                "mean_coverage":      0.0,
                "mean_blind_rate":    0.0,
                "mean_probe_count":   0.0,
                "survival_rate":      0.0,
                "mean_num_nodes":     0.0,
                "mean_num_probeable": 0.0,
                "n_episodes":         0,
            }
        result = {
            "mean_reward":        float(np.mean([e["reward"]        for e in self.completed])),
            "mean_ep_len":        float(np.mean([e["length"]        for e in self.completed])),
            "mean_coverage":      float(np.mean([e["coverage"]      for e in self.completed])),
            "mean_blind_rate":    float(np.mean([e["blind_rate"]    for e in self.completed])),
            "mean_probe_count":   float(np.mean([e["probe_count"]   for e in self.completed])),
            "survival_rate":      float(np.mean([e["survived"]      for e in self.completed])),
            "mean_num_nodes":     float(np.mean([e["num_nodes"]     for e in self.completed])),
            "mean_num_probeable": float(np.mean([e["num_probeable"] for e in self.completed])),
            "n_episodes":         len(self.completed),
        }
        self.completed.clear()
        return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full PPO training loop for the probe placement POMDP.

    Parameters
    ----------
    config : TrainerConfig
        All training hyperparameters.
    """

    def __init__(self, config: TrainerConfig):
        self.cfg    = config
        self.device = torch.device(config.device)

        # Reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # ── Components ────────────────────────────────────────────────
        self.env    = ProbeEnv(config.env_config)
        self.policy = PolicyNetwork(config.gnn_config, config.policy_config)
        self.policy.to(self.device)

        self.buffer = RolloutBuffer(
            rollout_len=config.rollout_len,
            device=self.device,
        )

        self.ppo = PPO(
            policy=self.policy,
            config=config.ppo_config,
            device=self.device,
        )

        self.curriculum = CurriculumScheduler(
            config=config.curriculum_config,
            ppo=self.ppo,
            env_config=config.env_config,
        )

        self.logger = Logger(
            run_name=config.run_name,
            log_dir=config.log_dir,
            use_tensorboard=config.use_tensorboard,
        )

        self.checkpointer = Checkpointer(
            run_name=config.run_name,
            save_dir=config.checkpoint_dir,
        )

        self.tracker = EpisodeTracker()

        # State that persists across rollouts
        self._obs:        Optional[dict] = None
        self._ep                         = None
        self._env_steps:  int            = 0
        self._start_iter: int            = 0

        # Resume from checkpoint if specified
        if config.resume_from is not None:
            meta = self.checkpointer.load(
                config.resume_from, self.policy, self.ppo.optimizer
            )
            self._start_iter  = meta["iteration"] + 1
            self._env_steps   = meta["env_steps"]
            self.curriculum._stage = meta["curriculum_stage"]
            print(f"[Trainer] Resumed from {config.resume_from} "
                  f"at iteration {meta['iteration']}")

        # Initial env reset
        self._reset_env()

    # ------------------------------------------------------------------
    # train()
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        print(f"[Trainer] Starting run '{self.cfg.run_name}' | "
              f"device={self.cfg.device} | "
              f"total_iters={self.cfg.total_iterations} | "
              f"rollout={self.cfg.rollout_len}")

        for iteration in range(self._start_iter, self.cfg.total_iterations):

            # ── 1. Collect rollout ────────────────────────────────────
            last_value = self.collect_rollout()

            # ── 2. GAE ───────────────────────────────────────────────
            self.buffer.compute_gae(
                last_value=last_value,
                gamma=0.99,
                gae_lambda=0.95,
            )

            # ── 3. PPO update ─────────────────────────────────────────
            ppo_stats = self.ppo.update(self.buffer)

            # ── 4. Reset buffer ───────────────────────────────────────
            self.buffer.reset()

            # ── 5. Episode metrics ────────────────────────────────────
            ep_metrics = self.tracker.aggregate()

            # ── 6. Curriculum ─────────────────────────────────────────
            curr_info = self.curriculum.step(
                mean_ep_reward=ep_metrics["mean_reward"],
                iteration=iteration,
                total_iters=self.cfg.total_iterations,
            )
            if curr_info["promoted"]:
                print(f"[Curriculum] Promoted to stage {curr_info['stage']} "
                      f"(K={curr_info['K']}) at iteration {iteration}")
                # Re-create env with updated K
                self._reset_env()

            # ── 7. Logging ────────────────────────────────────────────
            self.logger.log(
                iteration=iteration,
                env_steps=self._env_steps,
                ppo_stats=ppo_stats,
                curriculum_info=curr_info,
                episode_metrics=ep_metrics,
            )
            if iteration % self.cfg.log_interval == 0:
                self.logger.print_summary(
                    iteration=iteration,
                    total_iters=self.cfg.total_iterations,
                    env_steps=self._env_steps,
                    ppo_stats=ppo_stats,
                    curriculum_info=curr_info,
                    episode_metrics=ep_metrics,
                )

            # ── 8. Checkpoint ─────────────────────────────────────────
            if (self.cfg.save_interval > 0 and
                    iteration % self.cfg.save_interval == 0):
                path = self.checkpointer.save(
                    iteration=iteration,
                    env_steps=self._env_steps,
                    policy=self.policy,
                    optimizer=self.ppo.optimizer,
                    mean_reward=ep_metrics["mean_reward"],
                    curriculum_stage=self.curriculum.current_stage,
                )

        self.logger.close()
        print(f"[Trainer] Training complete. "
              f"Total env steps: {self._env_steps:,}")

    # ------------------------------------------------------------------
    # collect_rollout()
    # ------------------------------------------------------------------

    def collect_rollout(self) -> float:
        """
        Collect rollout_len transitions into the buffer.

        Returns
        -------
        last_value : float
            V(s_T) — bootstrapped value of the state after the last
            stored transition.  Used by compute_gae().
        """
        self.policy.eval()
        last_value = 0.0

        while not self.buffer.is_full:
            # ── Act ───────────────────────────────────────────────────
            action, log_prob, value, _ = self.policy.act(
                self._obs, self._ep, deterministic=False
            )

            # ── Step ──────────────────────────────────────────────────
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # ── Store ─────────────────────────────────────────────────
            self.buffer.add(
                obs_dict=self._obs,
                episode_graph=self._ep,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
            )
            self._env_steps += 1

            # ── Track episode stats ───────────────────────────────────
            self.tracker.step(reward, info)

            # ── Per-step debug print (active during debugging) ────────
            if getattr(self.cfg, 'debug_print', False):
                print(
                    f"t={self.env._t:3d} | "
                    f"{info['action_str']:<32} | "
                    f"probes={len(info['probe_set']):2d}/{self._ep.num_probeable} "
                    f"(nodes={self._ep.num_nodes}) | "
                    f"R={reward:+.3f} "
                    f"(obs={info['reward_breakdown'].observability:+.3f} "
                    f"blind={info['reward_breakdown'].blind_violation:+.3f}) | "
                    f"cum_blind={info['cum_blind']}"
                )

            # ── Episode boundary ──────────────────────────────────────
            if done:
                survived = truncated and not terminated
                self.tracker.end_episode(survived=survived)
                self._reset_env()
                # Record graph size for the new episode
                self.tracker._num_nodes     = self._ep.num_nodes
                self.tracker._num_probeable = self._ep.num_probeable
                last_value = 0.0
            else:
                self._obs = obs_next
                last_value = value   # will be overwritten until loop ends

        # Bootstrap: get V(s_T) for the state AFTER the last transition
        if not (terminated if 'terminated' in dir() else False):
            with torch.no_grad():
                _, _, last_value, _ = self.policy.act(
                    self._obs, self._ep, deterministic=False
                )

        return float(last_value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_env(self) -> None:
        """Reset env and update obs/ep state."""
        self._obs, _ = self.env.reset()
        self._ep     = self.env.current_graph
        # Keep tracker informed of current episode graph size
        self.tracker._num_nodes     = self._ep.num_nodes
        self.tracker._num_probeable = self._ep.num_probeable