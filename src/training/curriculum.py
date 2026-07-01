"""
curriculum.py
-------------
Curriculum scheduler for the probe placement POMDP.

The curriculum progressively tightens training difficulty to prevent
the agent from finding degenerate solutions on easy episodes.

─────────────────────────────────────────────────────────────────
What the curriculum controls
─────────────────────────────────────────────────────────────────

  K  — blind violation budget (ProbeEnvConfig.blind_violation_budget)
       Starts large (easy: agent allowed many blind violations) and
       decreases toward the target K as training progresses.

  entropy_coef — PPO exploration bonus weight
       Starts high (explore broadly) and anneals toward a low value
       to exploit the learned policy.

  learning_rate — Adam lr
       Optionally decayed on a cosine or linear schedule.

─────────────────────────────────────────────────────────────────
Promotion criterion
─────────────────────────────────────────────────────────────────
The curriculum advances a stage when the agent achieves a minimum
mean episode reward above a threshold for a sustained window of
iterations.  This prevents premature promotion due to noisy rewards.

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  CurriculumConfig        — schedule hyperparameters
  CurriculumScheduler(config, ppo, env_config)
    .step(mean_ep_reward, iteration) → dict of updated values
    .current_K            → current blind violation budget
    .current_stage        → stage index (0 = easiest)
    .should_promote       → True if promotion criteria met
"""

from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional, Tuple

from src.training.ppo import PPO
from src.simulator.env.probe_env import ProbeEnvConfig


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    """
    Curriculum schedule hyperparameters.

    Attributes
    ----------
    k_stages : List[int]
        Sequence of K values from easiest to hardest.
        E.g. [50, 20, 10, 5, 2] — agent must master each stage before
        progressing to the next tighter budget.
    entropy_coef_start : float
        Initial entropy coefficient (high exploration).
    entropy_coef_end : float
        Final entropy coefficient (low exploration at last stage).
    lr_start : float
        Initial learning rate.
    lr_end : float
        Final learning rate (after all stages, or at total_iterations).
    promotion_threshold : float
        Minimum mean episode reward required to advance to the next stage.
    promotion_window : int
        Number of consecutive iterations above threshold before promotion.
    """
    k_stages:             List[int] = field(default_factory=lambda: [50, 20, 10, 5, 2])
    entropy_coef_start:   float     = 0.05
    entropy_coef_end:     float     = 0.005
    lr_start:             float     = 3e-4
    lr_end:               float     = 1e-4
    promotion_threshold:  float     = 0.5
    promotion_window:     int       = 20

    def __post_init__(self):
        assert len(self.k_stages) >= 1,      "Need at least one K stage"
        assert self.entropy_coef_start > 0
        assert self.entropy_coef_end   > 0
        assert self.lr_start           > 0
        assert self.lr_end             > 0
        assert self.promotion_window   >= 1


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Manages the curriculum across training iterations.

    Parameters
    ----------
    config : CurriculumConfig
        Schedule hyperparameters.
    ppo : PPO
        The PPO engine (to call set_learning_rate / set_entropy_coef).
    env_config : ProbeEnvConfig
        The environment config whose blind_violation_budget will be updated.
    """

    def __init__(
        self,
        config:     CurriculumConfig,
        ppo:        PPO,
        env_config: ProbeEnvConfig,
    ):
        self.cfg        = config
        self.ppo        = ppo
        self.env_config = env_config

        self._stage       = 0
        self._reward_hist = deque(maxlen=config.promotion_window)
        self._n_stages    = len(config.k_stages)

        # Apply initial curriculum values
        self._apply_stage(self._stage)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> int:
        return self._stage

    @property
    def current_K(self) -> int:
        return self.cfg.k_stages[self._stage]

    @property
    def should_promote(self) -> bool:
        """True if promotion window is full and all rewards above threshold."""
        if len(self._reward_hist) < self.cfg.promotion_window:
            return False
        return all(r >= self.cfg.promotion_threshold for r in self._reward_hist)

    @property
    def is_final_stage(self) -> bool:
        return self._stage == self._n_stages - 1

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        mean_ep_reward: float,
        iteration:      int,
        total_iters:    int,
    ) -> dict:
        """
        Update the curriculum based on recent performance.

        Called once per training iteration after the PPO update.

        Parameters
        ----------
        mean_ep_reward : float
            Mean episode reward over the last rollout.
        iteration : int
            Current iteration index (0-based).
        total_iters : int
            Total number of training iterations (for lr annealing).

        Returns
        -------
        dict with keys: stage, K, entropy_coef, lr, promoted
        """
        self._reward_hist.append(mean_ep_reward)

        promoted = False

        # Check for promotion
        if self.should_promote and not self.is_final_stage:
            self._stage += 1
            self._reward_hist.clear()
            self._apply_stage(self._stage)
            promoted = True

        # Anneal learning rate linearly across all iterations
        lr_frac  = iteration / max(total_iters - 1, 1)
        new_lr   = self.cfg.lr_start + lr_frac * (self.cfg.lr_end - self.cfg.lr_start)
        self.ppo.set_learning_rate(new_lr)

        # Anneal entropy_coef linearly across stages
        stage_frac       = self._stage / max(self._n_stages - 1, 1)
        new_entropy_coef = (
            self.cfg.entropy_coef_start
            + stage_frac * (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start)
        )
        self.ppo.set_entropy_coef(new_entropy_coef)

        return {
            "stage":        self._stage,
            "K":            self.current_K,
            "entropy_coef": new_entropy_coef,
            "lr":           new_lr,
            "promoted":     promoted,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_stage(self, stage: int) -> None:
        """Apply the curriculum values for the given stage."""
        # Update env config K (trainer re-creates env or updates config directly)
        self.env_config.blind_violation_budget = self.cfg.k_stages[stage]

        # Set entropy coef for this stage
        stage_frac = stage / max(self._n_stages - 1, 1)
        entropy_coef = (
            self.cfg.entropy_coef_start
            + stage_frac * (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start)
        )
        self.ppo.set_entropy_coef(entropy_coef)