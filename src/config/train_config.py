"""
train_config.py
---------------
Single entry point for all training hyperparameters.

This is the file a researcher opens first when starting a new run.
It assembles all sub-configs into a ready-to-use TrainerConfig and
provides named presets for common scenarios.

─────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────
    from src.config.train_config import get_config
    from src.training.trainer import Trainer

    # Use a preset
    cfg = get_config("full")

    # Or customise
    cfg = get_config("full")
    cfg.ppo_config.learning_rate = 1e-4
    cfg.run_name = "my_experiment"

    trainer = Trainer(cfg)
    trainer.train()

─────────────────────────────────────────────────────────────────
Presets
─────────────────────────────────────────────────────────────────
  "debug"  — tiny config for verifying the pipeline runs (< 30s)
  "fast"   — moderate config for quick experiments (~10 min)
  "full"   — full training config as described in the formulation

─────────────────────────────────────────────────────────────────
Hyperparameter reference
─────────────────────────────────────────────────────────────────
All fields with their meaning and default values are documented
inline below.  Sub-configs group related parameters:

  env_config        → ProbeEnvConfig   (episode structure)
  reward_config     → RewardConfig     (λ, μ, ρ)
  gnn_config        → GNNConfig        (encoder architecture)
  policy_config     → PolicyConfig     (actor-critic heads)
  ppo_config        → PPOConfig        (PPO update)
  curriculum_config → CurriculumConfig (K annealing)
  eval_config       → EvalConfig       (evaluation episodes)

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  get_config(preset)  → TrainerConfig
  describe(cfg)       → prints a human-readable summary
"""

from dataclasses import replace
from typing import Literal

from src.simulator.env.probe_env import ProbeEnvConfig
from src.simulator.env.reward import RewardConfig
from src.models.gnn_encoder import GNNConfig
from src.models.policy_network import PolicyConfig
from src.training.ppo import PPOConfig
from src.training.curriculum import CurriculumConfig
from src.training.trainer import TrainerConfig
from src.evaluation.evaluate import EvalConfig


# ---------------------------------------------------------------------------
# Default sub-configs
# ---------------------------------------------------------------------------

def _default_reward() -> RewardConfig:
    return RewardConfig(
        lam             = 0.15,    # probe overhead penalty per active probe
        mu              = 2.0,     # blind violation penalty weight
        rho             = 0.5,     # removal risk sensitivity
        terminal_bonus  = 5.0,     # reward for surviving T steps
        terminal_penalty= -10.0,   # penalty for exceeding blind budget K
    )


def _default_env(K: int = 10) -> ProbeEnvConfig:
    return ProbeEnvConfig(
        episode_length          = 100,   # T — timesteps per episode
        blind_violation_budget  = K,     # K — set by curriculum
        min_failures              = 0,     # minimum failure events injected per episode
        max_failures              = 3,     # maximum failure events injected per episode
        window_size             = 4,     # N — SLI rolling history window
        graph_seed              = None,  # None = non-deterministic across episodes
        workload_seed           = None,
        diurnal_amplitude       = 0.3,   # sine wave amplitude for SLI patterns
        reward_config           = _default_reward(),
    )


def _default_gnn() -> GNNConfig:
    return GNNConfig(
        input_dim  = 62,    # NODE_FEATURE_DIM — must match observation_builder
        hidden_dim = 128,   # node embedding size
        num_layers = 3,     # L — message-passing depth (≥2 for multi-hop stress)
        dropout    = 0.1,
    )


def _default_policy() -> PolicyConfig:
    return PolicyConfig(
        actor_hidden_dim  = 64,    # add/remove head hidden size
        critic_hidden_dim = 128,   # value head hidden size
        noop_hidden_dim   = 64,    # no_op head hidden size
        dropout           = 0.1,
    )


def _default_ppo() -> PPOConfig:
    return PPOConfig(
        learning_rate        = 3e-4,
        n_epochs             = 4,      # PPO epochs per rollout
        batch_size           = 64,     # mini-batch size
        clip_epsilon         = 0.2,    # ratio clipping range
        value_coef           = 0.05,    # c1 — value loss weight
        entropy_coef         = 0.01,   # c2 — entropy bonus (annealed by curriculum)
        max_grad_norm        = 0.5,    # gradient clipping
        clip_value_loss      = True,
        normalize_advantages = True,
    )


def _default_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        # K stages: start permissive, tighten progressively
        # Agent must master each before progressing to the next
        k_stages            = [50, 20, 10, 5, 2],
        entropy_coef_start  = 0.05,    # high exploration at start
        entropy_coef_end    = 0.005,   # low exploration at end
        lr_start            = 3e-4,    # initial learning rate
        lr_end              = 1e-4,    # final learning rate (cosine decay)
        promotion_threshold = 30,     # min mean reward to advance stage
        promotion_window    = 20,      # consecutive iterations above threshold
    )


def _default_eval() -> EvalConfig:
    return EvalConfig(
        n_episodes    = 20,
        deterministic = True,
        env_config    = _default_env(K=2),   # evaluate at hardest K
        device        = "cpu",
        seed          = 1000,    # separate seed from training
        verbose       = False,
    )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _debug_config() -> TrainerConfig:
    """
    Tiny config for verifying the full pipeline runs.
    Completes in under 30 seconds on CPU.
    Not suitable for learning — use only for smoke-testing.
    """
    return TrainerConfig(
        run_name          = "debug",
        total_iterations  = 5,
        rollout_len       = 32,
        log_interval      = 1,
        eval_interval     = 0,
        save_interval     = 0,
        device            = "cpu",
        seed              = 42,
        log_dir           = "logs",
        checkpoint_dir    = "checkpoints",
        use_tensorboard   = True,
        env_config        = ProbeEnvConfig(
            episode_length         = 10,
            blind_violation_budget = 100,
            min_failures             = 0,
            max_failures             = 0, # no failures, just for test
            window_size            = 4,
            reward_config          = _default_reward(),
        ),
        gnn_config        = GNNConfig(hidden_dim=32, num_layers=2, dropout=0.0),
        policy_config     = PolicyConfig(actor_hidden_dim=16, critic_hidden_dim=32,
                                         noop_hidden_dim=16, dropout=0.0),
        ppo_config        = PPOConfig(n_epochs=1, batch_size=16, learning_rate=1e-3),
        curriculum_config = CurriculumConfig(
            k_stages=[100], promotion_threshold=1e9, promotion_window=1,
        ),
    )


def _fast_config() -> TrainerConfig:
    """
    Moderate config for quick experiments on GPU (~5-10 minutes).
    Enough iterations to see the policy begin to learn.

    Key calibrations vs original:
    - rollout_len 256 → 512: more complete episodes per rollout (~5 vs ~2.5)
    - value_coef 0.5 → 0.05: reward scale ~20-85 per episode; large returns
      cause value loss explosion without scaling down the critic coefficient
    - promotion_threshold 0.3 → 30.0: calibrated to actual reward scale
    - promotion_window 10 → 20: require sustained performance before promoting
    - batch_size 32 → 64: more stable gradient estimates
    """
    return TrainerConfig(
        run_name          = "fast",
        total_iterations  = 100,
        rollout_len       = 512,
        log_interval      = 10,
        eval_interval     = 50,
        save_interval     = 50,
        device            = "cpu", # this is just a safe default, in train.py its override to gpu
        seed              = 42,
        log_dir           = "logs",
        checkpoint_dir    = "checkpoints",
        use_tensorboard   = False,
        env_config        = _default_env(K=50),
        gnn_config        = GNNConfig(hidden_dim=64, num_layers=2, dropout=0.1),
        policy_config     = _default_policy(),
        ppo_config        = PPOConfig(
            n_epochs        = 2,
            batch_size      = 64,
            learning_rate   = 3e-4,
            value_coef      = 0.05,   # scaled down for large episode returns
        ),
        curriculum_config = CurriculumConfig(
            k_stages            = [50, 20, 10, 5, 2],
            promotion_threshold = 30.0,   # calibrated to actual reward scale
            promotion_window    = 20,     # require sustained performance
        ),
    )


def _full_config() -> TrainerConfig:
    """
    Full training config as described in the formulation.
    Recommended for producing results.
    ~2-4 hours on CPU, ~20-40 minutes on GPU.
    """
    return TrainerConfig(
        run_name          = "probe_placement_full",
        total_iterations  = 10000,
        rollout_len       = 512,
        log_interval      = 10,
        eval_interval     = 50,
        save_interval     = 50,
        device            = "cpu", # this is just a safe default, in train.py its override to gpu
        seed              = 42,
        log_dir           = "logs",
        checkpoint_dir    = "checkpoints",
        use_tensorboard   = False,
        env_config        = _default_env(K=50),
        gnn_config        = _default_gnn(),
        policy_config     = _default_policy(),
        ppo_config        = _default_ppo(),
        curriculum_config = _default_curriculum(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PRESETS = {
    "debug": _debug_config,
    "fast":  _fast_config,
    "full":  _full_config,
}


def get_config(
    preset: Literal["debug", "fast", "full"] = "full",
) -> TrainerConfig:
    """
    Return a TrainerConfig for the given preset.

    Parameters
    ----------
    preset : str
        One of "debug", "fast", "full".

    Returns
    -------
    TrainerConfig
        A fully configured TrainerConfig ready to pass to Trainer().
        Modify individual fields after calling this function to customise.

    Example
    -------
    >>> cfg = get_config("full")
    >>> cfg.run_name = "my_experiment"
    >>> cfg.ppo_config.learning_rate = 1e-4
    >>> cfg.device = "cuda"
    >>> trainer = Trainer(cfg)
    >>> trainer.train()
    """
    if preset not in _PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(_PRESETS.keys())}"
        )
    return _PRESETS[preset]()


def describe(cfg: TrainerConfig) -> None:
    """
    Print a human-readable summary of all hyperparameters in a TrainerConfig.

    Parameters
    ----------
    cfg : TrainerConfig
    """
    W = 55
    print(f"{'═'*W}")
    print(f"  Run: {cfg.run_name}")
    print(f"{'─'*W}")
    print(f"  Training")
    print(f"    total_iterations : {cfg.total_iterations:,}")
    print(f"    rollout_len      : {cfg.rollout_len}")
    print(f"    total_env_steps  : {cfg.total_iterations * cfg.rollout_len:,}")
    print(f"    device           : {cfg.device}")
    print(f"    seed             : {cfg.seed}")
    print(f"    log_interval     : {cfg.log_interval}")
    print(f"    eval_interval    : {cfg.eval_interval}")
    print(f"    save_interval    : {cfg.save_interval}")
    print(f"{'─'*W}")
    print(f"  Environment")
    env = cfg.env_config
    print(f"    episode_length   : {env.episode_length}")
    print(f"    initial K        : {env.blind_violation_budget}")
    print(f"    min_failures     : {env.min_failures}")
    print(f"    max_failures     : {env.max_failures}")
    print(f"    window_size      : {env.window_size}")
    print(f"    diurnal_ampl.    : {env.diurnal_amplitude}")
    r = env.reward_config
    print(f"    λ (overhead)     : {r.lam}")
    print(f"    μ (blind viol.)  : {r.mu}")
    print(f"    ρ (remove risk)  : {r.rho}")
    print(f"    terminal bonus   : {r.terminal_bonus}")
    print(f"    terminal penalty : {r.terminal_penalty}")
    print(f"{'─'*W}")
    print(f"  GNN Encoder")
    g = cfg.gnn_config
    print(f"    hidden_dim       : {g.hidden_dim}")
    print(f"    num_layers       : {g.num_layers}")
    print(f"    dropout          : {g.dropout}")
    print(f"{'─'*W}")
    print(f"  Policy Network")
    p = cfg.policy_config
    print(f"    actor_hidden_dim : {p.actor_hidden_dim}")
    print(f"    critic_hidden_dim: {p.critic_hidden_dim}")
    print(f"    noop_hidden_dim  : {p.noop_hidden_dim}")
    print(f"    dropout          : {p.dropout}")
    print(f"{'─'*W}")
    print(f"  PPO")
    ppo = cfg.ppo_config
    print(f"    learning_rate    : {ppo.learning_rate}")
    print(f"    n_epochs         : {ppo.n_epochs}")
    print(f"    batch_size       : {ppo.batch_size}")
    print(f"    clip_epsilon     : {ppo.clip_epsilon}")
    print(f"    value_coef (c1)  : {ppo.value_coef}")
    print(f"    entropy_coef(c2) : {ppo.entropy_coef}")
    print(f"    max_grad_norm    : {ppo.max_grad_norm}")
    print(f"    clip_value_loss  : {ppo.clip_value_loss}")
    print(f"{'─'*W}")
    print(f"  Curriculum")
    c = cfg.curriculum_config
    print(f"    k_stages         : {c.k_stages}")
    print(f"    entropy_coef     : {c.entropy_coef_start} → {c.entropy_coef_end}")
    print(f"    lr               : {c.lr_start} → {c.lr_end}")
    print(f"    promotion thresh.: {c.promotion_threshold}")
    print(f"    promotion window : {c.promotion_window}")
    print(f"{'═'*W}")