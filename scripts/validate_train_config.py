"""
validate_train_config.py
------------------------
Phase: train_config validation.

Run from project root:
    python scripts/validate_train_config.py

Checks:
  1.  get_config("debug") returns valid TrainerConfig
  2.  get_config("fast")  returns valid TrainerConfig
  3.  get_config("full")  returns valid TrainerConfig
  4.  Unknown preset raises ValueError
  5.  All presets: batch_size <= rollout_len
  6.  All presets: gnn hidden_dim matches policy expectations
  7.  describe() prints without error for all presets
  8.  Configs are independent (modifying one doesn't affect another)
  9.  Full pipeline: debug config runs end-to-end with Trainer
 10.  get_config returns a fresh object each call (not cached reference)
"""

import sys
sys.path.insert(0, ".")

from src.config.train_config import get_config, describe
from src.training.trainer import Trainer

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1-3. All presets return valid TrainerConfig ───────────────────────────────
for preset in ["debug", "fast", "full"]:
    section(f"{['1','2','3'][['debug','fast','full'].index(preset)]}. get_config('{preset}')")
    cfg = get_config(preset)
    assert cfg.total_iterations  > 0
    assert cfg.rollout_len       > 0
    assert cfg.gnn_config.hidden_dim > 0
    assert cfg.gnn_config.num_layers >= 1
    assert cfg.ppo_config.learning_rate > 0
    assert len(cfg.curriculum_config.k_stages) >= 1
    assert cfg.env_config.episode_length > 0
    assert cfg.env_config.reward_config is not None
    print(f"  '{preset}': iters={cfg.total_iterations}, "
          f"rollout={cfg.rollout_len}, "
          f"K_stages={cfg.curriculum_config.k_stages}  {PASS}")


# ── 4. Unknown preset raises ValueError ───────────────────────────────────────
section("4. Unknown preset raises ValueError")

try:
    get_config("unknown_preset")
    assert False, f"{FAIL} Should have raised ValueError"
except ValueError as e:
    assert "unknown_preset" in str(e)
print(f"  ValueError raised with informative message  {PASS}")


# ── 5. batch_size <= rollout_len for all presets ──────────────────────────────
section("5. batch_size <= rollout_len for all presets")

for preset in ["debug", "fast", "full"]:
    cfg = get_config(preset)
    assert cfg.ppo_config.batch_size <= cfg.rollout_len, \
        f"{FAIL} '{preset}': batch_size={cfg.ppo_config.batch_size} > rollout_len={cfg.rollout_len}"
    print(f"  '{preset}': batch_size={cfg.ppo_config.batch_size} <= rollout_len={cfg.rollout_len}  {PASS}")


# ── 6. GNN input_dim consistent ───────────────────────────────────────────────
section("6. GNN input_dim=62 matches NODE_FEATURE_DIM")

from src.simulator.env.observation_builder import NODE_FEATURE_DIM

for preset in ["debug", "fast", "full"]:
    cfg = get_config(preset)
    # debug uses smaller hidden_dim but still must have correct input_dim
    # full config explicitly sets input_dim=62
    if hasattr(cfg.gnn_config, 'input_dim'):
        assert cfg.gnn_config.input_dim == NODE_FEATURE_DIM, \
            f"{FAIL} '{preset}': input_dim={cfg.gnn_config.input_dim} != {NODE_FEATURE_DIM}"
print(f"  GNN input_dim={NODE_FEATURE_DIM} consistent across presets  {PASS}")


# ── 7. describe() works for all presets ───────────────────────────────────────
section("7. describe() prints without error")

import io, sys as _sys
for preset in ["debug", "fast", "full"]:
    cfg = get_config(preset)
    # Capture stdout
    buf = io.StringIO()
    old = _sys.stdout
    _sys.stdout = buf
    describe(cfg)
    _sys.stdout = old
    output = buf.getvalue()
    assert len(output) > 100, f"{FAIL} '{preset}': describe() output too short"
    assert cfg.run_name in output
    assert "PPO" in output
    assert "Curriculum" in output
print(f"  describe() output contains run_name, PPO, Curriculum  {PASS}")


# ── 8. Configs are independent ────────────────────────────────────────────────
section("8. Modifying one config doesn't affect another")

cfg_a = get_config("full")
cfg_b = get_config("full")

cfg_a.run_name = "modified_a"
cfg_a.ppo_config.learning_rate = 9e-9

assert cfg_b.run_name != "modified_a", \
    f"{FAIL} cfg_b.run_name was affected by cfg_a modification"
assert cfg_b.ppo_config.learning_rate != 9e-9, \
    f"{FAIL} cfg_b.ppo_config was affected by cfg_a modification"
print(f"  Configs are independent objects  {PASS}")


# ── 9. Debug config runs end-to-end ───────────────────────────────────────────
section("9. Debug config runs full Trainer.train() without error")

import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    cfg = get_config("debug")
    cfg.log_dir        = os.path.join(tmpdir, "logs")
    cfg.checkpoint_dir = os.path.join(tmpdir, "ckpt")
    cfg.run_name       = "validate_debug"

    trainer = Trainer(cfg)
    trainer.train()

    assert trainer._env_steps == cfg.rollout_len * cfg.total_iterations
print(f"  Debug config: {cfg.total_iterations} iterations, "
      f"{cfg.rollout_len * cfg.total_iterations} env steps  {PASS}")


# ── 10. get_config returns fresh object each call ────────────────────────────
section("10. get_config returns fresh object each call")

cfg1 = get_config("full")
cfg2 = get_config("full")
assert cfg1 is not cfg2, f"{FAIL} Same object returned (should be fresh each call)"
assert cfg1.ppo_config is not cfg2.ppo_config, \
    f"{FAIL} Shared ppo_config reference"
print(f"  Fresh TrainerConfig returned each call  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ALL TRAIN_CONFIG ASSERTIONS PASSED ✓")
print(f"{'═'*60}\n")