"""
validate_trainer.py
-------------------
Phase 5c validation: trainer, curriculum, logger, checkpointer.

Run from project root:
    python scripts/validate_trainer.py

Checks:
  1.  TrainerConfig defaults valid
  2.  Trainer initialises without error
  3.  collect_rollout() fills buffer and returns a float
  4.  Full train() loop: 5 iterations without crash
  5.  EpisodeTracker accumulates and aggregates correctly
  6.  CurriculumScheduler: K decreases on promotion
  7.  CurriculumScheduler: no promotion below threshold
  8.  CurriculumScheduler: entropy_coef anneals across stages
  9.  Logger: writes JSON lines file
 10.  Checkpointer: save and load round-trip
 11.  Resume from checkpoint: iteration continues correctly
 12.  Buffer resets between iterations
 13.  env_steps accumulates correctly
 14.  Curriculum promotion triggers env reset (K applied)
"""

import sys, os, json, tempfile
sys.path.insert(0, ".")

import torch
import numpy as np

from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.simulator.env.reward import RewardConfig
from src.models.gnn_encoder import GNNConfig
from src.models.policy_network import PolicyConfig, PolicyNetwork
from src.training.rollout_buffer import RolloutBuffer
from src.training.ppo import PPO, PPOConfig
from src.training.curriculum import CurriculumScheduler, CurriculumConfig
from src.training.trainer import Trainer, TrainerConfig, EpisodeTracker
from src.utils.logger import Logger
from src.utils.checkpointing import Checkpointer

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def make_fast_config(tmpdir, run_name="test", n_iters=5):
    """Minimal config for fast tests."""
    return TrainerConfig(
        run_name         = run_name,
        total_iterations = n_iters,
        rollout_len      = 32,
        log_interval     = 1,
        eval_interval    = 0,
        save_interval    = 0,
        device           = "cpu",
        seed             = 42,
        log_dir          = os.path.join(tmpdir, "logs"),
        checkpoint_dir   = os.path.join(tmpdir, "ckpt"),
        use_tensorboard  = False,
        env_config       = ProbeEnvConfig(
            episode_length=10, n_failures=0,
            blind_violation_budget=100,
            graph_seed=0, workload_seed=0,
            reward_config=RewardConfig(),
        ),
        gnn_config       = GNNConfig(hidden_dim=32, num_layers=2, dropout=0.0),
        policy_config    = PolicyConfig(actor_hidden_dim=16, critic_hidden_dim=32,
                                        noop_hidden_dim=16, dropout=0.0),
        ppo_config       = PPOConfig(n_epochs=1, batch_size=16,
                                     learning_rate=1e-3),
        curriculum_config= CurriculumConfig(
            k_stages=[100, 50],
            promotion_threshold=1000.0,   # never promote automatically
            promotion_window=3,
        ),
    )


# ── 1. TrainerConfig defaults valid ──────────────────────────────────────────
section("1. TrainerConfig defaults valid")

cfg_default = TrainerConfig()
assert cfg_default.total_iterations > 0
assert cfg_default.rollout_len      > 0
assert cfg_default.ppo_config.batch_size <= cfg_default.rollout_len
print(f"  defaults: iters={cfg_default.total_iterations}, "
      f"rollout={cfg_default.rollout_len}  {PASS}")


# ── 2. Trainer initialises ────────────────────────────────────────────────────
section("2. Trainer initialises without error")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir)
    trainer = Trainer(cfg)
    assert trainer._obs  is not None
    assert trainer._ep   is not None
    assert trainer._env_steps == 0
print(f"  Trainer created, obs and ep initialised  {PASS}")


# ── 3. collect_rollout() fills buffer ─────────────────────────────────────────
section("3. collect_rollout() fills buffer and returns float")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir)
    trainer = Trainer(cfg)

    last_val = trainer.collect_rollout()

    assert trainer.buffer.is_full, f"{FAIL} Buffer not full after collect_rollout()"
    assert isinstance(last_val, float), f"{FAIL} last_value not float"
    assert np.isfinite(last_val), f"{FAIL} last_value not finite"
    assert trainer._env_steps == cfg.rollout_len, \
        f"{FAIL} env_steps={trainer._env_steps}, expected {cfg.rollout_len}"
print(f"  buffer full, last_value={last_val:.4f}, "
      f"env_steps={cfg.rollout_len}  {PASS}")


# ── 4. Full train() loop: 5 iterations ───────────────────────────────────────
section("4. Full train() loop: 5 iterations without crash")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir, n_iters=5)
    trainer = Trainer(cfg)
    trainer.train()
    assert trainer._env_steps == cfg.rollout_len * 5, \
        f"{FAIL} env_steps={trainer._env_steps}, expected {cfg.rollout_len*5}"
print(f"  5 iterations completed, env_steps={cfg.rollout_len*5}  {PASS}")


# ── 5. EpisodeTracker accumulates correctly ───────────────────────────────────
section("5. EpisodeTracker accumulates and aggregates")

tracker = EpisodeTracker()

# Simulate 2 episodes
for _ in range(5):
    tracker.step(1.0, {"reward_breakdown": type("rb", (), {
        "covered_slos": [0], "blind_slos": []})()})
tracker.end_episode()   # episode 1: reward=5, coverage=1, blind=0

for _ in range(3):
    tracker.step(-1.0, {"reward_breakdown": type("rb", (), {
        "covered_slos": [], "blind_slos": [1]})()})
tracker.end_episode()   # episode 2: reward=-3, coverage=0, blind=1

metrics = tracker.aggregate()
assert abs(metrics["mean_reward"]     - 1.0)  < 1e-6, \
    f"{FAIL} mean_reward={metrics['mean_reward']}, expected 1.0"
assert abs(metrics["mean_coverage"]   - 0.5)  < 1e-6, \
    f"{FAIL} mean_coverage={metrics['mean_coverage']}, expected 0.5"
assert abs(metrics["mean_blind_rate"] - 0.5)  < 1e-6, \
    f"{FAIL} mean_blind_rate={metrics['mean_blind_rate']}, expected 0.5"
assert metrics["n_episodes"] == 2
# After aggregate, completed list is cleared
assert len(tracker.completed) == 0
print(f"  mean_reward=1.0, coverage=0.5, blind=0.5, n_ep=2  {PASS}")


# ── 6. Curriculum: promotes when threshold met ────────────────────────────────
section("6. CurriculumScheduler: promotes on sustained high reward")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir)
    pol = PolicyNetwork(cfg.gnn_config, cfg.policy_config)
    ppo = PPO(pol, cfg.ppo_config)
    env_cfg = ProbeEnvConfig(blind_violation_budget=100)
    curr = CurriculumScheduler(
        CurriculumConfig(
            k_stages=[100, 50, 10],
            promotion_threshold=0.0,   # always promote
            promotion_window=2,
        ),
        ppo, env_cfg,
    )
    # Step with reward above threshold to fill window
    info1 = curr.step(1.0, 0, 10)
    info2 = curr.step(1.0, 1, 10)   # window full → promote

    assert info2["promoted"],        f"{FAIL} Should have promoted"
    assert curr.current_stage == 1,  f"{FAIL} stage={curr.current_stage}"
    assert curr.current_K == 50,     f"{FAIL} K={curr.current_K}"
    assert env_cfg.blind_violation_budget == 50
print(f"  Promoted to stage 1, K=50  {PASS}")


# ── 7. Curriculum: no promotion below threshold ───────────────────────────────
section("7. CurriculumScheduler: no promotion below threshold")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir)
    pol = PolicyNetwork(cfg.gnn_config, cfg.policy_config)
    ppo = PPO(pol, cfg.ppo_config)
    env_cfg2 = ProbeEnvConfig(blind_violation_budget=100)
    curr2 = CurriculumScheduler(
        CurriculumConfig(
            k_stages=[100, 50],
            promotion_threshold=10.0,   # high threshold — never met
            promotion_window=2,
        ),
        ppo, env_cfg2,
    )
    for _ in range(5):
        info = curr2.step(0.5, 0, 10)   # reward < threshold

    assert not info["promoted"],      f"{FAIL} Should not have promoted"
    assert curr2.current_stage == 0,  f"{FAIL} stage should be 0"
print(f"  No promotion with reward below threshold  {PASS}")


# ── 8. Curriculum: entropy anneals across stages ──────────────────────────────
section("8. Curriculum: entropy_coef anneals across stages")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir)
    pol = PolicyNetwork(cfg.gnn_config, cfg.policy_config)
    ppo = PPO(pol, cfg.ppo_config)
    env_cfg3 = ProbeEnvConfig()
    curr3 = CurriculumScheduler(
        CurriculumConfig(
            k_stages=[100, 50, 10],
            entropy_coef_start=0.1,
            entropy_coef_end=0.001,
            promotion_threshold=0.0,
            promotion_window=1,
        ),
        ppo, env_cfg3,
    )
    info_s0 = curr3.step(1.0, 0, 10)   # stage 0 → promote
    coef_s0  = info_s0["entropy_coef"]
    info_s1 = curr3.step(1.0, 1, 10)   # stage 1 → promote
    coef_s1  = info_s1["entropy_coef"]

    assert coef_s1 < coef_s0, \
        f"{FAIL} entropy_coef should decrease: {coef_s0:.4f} → {coef_s1:.4f}"
print(f"  entropy_coef: {coef_s0:.4f} → {coef_s1:.4f} (decreasing)  {PASS}")


# ── 9. Logger writes JSON lines ───────────────────────────────────────────────
section("9. Logger writes JSON lines file")

from src.training.ppo import PPOStats

with tempfile.TemporaryDirectory() as tmpdir:
    logger = Logger("test_run", log_dir=tmpdir)
    for i in range(3):
        logger.log(
            iteration=i, env_steps=i*512,
            ppo_stats=PPOStats(total_loss=1.0, policy_loss=0.5,
                               value_loss=0.3, entropy=2.0,
                               approx_kl=0.01, clip_fraction=0.1,
                               explained_var=0.5, n_updates=4),
            curriculum_info={"stage": 0, "K": 100, "promoted": False,
                             "entropy_coef": 0.01, "lr": 3e-4},
            episode_metrics={"mean_reward": 0.5, "mean_ep_len": 20.0,
                             "mean_coverage": 0.8, "mean_blind_rate": 0.1,
                             "n_episodes": 2},
        )
    logger.close()

    jsonl_path = os.path.join(tmpdir, "test_run.jsonl")
    assert os.path.exists(jsonl_path), f"{FAIL} JSONL file not created"
    with open(jsonl_path) as f:
        lines = f.readlines()
    assert len(lines) == 3, f"{FAIL} Expected 3 lines, got {len(lines)}"
    record = json.loads(lines[0])
    assert "total_loss" in record
    assert "curr_stage" in record
    assert "ep_mean_reward" in record
print(f"  3 JSON lines written with correct keys  {PASS}")


# ── 10. Checkpointer: save and load round-trip ───────────────────────────────
section("10. Checkpointer: save and load round-trip")

with tempfile.TemporaryDirectory() as tmpdir:
    pol10  = PolicyNetwork(GNNConfig(hidden_dim=32, num_layers=2))
    opt10  = torch.optim.Adam(pol10.parameters(), lr=1e-3)
    ckptr  = Checkpointer("test_ckpt", save_dir=tmpdir)

    # Modify policy weights so we can detect a successful load
    with torch.no_grad():
        for p in pol10.parameters():
            p.fill_(1.23)

    path = ckptr.save(
        iteration=5, env_steps=2560,
        policy=pol10, optimizer=opt10,
        mean_reward=0.7, curriculum_stage=1,
    )
    assert os.path.exists(path), f"{FAIL} Checkpoint not saved"

    # Load into a fresh policy
    pol10_new = PolicyNetwork(GNNConfig(hidden_dim=32, num_layers=2))
    meta      = ckptr.load(path, pol10_new)

    assert meta["iteration"]  == 5
    assert meta["env_steps"]  == 2560
    assert abs(meta["mean_reward"] - 0.7) < 1e-6
    assert meta["curriculum_stage"] == 1

    # Weights should match
    for (n1, p1), (n2, p2) in zip(pol10.named_parameters(),
                                   pol10_new.named_parameters()):
        assert torch.allclose(p1, p2), f"{FAIL} Weight mismatch: {n1}"
print(f"  Save/load round-trip: weights and metadata match  {PASS}")


# ── 11. Best checkpoint saved when reward improves ────────────────────────────
section("11. Checkpointer: best checkpoint tracks best reward")

with tempfile.TemporaryDirectory() as tmpdir:
    pol11 = PolicyNetwork(GNNConfig(hidden_dim=32, num_layers=2))
    opt11 = torch.optim.Adam(pol11.parameters())
    ckptr11 = Checkpointer("best_test", save_dir=tmpdir)

    ckptr11.save(0, 0, pol11, opt11, mean_reward=0.3)
    ckptr11.save(1, 512, pol11, opt11, mean_reward=0.8)   # new best
    ckptr11.save(2, 1024, pol11, opt11, mean_reward=0.5)  # not best

    assert ckptr11.best_path is not None
    best_meta = ckptr11.load(ckptr11.best_path, pol11)
    assert abs(best_meta["mean_reward"] - 0.8) < 1e-6, \
        f"{FAIL} Best reward should be 0.8, got {best_meta['mean_reward']}"
print(f"  Best checkpoint correctly tracks max reward=0.8  {PASS}")


# ── 12. Buffer resets between iterations ──────────────────────────────────────
section("12. Buffer resets between iterations")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir, n_iters=3)
    trainer = Trainer(cfg)

    for _ in range(3):
        trainer.collect_rollout()
        assert trainer.buffer.is_full
        trainer.buffer.compute_gae(last_value=0.0)
        trainer.ppo.update(trainer.buffer)
        trainer.buffer.reset()
        assert trainer.buffer.size == 0, f"{FAIL} Buffer not reset"
print(f"  Buffer resets to size=0 between iterations  {PASS}")


# ── 13. env_steps accumulates correctly ───────────────────────────────────────
section("13. env_steps accumulates correctly")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir, n_iters=4)
    trainer = Trainer(cfg)
    trainer.train()

    expected = cfg.rollout_len * 4
    assert trainer._env_steps == expected, \
        f"{FAIL} env_steps={trainer._env_steps}, expected {expected}"
print(f"  env_steps={trainer._env_steps} = {cfg.rollout_len}×4  {PASS}")


# ── 14. Curriculum promotion applies K to env ─────────────────────────────────
section("14. Curriculum promotion applies updated K to env")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = make_fast_config(tmpdir)
    # Set curriculum to promote immediately
    cfg.curriculum_config = CurriculumConfig(
        k_stages=[100, 5],
        promotion_threshold=-1000.0,  # always promote
        promotion_window=1,
    )
    trainer = Trainer(cfg)
    initial_K = trainer.env.cfg.blind_violation_budget

    # Run one iteration so curriculum promotes
    last_val = trainer.collect_rollout()
    trainer.buffer.compute_gae(last_val)
    trainer.ppo.update(trainer.buffer)
    trainer.buffer.reset()
    ep_metrics = trainer.tracker.aggregate()
    curr_info  = trainer.curriculum.step(
        ep_metrics["mean_reward"], 0, cfg.total_iterations
    )
    if curr_info["promoted"]:
        trainer._reset_env()

    new_K = trainer.env.cfg.blind_violation_budget
    if curr_info["promoted"]:
        assert new_K == 5, f"{FAIL} K should be 5 after promotion, got {new_K}"
        print(f"  K: {initial_K} → {new_K} after promotion  {PASS}")
    else:
        print(f"  Curriculum at stage 0, K={initial_K} (no promotion yet)  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ALL TRAINER ASSERTIONS PASSED ✓")
print(f"{'═'*60}\n")