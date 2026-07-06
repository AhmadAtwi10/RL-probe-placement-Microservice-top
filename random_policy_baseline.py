import sys, numpy as np
sys.path.insert(0, ".")

from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.models.gnn_encoder import GNNConfig
from src.models.policy_network import PolicyNetwork, PolicyConfig, build_probeable_indices

policy = PolicyNetwork(
    GNNConfig(hidden_dim=64, num_layers=2, dropout=0.0),
    PolicyConfig(actor_hidden_dim=64, critic_hidden_dim=128, noop_hidden_dim=64, dropout=0.0),
)
policy.eval()

N = 50
returns, probes, blinds, coverages = [], [], [], []

for i in range(N):
    env = ProbeEnv(ProbeEnvConfig(
        episode_length=100, blind_violation_budget=50,
        min_failures=0, max_failures=3,
        graph_seed=i, workload_seed=i,
    ))
    obs, _ = env.reset()
    ep = env.current_graph
    total_r, probe_sum, blind_steps, cov_steps = 0.0, 0, 0, 0

    for _ in range(100):
        action, _, _, _ = policy.act(obs, ep, deterministic=False)
        obs, r, terminated, truncated, info = env.step(action)
        ep = env.current_graph
        total_r   += r
        probe_sum += len(info["probe_set"])
        blind_steps += 1 if info["reward_breakdown"].blind_violation < 0 else 0
        cov_steps   += 1 if info["reward_breakdown"].observability  > 0 else 0
        if terminated or truncated:
            break

    returns.append(total_r)
    probes.append(probe_sum / 100)
    blinds.append(blind_steps / 100)
    coverages.append(cov_steps / 100)
    print(f"ep={i:3d}  R={total_r:+8.2f}  probes={probe_sum/100:.1f}  blind={blind_steps/100:.3f}  cov={cov_steps/100:.3f}")

print(f"\n{'='*60}")
print(f"RANDOM POLICY BASELINE ({N} episodes)")
print(f"  mean_reward      : {np.mean(returns):+.2f} ± {np.std(returns):.2f}")
print(f"  mean_probe_count : {np.mean(probes):.2f} ± {np.std(probes):.2f}")
print(f"  blind_rate       : {np.mean(blinds):.3f} ± {np.std(blinds):.3f}")
print(f"  coverage_rate    : {np.mean(coverages):.3f} ± {np.std(coverages):.3f}")
print(f"  MSE baseline     : {np.std(returns)**2:.1f}  (val_loss must beat this)")
print(f"{'='*60}")