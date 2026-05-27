"""
checkpointing.py
----------------
Save and load training state for the probe placement POMDP.

Saves:
  - policy network weights (state_dict)
  - optimizer state (for resuming training)
  - training metadata (iteration, env_steps, best_reward)
  - PPO config and curriculum state

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  Checkpointer(run_name, save_dir)
    .save(iteration, env_steps, policy, optimizer,
          mean_reward, curriculum_stage)
    .load(path, policy, optimizer)  → metadata dict
    .best_path   → path to best checkpoint
    .latest_path → path to latest checkpoint
"""

import os
import glob
from typing import Optional

import torch

from src.models.policy_network import PolicyNetwork


class Checkpointer:
    """
    Manages saving and loading of training checkpoints.

    Parameters
    ----------
    run_name : str
        Unique name for this training run.
    save_dir : str
        Directory for checkpoint files.
    keep_last : int
        Number of most recent checkpoints to keep.
        Older ones are deleted automatically.
    """

    def __init__(
        self,
        run_name: str,
        save_dir: str = "checkpoints",
        keep_last: int = 3,
    ):
        self.run_name  = run_name
        self.save_dir  = save_dir
        self.keep_last = keep_last
        os.makedirs(save_dir, exist_ok=True)

        self._best_reward: float = float("-inf")
        self._best_path:   Optional[str] = None
        self._latest_path: Optional[str] = None

    @property
    def best_path(self) -> Optional[str]:
        return self._best_path

    @property
    def latest_path(self) -> Optional[str]:
        return self._latest_path

    def save(
        self,
        iteration:         int,
        env_steps:         int,
        policy:            PolicyNetwork,
        optimizer:         torch.optim.Optimizer,
        mean_reward:       float,
        curriculum_stage:  int  = 0,
    ) -> str:
        """
        Save a checkpoint.

        Always saves a 'latest' checkpoint.
        Also saves a 'best' checkpoint when mean_reward improves.

        Returns
        -------
        str — path of the saved checkpoint.
        """
        state = {
            "iteration":        iteration,
            "env_steps":        env_steps,
            "mean_reward":      mean_reward,
            "curriculum_stage": curriculum_stage,
            "policy_state":     policy.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
        }

        # Latest checkpoint
        latest = os.path.join(
            self.save_dir, f"{self.run_name}_iter{iteration:06d}.pt"
        )
        torch.save(state, latest)
        self._latest_path = latest

        # Best checkpoint
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            best = os.path.join(self.save_dir, f"{self.run_name}_best.pt")
            torch.save(state, best)
            self._best_path = best

        # Prune old checkpoints (keep_last most recent)
        self._prune()

        return latest

    def load(
        self,
        path:      str,
        policy:    PolicyNetwork,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        """
        Load a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        policy : PolicyNetwork
            Policy network to load weights into.
        optimizer : optional
            Optimizer to restore state (None = skip optimizer state).

        Returns
        -------
        dict with keys: iteration, env_steps, mean_reward,
                        curriculum_stage.
        """
        state = torch.load(path, map_location="cpu")
        policy.load_state_dict(state["policy_state"])
        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])

        return {
            "iteration":        state.get("iteration",        0),
            "env_steps":        state.get("env_steps",        0),
            "mean_reward":      state.get("mean_reward",      float("-inf")),
            "curriculum_stage": state.get("curriculum_stage", 0),
        }

    def _prune(self) -> None:
        """Delete old checkpoints, keeping only keep_last most recent."""
        pattern = os.path.join(
            self.save_dir, f"{self.run_name}_iter*.pt"
        )
        checkpoints = sorted(glob.glob(pattern))
        for old in checkpoints[: -self.keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass