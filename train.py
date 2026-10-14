"""train.py — Training entry-point for Super Mario Bros RL agents.

Supports four algorithm choices via ``--algo``:

  dqn   : Vanilla Deep Q-Network (Nature DQN hyperparameters from spec §3.4)
  ddqn  : Double DQN — custom subclass that fixes DQN's overestimation bias
           by decoupling action selection (online net) from evaluation (target net)
  ppo   : Proximal Policy Optimization (on-policy, SB3 defaults)
  a2c   : Advantage Actor-Critic (on-policy, SB3 defaults)

Usage
-----
    python train.py --algo dqn  --timesteps 1000000
    python train.py --algo ddqn --timesteps 1000000
    python train.py --algo ppo  --timesteps 500000 --save_dir ./my_models

Outputs
-------
    ./models/{algo}_final.zip              — final trained weights
    ./models/{algo}_checkpoints/           — periodic snapshots every 50 k steps
    ./logs/{algo}_mario_*/                 — TensorBoard event files
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

from env_setup import make_env


# ---------------------------------------------------------------------------
# Double DQN — custom subclass
# ---------------------------------------------------------------------------
# Vanilla DQN uses the target network for BOTH selecting and evaluating the
# best next action.  This causes systematic overestimation of Q-values
# (van Hasselt et al., 2016).  Double DQN fixes this by decoupling the two:
#
#   Standard DQN target:
#       y = r + γ · max_a' Q_target(s', a')
#
#   Double DQN target:
#       a* = argmax_a' Q_online(s', a')      ← online net SELECTS
#       y  = r + γ · Q_target(s', a*)         ← target net EVALUATES
#
# The only code change is inside train() — everything else (replay buffer,
# ε-greedy, target sync) is inherited from SB3's DQN.
# ---------------------------------------------------------------------------


class DoubleDQN(DQN):
    """Double DQN (van Hasselt et al., 2016) built on top of SB3's DQN.

    Overrides ``train()`` to decouple action selection from evaluation in
    the TD-target, eliminating the maximisation bias of vanilla DQN.
    All other behaviour (replay buffer, exploration, target-net sync,
    logging, saving/loading) is inherited unchanged.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch networks to training mode
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule (omitting this was a bug)
        self._update_learning_rate([self.policy.optimizer])


        losses = []

        for _ in range(gradient_steps):
            # --- Sample a mini-batch from replay buffer ------------------
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env,
            )

            with th.no_grad():
                # --- Double DQN target -----------------------------------
                # 1. ONLINE network selects the best next action
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                # 2. TARGET network evaluates that action
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(next_q_target, dim=1, index=next_actions)

                # 3. TD target: r + γ · Q_target(s', a*) — zero if terminal
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # --- Current Q-values for the actions that were taken --------
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long(),
            )

            # --- Huber loss (smooth L1) & gradient step ------------------
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — inherited max_grad_norm from DQN
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm,
            )
            self.policy.optimizer.step()

            self._n_updates += 1

        # --- Bookkeeping (matches SB3's DQN.train() logging) -------------
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ---------------------------------------------------------------------------
# Hyperparameters — project_spec.md §3.4
# ---------------------------------------------------------------------------
# Applied to DQN and DDQN only.  PPO / A2C use SB3 sensible defaults (tuned
# for Atari-scale problems) so we don't override them here.
# ---------------------------------------------------------------------------

DQN_HYPERPARAMS = {
    "learning_rate":          2.5e-4,      # Adam optimiser
    "buffer_size":            100_000,     # replay buffer capacity (transitions)
    "batch_size":             32,          # mini-batch sampled per gradient step
    "gamma":                  0.99,        # discount factor
    "target_update_interval": 10_000,      # hard-copy θ → θ⁻ every N steps
    "exploration_initial_eps": 1.0,        # ε-start: fully random
    "exploration_final_eps":  0.02,        # ε-end: small residual exploration
    "exploration_fraction":   0.10,        # decay ε linearly over 10 % of training
    "learning_starts":        10_000,      # fill buffer before first gradient step
    "train_freq":             4,           # gradient step every 4 env steps
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    parser = argparse.ArgumentParser(
        description="Train an RL agent to play Super Mario Bros (World 1-1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ddqn", "ppo", "a2c"],
        default="dqn",
        help="RL algorithm to use.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps to train for.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Root directory for saved model weights and checkpoints.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(algo: str, env, log_dir: str):
    """Instantiate the SB3 model for the chosen algorithm.

    Parameters
    ----------
    algo : str
        One of ``"dqn"``, ``"ddqn"``, ``"ppo"``, ``"a2c"``.
    env : VecEnv
        The SB3-compatible vectorized environment from ``make_env()``.
    log_dir : str
        Path for TensorBoard event files.

    Returns
    -------
    stable_baselines3.common.base_class.BaseAlgorithm
    """
    # Arguments shared by every algorithm
    common_kwargs = {
        "policy":          "CnnPolicy",
        "env":             env,
        "tensorboard_log": log_dir,
        "verbose":         1,
    }

    if algo == "dqn":
        return DQN(**common_kwargs, **DQN_HYPERPARAMS)

    if algo == "ddqn":
        return DoubleDQN(**common_kwargs, **DQN_HYPERPARAMS)

    if algo == "ppo":
        return PPO(**common_kwargs)

    if algo == "a2c":
        return A2C(**common_kwargs)

    # Should be unreachable (argparse validates choices), but just in case
    raise ValueError(f"Unknown algorithm: {algo!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    algo       = args.algo
    timesteps  = args.timesteps
    save_dir   = args.save_dir

    # --- Directory setup -----------------------------------------------------
    log_dir        = "./logs"
    checkpoint_dir = os.path.join(save_dir, f"{algo}_checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Banner --------------------------------------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 64)
    print("  Super Mario Bros — RL Training")
    print("=" * 64)
    print(f"  Algorithm   : {algo.upper()}")
    print(f"  Timesteps   : {timesteps:,}")
    print(f"  Save dir    : {save_dir}")
    print(f"  Checkpoints : {checkpoint_dir}")
    print(f"  TB logs     : {log_dir}")
    print(f"  Run ID      : {run_id}")
    print("=" * 64)

    # --- Environment ---------------------------------------------------------
    env = make_env()
    print(f"\n  Observation space : {env.observation_space.shape}")
    print(f"  Action space      : {env.action_space}\n")

    # --- Model ---------------------------------------------------------------
    model = build_model(algo, env, log_dir)

    # --- Checkpoint callback -------------------------------------------------
    # Saves a snapshot every 50 k timesteps so training can be resumed or
    # the best intermediate model can be cherry-picked later.
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_dir,
        name_prefix=f"{algo}_mario_{run_id}",
        verbose=1,
    )

    # --- Training ------------------------------------------------------------
    print(f"Starting training for {timesteps:,} timesteps …\n")
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        tb_log_name=f"{algo}_mario",
    )

    # --- Save final model ----------------------------------------------------
    final_path = os.path.join(save_dir, f"{algo}_final")
    model.save(final_path)

    env.close()

    # --- Confirmation --------------------------------------------------------
    print("\n" + "=" * 64)
    print("  Training complete!")
    print("=" * 64)
    print(f"  Final model saved to : {final_path}.zip")
    print(f"  Checkpoints saved in : {checkpoint_dir}/")
    print(f"  TensorBoard logs in  : {log_dir}/")
    print()
    print("  To view TensorBoard:")
    print(f"    tensorboard --logdir {log_dir}")
    print()
    print("  To download from Kaggle, look for:")
    print(f"    /kaggle/working/{final_path}.zip")
    print("=" * 64)


if __name__ == "__main__":
    main()
