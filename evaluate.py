"""
evaluate.py — Evaluate a trained RL agent on Super Mario Bros.

Loads a saved model checkpoint, runs it for a configurable number of
episodes with rendering enabled, and prints per-episode and aggregate
reward statistics.

Usage
-----
    python evaluate.py --algo dqn  --model_path ./models/dqn_final.zip
    python evaluate.py --algo ddqn --model_path ./models/ddqn_final.zip --episodes 10
    python evaluate.py --algo ppo  --model_path ./models/ppo_final.zip

Notes
-----
    - For ``--algo ddqn``, the custom ``DoubleDQN`` class is imported from
      ``train.py`` so that SB3 can deserialise the saved weights correctly.
    - ``env.render()`` is called every step so the NES emulator window is
      visible on screen (requires a display / X server).
    - Actions are chosen deterministically (``deterministic=True``) to see
      the agent's learned policy without exploration noise.
"""

import argparse
import numpy as np

from stable_baselines3 import DQN, PPO, A2C

from env_setup import make_env
from train import DoubleDQN


# ---------------------------------------------------------------------------
# Algorithm → SB3 model class mapping
# ---------------------------------------------------------------------------
# DoubleDQN is needed for --algo ddqn so that SB3 loads the weights into the
# correct subclass (otherwise custom attributes / overrides would be lost).
# ---------------------------------------------------------------------------

ALGO_CLASS = {
    "dqn":  DQN,
    "ddqn": DoubleDQN,
    "ppo":  PPO,
    "a2c":  A2C,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on Super Mario Bros (World 1-1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ddqn", "ppo", "a2c"],
        required=True,
        help="RL algorithm the model was trained with.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model file (e.g. ./models/dqn_final.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(algo: str, model_path: str, episodes: int) -> None:
    """Load a trained model and run evaluation episodes with rendering.

    Parameters
    ----------
    algo : str
        Algorithm key (``"dqn"``, ``"ddqn"``, ``"ppo"``, ``"a2c"``).
    model_path : str
        Path to the saved ``.zip`` model file.
    episodes : int
        Number of episodes to play.
    """
    # --- Environment ---------------------------------------------------------
    env = make_env()

    # --- Load model ----------------------------------------------------------
    model_cls = ALGO_CLASS[algo]
    model = model_cls.load(model_path, env=env)

    print("=" * 64)
    print("  Super Mario Bros — Evaluation")
    print("=" * 64)
    print(f"  Algorithm  : {algo.upper()}")
    print(f"  Model      : {model_path}")
    print(f"  Episodes   : {episodes}")
    print("=" * 64)

    # --- Play episodes -------------------------------------------------------
    episode_rewards = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Deterministic action — no exploration noise
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # reward is a 1-element array from VecEnv
            steps += 1

            # Render the NES emulator window
            env.render()

        episode_rewards.append(total_reward)
        print(f"  Episode {ep:>3d}/{episodes}  |  "
              f"Reward: {total_reward:>8.1f}  |  "
              f"Steps: {steps:>5d}")

    # --- Summary -------------------------------------------------------------
    rewards = np.array(episode_rewards)
    print("\n" + "-" * 64)
    print(f"  Mean reward  : {rewards.mean():.1f}")
    print(f"  Std reward   : {rewards.std():.1f}")
    print(f"  Min reward   : {rewards.min():.1f}")
    print(f"  Max reward   : {rewards.max():.1f}")
    print("-" * 64)

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    evaluate(
        algo=args.algo,
        model_path=args.model_path,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
