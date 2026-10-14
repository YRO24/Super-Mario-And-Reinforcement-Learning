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
    - The environment uses ``render_mode="rgb_array"`` so frames are returned
      as numpy arrays.  Each frame is scaled 4× with OpenCV and displayed in
      a resizable window at a watchable speed (~20 FPS).
    - Press **q** or close the window to abort evaluation early.
    - Actions are chosen deterministically (``deterministic=True``) to see
      the agent's learned policy without exploration noise.
"""

import argparse
import time

import cv2
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
# Display constants
# ---------------------------------------------------------------------------
SCALE_FACTOR = 4          # 240×256 → 960×1024 — comfortably visible on a Mac
WINDOW_NAME = "Super Mario Bros — Evaluation"
FRAME_DELAY = 0.05        # seconds between rendered steps (~20 effective FPS)


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
# Helpers
# ---------------------------------------------------------------------------

def _window_is_open() -> bool:
    """Return True if the OpenCV display window still exists."""
    try:
        # getWindowProperty returns -1.0 when the window has been closed
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1.0
    except cv2.error:
        return False


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
    env = make_env(render_mode="rgb_array")

    # --- Load model ----------------------------------------------------------
    model_cls = ALGO_CLASS[algo]
    model = model_cls.load(model_path, env=env)

    print("=" * 64)
    print("  Super Mario Bros — Evaluation")
    print("=" * 64)
    print(f"  Algorithm  : {algo.upper()}")
    print(f"  Model      : {model_path}")
    print(f"  Episodes   : {episodes}")
    print(f"  Scale      : {SCALE_FACTOR}×  (press 'q' or close window to stop)")
    print("=" * 64)

    # --- Create OpenCV window ------------------------------------------------
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --- Play episodes -------------------------------------------------------
    episode_rewards = []
    aborted = False          # set True if user closes window / presses 'q'

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

            # --- Render frame via OpenCV -------------------------------------
            # env.render() returns a list of RGB frames (one per sub-env).
            frame = env.render()
            if frame is not None:
                # VecEnv.render() may return a list or a single array
                if isinstance(frame, list):
                    frame = frame[0]

                # Scale up with nearest-neighbour to keep pixel-art sharp
                h, w = frame.shape[:2]
                scaled = cv2.resize(
                    frame,
                    (w * SCALE_FACTOR, h * SCALE_FACTOR),
                    interpolation=cv2.INTER_NEAREST,
                )

                # OpenCV expects BGR; gym returns RGB
                scaled_bgr = cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)
                cv2.imshow(WINDOW_NAME, scaled_bgr)

            # --- Throttle & handle input -------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not _window_is_open():
                aborted = True
                break

            time.sleep(FRAME_DELAY)

        episode_rewards.append(total_reward)
        print(f"  Episode {ep:>3d}/{episodes}  |  "
              f"Reward: {total_reward:>8.1f}  |  "
              f"Steps: {steps:>5d}")

        if aborted:
            print("\n  ⏹  Evaluation aborted by user.")
            break

    # --- Summary -------------------------------------------------------------
    rewards = np.array(episode_rewards)
    print("\n" + "-" * 64)
    print(f"  Mean reward  : {rewards.mean():.1f}")
    print(f"  Std reward   : {rewards.std():.1f}")
    print(f"  Min reward   : {rewards.min():.1f}")
    print(f"  Max reward   : {rewards.max():.1f}")
    print("-" * 64)

    # --- Cleanup -------------------------------------------------------------
    cv2.destroyAllWindows()
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

