"""
env_setup.py — Environment creation and wrapper pipeline for Super Mario Bros DQN.

Builds an SB3-compatible vectorized environment with the preprocessing pipeline
specified in project_spec.md §2:

  Raw env (240×256×3 RGB)
    → JoypadSpace(SIMPLE_MOVEMENT)   — 256 → 7 discrete actions
    → SkipFrame(skip=4)              — act every 4th frame, sum rewards
    → GrayScaleObservation           — RGB → grayscale (H×W×1)
    → ResizeObservation(84)          — downsample to 84×84
    → DummyVecEnv                    — SB3 vectorized env wrapper
    → VecFrameStack(n_stack=4)       — stack 4 frames → (84, 84, 4)

Note on wrapper ordering:
    SkipFrame is applied BEFORE grayscale/resize so we only transform the
    frames the agent actually sees (every 4th), not the skipped ones.
    This is both correct and ~4× cheaper on the preprocessing side.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# ---------------------------------------------------------------------------
# Custom gymnasium wrappers
# ---------------------------------------------------------------------------
# We write these by hand instead of pulling from gymnasium.wrappers because
# writing them is a good learning exercise and keeps the pipeline explicit.
# ---------------------------------------------------------------------------


class SkipFrame(gym.Wrapper):
    """Return only every ``skip``-th frame.

    The selected action is repeated for ``skip`` consecutive steps and the
    rewards are summed.  If the episode terminates or is truncated mid-skip
    the loop breaks immediately so no post-done frames are consumed.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    skip : int
        Number of frames to skip (default: 4).
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # Cast to Python int to prevent nes-py OverflowError on 64-bit
        # Windows where numpy int64 exceeds the C int range.
        action = int(action)
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB observations to single-channel grayscale.

    Uses the ITU-R BT.601 luminance formula::

        Y = 0.299 R + 0.587 G + 0.114 B

    Output retains a trailing channel dimension ``(H, W, 1)`` so that
    SB3's ``VecFrameStack`` can stack along ``axis=-1`` consistently.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]  # (H, W)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        return gray.astype(np.uint8)[..., np.newaxis]


class ResizeObservation(gym.ObservationWrapper):
    """Downsample observations to ``(size, size)`` using area interpolation.

    Area interpolation (``cv2.INTER_AREA``) is preferred over bilinear for
    *downsampling* because it averages pixel blocks rather than point-sampling,
    which avoids aliasing artifacts on Mario's pixel-art sprites.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    size : int
        Target height and width in pixels (default: 84).
    """

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self._size = (size, size)

        old_shape = self.observation_space.shape  # (H, W) or (H, W, C)
        new_shape = (size, size) + old_shape[2:]  # preserve channel dim
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2  # lazy import — fails fast with a clear error if missing

        # cv2.resize takes target as (width, height)
        resized = cv2.resize(obs, self._size, interpolation=cv2.INTER_AREA)

        # cv2 drops the channel dim when it's 1 — restore it
        if resized.ndim == 2:
            resized = resized[..., np.newaxis]

        return resized.astype(np.uint8)




# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_env(render_mode: str | None = "rgb_array") -> VecFrameStack:
    """Create a fully wrapped, SB3-compatible Super Mario Bros environment.

    The returned environment has:
        - Action space : Discrete(7)  — SIMPLE_MOVEMENT
        - Observation  : Box(84, 84, 4, uint8)  — 4 stacked grayscale frames

    Parameters
    ----------
    render_mode : str | None
        Render mode forwarded to ``gym_super_mario_bros.make()``.
        Use ``"rgb_array"`` for evaluation (frames rendered to numpy arrays
        and displayed via OpenCV) or ``None`` for headless training.

    Returns
    -------
    VecFrameStack
        A vectorized environment ready to be passed directly to an SB3 agent
        (e.g. ``DQN("CnnPolicy", env)``).

    Example
    -------
    >>> from env_setup import make_env
    >>> env = make_env()
    >>> obs = env.reset()
    >>> print(obs.shape)   # (1, 84, 84, 4)
    """

    def _init() -> gym.Env:
        """Build and return a single wrapped Mario environment."""

        # 1. Create base NES environment — SuperMarioBros World 1-1
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode=render_mode,
        )

        # 2. Restrict action space: 256 raw NES button combos → 7 meaningful actions
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        # 3. Frame skip: repeat each action for 4 frames, sum rewards
        #    This reduces effective FPS from 60 → 15, matching human reaction time
        env = SkipFrame(env, skip=4)

        # 4. Grayscale: (240, 256, 3) → (240, 256, 1)
        #    Color carries almost no decision-relevant information in Mario
        env = GrayScaleObservation(env)

        # 5. Resize: (240, 256, 1) → (84, 84, 1)
        #    Matches the input size from the Nature DQN paper
        env = ResizeObservation(env, size=84)

        return env

    # --- Vectorize for SB3 ---------------------------------------------------

    # DummyVecEnv wraps a list of env-factory callables into a single VecEnv.
    # We use a single env (n_envs=1) for vanilla DQN; scale up later if needed.
    env = DummyVecEnv([_init])

    # Stack 4 consecutive frames along the channel axis (channels-last).
    # Observation shape: (n_envs, 84, 84, 4)
    # SB3's CnnPolicy auto-transposes to channels-first (4, 84, 84) for PyTorch.
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    return env


# ---------------------------------------------------------------------------
# Quick smoke test (won't run on import)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = make_env()
    obs = env.reset()
    print(f"Observation shape : {obs.shape}")       # (1, 84, 84, 4)
    print(f"Action space      : {env.action_space}") # Discrete(7)
    print("Environment created successfully ✓")
    env.close()
