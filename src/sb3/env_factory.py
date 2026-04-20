"""
Environment factory functions for creating vectorized environments.
These can be instantiated via Hydra's instantiate mechanism.
"""
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from typing import Optional


def create_atari_env(
    env_id: str,
    n_envs: int = 4,
    n_stack: int = 4,
    seed: Optional[int] = None,
    **kwargs
):
    """
    Factory function to create a vectorized Atari environment.

    Args:
        env_id: The Atari environment ID (e.g., "BreakoutNoFrameskip-v4")
        n_envs: Number of parallel environments
        n_stack: Number of frames to stack (default: 4)
        seed: Random seed for environment
        **kwargs: Additional arguments passed to make_atari_env

    Returns:
        A vectorized and frame-stacked Atari environment
    """
    env = make_atari_env(env_id, n_envs=n_envs, **kwargs)
    env = VecFrameStack(env, n_stack=n_stack)
    if seed is not None:
        env.seed(seed)
    return env


def create_mujoco_env(
    env_id: str,
    n_envs: int = 1,
    render_mode: str = "rgb_array",
    seed: Optional[int] = None,
    **kwargs
):
    """
    Factory function to create a vectorized MuJoCo/standard Gymnasium environment.

    Args:
        env_id: The environment ID (e.g., "HalfCheetah-v5")
        n_envs: Number of parallel environments
        render_mode: Render mode for the environment
        seed: Random seed for environment
        **kwargs: Additional arguments passed to gym.make

    Returns:
        A vectorized environment wrapped with Monitor
    """
    def make_env():
        env = gym.make(env_id, render_mode=render_mode, **kwargs)
        env = Monitor(env)  # record stats such as returns
        return env

    envs = [make_env for _ in range(n_envs)]
    env = DummyVecEnv(envs)
    if seed is not None:
        env.seed(seed)
    return env

