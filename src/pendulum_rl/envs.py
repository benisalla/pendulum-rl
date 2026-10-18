"""The two environments and the one wrapper the repo needs.

* ``pendulum`` = gymnasium ``Pendulum-v1``. Observation ``(cos theta, sin theta, theta_dot)``,
  one continuous action: the torque in [-2, 2]. Reward ``-(theta^2 + 0.1 theta_dot^2 + 0.001 u^2)``
  every step, 200 steps per episode, never terminates (only truncation). A return of about
  -150 is a good swing-up policy; we call the task solved at -200.
* ``acrobot`` = gymnasium ``Acrobot-v1``. Two-link arm, observation of 6 numbers (cos/sin of
  both joint angles and the two angular velocities), 3 discrete actions (torque -1, 0, +1 on
  the elbow), reward -1 per step until the tip swings above the bar, at most 500 steps.
  Solved at -100.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    gym_id: str
    continuous: bool
    solved_return: float  # threshold used for "steps to reach" in the comparison table
    max_steps: int


ENVS: dict[str, EnvSpec] = {
    "pendulum": EnvSpec("Pendulum-v1", continuous=True, solved_return=-200.0, max_steps=200),
    "acrobot": EnvSpec("Acrobot-v1", continuous=False, solved_return=-100.0, max_steps=500),
}


class DiscretizedAction(gym.ActionWrapper):
    """Turn a 1-D Box action space into ``Discrete(n_bins)``: bin k -> evenly spaced torque.

    With 5 bins on Pendulum the agent can push -2, -1, 0, +1, +2; that is enough for a
    swing-up, which is why plain DQN can solve a "continuous" task.
    """

    def __init__(self, env: gym.Env, n_bins: int = 5):
        super().__init__(env)
        low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
        self.torques = np.linspace(low, high, n_bins, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_bins)

    def action(self, a):
        return np.array([self.torques[int(a)]], dtype=np.float32)


def make_env(name: str, seed: int | None = None, render: bool = False, n_bins: int | None = None) -> gym.Env:
    """Create ``pendulum`` or ``acrobot``. ``n_bins`` discretizes Pendulum's torque."""
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}, choose from {list(ENVS)}")
    env = gym.make(ENVS[name].gym_id, render_mode="rgb_array" if render else None)
    if n_bins:
        env = DiscretizedAction(env, n_bins)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def obs_dim(env: gym.Env) -> int:
    return int(np.prod(env.observation_space.shape))


def act_dim(env: gym.Env) -> int:
    """Number of actions (discrete) or action dimensions (continuous)."""
    space = env.action_space
    return int(space.n) if isinstance(space, gym.spaces.Discrete) else int(np.prod(space.shape))


def is_discrete(env: gym.Env) -> bool:
    return isinstance(env.action_space, gym.spaces.Discrete)


def max_action(env: gym.Env) -> float:
    return 1.0 if is_discrete(env) else float(env.action_space.high[0])
