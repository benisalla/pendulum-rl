"""Replay buffer (off-policy methods) and GAE (on-policy methods)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size ring buffer of transitions (s, a, r, s', terminated).

    Only ``terminated`` (a real end, like Acrobot reaching the goal) stops bootstrapping.
    Pendulum episodes are *truncated* at 200 steps, which is not a real end: the target
    must still bootstrap from s', otherwise the last states look artificially good.
    """

    def __init__(self, s_dim: int, a_dim: int, capacity: int = 100_000, seed: int = 0):
        self.s = np.zeros((capacity, s_dim), np.float32)
        self.a = np.zeros((capacity, a_dim), np.float32)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, s_dim), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, terminated) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {k: torch.from_numpy(getattr(self, k)[idx]) for k in ("s", "a", "r", "s2", "done")}

    def __len__(self) -> int:
        return self.size


def compute_gae(rewards, values, terminated, truncated, trunc_values, last_value, gamma: float, lam: float):
    """Generalized Advantage Estimation over one rollout of T steps (all inputs length T).

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)            (V(s_{t+1}) = 0 if terminated)
    A_t     = delta_t + gamma * lam * A_{t+1}               (the sum is cut at episode ends)

    Which V(s_{t+1}) to use: 0 after a real termination; ``trunc_values[t]`` (V of the final
    observation) after a time-limit truncation; ``last_value`` = V(s_T) for the last step of
    a rollout that stops mid-episode; otherwise simply ``values[t + 1]``.
    lam = 1 gives Monte Carlo advantages (low bias, high variance), lam = 0 the one-step TD
    error (high bias, low variance). Returns (advantages, returns = advantages + values).
    """
    T = len(rewards)
    values = np.asarray(values, np.float32)
    adv = np.zeros(T, np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        if terminated[t]:
            next_v = 0.0
        elif truncated[t]:
            next_v = trunc_values[t]
        elif t == T - 1:
            next_v = last_value
        else:
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        end = terminated[t] or truncated[t]
        gae = delta + gamma * lam * (0.0 if end else 1.0) * gae
        adv[t] = gae
    return adv, adv + values
