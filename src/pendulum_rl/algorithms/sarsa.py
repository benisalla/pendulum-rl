"""SARSA(lambda) with tile coding: linear on-policy TD control, no neural network.

Tile coding turns a continuous state into a sparse binary feature vector: ``n_tilings``
grids, each shifted by a fraction of a tile, cover the (angles, velocities) box; a state
activates exactly one tile per grid. Nearby states share most tiles, so learning
generalises locally, and each update only touches ``n_tilings`` weights.

Q(s, a) = sum_{i in tiles(s)} w[a, i]
delta   = r + gamma * Q(s', a') - Q(s, a)        (a' is the action actually taken next)
w      += (alpha / n_tilings) * delta * e,       e = eligibility traces
Replacing traces: e[a, tiles(s)] = 1, then e *= gamma * lambda each step, so a reward
reaches back along the whole recent trajectory (lambda = 0 is one-step SARSA).

Acrobot gives -1 per step, so initialising w = 0 is *optimistic*: unseen actions look
better than tried ones, which explores systematically even with epsilon = 0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import Agent


class TileCoder:
    def __init__(self, low, high, n_tilings: int = 8, n_tiles: int = 6):
        self.low, self.high = np.asarray(low, np.float64), np.asarray(high, np.float64)
        self.n_tilings, self.n_tiles, self.dims = n_tilings, n_tiles, len(low)
        self.per_tiling = (n_tiles + 1) ** self.dims  # +1: shifted grids need one extra tile
        self.n_features = n_tilings * self.per_tiling
        # asymmetric offsets (1, 3, 5, ...) avoid the diagonal artefacts of uniform shifts
        disp = np.array([2 * k + 1 for k in range(self.dims)], np.float64)
        self.offsets = (np.arange(n_tilings)[:, None] * disp[None, :] / n_tilings) % 1.0
        self.strides = (n_tiles + 1) ** np.arange(self.dims)

    def __call__(self, x) -> np.ndarray:
        """Indices of the ``n_tilings`` active features for state ``x``."""
        z = (np.clip(x, self.low, self.high) - self.low) / (self.high - self.low) * self.n_tiles
        coords = np.floor(z[None, :] + self.offsets).astype(np.int64)  # (n_tilings, dims)
        return np.arange(self.n_tilings) * self.per_tiling + coords @ self.strides


def acrobot_features(obs) -> np.ndarray:
    """6 numbers (cos/sin of 2 angles, 2 velocities) -> the 4 underlying state variables."""
    c1, s1, c2, s2, d1, d2 = obs
    return np.array([np.arctan2(s1, c1), np.arctan2(s2, c2), d1, d2])


class _Weights(nn.Module):
    def __init__(self, n_actions, n_features):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_actions, n_features, dtype=torch.float64), requires_grad=False)


class SarsaLambda(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        if not self.discrete:
            raise ValueError("sarsa needs discrete actions")
        self.coder = TileCoder([-np.pi, -np.pi, -4 * np.pi, -9 * np.pi], [np.pi, np.pi, 4 * np.pi, 9 * np.pi],
                               cfg.n_tilings, cfg.n_tiles)
        self.weights = _Weights(self.a_dim, self.coder.n_features)
        self.w = self.weights.w.data.numpy()  # shares memory with the saved tensor
        self.rng = np.random.default_rng(cfg.seed)

    def modules(self):
        return {"weights": self.weights}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.w = self.weights.w.data.numpy()

    def q(self, obs) -> np.ndarray:  # (n_actions,)
        return self.w[:, self.coder(acrobot_features(obs))].sum(-1)

    def act_batch(self, obs):
        return np.array([int(np.argmax(self.q(o))) for o in obs])

    def value_batch(self, obs):
        return np.array([self.q(o).max() for o in obs])

    def policy(self, tiles) -> int:
        if self.rng.random() < self.cfg.sarsa_eps:
            return int(self.rng.integers(self.a_dim))
        q = self.w[:, tiles].sum(-1)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # random tie-break

    def learn(self, env, monitor):
        cfg = self.cfg
        alpha = cfg.sarsa_alpha / cfg.n_tilings
        s, _ = env.reset(seed=cfg.seed)
        while not monitor.done:
            e = np.zeros_like(self.w)
            tiles = self.coder(acrobot_features(s))
            a = self.policy(tiles)
            done, ep_ret = False, 0.0
            while not done:
                s, r, term, trunc, _ = env.step(a)
                ep_ret += r
                done = term or trunc
                delta = r - self.w[a, tiles].sum()
                e[a, tiles] = 1.0  # replacing traces
                if not term:  # bootstrap unless the episode really ended
                    tiles2 = self.coder(acrobot_features(s))
                    a2 = self.policy(tiles2)
                    delta += cfg.gamma * self.w[a2, tiles2].sum()
                self.w += alpha * delta * e
                e *= cfg.gamma * cfg.sarsa_lambda
                if not term:
                    tiles, a = tiles2, a2
                monitor.step(self)
                if monitor.done:
                    break
            monitor.episode(ep_ret)
            s, _ = env.reset()
