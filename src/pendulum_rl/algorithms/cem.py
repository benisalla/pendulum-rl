"""Cross-entropy method: black-box search directly in the space of policy weights.

No gradients, no value function, no Bellman equation. Keep a Gaussian N(mu, sigma^2) over
the flat weight vector of a small policy network; each generation:

1. sample ``cem_population`` weight vectors,
2. run ``cem_episodes`` episodes with each (all candidates on the same start states, so
   they are compared fairly; several start states stop the elites from overfitting one),
3. keep the best ``cem_elite_frac`` of them (the elites),
4. refit mu and sigma to the elites (plus a little decaying extra noise so sigma does not
   collapse too early).

It works when the policy is small (here 3 -> 16 -> 1, about 80 weights). Its cost is
sample efficiency: every generation spends ``population`` whole episodes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import Agent


class NumpyPolicy:
    """s -> tanh(W1 s + b1) -> W2 h + b2, with weights unpacked from one flat vector."""

    def __init__(self, s_dim: int, out_dim: int, hidden: int):
        self.shapes = [(hidden, s_dim), (hidden,), (out_dim, hidden), (out_dim,)]
        self.n_params = sum(int(np.prod(sh)) for sh in self.shapes)

    def forward(self, theta: np.ndarray, obs: np.ndarray) -> np.ndarray:  # (N, S) -> (N, out)
        parts, i = [], 0
        for sh in self.shapes:
            n = int(np.prod(sh))
            parts.append(theta[i:i + n].reshape(sh))
            i += n
        w1, b1, w2, b2 = parts
        return np.tanh(obs @ w1.T + b1) @ w2.T + b2


class _Params(nn.Module):  # lets the mean weights be saved like any other module
    def __init__(self, n):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(n), requires_grad=False)
        self.std = nn.Parameter(torch.ones(n), requires_grad=False)


class CEM(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.net = NumpyPolicy(self.s_dim, self.a_dim, cfg.hidden)
        self.params = _Params(self.net.n_params)
        self.params.std.data.fill_(cfg.cem_init_std)
        self.rng = np.random.default_rng(cfg.seed)

    def modules(self):
        return {"params": self.params}

    def _act(self, theta, obs):
        out = self.net.forward(theta, np.asarray(obs, np.float64))
        if self.discrete:
            return out.argmax(-1)
        return self.max_action * np.tanh(out)

    def act_batch(self, obs):
        return self._act(self.params.mean.numpy().astype(np.float64), obs)

    def _episode(self, env, theta, seed, monitor) -> float:
        s, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            a = self._act(theta, s[None])[0]
            s, r, term, trunc, _ = env.step(int(a) if self.discrete else a.astype(np.float32))
            total, done = total + r, term or trunc
            monitor.step(self)
        monitor.episode(total)
        return total

    def learn(self, env, monitor):
        cfg = self.cfg
        n_elite = max(2, int(cfg.cem_population * cfg.cem_elite_frac))
        mean = self.params.mean.numpy().astype(np.float64)
        std = self.params.std.numpy().astype(np.float64)
        generation = 0
        while not monitor.done:
            # the same start states for the whole population ("common random numbers")
            seeds = [int(x) for x in self.rng.integers(1 << 30, size=cfg.cem_episodes)]
            pop = mean + std * self.rng.standard_normal((cfg.cem_population, mean.size))
            scores = []
            for theta in pop:
                scores.append(np.mean([self._episode(env, theta, sd, monitor) for sd in seeds]))
                if monitor.done:
                    break
            elite = pop[np.argsort(scores)[-n_elite:]] if len(scores) == len(pop) else pop[:0]
            if len(elite):
                extra = cfg.cem_extra_std * max(0.0, 1.0 - generation / 50)
                mean, std = elite.mean(0), np.sqrt(elite.var(0) + extra**2)
                self.params.mean.data = torch.from_numpy(mean).float()
                self.params.std.data = torch.from_numpy(std).float()
            generation += 1
