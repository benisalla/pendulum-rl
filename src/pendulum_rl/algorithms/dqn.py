"""DQN and its two classic upgrades, for discrete actions.

* ``dqn`` (Mnih et al. 2015): Q(s, .) network, epsilon-greedy behaviour, replay buffer,
  target network copied every ``target_update`` steps, target y = r + gamma * max_a Q'(s', a).
* ``double-dqn`` (van Hasselt et al. 2016): the online net *chooses* a* = argmax_a Q(s', a),
  the target net *evaluates* it: y = r + gamma * Q'(s', a*). The max of noisy estimates is
  biased upward; decoupling choice and evaluation removes most of that bias.
* ``dueling-dqn`` (Wang et al. 2016): double DQN with the V + A head of ``DuelingQNetwork``.
* ``discretized-dqn``: double DQN on Pendulum with the torque cut into ``n_bins`` values.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..common import to_tensor
from ..networks import DuelingQNetwork, QNetwork
from .base import OffPolicyAgent


class DQN(OffPolicyAgent):
    def __init__(self, env, cfg, double: bool = False, dueling: bool = False):
        super().__init__(env, cfg)
        net = DuelingQNetwork if dueling else QNetwork
        self.q = net(self.s_dim, self.a_dim, cfg.hidden)
        self.q_t = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.double = double
        self.rng = np.random.default_rng(cfg.seed)

    def modules(self):
        return {"q": self.q}

    @torch.no_grad()
    def act_batch(self, obs):
        return self.q(to_tensor(obs)).argmax(-1).numpy()

    @torch.no_grad()
    def value_batch(self, obs):
        return self.q(to_tensor(obs)).max(-1).values.numpy()  # V(s) = max_a Q(s, a)

    def epsilon(self, step: int) -> float:
        cfg = self.cfg
        frac = min(1.0, step / max(1.0, cfg.eps_fraction * cfg.total_steps))
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    def explore(self, obs, step):
        if self.rng.random() < self.epsilon(step):
            return int(self.rng.integers(self.a_dim))
        return self.act(obs)

    def update(self, b):
        a = b["a"].long().squeeze(-1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(b["s2"]).argmax(-1, keepdim=True)
                q2 = self.q_t(b["s2"]).gather(-1, a2).squeeze(-1)
            else:
                q2 = self.q_t(b["s2"]).max(-1).values
            y = b["r"] + self.cfg.gamma * (1.0 - b["done"]) * q2
        q = self.q(b["s"]).gather(-1, a[:, None]).squeeze(-1)
        loss = F.smooth_l1_loss(q, y)  # Huber: robust to the occasional huge TD error
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.n_updates % self.cfg.target_update == 0:
            self.q_t.load_state_dict(self.q.state_dict())
