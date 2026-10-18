"""DDPG (Lillicrap et al. 2016): deterministic actor-critic for continuous actions.

Critic: Q(s, a) regressed on y = r + gamma * Q'(s', mu'(s')) (primed = slow target nets).
Actor:  mu(s) moves uphill on the critic, loss = -Q(s, mu(s)) (the deterministic policy
        gradient: dQ/da * dmu/dtheta, computed by autograd through the critic).
Exploration: the actor is deterministic, so Gaussian noise is added to its action.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..common import to_tensor
from ..networks import DeterministicActor, QCritic, soft_update
from .base import OffPolicyAgent


class DDPG(OffPolicyAgent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.actor = DeterministicActor(self.s_dim, self.a_dim, self.max_action, cfg.hidden)
        self.critic = QCritic(self.s_dim, self.a_dim, cfg.hidden)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.rng = np.random.default_rng(cfg.seed)

    def modules(self):
        return {"actor": self.actor, "critic": self.critic}

    @torch.no_grad()
    def act_batch(self, obs):
        return self.actor(to_tensor(obs)).numpy()

    @torch.no_grad()
    def value_batch(self, obs):
        s = to_tensor(obs)
        return self.critic(s, self.actor(s)).numpy()  # V(s) = Q(s, mu(s))

    def explore(self, obs, step):
        a = self.act(obs) + self.rng.normal(0.0, self.cfg.exploration_noise * self.max_action, self.a_dim)
        return np.clip(a, -self.max_action, self.max_action).astype(np.float32)

    def critic_target(self, b):
        with torch.no_grad():
            return b["r"] + self.cfg.gamma * (1.0 - b["done"]) * self.critic_t(b["s2"], self.actor_t(b["s2"]))

    def update(self, b):
        critic_loss = F.mse_loss(self.critic(b["s"], b["a"]), self.critic_target(b))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(b["s"], self.actor(b["s"])).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.critic_t, self.critic, self.cfg.tau)
        soft_update(self.actor_t, self.actor, self.cfg.tau)
