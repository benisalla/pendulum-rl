"""SAC (Haarnoja et al. 2018): maximum-entropy actor-critic with a stochastic policy.

Objective: maximise E[sum_t r_t + alpha * H(pi(.|s_t))]: reward plus a bonus for staying
random. The bonus keeps exploring without hand-made noise and makes the policy robust.

Critic target (soft Bellman, twin critics as in TD3, no target actor):
    y = r + gamma * (min(Q1', Q2')(s', a') - alpha * log pi(a'|s')),  a' ~ pi(.|s')
Actor loss (reparameterised sample a = tanh(mu + sigma * eps)):
    E[alpha * log pi(a|s) - min(Q1, Q2)(s, a)]
Automatic temperature: alpha is learned so that the policy entropy stays near a target
(-dim(A)), loss = -log_alpha * (log pi(a|s) + target_entropy). When the policy is more
random than the target, alpha shrinks; when it gets too deterministic, alpha grows.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..common import to_tensor
from ..networks import SquashedGaussianActor, TwinQCritic, soft_update
from .base import OffPolicyAgent


class SAC(OffPolicyAgent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.actor = SquashedGaussianActor(self.s_dim, self.a_dim, self.max_action, cfg.hidden)
        self.critic = TwinQCritic(self.s_dim, self.a_dim, cfg.hidden)
        self.critic_t = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = -float(self.a_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def modules(self):
        return {"actor": self.actor, "critic": self.critic}

    @torch.no_grad()
    def act_batch(self, obs):
        return self.actor.deterministic(to_tensor(obs)).numpy()  # evaluation: the mean action

    @torch.no_grad()
    def value_batch(self, obs):
        s = to_tensor(obs)
        return torch.min(*self.critic(s, self.actor.deterministic(s))).numpy()

    @torch.no_grad()
    def explore(self, obs, step):
        a, _ = self.actor.sample(to_tensor(obs)[None])  # exploration = sampling the policy
        return a[0].numpy().astype(np.float32)

    def update(self, b):
        cfg = self.cfg
        with torch.no_grad():
            a2, logp2 = self.actor.sample(b["s2"])
            q_t = torch.min(*self.critic_t(b["s2"], a2)) - self.alpha * logp2
            y = b["r"] + cfg.gamma * (1.0 - b["done"]) * q_t
        q1, q2 = self.critic(b["s"], b["a"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a, logp = self.actor.sample(b["s"])
        actor_loss = (self.alpha * logp - torch.min(*self.critic(b["s"], a))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        soft_update(self.critic_t, self.critic, cfg.tau)
