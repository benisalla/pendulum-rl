"""TD3 (Fujimoto et al. 2018): DDPG plus three fixes for Q overestimation.

1. Clipped double Q: two critics, the target uses min(Q1', Q2'). One critic's errors are
   exploited by the actor (it climbs to wherever Q is wrongly high); the min is pessimistic.
2. Delayed policy updates: the actor and the targets move every ``policy_delay`` critic
   steps, so the actor follows a critic that has had time to settle.
3. Target policy smoothing: noise on the target action, a' = clip(mu'(s') + clip(eps)),
   so Q is fitted on a small neighbourhood of actions and cannot have sharp, fake peaks.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..common import to_tensor
from ..networks import DeterministicActor, TwinQCritic, soft_update
from .ddpg import DDPG


class TD3(DDPG):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.critic = TwinQCritic(self.s_dim, self.a_dim, cfg.hidden)  # fix 1
        self.critic_t = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        assert isinstance(self.actor, DeterministicActor)

    @torch.no_grad()
    def value_batch(self, obs):
        s = to_tensor(obs)
        return torch.min(*self.critic(s, self.actor(s))).numpy()

    def update(self, b):
        cfg, m = self.cfg, self.max_action
        with torch.no_grad():
            noise = (torch.randn_like(b["a"]) * cfg.target_noise * m).clamp(-cfg.noise_clip * m, cfg.noise_clip * m)
            a2 = (self.actor_t(b["s2"]) + noise).clamp(-m, m)  # fix 3
            q1_t, q2_t = self.critic_t(b["s2"], a2)
            y = b["r"] + cfg.gamma * (1.0 - b["done"]) * torch.min(q1_t, q2_t)  # fix 1
        q1, q2 = self.critic(b["s"], b["a"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.n_updates % cfg.policy_delay == 0:  # fix 2
            actor_loss = -self.critic.q1(b["s"], self.actor(b["s"])).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.critic_t, self.critic, cfg.tau)
            soft_update(self.actor_t, self.actor, cfg.tau)
