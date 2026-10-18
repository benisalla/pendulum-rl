"""On-policy policy-gradient methods: REINFORCE, A2C and PPO, for both action types.

All three follow grad J = E[grad log pi(a|s) * A], they only differ in what A is and how
far one batch of data is pushed:

* ``reinforce`` (Williams 1992): A = the discounted return-to-go G_t of complete episodes,
  normalised over the batch (the mean acts as a baseline). No critic, high variance.
* ``a2c`` (Mnih et al. 2016, synchronous): a critic V(s) is learned, A = GAE over short
  rollouts of ``n_steps``; one gradient step per rollout, then the data is thrown away.
* ``ppo`` (Schulman et al. 2017): long rollouts, GAE, then ``epochs`` passes of minibatch
  steps on the clipped surrogate
      L = E[min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)],  rho = pi(a|s) / pi_old(a|s).
  The clip removes the incentive to move pi more than ~eps away from the policy that
  collected the data, which is what makes reusing the batch ten times safe.

The policy is Gaussian (continuous, Pendulum) or categorical (discrete, Acrobot).
"""

from __future__ import annotations

import numpy as np
import torch

from ..buffers import compute_gae
from ..common import to_tensor
from ..networks import CategoricalPolicy, GaussianPolicy, mlp
from .base import Agent


class PolicyGradientAgent(Agent):
    uses_critic = True

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        if self.discrete:
            self.policy = CategoricalPolicy(self.s_dim, self.a_dim, cfg.hidden)
        else:
            self.policy = GaussianPolicy(self.s_dim, self.a_dim, cfg.hidden, cfg.init_log_std)
        params = list(self.policy.parameters())
        if self.uses_critic:
            self.critic = mlp(self.s_dim, 1, cfg.hidden, out_gain=1.0)
            params += list(self.critic.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

    def modules(self):
        return {"policy": self.policy, **({"critic": self.critic} if self.uses_critic else {})}

    @torch.no_grad()
    def act_batch(self, obs):
        a = self.policy.mode(to_tensor(obs)).numpy()
        return a if self.discrete else np.clip(a, -self.max_action, self.max_action)

    @torch.no_grad()
    def value_batch(self, obs):
        if not self.uses_critic:
            return None
        return self.critic(to_tensor(obs)).squeeze(-1).numpy() / self.cfg.reward_scale

    @torch.no_grad()
    def sample(self, obs):
        """Sample from pi for one observation. Returns (env action, stored action, log pi)."""
        s = to_tensor(obs)[None]
        dist = self.policy.dist(s)
        a = dist.sample()
        logp = float(self.policy.log_prob(dist, a))
        if self.discrete:
            return int(a), np.array([int(a)], np.float32), logp
        a = a[0].numpy()
        return np.clip(a, -self.max_action, self.max_action), a, logp  # log pi of the unclipped sample

    @torch.no_grad()
    def v(self, obs) -> float:
        return float(self.critic(to_tensor(obs)[None]))

    def collect(self, env, n_steps, monitor, state):
        """Run the current policy for ``n_steps`` (or until the budget ends) and record it."""
        keys = ("s", "a", "logp", "r", "v", "term", "trunc", "trunc_v")
        roll = {k: [] for k in keys}
        s, ep_ret = state
        for _ in range(n_steps):
            a_env, a, logp = self.sample(s)
            s2, r, term, trunc, _ = env.step(a_env)
            roll["s"].append(s)
            roll["a"].append(a)
            roll["logp"].append(logp)
            roll["r"].append(r * self.cfg.reward_scale)
            roll["v"].append(self.v(s))
            roll["term"].append(term)
            roll["trunc"].append(trunc)
            roll["trunc_v"].append(self.v(s2) if trunc and not term else 0.0)
            s, ep_ret = s2, ep_ret + r
            if term or trunc:
                monitor.episode(ep_ret)
                s, _ = env.reset()
                ep_ret = 0.0
            monitor.step(self)
            if monitor.done:
                break
        roll["last_v"] = self.v(s)
        return roll, (s, ep_ret)

    def advantages(self, roll):
        adv, ret = compute_gae(roll["r"], roll["v"], roll["term"], roll["trunc"], roll["trunc_v"],
                               roll["last_v"], self.cfg.gamma, self.cfg.gae_lambda)
        return torch.from_numpy(adv), torch.from_numpy(ret)

    def loss(self, s, a, old_logp, adv, ret):
        raise NotImplementedError

    def step_optimizer(self, loss):
        self.opt.zero_grad()
        loss.backward()
        params = [p for g in self.opt.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        self.opt.step()

    def learn(self, env, monitor):
        s, _ = env.reset(seed=self.cfg.seed)
        state = (s, 0.0)
        while not monitor.done:
            roll, state = self.collect(env, self.cfg.n_steps, monitor, state)
            self.update(roll)

    def update(self, roll):
        raise NotImplementedError


def _tensors(roll):
    return (to_tensor(np.array(roll["s"])), to_tensor(np.array(roll["a"])), to_tensor(np.array(roll["logp"])))


class A2C(PolicyGradientAgent):
    def update(self, roll):
        s, a, _ = _tensors(roll)
        adv, ret = self.advantages(roll)
        dist = self.policy.dist(s)
        logp = self.policy.log_prob(dist, a)
        policy_loss = -(logp * adv).mean()
        value_loss = (self.critic(s).squeeze(-1) - ret).pow(2).mean()
        entropy = self.policy.entropy(dist).mean()
        self.step_optimizer(policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy)


class PPO(PolicyGradientAgent):
    def update(self, roll):
        cfg = self.cfg
        s, a, old_logp = _tensors(roll)
        adv, ret = self.advantages(roll)
        n = len(adv)
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n).split(cfg.minibatch):
                mb_adv = adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                dist = self.policy.dist(s[idx])
                ratio = (self.policy.log_prob(dist, a[idx]) - old_logp[idx]).exp()
                surrogate = torch.min(ratio * mb_adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * mb_adv)
                value_loss = (self.critic(s[idx]).squeeze(-1) - ret[idx]).pow(2).mean()
                entropy = self.policy.entropy(dist).mean()
                self.step_optimizer(-surrogate.mean() + cfg.vf_coef * value_loss - cfg.ent_coef * entropy)


class Reinforce(PolicyGradientAgent):
    uses_critic = False

    def learn(self, env, monitor):
        cfg = self.cfg
        s, _ = env.reset(seed=cfg.seed)
        while not monitor.done:
            states, actions, returns = [], [], []
            for _ in range(cfg.episodes_per_batch):  # complete episodes: Monte Carlo returns
                rewards, done, ep_ret = [], False, 0.0
                while not done:
                    a_env, a, _ = self.sample(s)
                    s2, r, term, trunc, _ = env.step(a_env)
                    states.append(s)
                    actions.append(a)
                    rewards.append(r)
                    s, ep_ret, done = s2, ep_ret + r, term or trunc
                    monitor.step(self)
                monitor.episode(ep_ret)
                s, _ = env.reset()
                g, rtg = 0.0, []
                for r in reversed(rewards):  # G_t = r_t + gamma * G_{t+1}
                    g = r + cfg.gamma * g
                    rtg.append(g)
                returns += rtg[::-1]
                if monitor.done:
                    break
            s_t, a_t = to_tensor(np.array(states)), to_tensor(np.array(actions))
            g_t = to_tensor(np.array(returns))
            g_t = (g_t - g_t.mean()) / (g_t.std() + 1e-8)  # baseline + scale
            dist = self.policy.dist(s_t)
            loss = -(self.policy.log_prob(dist, a_t) * g_t).mean() - cfg.ent_coef * self.policy.entropy(dist).mean()
            self.step_optimizer(loss)
