"""What every agent has in common, and the shared off-policy interaction loop.

An agent exposes four things:

* ``act_batch(obs)``: the deterministic (greedy) action for a batch of observations, used
  by evaluation, the GIF and the policy heatmap;
* ``value_batch(obs)``: its value estimate V(s) (or ``None`` if it has no critic), used by
  the value heatmap;
* ``learn(env, monitor)``: interacts with ``env`` until ``monitor.done``; it calls
  ``monitor.step(self)`` after every environment step and ``monitor.episode(ret)`` after
  every episode, and the monitor takes care of periodic evaluation and logging;
* ``modules()``: the networks to save / load.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..buffers import ReplayBuffer
from ..config import TrainConfig
from ..envs import act_dim, is_discrete, max_action, obs_dim


class Agent:
    def __init__(self, env, cfg: TrainConfig):
        self.cfg = cfg
        self.s_dim, self.a_dim = obs_dim(env), act_dim(env)
        self.discrete = is_discrete(env)
        self.max_action = max_action(env)
        self.torques = getattr(env, "torques", None)  # set when the action space is discretized

    # --- to implement -------------------------------------------------------------------
    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def value_batch(self, obs: np.ndarray) -> np.ndarray | None:
        return None

    def learn(self, env, monitor) -> None:
        raise NotImplementedError

    def modules(self) -> dict[str, nn.Module]:
        return {}

    # --- shared -------------------------------------------------------------------------
    def act(self, obs):
        """Deterministic action for one observation, in the env's own format."""
        a = self.act_batch(np.asarray(obs, np.float32)[None])[0]
        return int(a) if self.discrete else np.asarray(a, np.float32).reshape(self.a_dim)

    def state_dict(self) -> dict:
        return {k: m.state_dict() for k, m in self.modules().items()}

    def load_state_dict(self, state: dict) -> None:
        for k, m in self.modules().items():
            m.load_state_dict(state[k])


class OffPolicyAgent(Agent):
    """Act, store the transition in a replay buffer, take one gradient step on a random batch.

    Sub-classes provide ``explore(obs)`` (the behaviour policy: noisy / sampled / epsilon-greedy)
    and ``update(batch)`` (one gradient step).
    """

    def __init__(self, env, cfg: TrainConfig):
        super().__init__(env, cfg)
        self.buffer = ReplayBuffer(self.s_dim, 1 if self.discrete else self.a_dim, cfg.buffer_size, cfg.seed)
        self.n_updates = 0

    def explore(self, obs: np.ndarray, step: int):
        raise NotImplementedError

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def learn(self, env, monitor) -> None:
        cfg = self.cfg
        s, _ = env.reset(seed=cfg.seed)
        ep_ret = 0.0
        while not monitor.done:
            if monitor.steps < cfg.warmup_steps:
                a = env.action_space.sample()  # pure exploration fills the buffer first
            else:
                a = self.explore(s, monitor.steps)
            s2, r, terminated, truncated, _ = env.step(a)
            self.buffer.add(s, a, r * cfg.reward_scale, s2, terminated)
            s, ep_ret = s2, ep_ret + r
            if terminated or truncated:
                monitor.episode(ep_ret)
                s, _ = env.reset()
                ep_ret = 0.0
            if monitor.steps >= cfg.warmup_steps and len(self.buffer) >= cfg.batch_size:
                self.update(self.buffer.sample(cfg.batch_size))
                self.n_updates += 1
            monitor.step(self)
