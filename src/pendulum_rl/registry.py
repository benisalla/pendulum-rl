"""Algorithm name -> agent class, plus checkpoint save / load."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from .algorithms import A2C, CEM, DDPG, DQN, PPO, SAC, TD3, Reinforce, SarsaLambda
from .config import TrainConfig
from .envs import make_env

AGENTS = {
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "ppo": PPO,
    "a2c": A2C,
    "reinforce": Reinforce,
    "cem": CEM,
    "dqn": DQN,
    "double-dqn": partial(DQN, double=True),
    "dueling-dqn": partial(DQN, double=True, dueling=True),
    "discretized-dqn": partial(DQN, double=True),
    "sarsa": SarsaLambda,
}


def env_bins(cfg: TrainConfig) -> int | None:
    """discretized-dqn sees Pendulum through the DiscretizedAction wrapper."""
    return cfg.n_bins if cfg.algo == "discretized-dqn" else None


def make_agent(cfg: TrainConfig, env=None):
    env = env if env is not None else make_env(cfg.env, n_bins=env_bins(cfg))
    return AGENTS[cfg.algo](env, cfg)


def save_agent(agent, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": agent.cfg.to_dict(), "state": agent.state_dict()}, path)


def load_agent(path: str | Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(ckpt["config"])
    agent = make_agent(cfg)
    agent.load_state_dict(ckpt["state"])
    return agent
