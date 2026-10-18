"""One module per family of methods; see ``pendulum_rl.registry`` for the name -> class map."""

from .base import Agent, OffPolicyAgent
from .cem import CEM
from .ddpg import DDPG
from .dqn import DQN
from .policy_gradient import A2C, PPO, Reinforce
from .sac import SAC
from .sarsa import SarsaLambda
from .td3 import TD3

__all__ = ["A2C", "Agent", "CEM", "DDPG", "DQN", "OffPolicyAgent", "PPO", "Reinforce", "SAC", "SarsaLambda", "TD3"]
