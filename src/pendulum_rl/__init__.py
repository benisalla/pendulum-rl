"""pendulum_rl: every classic RL method on Pendulum-v1 (continuous) and Acrobot-v1 (discrete).

The public API, in the order one uses it:

    from pendulum_rl import TrainConfig, train, evaluate, load_agent
    cfg = TrainConfig(env="pendulum", algo="sac", total_steps=20_000)
    result = train(cfg)                       # trains, evaluates, saves runs/pendulum/sac/
    agent = load_agent("runs/pendulum/sac/model.pt")
    print(evaluate(agent, "pendulum", n_episodes=10))
"""

from .config import ALGOS, TrainConfig, default_config
from .envs import ENVS, make_env
from .evaluate import evaluate
from .registry import load_agent, make_agent
from .train import train

__all__ = [
    "ALGOS", "ENVS", "TrainConfig", "default_config", "evaluate", "load_agent", "make_agent",
    "make_env", "train",
]
__version__ = "0.1.0"
