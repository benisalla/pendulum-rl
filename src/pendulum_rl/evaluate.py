"""The one evaluation every method shares: mean return of the deterministic policy.

Same environment, same N start states (seeds ``seed, seed + 1, ...``, disjoint from the
training seeds), no exploration noise. ``pendulum-eval runs/pendulum/sac/model.pt``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .envs import make_env


def run_episode(agent, env, seed: int | None = None, frames: list | None = None) -> float:
    s, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        if frames is not None:
            frames.append(env.render())
        s, r, term, trunc, _ = env.step(agent.act(s))
        total, done = total + r, term or trunc
    return total


def evaluate(agent, env_name: str | None = None, n_episodes: int = 10, seed: int = 10_000, env=None) -> dict:
    """Mean / std return over ``n_episodes`` deterministic episodes."""
    if env is None:
        from .registry import env_bins

        env = make_env(env_name or agent.cfg.env, n_bins=env_bins(agent.cfg))
    returns = [run_episode(agent, env, seed + i) for i in range(n_episodes)]
    return {"mean": float(np.mean(returns)), "std": float(np.std(returns)), "returns": [float(r) for r in returns]}


def main(argv=None) -> None:
    from .common import set_threads
    from .registry import load_agent

    p = argparse.ArgumentParser(description="Evaluate a saved agent: mean return over N episodes.")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000)
    args = p.parse_args(argv)
    set_threads(1)
    agent = load_agent(args.checkpoint)
    res = evaluate(agent, n_episodes=args.episodes, seed=args.seed)
    print(json.dumps({"env": agent.cfg.env, "algo": agent.cfg.algo, "mean": round(res["mean"], 1),
                      "std": round(res["std"], 1), "episodes": args.episodes}))


if __name__ == "__main__":
    main()
