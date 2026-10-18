"""Watch a trained agent: ``pendulum-play runs/pendulum/sac/model.pt --gif sac.gif``.

Prints the return of each episode; ``--gif`` renders the first one (works headless).
"""

from __future__ import annotations

import argparse

from .common import set_threads
from .envs import make_env
from .evaluate import run_episode
from .registry import env_bins, load_agent


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run a saved agent and optionally save a GIF.")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--gif", default=None, help="write the first episode to this GIF file")
    args = p.parse_args(argv)
    set_threads(1)
    agent = load_agent(args.checkpoint)
    if args.gif:
        from .viz import save_gif

        print(f"episode 0: return {save_gif(agent, args.gif, seed=args.seed):.1f} -> {args.gif}")
    env = make_env(agent.cfg.env, n_bins=env_bins(agent.cfg))
    for i in range(1 if args.gif else 0, args.episodes):
        print(f"episode {i}: return {run_episode(agent, env, seed=args.seed + i):.1f}")


if __name__ == "__main__":
    main()
