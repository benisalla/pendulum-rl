"""Train any (env, algo) pair: ``pendulum-train --env pendulum --algo sac``.

Every field of ``TrainConfig`` is a flag (``--total-steps``, ``--lr``, ...). Without flags
the tuned defaults of ``config.DEFAULTS`` are used. Results go to ``runs/<env>/<algo>/``:
``model.pt`` (load with ``load_agent``) and ``metrics.json`` (curves, final evaluation).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

from .common import set_seed, set_threads
from .config import ALGOS, TrainConfig, default_config
from .envs import ENVS, make_env
from .evaluate import evaluate
from .metrics import Monitor, make_eval_env, steps_to_threshold
from .registry import env_bins, make_agent, save_agent


def train(cfg: TrainConfig, verbose: bool = True, save: bool = True) -> dict:
    """Train, evaluate on ``final_eval_episodes`` and (optionally) save. Returns the metrics."""
    set_threads(cfg.threads)
    set_seed(cfg.seed)
    env = make_env(cfg.env, seed=cfg.seed, n_bins=env_bins(cfg))
    agent = make_agent(cfg, env)
    monitor = Monitor(cfg, make_eval_env(cfg), verbose)
    t0 = time.time()
    agent.learn(env, monitor)
    wall = time.time() - t0
    final = evaluate(agent, n_episodes=cfg.final_eval_episodes, env=make_eval_env(cfg))
    metrics = {
        "config": cfg.to_dict(),
        "evals": monitor.evals,
        "episodes": monitor.episodes,
        "final": {"mean": final["mean"], "std": final["std"], "n": cfg.final_eval_episodes},
        "wall_time": wall,
        "steps_to_threshold": steps_to_threshold(monitor.evals, ENVS[cfg.env].solved_return),
    }
    if save:
        out = Path(cfg.out_dir) / cfg.env / cfg.algo
        out.mkdir(parents=True, exist_ok=True)
        save_agent(agent, out / "model.pt")
        (out / "metrics.json").write_text(json.dumps(metrics))
    if verbose:
        print(f"[{cfg.env}/{cfg.algo}] final return {final['mean']:.1f} +- {final['std']:.1f} "
              f"over {cfg.final_eval_episodes} episodes, {wall / 60:.1f} min")
    metrics["agent"] = agent
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one RL method on Pendulum-v1 or Acrobot-v1.")
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "env":
            p.add_argument(flag, default="pendulum", choices=list(ENVS))
        elif f.name == "algo":
            p.add_argument(flag, default="sac", choices=list(ALGOS))
        elif f.type in (bool, "bool"):
            p.add_argument(flag, type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__)
            p.add_argument(flag, type=typ, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("env", "algo", "quiet")}
    cfg = default_config(args.env, args.algo, **overrides)
    train(cfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
