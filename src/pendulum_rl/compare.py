"""Compare methods on one environment: ``pendulum-compare --env pendulum [--train]``.

With ``--train`` every applicable method is trained one after the other with its default
budget (or ``--steps-scale`` times it). Then, from ``runs/<env>/*/metrics.json``, it writes:

* ``<assets>/<env>_results.md``: the table (steps to reach the threshold, final return, time)
* ``<assets>/<env>_curves.png``: evaluation return curves of all methods
* ``<assets>/pendulum_heatmaps.png``: value and policy over (theta, theta_dot) (Pendulum only)
* ``<assets>/<env>_<algo>.gif`` with ``--gif``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import set_threads
from .config import ALGOS, default_config
from .metrics import load_metrics, results_table
from .registry import load_agent


def methods_for(env: str) -> list[str]:
    return [a for a, envs in ALGOS.items() if env in envs]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train (optional) and compare every method on one env.")
    p.add_argument("--env", default="pendulum", choices=["pendulum", "acrobot"])
    p.add_argument("--algos", nargs="*", default=None, help="subset of methods (default: all applicable)")
    p.add_argument("--train", action="store_true", help="train the methods first (one at a time)")
    p.add_argument("--steps-scale", type=float, default=1.0, help="multiply every default budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--assets", default="assets")
    p.add_argument("--gif", action="store_true", help="also save one GIF per method")
    args = p.parse_args(argv)
    set_threads(1)

    algos = args.algos or methods_for(args.env)
    if args.train:
        from .train import train

        for algo in algos:
            base = default_config(args.env, algo)
            cfg = default_config(args.env, algo, seed=args.seed, out_dir=args.runs_dir,
                                 total_steps=int(base.total_steps * args.steps_scale),
                                 eval_every=max(1, int(base.eval_every * args.steps_scale)))
            train(cfg)

    runs = Path(args.runs_dir) / args.env
    found = [a for a in algos if (runs / a / "metrics.json").exists()]
    if not found:
        raise SystemExit(f"no runs found in {runs}; use --train")
    metrics = [load_metrics(runs / a) for a in found]
    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)

    from .viz import plot_curves, plot_heatmaps, save_gif

    table = results_table(metrics)
    (assets / f"{args.env}_results.md").write_text(table + "\n")
    print(table)
    plot_curves(metrics, assets / f"{args.env}_curves.png")
    agents = {a: load_agent(runs / a / "model.pt") for a in found}
    if args.env == "pendulum":
        plot_heatmaps(agents, assets / "pendulum_heatmaps.png")
    if args.gif:
        for a, ag in agents.items():
            save_gif(ag, assets / f"{args.env}_{a}.gif")
    print(f"figures written to {assets}/")


if __name__ == "__main__":
    main()
