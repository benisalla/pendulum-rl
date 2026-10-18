"""Training log and the numbers of the comparison table."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .envs import ENVS, make_env
from .evaluate import evaluate


class Monitor:
    """Counts environment steps, evaluates every ``eval_every`` steps, logs everything.

    Agents call ``step(agent)`` after each env step and ``episode(ret)`` after each episode;
    ``done`` turns True when the step budget is spent. That is the only contact between the
    algorithms and the bookkeeping.
    """

    def __init__(self, cfg, eval_env, verbose: bool = True):
        self.cfg, self.eval_env, self.verbose = cfg, eval_env, verbose
        self.steps = 0
        self.episodes: list[tuple[int, float]] = []   # (step, training-episode return)
        self.evals: list[tuple[int, float, float, float]] = []  # (step, mean, std, wall seconds)
        self.t0 = time.time()

    @property
    def done(self) -> bool:
        return self.steps >= self.cfg.total_steps

    def episode(self, ret: float) -> None:
        self.episodes.append((self.steps, float(ret)))

    def step(self, agent) -> None:
        self.steps += 1
        if self.steps % self.cfg.eval_every == 0 or self.steps == self.cfg.total_steps:
            res = evaluate(agent, n_episodes=self.cfg.eval_episodes, env=self.eval_env)
            self.evals.append((self.steps, res["mean"], res["std"], time.time() - self.t0))
            if self.verbose:
                print(f"[{self.cfg.env}/{self.cfg.algo}] step {self.steps:>7d}  eval return "
                      f"{res['mean']:8.1f} +- {res['std']:5.1f}  ({time.time() - self.t0:5.0f} s)", flush=True)


def steps_to_threshold(evals, threshold: float) -> int | None:
    """First evaluation step at which the mean return reached ``threshold`` (None = never)."""
    for step, mean, *_ in evals:
        if mean >= threshold:
            return int(step)
    return None


def load_metrics(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text())


def results_table(metrics: list[dict]) -> str:
    """Markdown table: steps to reach the threshold, final return, wall time."""
    if not metrics:
        return ""
    env = metrics[0]["config"]["env"]
    thr = ENVS[env].solved_return
    rows = [f"| method | env steps | steps to reach {thr:.0f} | final return (mean +- std, "
            f"{metrics[0]['final']['n']} episodes) | best eval | wall time |", "|---|---|---|---|---|---|"]
    for m in sorted(metrics, key=lambda m: -m["final"]["mean"]):
        reach = m.get("steps_to_threshold")
        best = max(e[1] for e in m["evals"]) if m["evals"] else float("nan")
        rows.append(f"| `{m['config']['algo']}` | {m['config']['total_steps']:,} | "
                    f"{f'{reach:,}' if reach else 'not reached'} | {m['final']['mean']:.1f} +- {m['final']['std']:.1f} | "
                    f"{best:.1f} | {m['wall_time'] / 60:.1f} min |")
    return "\n".join(rows)


def make_eval_env(cfg):
    from .registry import env_bins

    return make_env(cfg.env, n_bins=env_bins(cfg))
