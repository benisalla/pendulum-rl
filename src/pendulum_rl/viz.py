"""Figures: return curves, Pendulum value / policy heatmaps, GIFs of an episode."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .envs import ENVS  # noqa: E402

# one fixed colour per method, the same in every figure
COLORS = {
    "sac": "#1f77b4", "td3": "#ff7f0e", "ddpg": "#2ca02c", "discretized-dqn": "#d62728", "ppo": "#9467bd",
    "a2c": "#8c564b", "reinforce": "#e377c2", "cem": "#7f7f7f", "dqn": "#bcbd22", "double-dqn": "#17becf",
    "dueling-dqn": "#d62728", "sarsa": "#2ca02c",
}


def plot_curves(metrics: list[dict], path: str | Path | None = None, log_x: bool = True, title: str | None = None):
    """Evaluation return vs environment steps, one line per method, with the solved line."""
    env = metrics[0]["config"]["env"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in metrics:
        algo = m["config"]["algo"]
        ev = np.array(m["evals"])
        if len(ev) == 0:
            continue
        ax.plot(ev[:, 0], ev[:, 1], label=algo, color=COLORS.get(algo), lw=2)
        ax.fill_between(ev[:, 0], ev[:, 1] - ev[:, 2], ev[:, 1] + ev[:, 2], color=COLORS.get(algo), alpha=0.12)
    ax.axhline(ENVS[env].solved_return, color="k", ls="--", lw=1, label=f"solved ({ENVS[env].solved_return:.0f})")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("environment steps" + (" (log scale)" if log_x else ""))
    ax.set_ylabel("evaluation return (deterministic policy)")
    ax.set_title(title or f"{ENVS[env].gym_id}: every method, same evaluation")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def pendulum_grid(n: int = 61):
    """Observations covering (theta, theta_dot) in [-pi, pi] x [-8, 8]; theta = 0 is upright."""
    theta = np.linspace(-np.pi, np.pi, n)
    theta_dot = np.linspace(-8, 8, n)
    T, D = np.meshgrid(theta, theta_dot)
    obs = np.stack([np.cos(T), np.sin(T), D], -1).reshape(-1, 3).astype(np.float32)
    return theta, theta_dot, obs


def policy_torque(agent, obs) -> np.ndarray:
    a = np.asarray(agent.act_batch(obs))
    if agent.torques is not None:  # discretized-dqn: bin index -> torque
        return agent.torques[a.astype(int).reshape(-1)]
    return a.reshape(len(obs), -1)[:, 0]


def plot_heatmaps(agents: dict, path: str | Path | None = None, n: int = 61):
    """For each Pendulum agent: V(s) (top row) and the torque mu(s) (bottom row) over (theta, theta_dot)."""
    theta, theta_dot, obs = pendulum_grid(n)
    names = list(agents)
    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 6.0), squeeze=False)
    ext = [theta[0], theta[-1], theta_dot[0], theta_dot[-1]]
    for j, name in enumerate(names):
        ag = agents[name]
        v = ag.value_batch(obs)
        ax = axes[0, j]
        if v is None:
            ax.text(0.5, 0.5, "no value\nfunction", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            im = ax.imshow(np.asarray(v).reshape(n, n), origin="lower", extent=ext, aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{name}: V(s)", fontsize=10)
        ax = axes[1, j]
        im = ax.imshow(policy_torque(ag, obs).reshape(n, n), origin="lower", extent=ext, aspect="auto",
                       cmap="coolwarm", vmin=-2, vmax=2)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{name}: torque", fontsize=10)
        ax.set_xlabel("theta (0 = upright)")
    for i in range(2):
        axes[i, 0].set_ylabel("theta_dot")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def save_gif(agent, path: str | Path, seed: int = 10_000, every: int = 2, size: int = 200) -> float:
    """Render one deterministic episode to a GIF (headless: SDL dummy video driver). Returns its return."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from PIL import Image

    from .envs import make_env
    from .evaluate import run_episode
    from .registry import env_bins

    env = make_env(agent.cfg.env, render=True, n_bins=env_bins(agent.cfg))
    frames: list = []
    ret = run_episode(agent, env, seed=seed, frames=frames)
    images = [Image.fromarray(f).resize((size, size)) for f in frames[::every]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=40 * every, loop=0)
    env.close()
    return ret
