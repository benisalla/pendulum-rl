"""02: load the SAC agent of example 01, evaluate it, draw what it learned and record a GIF.

* ``evaluate``: mean return of the deterministic policy over 10 fixed start states;
* ``plot_heatmaps``: the critic's V(s) = min(Q1, Q2)(s, mu(s)) and the torque mu(s) over
  (theta, theta_dot). theta = 0 is upright. Look for the "spiral" the swing-up policy
  draws: push one way to build up energy, brake near the top;
* ``save_gif``: one episode rendered headless (SDL dummy driver, no display needed).

    python examples/02_evaluate_heatmap_gif.py          # needs example 01, ~20 s
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pendulum_rl import evaluate, load_agent  # noqa: E402
from pendulum_rl.viz import plot_heatmaps, save_gif  # noqa: E402

agent = load_agent("examples/output/runs/pendulum/sac/model.pt")
res = evaluate(agent, n_episodes=10)
print(f"SAC on Pendulum-v1: {res['mean']:.1f} +- {res['std']:.1f} over 10 episodes")
print("per-episode returns:", [round(r) for r in res["returns"]])

plot_heatmaps({"sac": agent}, "examples/output/sac_heatmaps.png")
ret = save_gif(agent, "examples/output/sac_pendulum.gif")
print(f"wrote examples/output/sac_heatmaps.png and sac_pendulum.gif (episode return {ret:.1f})")
