"""06: the comparison, small: every run found in examples/output/runs -> table + figures.

Run some of examples 01-05 first (each saves into examples/output/runs). This script then
does what ``pendulum-compare`` does for the full runs: the results table (steps to reach
the threshold, final return, wall time), the return curves and, for Pendulum, the value
and policy heatmaps of every agent side by side.

    python examples/06_compare_methods.py              # ~10 s
"""

from pathlib import Path

from pendulum_rl.metrics import load_metrics, results_table
from pendulum_rl.registry import load_agent
from pendulum_rl.viz import plot_curves, plot_heatmaps

out = Path("examples/output")
for env in ("pendulum", "acrobot"):
    runs = sorted(p.parent for p in (out / "runs" / env).glob("*/metrics.json"))
    if not runs:
        continue
    metrics = [load_metrics(r) for r in runs]
    print(f"\n{env}\n{results_table(metrics)}")
    plot_curves(metrics, out / f"{env}_curves.png")
    if env == "pendulum":
        plot_heatmaps({r.name: load_agent(r / "model.pt") for r in runs}, out / "pendulum_heatmaps.png")
print(f"\nfigures in {out}/")
