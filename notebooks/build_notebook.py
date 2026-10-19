"""Build notebooks/train_colab_kaggle.ipynb from the package sources.

    python notebooks/build_notebook.py

The notebook installs the package from GitHub, explains each method in markdown, shows
the exact code of that method (extracted from src/ at build time, so it never drifts), and
runs the full comparison with larger budgets than the local CPU runs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pendulum_rl"
OUT = ROOT / "notebooks" / "train_colab_kaggle.ipynb"


def source_of(rel: str, name: str) -> str:
    """Source code of class/function ``name`` (``Class.method`` allowed) in src/pendulum_rl/<rel>."""
    text = (SRC / rel).read_text()
    tree = ast.parse(text)
    parts = name.split(".")
    nodes = tree.body
    node = None
    for p in parts:
        node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == p)
        nodes = node.body
    lines = text.splitlines()
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
    body = lines[start:node.end_lineno]
    indent = len(body[0]) - len(body[0].lstrip())
    return "\n".join(line[indent:] for line in body)


cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


def show(rel: str, *names: str, intro: str = "") -> None:
    blocks = "\n\n".join(f"```python\n# src/pendulum_rl/{rel}\n{source_of(rel, n)}\n```" for n in names)
    md((intro + "\n\n" if intro else "") + blocks)


md("""
# Pendulum swing-up and Acrobot with every RL method

This notebook is generated from the `pendulum_rl` package (`notebooks/build_notebook.py`), so the
code shown in each section is exactly the code that runs. It goes through the methods one at a
time, in the order that makes the ideas build on each other:

1. the environments and what "continuous action" changes,
2. deterministic vs stochastic policies,
3. DDPG (actor-critic), 4. TD3 (three fixes), 5. SAC (entropy),
6. policy gradients: REINFORCE, A2C, GAE and PPO's clipping,
7. CEM, discretized DQN, and the Acrobot methods (DQN variants, tile-coded SARSA),
8. the full comparison: curves, table, value/policy heatmaps, GIFs.

Runs on CPU (Kaggle / Colab CPU is fine: these networks are tiny and **one thread is the fastest**;
a GPU does not help here). The full comparison at the end takes about 30-60 minutes.
""")

md("""
## 0. Install

The repository is private: on Kaggle add a GitHub token as a secret (`GITHUB_TOKEN`) or upload the
repo as a dataset and `pip install` it from there. On a machine with the repo cloned, `pip install -e .`.
""")
code("""
import os, sys, subprocess
os.environ["SDL_VIDEODRIVER"] = "dummy"          # headless rendering for GIFs
os.environ["OMP_NUM_THREADS"] = "1"
try:
    import pendulum_rl
except ImportError:
    token = os.environ.get("GITHUB_TOKEN", "")
    try:  # Kaggle secret
        from kaggle_secrets import UserSecretsClient
        token = token or UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        pass
    auth = f"{token}@" if token else ""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"git+https://{auth}github.com/benisalla/pendulum-rl.git"], check=True)
import torch, numpy as np, matplotlib.pyplot as plt
torch.set_num_threads(1)
import pendulum_rl
from pendulum_rl import default_config, train, evaluate, make_env, load_agent, ALGOS
print(pendulum_rl.__version__, torch.__version__)
""")

md("""
## 1. The environments

**Pendulum-v1.** Observation `(cos theta, sin theta, theta_dot)`, one action: a torque in `[-2, 2]`.
Reward `-(theta^2 + 0.1 theta_dot^2 + 0.001 u^2)` every step, 200 steps, never terminates.
The motor is too weak to lift the rod directly, so the policy must rock it to build energy, then
brake at the top. Random: about -1200. Good: about -150. We call it solved at **-200**.

**Acrobot-v1.** Two links, motor on the elbow only, 3 discrete torques, -1 per step until the tip
swings above the line (then it terminates), at most 500 steps. Random: -500. Solved at **-100**.

Why it matters that Pendulum is *truncated* and not *terminated*: step 200 is not a real end, so
value targets still bootstrap from the last state. The buffer stores only `terminated` as "done".
""")
code("""
for name in ("pendulum", "acrobot"):
    env = make_env(name, seed=0)
    s, _ = env.reset(seed=0)
    print(name, "obs", s.round(3), "action space", env.action_space)

env = make_env("pendulum", render=True)
env.reset(seed=1)
plt.imshow(env.render()); plt.axis("off"); plt.title("Pendulum-v1 (theta = 0 is upright)"); plt.show()
""")

md("""
## 2. Continuous actions and deterministic vs stochastic policies

DQN acts with `argmax_a Q(s, a)`. With a real-valued torque that max is an optimisation problem at
every step. The way out is to *learn a network that outputs the action*:

* a **deterministic** actor `a = mu(s)` (DDPG, TD3): exploration must be added from outside as noise,
  and the actor is trained by backpropagating through the critic `Q(s, mu(s))`;
* a **stochastic** policy `a ~ pi(.|s)` (SAC, PPO, A2C, REINFORCE): sampling *is* the exploration;
  trained with `grad log pi(a|s) * advantage` or, in SAC, by reparameterisation.

Other ways: search the policy weights with no gradient (CEM), or discretize the torque and keep DQN.
""")
show("networks.py", "DeterministicActor", "GaussianPolicy", "SquashedGaussianActor",
     intro="The three continuous policies:")

md("""
## 3. DDPG: the deterministic actor-critic

The critic learns "how good is action a in state s" by regression on a bootstrapped target from
slow **target networks**; the actor moves its output in the direction the critic says is better.
`actor_loss = -critic(s, actor(s)).mean()`: autograd does `dQ/da * dmu/dtheta` for us.

Every off-policy method uses the same loop: act (random during warm-up, then `explore`), store the
transition, take one gradient step on a random batch from the replay buffer.
""")
show("algorithms/base.py", "OffPolicyAgent.learn")
show("algorithms/ddpg.py", "DDPG.explore", "DDPG.critic_target", "DDPG.update")
code("""
ddpg = train(default_config("pendulum", "ddpg", total_steps=15_000))
""")

md("""
## 4. TD3: three fixes for overestimation

The DDPG actor is an optimiser of the critic: wherever Q is wrongly high, it goes there. TD3:

1. **clipped double Q**: two critics, the target uses the min;
2. **delayed policy updates**: actor and targets move every 2 critic steps;
3. **target policy smoothing**: clipped noise on the target action, so Q cannot have sharp fake peaks.
""")
show("algorithms/td3.py", "TD3.update")
code("""
td3 = train(default_config("pendulum", "td3", total_steps=15_000))
""")

md("""
## 5. SAC: reward plus entropy

SAC maximises `E[sum r + alpha * H(pi)]`. The entropy bonus replaces hand-made exploration noise.
The actor samples `a = 2 tanh(mu + sigma * eps)` (reparameterised, so gradients flow through the
critics) and the log-probability is corrected for the tanh. The temperature `alpha` is *learned*:
it drops when the policy is more random than the target entropy (-1) and rises when it collapses.
""")
show("networks.py", "SquashedGaussianActor.sample")
show("algorithms/sac.py", "SAC.update")
code("""
sac = train(default_config("pendulum", "sac", total_steps=15_000))
print("learned alpha:", float(sac["agent"].alpha))
""")

md("""
## 6. Policy gradients: REINFORCE, A2C, GAE and PPO

All three ascend `E[grad log pi(a|s) * A]`.

* **REINFORCE**: `A` = normalised return-to-go of complete episodes. Unbiased, very noisy.
* **A2C**: a learned critic V(s), `A` from GAE over short rollouts, one gradient step per rollout.
* **GAE**: `A_t = sum_k (gamma lambda)^k delta_{t+k}` with `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`.
  lambda = 0 trusts the critic, lambda = 1 ignores it; 0.95 is the compromise.
* **PPO**: long rollouts reused for 10 epochs. The clipped ratio `min(rho A, clip(rho, 0.8, 1.2) A)`
  stops the gradient once an action's probability has moved 20% from the policy that collected
  the data: that is what makes the reuse safe.
""")
show("buffers.py", "compute_gae")
show("algorithms/policy_gradient.py", "PPO.update", "A2C.update")
code("""
ppo = train(default_config("pendulum", "ppo"))          # 100k steps
a2c = train(default_config("pendulum", "a2c"))          # 150k steps
reinforce = train(default_config("pendulum", "reinforce"))  # 600k steps
""")

md("""
## 7. CEM and discretized DQN

**CEM** keeps a Gaussian over the 81 weights of a tiny policy, samples 50 of them, runs one episode
each (3 episodes per candidate, on the same start states), refits the Gaussian on the best 20%. No gradient at all.

**Discretized DQN**: 5 torques (-2, -1, 0, 1, 2) are enough for a swing-up, so double DQN works.
""")
show("algorithms/cem.py", "CEM.learn")
show("algorithms/dqn.py", "DQN.update")
code("""
cem = train(default_config("pendulum", "cem"))
ddqn = train(default_config("pendulum", "discretized-dqn"))
""")

md("""
## 8. Pendulum comparison: curves, table, heatmaps

Same evaluation for everyone: deterministic policy, the same start states. The heatmaps show what
each agent learned over `(theta, theta_dot)`: V(s) (top) and the torque (bottom). A swing-up policy
draws a spiral: push in one direction to gain energy, reverse near the top.
""")
code("""
from pendulum_rl.metrics import results_table
from pendulum_rl.viz import plot_curves, plot_heatmaps
from IPython.display import Markdown, display
pend = {"ddpg": ddpg, "td3": td3, "sac": sac, "ppo": ppo, "a2c": a2c, "reinforce": reinforce,
        "cem": cem, "discretized-dqn": ddqn}
display(Markdown(results_table(list(pend.values()))))
plot_curves(list(pend.values())); plt.show()
plot_heatmaps({k: v["agent"] for k, v in pend.items()}); plt.show()
""")

md("""
## 9. Acrobot: the discrete methods

Every step costs -1 until the goal: nothing is learned until exploration reaches it once.
**DQN** (max target), **double DQN** (online net picks the action, target net evaluates it),
**dueling DQN** (`Q = V + A - mean A`), the policy-gradient trio with a categorical policy, CEM,
and **SARSA(lambda) with tile coding**: a linear Q over 8 shifted grids, eligibility traces, and
optimistic zero initialisation that explores without epsilon.
""")
show("algorithms/sarsa.py", "TileCoder", "SarsaLambda.learn")
code("""
acro = {}
for algo in ["dqn", "double-dqn", "dueling-dqn", "ppo", "a2c", "reinforce", "cem", "sarsa"]:
    acro[algo] = train(default_config("acrobot", algo))
display(Markdown(results_table(list(acro.values()))))
plot_curves(list(acro.values())); plt.show()
""")

md("""
## 10. Watch them

`save_gif` renders one deterministic episode headless.
""")
code("""
from pendulum_rl.viz import save_gif
from IPython.display import Image as IPImage
for name in ("sac", "td3", "ppo"):
    save_gif(pend[name]["agent"], f"{name}.gif")
display(IPImage(filename="sac.gif"))
""")

md("""
## 11. Going further

* several seeds per method (`default_config(..., seed=k)`): RL results vary a lot between seeds;
* bigger budgets for REINFORCE / A2C, or PPO with `gamma=0.99` to see why the shorter horizon helps;
* `n_bins=11` for discretized DQN, `sarsa_lambda=0` for one-step SARSA, `auto_alpha=False` for SAC.
Checkpoints saved in `runs/` load in the repo's CLIs: `pendulum-play runs/pendulum/sac/model.pt --gif sac.gif`.
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")
