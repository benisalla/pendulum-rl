"""04: Acrobot without neural networks: tile-coded SARSA(lambda) and the cross-entropy method.

Two opposite ideas on the same discrete-action task:
* SARSA(lambda): linear Q over tile-coded (angles, velocities), TD updates every step;
* CEM: no value function at all, a Gaussian search over the 150 weights of a small policy.

    python examples/04_acrobot_sarsa_vs_cem.py         # ~1 min
"""

from pendulum_rl import default_config, train
from pendulum_rl.algorithms.sarsa import TileCoder, acrobot_features
from pendulum_rl.envs import make_env

env = make_env("acrobot", seed=0)
s, _ = env.reset(seed=0)
coder = TileCoder([-3.1416, -3.1416, -12.57, -28.27], [3.1416, 3.1416, 12.57, 28.27], n_tilings=8, n_tiles=6)
print(f"observation {s.round(3)} -> state {acrobot_features(s).round(3)}")
print(f"active tiles: {coder(acrobot_features(s))} (8 of {coder.n_features} features)\n")

for algo, steps in (("sarsa", 30_000), ("cem", 100_000)):  # CEM: cheap steps, but needs ~80k on Acrobot
    m = train(default_config("acrobot", algo, total_steps=steps, eval_every=5_000,
                             final_eval_episodes=10, out_dir="examples/output/runs"))
    print(f"{algo}: {len(m['episodes'])} training episodes, final return {m['final']['mean']:.1f}\n")
