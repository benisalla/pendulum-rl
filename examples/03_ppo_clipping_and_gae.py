"""03: PPO step by step: one rollout, GAE, the clipped ratio, then a full training run.

First the pieces, on one real rollout of an untrained Gaussian policy:
  1. collect 2048 steps, 2. compute GAE advantages, 3. take a few PPO epochs and print how
  far the ratio pi_new / pi_old moved and how many samples hit the clip.
Then PPO is trained for 40 000 steps (it usually reaches about -200 within 20-40k).

    python examples/03_ppo_clipping_and_gae.py          # ~2-4 min
"""

import numpy as np
import torch

from pendulum_rl import default_config, make_env, train
from pendulum_rl.metrics import Monitor
from pendulum_rl.registry import make_agent

cfg = default_config("pendulum", "ppo", total_steps=2048, eval_every=10**9)
env = make_env("pendulum", seed=0)
agent = make_agent(cfg, env)
s, _ = env.reset(seed=0)
roll, _ = agent.collect(env, cfg.n_steps, Monitor(cfg, None, verbose=False), (s, 0.0))
adv, ret = agent.advantages(roll)
print(f"rollout: {len(roll['r'])} steps, mean scaled reward {np.mean(roll['r']):.3f}")
print(f"GAE advantages: mean {adv.mean():.3f}, std {adv.std():.3f}; value targets mean {ret.mean():.3f}")

states = torch.tensor(np.array(roll["s"]), dtype=torch.float32)
actions = torch.tensor(np.array(roll["a"]), dtype=torch.float32)
old_logp = torch.tensor(roll["logp"], dtype=torch.float32)
agent.update(roll)  # 10 epochs of clipped minibatch steps
with torch.no_grad():
    ratio = (agent.policy.log_prob(agent.policy.dist(states), actions) - old_logp).exp()
clipped = ((ratio < 1 - cfg.clip) | (ratio > 1 + cfg.clip)).float().mean()
print(f"after one PPO update: ratio in [{ratio.min():.2f}, {ratio.max():.2f}], "
      f"{100 * clipped:.0f}% of samples outside the clip range 1 +- {cfg.clip}")
print("-> the clip stops the gradient for those samples, so the policy cannot run away from the data.\n")

result = train(default_config("pendulum", "ppo", total_steps=40_000, eval_every=5_000,
                              final_eval_episodes=10, out_dir="examples/output/runs"))
