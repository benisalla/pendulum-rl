"""05: one command per method: train any (env, algo) pair with small CPU defaults.

    python examples/05_train_any_method.py --algo td3
    python examples/05_train_any_method.py --algo discretized-dqn
    python examples/05_train_any_method.py --env acrobot --algo dueling-dqn
    python examples/05_train_any_method.py --list

The small budgets below are enough to see each method learn (or, for REINFORCE, to see
how slowly it learns). The full budgets are in pendulum_rl.config.DEFAULTS.
"""

import argparse

from pendulum_rl import ALGOS, default_config, train

SMALL = {  # environment steps for a quick, convincing run on one CPU thread
    ("pendulum", "ddpg"): 6_000, ("pendulum", "td3"): 6_000, ("pendulum", "sac"): 6_000,
    ("pendulum", "discretized-dqn"): 15_000, ("pendulum", "ppo"): 40_000, ("pendulum", "a2c"): 60_000,
    ("pendulum", "reinforce"): 60_000, ("pendulum", "cem"): 60_000,
    ("acrobot", "dqn"): 15_000, ("acrobot", "double-dqn"): 15_000, ("acrobot", "dueling-dqn"): 15_000,
    ("acrobot", "ppo"): 60_000, ("acrobot", "a2c"): 60_000, ("acrobot", "reinforce"): 60_000,
    ("acrobot", "cem"): 100_000, ("acrobot", "sarsa"): 30_000,
}

p = argparse.ArgumentParser()
p.add_argument("--env", default="pendulum", choices=["pendulum", "acrobot"])
p.add_argument("--algo", default="td3", choices=list(ALGOS))
p.add_argument("--steps", type=int, default=None)
p.add_argument("--list", action="store_true")
args = p.parse_args()
if args.list:
    for (env, algo), steps in SMALL.items():
        print(f"{env:9s} {algo:16s} {steps:>7,d} steps")
    raise SystemExit

steps = args.steps or SMALL[(args.env, args.algo)]
cfg = default_config(args.env, args.algo, total_steps=steps, eval_every=max(1_000, steps // 8),
                     final_eval_episodes=10, out_dir="examples/output/runs")
train(cfg)
print(f"saved to examples/output/runs/{args.env}/{args.algo}/model.pt; "
      f"try: pendulum-play examples/output/runs/{args.env}/{args.algo}/model.pt --gif out.gif")
