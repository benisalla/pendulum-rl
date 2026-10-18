"""01: train SAC on Pendulum-v1 from Python and watch the evaluation return climb.

SAC is the most reliable method here: a stochastic squashed-Gaussian actor, two critics,
and an entropy temperature that tunes itself. 6 000 environment steps are enough for it
to swing the pendulum up and hold it (random policy: about -1200; good policy: about -150).

    python examples/01_train_sac_pendulum.py            # ~2-6 min on one CPU thread

Writes examples/output/runs/pendulum/sac/{model.pt, metrics.json}, used by example 02.
"""

from pendulum_rl import default_config, train

cfg = default_config(
    env="pendulum",
    algo="sac",
    total_steps=6_000,    # the full comparison uses 15 000
    eval_every=1_000,     # deterministic evaluation (mean action) every 1 000 steps
    eval_episodes=5,
    final_eval_episodes=10,
    out_dir="examples/output/runs",
)
print("config:", {k: getattr(cfg, k) for k in ("hidden", "lr", "gamma", "batch_size", "tau", "init_alpha")})
result = train(cfg)

agent = result["agent"]
print(f"learned temperature alpha = {float(agent.alpha):.3f} (started at {cfg.init_alpha})")
print(f"final return {result['final']['mean']:.1f} +- {result['final']['std']:.1f}")
