"""Do the methods actually learn? Short runs that must clearly beat a random policy."""

import numpy as np
import pytest

from pendulum_rl import default_config, make_env, train
from pendulum_rl.algorithms.dqn import DQN


def random_return(env_name, n=5):
    env = make_env(env_name, seed=0)
    out = []
    for i in range(n):
        env.reset(seed=10_000 + i)
        done, tot = False, 0.0
        while not done:
            _, r, te, tr, _ = env.step(env.action_space.sample())
            tot, done = tot + r, te or tr
        out.append(tot)
    return float(np.mean(out))


@pytest.mark.slow
def test_sac_learns_pendulum_swing_up(tmp_path):
    """SAC goes from about -1200 (random) to better than -400 in 5000 steps (opt-in: pytest -m slow)."""
    cfg = default_config("pendulum", "sac", total_steps=5_000, eval_every=5_000, eval_episodes=3,
                         final_eval_episodes=5, warmup_steps=500, out_dir=str(tmp_path))
    m = train(cfg, verbose=False, save=False)
    assert random_return("pendulum") < -900
    assert m["final"]["mean"] > -400, m["final"]


def test_dqn_overfits_a_fixed_batch():
    """Repeated updates on one batch drive the TD loss of that batch near zero."""
    import torch

    cfg = default_config("acrobot", "dqn", target_update=10**9, lr=3e-3, gamma=0.0)
    env = make_env("acrobot", seed=0)
    agent = DQN(env, cfg)
    rng = np.random.default_rng(0)
    for _ in range(64):
        agent.buffer.add(rng.normal(size=6), [rng.integers(3)], rng.normal(), rng.normal(size=6), False)
    batch = agent.buffer.sample(64)
    for _ in range(400):
        agent.update(batch)
    q = agent.q(batch["s"]).gather(-1, batch["a"].long()).squeeze(-1)
    assert torch.mean((q - batch["r"]) ** 2).item() < 0.05


def test_sarsa_learns_acrobot(tmp_path):
    """Tile-coded SARSA(lambda) beats the random policy (-500) on Acrobot within 15 000 steps."""
    cfg = default_config("acrobot", "sarsa", total_steps=15_000, eval_every=15_000, eval_episodes=3,
                         final_eval_episodes=3, out_dir=str(tmp_path))
    m = train(cfg, verbose=False, save=False)
    assert m["final"]["mean"] > -350, m["final"]

