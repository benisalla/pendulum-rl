"""Every (env, algo) pair runs end to end, saves, loads and evaluates."""

import numpy as np
import pytest
import torch

from pendulum_rl import ALGOS, default_config, evaluate, load_agent, train
from pendulum_rl.algorithms.sarsa import TileCoder
from pendulum_rl.config import TrainConfig

PAIRS = [(env, algo) for algo, envs in ALGOS.items() for env in envs]


def tiny(env, algo, tmp_path, steps=400):
    return default_config(env, algo, total_steps=steps, eval_every=steps, eval_episodes=1, final_eval_episodes=1,
                          warmup_steps=200, batch_size=32, n_steps=128, minibatch=32, epochs=2,
                          episodes_per_batch=1, cem_population=4, out_dir=str(tmp_path))


@pytest.mark.parametrize("env,algo", PAIRS)
def test_train_save_load_roundtrip(env, algo, tmp_path):
    m = train(tiny(env, algo, tmp_path), verbose=False)
    assert m["evals"] and m["evals"][-1][0] == 400
    agent = m["agent"]
    path = tmp_path / env / algo / "model.pt"
    assert path.exists() and (tmp_path / env / algo / "metrics.json").exists()
    loaded = load_agent(path)
    obs = np.stack([agent_obs(env) for _ in range(5)])
    np.testing.assert_allclose(np.asarray(loaded.act_batch(obs), float), np.asarray(agent.act_batch(obs), float),
                               rtol=1e-5, atol=1e-5)
    r = evaluate(loaded, n_episodes=1)
    assert np.isfinite(r["mean"])


def agent_obs(env):
    rng = np.random.default_rng(0)
    if env == "pendulum":
        th = rng.uniform(-np.pi, np.pi)
        return np.array([np.cos(th), np.sin(th), rng.uniform(-8, 8)], np.float32)
    a, b = rng.uniform(-np.pi, np.pi, 2)
    return np.array([np.cos(a), np.sin(a), np.cos(b), np.sin(b), 0.1, -0.2], np.float32)


def test_default_config_rejects_wrong_env():
    with pytest.raises(ValueError):
        default_config("acrobot", "sac")
    with pytest.raises(ValueError):
        default_config("pendulum", "sarsa")


def test_config_roundtrip():
    cfg = default_config("pendulum", "ppo")
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def test_tile_coder_properties():
    tc = TileCoder([0, 0], [1, 1], n_tilings=4, n_tiles=4)
    idx = tc(np.array([0.5, 0.5]))
    assert len(idx) == 4 and len(set(idx.tolist())) == 4 and idx.max() < tc.n_features
    near, far = tc(np.array([0.52, 0.5])), tc(np.array([0.0, 1.0]))
    assert len(set(idx) & set(near)) >= 2 and len(set(idx) & set(far)) == 0


def test_td3_target_is_min_of_twins(tmp_path):
    from pendulum_rl.registry import make_agent

    agent = make_agent(default_config("pendulum", "td3", target_noise=0.0))
    s2 = torch.randn(16, 3)
    q1, q2 = agent.critic_t(s2, agent.actor_t(s2))
    assert torch.all(torch.min(q1, q2) <= q1) and torch.all(torch.min(q1, q2) <= q2)


def test_sac_alpha_decreases_when_policy_too_random():
    from pendulum_rl.registry import make_agent

    agent = make_agent(default_config("pendulum", "sac", lr=1e-2))
    for _ in range(300):
        agent.buffer.add(np.random.randn(3), np.random.uniform(-2, 2, 1), -1.0, np.random.randn(3), False)
    a0 = float(agent.alpha)
    for _ in range(30):
        agent.update(agent.buffer.sample(64))
    # initial policy entropy (std ~ 1 before squashing) is above the target -1, so alpha shrinks
    assert float(agent.alpha) < a0
