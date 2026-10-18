import numpy as np
import pytest

from pendulum_rl.buffers import ReplayBuffer, compute_gae
from pendulum_rl.envs import ENVS, act_dim, is_discrete, make_env, max_action, obs_dim


@pytest.mark.parametrize("name", list(ENVS))
def test_make_env_shapes(name):
    env = make_env(name, seed=0)
    s, _ = env.reset(seed=0)
    assert s.shape == (obs_dim(env),)
    assert is_discrete(env) == (not ENVS[name].continuous)


def test_discretized_pendulum():
    env = make_env("pendulum", seed=0, n_bins=5)
    assert is_discrete(env) and act_dim(env) == 5
    np.testing.assert_allclose(env.torques, [-2, -1, 0, 1, 2])
    assert env.action(4)[0] == pytest.approx(2.0)
    env.reset(seed=0)
    env.step(3)


def test_max_action():
    assert max_action(make_env("pendulum")) == 2.0


def test_replay_buffer_ring_and_sample():
    buf = ReplayBuffer(3, 1, capacity=10)
    for i in range(15):
        buf.add(np.full(3, i), [i], -i, np.full(3, i + 1), i % 2)
    assert len(buf) == 10
    b = buf.sample(32)
    assert b["s"].shape == (32, 3) and b["a"].shape == (32, 1) and b["r"].shape == (32,)
    assert set(b["s"][:, 0].tolist()) <= set(range(5, 15))  # oldest five overwritten
    np.testing.assert_allclose(b["s2"][:, 0].numpy(), b["s"][:, 0].numpy() + 1)


def test_gae_lambda_one_is_monte_carlo():
    r = [1.0, 1.0, 1.0]
    v = [0.5, 0.2, 0.1]
    adv, ret = compute_gae(r, v, [False, False, True], [False] * 3, [0.0] * 3, 0.0, gamma=0.9, lam=1.0)
    mc = [1 + 0.9 + 0.81, 1 + 0.9, 1.0]
    np.testing.assert_allclose(ret, mc, rtol=1e-6)
    np.testing.assert_allclose(adv, np.array(mc) - v, rtol=1e-6)


def test_gae_lambda_zero_is_td_error_and_truncation_bootstraps():
    r, v = [1.0, 1.0], [0.5, 0.2]
    adv, _ = compute_gae(r, v, [False, False], [False, True], [0.0, 3.0], 99.0, gamma=0.9, lam=0.0)
    np.testing.assert_allclose(adv, [1 + 0.9 * 0.2 - 0.5, 1 + 0.9 * 3.0 - 0.2], rtol=1e-6)


def test_gae_rollout_cut_uses_last_value():
    adv, _ = compute_gae([0.0], [0.0], [False], [False], [0.0], 10.0, gamma=0.5, lam=0.95)
    assert adv[0] == pytest.approx(5.0)
