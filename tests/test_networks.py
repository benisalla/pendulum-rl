import torch

from pendulum_rl.networks import (
    CategoricalPolicy,
    DeterministicActor,
    DuelingQNetwork,
    GaussianPolicy,
    QCritic,
    QNetwork,
    SquashedGaussianActor,
    TwinQCritic,
    soft_update,
)


def test_deterministic_actor_bounded():
    actor = DeterministicActor(3, 1, 2.0)
    a = actor(torch.randn(64, 3) * 100)
    assert a.shape == (64, 1) and a.abs().max() <= 2.0


def test_critics_shapes():
    s, a = torch.randn(8, 3), torch.randn(8, 1)
    assert QCritic(3, 1)(s, a).shape == (8,)
    q1, q2 = TwinQCritic(3, 1)(s, a)
    assert q1.shape == q2.shape == (8,) and not torch.allclose(q1, q2)


def test_squashed_gaussian_logprob_matches_change_of_variables():
    torch.manual_seed(0)
    pol = SquashedGaussianActor(3, 1, max_action=1.0)
    s = torch.randn(4, 3)
    a, logp = pol.sample(s)
    assert a.abs().max() < 1.0 and logp.shape == (4,)
    mu, log_std = pol(s)
    u = torch.atanh(a.clamp(-0.999999, 0.999999))
    ref = torch.distributions.Normal(mu, log_std.exp()).log_prob(u) - torch.log(1 - a.pow(2) + 1e-12)
    torch.testing.assert_close(logp, ref.sum(-1), atol=1e-3, rtol=1e-3)


def test_squashed_gaussian_reparameterised():
    pol = SquashedGaussianActor(3, 1, 2.0)
    a, _ = pol.sample(torch.randn(5, 3))
    a.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pol.parameters())


def test_policies():
    s = torch.randn(6, 3)
    g = GaussianPolicy(3, 1)
    d = g.dist(s)
    assert g.log_prob(d, d.sample()).shape == (6,)
    c = CategoricalPolicy(3, 4)
    d = c.dist(s)
    assert c.log_prob(d, d.sample().float()).shape == (6,) and c.mode(s).shape == (6,)


def test_dueling_mean_advantage_is_zero():
    net = DuelingQNetwork(6, 3)
    s = torch.randn(10, 6)
    q = net(s)
    h = net.body(s)
    torch.testing.assert_close(q.mean(-1), net.value(h).squeeze(-1))
    assert QNetwork(6, 3)(s).shape == (10, 3)


def test_soft_update():
    a, b = QNetwork(2, 2), QNetwork(2, 2)
    before = [p.clone() for p in a.parameters()]
    soft_update(a, b, 0.25)
    for pa, p0, pb in zip(a.parameters(), before, b.parameters()):
        torch.testing.assert_close(pa, 0.75 * p0 + 0.25 * pb)
