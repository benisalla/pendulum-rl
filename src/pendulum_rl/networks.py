"""Every neural network the algorithms use, all built from one ``mlp`` helper.

Shapes: B = batch, S = observation size, A = action size (continuous) or number of actions.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical, Normal


def mlp(in_dim: int, out_dim: int, hidden: int = 128, n_layers: int = 2, out_gain: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    last = nn.Linear(d, out_dim)
    if out_gain is not None:  # small last layer -> near-zero initial outputs (PPO trick)
        nn.init.orthogonal_(last.weight, out_gain)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """mu(s) = max_action * tanh(f(s)). DDPG and TD3: one action per state, no randomness."""

    def __init__(self, s_dim: int, a_dim: int, max_action: float, hidden: int = 128):
        super().__init__()
        self.net, self.max_action = mlp(s_dim, a_dim, hidden), max_action

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # (B, S) -> (B, A)
        return self.max_action * torch.tanh(self.net(s))


class QCritic(nn.Module):
    """Q(s, a) for continuous actions: the action is an input, the output is one number."""

    def __init__(self, s_dim: int, a_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(s_dim + a_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:  # -> (B,)
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class TwinQCritic(nn.Module):
    """Two independent Q networks (TD3, SAC): the minimum of the two fights overestimation."""

    def __init__(self, s_dim: int, a_dim: int, hidden: int = 128):
        super().__init__()
        self.q1, self.q2 = QCritic(s_dim, a_dim, hidden), QCritic(s_dim, a_dim, hidden)

    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class SquashedGaussianActor(nn.Module):
    """SAC policy: u ~ N(mu(s), sigma(s)), a = max_action * tanh(u).

    ``sample`` uses the reparameterisation trick (u = mu + sigma * eps) so the gradient of
    Q(s, a) flows into the actor, and corrects log pi for the tanh squashing:
    log pi(a|s) = log N(u) - sum log(1 - tanh(u)^2).
    """

    def __init__(self, s_dim: int, a_dim: int, max_action: float, hidden: int = 128):
        super().__init__()
        self.body = mlp(s_dim, 2 * a_dim, hidden)
        self.max_action = max_action

    def forward(self, s):
        mu, log_std = self.body(s).chunk(2, dim=-1)
        return mu, log_std.clamp(-5.0, 2.0)

    def sample(self, s):  # -> action (B, A), log_prob (B,)
        mu, log_std = self(s)
        dist = Normal(mu, log_std.exp())
        u = dist.rsample()
        a = torch.tanh(u)
        # log(1 - tanh(u)^2) written in a numerically stable form
        log_det = 2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))
        log_prob = (dist.log_prob(u) - log_det).sum(-1)
        return self.max_action * a, log_prob

    def deterministic(self, s):
        return self.max_action * torch.tanh(self(s)[0])


class GaussianPolicy(nn.Module):
    """REINFORCE / A2C / PPO continuous policy: pi(a|s) = N(mu(s), exp(log_std)).

    The standard deviation is a free parameter (not a function of s), the most common and
    stable choice for on-policy methods. Actions are clipped to the bounds by the caller,
    the log-probability is that of the unclipped sample.
    """

    def __init__(self, s_dim: int, a_dim: int, hidden: int = 64, init_log_std: float = 0.0):
        super().__init__()
        self.mu = mlp(s_dim, a_dim, hidden, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((a_dim,), init_log_std))

    def dist(self, s) -> Normal:
        return Normal(self.mu(s), self.log_std.exp().expand(s.shape[0], -1))

    def log_prob(self, dist, a):
        return dist.log_prob(a).sum(-1)

    def entropy(self, dist):
        return dist.entropy().sum(-1)

    def mode(self, s):
        return self.mu(s)


class CategoricalPolicy(nn.Module):
    """Discrete policy: pi(a|s) = softmax(logits(s))."""

    def __init__(self, s_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.logits = mlp(s_dim, n_actions, hidden, out_gain=0.01)

    def dist(self, s) -> Categorical:
        return Categorical(logits=self.logits(s))

    def log_prob(self, dist, a):
        return dist.log_prob(a.long().reshape(-1))

    def entropy(self, dist):
        return dist.entropy()

    def mode(self, s):
        return self.logits(s).argmax(-1)


class QNetwork(nn.Module):
    """DQN: one forward pass gives Q(s, a) for every discrete action, (B, S) -> (B, n_actions)."""

    def __init__(self, s_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(s_dim, n_actions, hidden)

    def forward(self, s):
        return self.net(s)


class DuelingQNetwork(nn.Module):
    """Dueling DQN: Q(s, a) = V(s) + A(s, a) - mean_a A(s, a), shared body, two heads.

    V learns from every transition, whatever action was taken; only the advantages have to
    tell the actions apart. Subtracting the mean makes the split V / A identifiable.
    """

    def __init__(self, s_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, s):
        h = self.body(s)
        adv = self.adv(h)
        return self.value(h) + adv - adv.mean(-1, keepdim=True)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: theta_target <- tau * theta + (1 - tau) * theta_target."""
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(1.0 - tau).add_(s, alpha=tau)
