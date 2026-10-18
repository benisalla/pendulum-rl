# Architecture: every method, its equations and the code that implements it

Notation: state s, action a, reward r, discount gamma, next state s', `d` = 1 if the
episode really *terminated* (never for Pendulum). Primed networks (Q', mu') are slow
**target** copies, updated by Polyak averaging `theta' <- tau * theta + (1 - tau) * theta'`
(`networks.soft_update`) or copied every N steps (DQN).

```
src/pendulum_rl/
  envs.py          Pendulum-v1 / Acrobot-v1, DiscretizedAction wrapper
  networks.py      mlp, DeterministicActor, QCritic, TwinQCritic, SquashedGaussianActor,
                   GaussianPolicy, CategoricalPolicy, QNetwork, DuelingQNetwork, soft_update
  buffers.py       ReplayBuffer, compute_gae
  algorithms/
    base.py        Agent interface, OffPolicyAgent.learn (the shared off-policy loop)
    ddpg.py td3.py sac.py dqn.py policy_gradient.py cem.py sarsa.py
  registry.py      name -> class, save / load
  metrics.py       Monitor (step counting + periodic evaluation), results table
  evaluate.py train.py play.py compare.py viz.py
```

## 1. Continuous actions: why DQN does not apply directly

DQN picks `argmax_a Q(s, a)` by evaluating every action. With a torque in [-2, 2] there
are infinitely many actions: the max is itself an optimisation problem at every step.
There are three ways out, and the repo has all three:

1. **Learn a policy network that outputs the action** (actor-critic: DDPG, TD3, SAC; or
   pure policy gradient: REINFORCE, A2C, PPO).
2. **Search the policy weights directly** without any gradient (CEM).
3. **Discretize the action** and keep DQN (`discretized-dqn`, 5 torques).

## 2. Deterministic vs stochastic policies

| | deterministic `a = mu(s)` | stochastic `a ~ pi(.|s)` |
|---|---|---|
| methods | DDPG, TD3 | SAC, PPO, A2C, REINFORCE, (CEM's search distribution) |
| exploration | must be added from outside (Gaussian noise on the action) | built in: sampling *is* exploration |
| gradient | deterministic policy gradient: backprop through Q(s, mu(s)) | score function: grad log pi(a|s) x advantage, or reparameterisation (SAC) |
| at evaluation | mu(s) | the mean / mode of pi (all methods are evaluated deterministically here) |

Code: `DeterministicActor` = `max_action * tanh(mlp(s))`; `GaussianPolicy` = `N(mlp(s),
exp(log_std))` with a state-independent std (on-policy); `SquashedGaussianActor` =
`max_action * tanh(u)`, `u ~ N(mu(s), sigma(s))` (SAC).

## 3. DDPG: actor-critic with a deterministic actor (`algorithms/ddpg.py`)

```
critic:  y = r + gamma * (1 - d) * Q'(s', mu'(s'))          L_Q  = (Q(s, a) - y)^2
actor:   L_mu = -Q(s, mu(s))                                (gradient = dQ/da * dmu/dtheta)
explore: a = clip(mu(s) + N(0, (0.1 * 2)^2), -2, 2)
```

In plain words: the critic learns "how good is this action here", and the actor moves its
output in the direction in which the critic says things get better. Autograd does the
chain rule through the critic: `actor_loss = -critic(s, actor(s)).mean()` and only the
actor's optimizer steps. Both target networks make the regression target move slowly.

**Weak point:** the actor is an optimiser of the critic. Wherever the critic is wrongly
high, the actor goes there, and the bootstrapped target spreads the error. On Pendulum
DDPG usually still works, but it is the least stable of the three.

## 4. TD3: three fixes to DDPG (`algorithms/td3.py`)

```
a'  = clip(mu'(s') + clip(N(0, 0.4^2), -1, 1), -2, 2)        # fix 3: target policy smoothing
y   = r + gamma * (1 - d) * min(Q1'(s', a'), Q2'(s', a'))    # fix 1: clipped double Q
L_Q = (Q1(s, a) - y)^2 + (Q2(s, a) - y)^2
every 2 critic steps:  L_mu = -Q1(s, mu(s));  update targets  # fix 2: delayed policy updates
```

1. **Clipped double Q.** Two critics trained on the same target; the target takes the
   smaller of the two. Overestimation is what the actor exploits, so a pessimistic target
   removes the fuel.
2. **Delayed policy updates.** The actor and the targets move half as often as the
   critics: the critic gets time to become accurate for the current actor before the
   actor moves again.
3. **Target policy smoothing.** Noise on the target action means Q is fitted on a small
   neighbourhood of actions, so a single sharp, wrong peak of Q cannot be exploited.

`TD3` subclasses `DDPG`: same actor, same exploration, only the critic and `update` change.

## 5. SAC: maximum entropy (`algorithms/sac.py`)

Objective: `max E[ sum_t r_t + alpha * H(pi(.|s_t)) ]`: be rewarded, but stay as random as
possible while doing it.

```
a' ~ pi(.|s'):   y = r + gamma * (1 - d) * (min(Q1', Q2')(s', a') - alpha * log pi(a'|s'))
critic:          L_Q = (Q1(s, a) - y)^2 + (Q2(s, a) - y)^2
actor:           a = 2 * tanh(mu(s) + sigma(s) * eps),  L_pi = alpha * log pi(a|s) - min(Q1, Q2)(s, a)
temperature:     L_alpha = -log(alpha) * (log pi(a|s) + target_entropy),  target_entropy = -dim(A) = -1
```

- **Why entropy.** The bonus keeps exploration alive without a hand-made noise schedule,
  and a policy that is random wherever it can afford to be is robust (it does not bet on
  one fragile action sequence).
- **Reparameterisation.** `a = tanh(mu + sigma * eps)` with `eps ~ N(0, 1)` is a
  differentiable function of the network output, so `L_pi` is backpropagated through the
  critics like in DDPG; `SquashedGaussianActor.sample` uses `rsample()`.
- **tanh correction.** Squashing changes the density: `log pi(a) = log N(u) - sum log(1 -
  tanh(u)^2)`, written in a stable softplus form (`networks.py`, tested against the plain
  formula in `test_networks.py`).
- **Automatic temperature.** If the policy's entropy is above the target, `log pi +
  target_entropy < 0` and gradient descent lowers alpha: less reward for randomness. If
  the policy collapses, alpha rises. One less hyperparameter.
- No target actor: the next action is sampled from the current policy.
- Evaluation uses the mean action `2 * tanh(mu(s))`.

## 6. Policy gradients: REINFORCE, A2C, PPO (`algorithms/policy_gradient.py`)

All three ascend `grad J = E[ grad log pi(a|s) * A ]`; A says how much better than usual
the action was. They differ in A and in how hard each batch is used.

**REINFORCE.** A = return-to-go `G_t = sum_k gamma^k r_{t+k}` of complete episodes,
normalised over the batch (subtracting the mean is the simplest baseline). Unbiased, very
noisy: on Pendulum one episode's return depends a lot on the random start state.

**A2C.** A critic `V(s)` is learned with the policy; A is the GAE (below) over short
rollouts of 16-32 steps; **one** gradient step on
`-log pi(a|s) * A + 0.5 * (V(s) - R)^2 - c_ent * H`, then the data is discarded.

**GAE** (`buffers.compute_gae`): with TD errors `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`,

```
A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
```

lambda = 0 trusts the critic completely (low variance, biased if V is wrong); lambda = 1
ignores it (Monte Carlo, unbiased, noisy); 0.95 is the usual compromise. The sum stops at
episode ends; after a truncation it bootstraps from V of the final observation.

**PPO.** Long rollouts (2048 steps), GAE, then 10 epochs of minibatch steps on

```
rho = pi_theta(a|s) / pi_old(a|s)
L   = -E[ min(rho * A, clip(rho, 1 - 0.2, 1 + 0.2) * A) ] + 0.5 * (V(s) - R)^2
```

Why the clip: reusing the same data ten times is only valid while the new policy stays
close to the one that collected it. When an action with A > 0 already became 20% more
likely (rho > 1.2), the clipped term is constant, its gradient is zero, and there is no
reason to push further; same for A < 0 and rho < 0.8. The `min` makes it a pessimistic
bound: the clip can only remove improvement, never add it. Example 03 prints how many
samples hit the clip after one update.

Shared details: advantages normalised per minibatch (PPO), gradient norm clipped at 0.5,
the Gaussian log-std is a free parameter starting at 0 (std 1), the last policy layer is
initialised small so the first actions are near 0 (the centre of the torque range).

## 7. CEM: black-box search (`algorithms/cem.py`)

```
theta_i ~ N(mu, diag(sigma^2)),  i = 1..50
score_i = return of one episode with policy theta_i (all on the same start state)
elite   = the 20% best theta_i
mu, sigma^2 <- mean(elite), var(elite) + extra_noise^2          (extra noise decays to 0)
```

No gradient, no critic, no Bellman equation; works as long as the policy has few weights
(here `3 -> 16 -> 1`, 81 weights, NumPy only). Running all candidates of a generation from
the same start state is "common random numbers": the ranking then compares policies, not
luck with the start state. Cost: a generation is 50 whole episodes (10 000 Pendulum steps).

## 8. DQN family (`algorithms/dqn.py`)

```
dqn:          y = r + gamma * (1 - d) * max_a Q'(s', a)
double-dqn:   y = r + gamma * (1 - d) * Q'(s', argmax_a Q(s', a))      # online chooses, target evaluates
dueling-dqn:  Q(s, a) = V(s) + A(s, a) - mean_a A(s, a), with the double-DQN target
loss:         Huber(Q(s, a) - y), gradient norm clipped at 10, Q' copied every 500 updates
behaviour:    epsilon-greedy, epsilon 1 -> 0.05 linearly over the first 30% of the budget
```

- **Why double.** `max` over noisy estimates is biased upward (the largest of several
  noisy numbers is usually an overestimate). Using two networks for "which action" and
  "how good" de-correlates the noise.
- **Why dueling.** In many states the action barely matters (Acrobot hanging still). The
  V stream learns from every transition regardless of the action; the advantage stream
  only has to learn the differences. Subtracting the mean advantage fixes the V / A split.
- **discretized-dqn** = double DQN on `DiscretizedAction(Pendulum, 5)`.

## 9. SARSA(lambda) with tile coding (`algorithms/sarsa.py`)

```
features: 8 shifted grids of 7^4 tiles over (theta1, theta2, theta1_dot, theta2_dot)
Q(s, a) = sum of w[a, i] over the 8 active tiles i of s
delta   = r + gamma * Q(s', a') - Q(s, a)            (a' = the action actually taken next: on-policy)
e[a, tiles(s)] = 1  (replacing traces);   w += (0.5 / 8) * delta * e;   e *= gamma * lambda
```

- **Tile coding** is a hand-made, sparse, local feature map: nearby states share most of
  their 8 tiles (generalisation), far states share none (no interference). Each update
  touches 8 weights, so a large step size is safe.
- **Eligibility traces** pass each TD error back to the recently visited (state, action)
  pairs, decayed by `gamma * lambda`: the rare "reached the goal" event updates the whole
  swing that led to it, not only the last step.
- **Optimistic initialisation.** Every step gives -1 and w starts at 0, so untried actions
  look better than tried ones: greedy action selection explores systematically, epsilon = 0.

## 10. Evaluation and figures

`evaluate(agent, n_episodes)` (`evaluate.py`) runs the deterministic policy (`act_batch`)
on start states with seeds 10 000, 10 001, ...: the same states for every method.
`viz.plot_heatmaps` evaluates every agent on a 61 x 61 grid of `(theta, theta_dot)`:
`value_batch` (Q(s, mu(s)) for DDPG/TD3/SAC, max_a Q for DQN, V(s) for A2C/PPO, nothing for
REINFORCE/CEM) and the torque `act_batch` (for discretized DQN, the bin's torque).
