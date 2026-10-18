# pendulum-rl

**Pendulum swing-up (continuous torque) and Acrobot (discrete actions) solved with every
classic RL method in one small, readable PyTorch code base: DDPG, TD3, SAC with automatic
entropy tuning, PPO with GAE and clipping, A2C, REINFORCE, the cross-entropy method, DQN /
double / dueling / discretized DQN and tile-coded SARSA(lambda). Every method uses the same
environment, the same evaluation, the same plots, trains in minutes on one CPU thread, and
is explained step by step in the docs and a Kaggle/Colab notebook.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-classic--control-0081a5.svg)](https://gymnasium.farama.org)
[![Tests](https://img.shields.io/badge/tests-__TESTS__%20passing-brightgreen.svg)](docs/testing.md)

The goal is to *see* how each method works: one file per family, the equations next to
the code, the same budget unit (environment steps) for everyone, and figures that show
what each agent learned, not only its score.

```mermaid
flowchart LR
    E["Pendulum-v1<br/>torque in [-2, 2]"] --> OFF
    E --> ON
    E --> BB
    E --> D["DiscretizedAction<br/>5 torques"] --> Q
    A["Acrobot-v1<br/>3 actions"] --> Q
    A --> ON
    A --> BB
    A --> T["tile coding"] --> S["SARSA(lambda)"]
    OFF["replay buffer + target nets<br/>DDPG | TD3 | SAC"] --> M
    ON["rollouts + GAE<br/>REINFORCE | A2C | PPO"] --> M
    BB["weight search<br/>CEM"] --> M
    Q["replay + epsilon-greedy<br/>DQN | double | dueling"] --> M
    S --> M
    M["Monitor: same step budget,<br/>same evaluation"] --> R["curves · table ·<br/>V(s) / policy heatmaps · GIFs"]
```

| method | `--algo` | env | how it works, in one line | code |
|---|---|---|---|---|
| DDPG | `ddpg` | Pendulum | deterministic actor `mu(s)` climbs the critic `Q(s, mu(s))`; Gaussian action noise explores | `algorithms/ddpg.py` |
| TD3 | `td3` | Pendulum | DDPG + min of twin critics, delayed actor updates, noise on the target action | `algorithms/td3.py` |
| SAC | `sac` | Pendulum | stochastic tanh-Gaussian actor maximising reward + alpha x entropy; alpha tuned automatically | `algorithms/sac.py` |
| PPO | `ppo` | both | rollouts of 2048 steps, GAE, 10 epochs on the clipped ratio `min(rho A, clip(rho) A)` | `algorithms/policy_gradient.py: PPO` |
| A2C | `a2c` | both | learned V(s), GAE over short rollouts, one gradient step per rollout | `algorithms/policy_gradient.py: A2C` |
| REINFORCE | `reinforce` | both | `log pi(a|s)` x normalised Monte Carlo return of whole episodes, no critic | `algorithms/policy_gradient.py: Reinforce` |
| cross-entropy method | `cem` | both | sample 50 weight vectors, keep the best 20%, refit the Gaussian; no gradient at all | `algorithms/cem.py` |
| discretized DQN | `discretized-dqn` | Pendulum | the torque cut into 5 values, then double DQN | `envs.DiscretizedAction` + `algorithms/dqn.py` |
| DQN | `dqn` | Acrobot | epsilon-greedy, replay, target net, `y = r + gamma max Q'(s', .)` | `algorithms/dqn.py` |
| double DQN | `double-dqn` | Acrobot | online net picks `argmax`, target net evaluates it: less overestimation | `algorithms/dqn.py` |
| dueling DQN | `dueling-dqn` | Acrobot | `Q = V(s) + A(s, a) - mean A`, double-DQN target | `networks.DuelingQNetwork` |
| SARSA(lambda) | `sarsa` | Acrobot | linear Q over 8 shifted tile grids, eligibility traces, optimistic init | `algorithms/sarsa.py` |

Each one is explained with its equations, the plain-words "why" and the code in
[docs/architecture.md](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/benisalla/pendulum-rl && cd pendulum-rl
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
source .venv/bin/activate

pendulum-train --env pendulum --algo sac                       # 15k steps, saves runs/pendulum/sac/
pendulum-eval runs/pendulum/sac/model.pt --episodes 20         # mean return of the deterministic policy
pendulum-play runs/pendulum/sac/model.pt --gif sac.gif         # headless GIF (SDL dummy driver)
pendulum-train --env acrobot --algo dueling-dqn
pendulum-compare --env pendulum --train                        # every method, table + curves + heatmaps
```

`uv` installs the CPU build of PyTorch (configured in `pyproject.toml`); plain `pip` uses
the default wheel. Every `TrainConfig` field is a flag (`--total-steps`, `--lr`,
`--gamma`, `--n-bins`, `--sarsa-lambda`, ...); see [docs/training.md](docs/training.md).

## Examples

[`examples/`](examples/README.md) holds six short scripts that use the package from
Python, in order: train SAC in 6 000 steps, evaluate it and draw its value / torque
heatmaps and a GIF, take PPO apart (one rollout, GAE, how often the ratio is clipped),
solve Acrobot without a neural network (tile-coded SARSA and CEM), train **any method in
one command** with small defaults, and compare everything you ran.

```bash
python examples/01_train_sac_pendulum.py && python examples/02_evaluate_heatmap_gif.py
python examples/05_train_any_method.py --algo td3          # or --env acrobot --algo dueling-dqn
```

## Results

__RESULTS__

## Train for real: Kaggle / Colab notebook

[`notebooks/train_colab_kaggle.ipynb`](notebooks/train_colab_kaggle.ipynb) is generated from
`src/pendulum_rl` by `notebooks/build_notebook.py`: each section explains one idea
(continuous actions, deterministic vs stochastic policies, DDPG's actor-critic, TD3's three
fixes, SAC's entropy objective, PPO's clipping and GAE, CEM, DQN variants, tile-coded
SARSA), shows the exact code of the method, trains it, and ends with the full comparison,
heatmaps and GIFs. CPU runtime is enough (and fastest) for networks this small.

## Repository layout

```
src/pendulum_rl/
  config.py        TrainConfig (every hyperparameter = a CLI flag), tuned DEFAULTS per (env, algo)
  envs.py          Pendulum-v1 / Acrobot-v1, DiscretizedAction wrapper
  networks.py      actors, critics, twin critics, Gaussian / categorical policies, Q / dueling nets
  buffers.py       replay buffer (terminated vs truncated), GAE
  algorithms/      base (shared off-policy loop), ddpg, td3, sac, dqn, policy_gradient, cem, sarsa
  registry.py      name -> agent class, checkpoint save / load
  metrics.py       Monitor (step budget + periodic evaluation), results table
  train.py         train(), `pendulum-train`
  evaluate.py      evaluate(), `pendulum-eval`
  play.py          `pendulum-play` (+ GIF)
  compare.py       `pendulum-compare`: train all, table, curves, heatmaps
  viz.py           return curves, value / policy heatmaps, GIFs
docs/              architecture, environment, training, evaluation, testing
examples/          six scripts, from one training run to the full comparison
tests/             __TESTS__ fast tests (+1 opt-in slow learning test)
notebooks/         build_notebook.py -> train_colab_kaggle.ipynb
scripts/run_all.sh every real run behind the tables above
assets/            the figures and tables of the Results section
```

## Documentation

- [docs/architecture.md](docs/architecture.md): every method, its equations, why it works, where the code is
- [docs/environment.md](docs/environment.md): Pendulum and Acrobot, terminated vs truncated, discretization, rendering
- [docs/training.md](docs/training.md): the shared loop, defaults and why, budgets, threads, seeds
- [docs/evaluation.md](docs/evaluation.md): the evaluation protocol and the runs behind the results
- [docs/testing.md](docs/testing.md): what the tests check
- [examples/](examples/README.md): the package from Python, step by step

## Legacy notebooks

None: this repository was written fresh, there is no earlier notebook for Pendulum or
Acrobot. The CartPole coursework notebooks live in their own repository.

## References

Lillicrap et al. 2016 (DDPG) · Fujimoto et al. 2018 (TD3) · Haarnoja et al. 2018 (SAC and
automatic temperature) · Schulman et al. 2016 (GAE), 2017 (PPO) · Mnih et al. 2015 (DQN),
2016 (A3C/A2C) · van Hasselt et al. 2016 (double DQN) · Wang et al. 2016 (dueling) ·
Williams 1992 (REINFORCE) · Rubinstein 1999, Szita & Lorincz 2006 (CEM) · Sutton & Barto
2018, ch. 10 and 12 (tile coding, SARSA(lambda), Acrobot).

## License

MIT, (c) 2026 Ismail Ben Alla
