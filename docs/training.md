# Training

```bash
pendulum-train --env pendulum --algo sac                        # tuned defaults
pendulum-train --env pendulum --algo td3 --total-steps 30000 --exploration-noise 0.2
pendulum-train --env acrobot --algo dueling-dqn --seed 3
pendulum-compare --env pendulum --train                         # every method, one after the other
```

Every field of `TrainConfig` (`src/pendulum_rl/config.py`) is a flag. `default_config(env,
algo)` starts from the dataclass defaults and applies the per-pair settings in
`DEFAULTS`; your flags go on top. Results go to `runs/<env>/<algo>/model.pt` and
`metrics.json`.

## The loop, in one picture

All methods share the same contract with `metrics.Monitor`:

```python
while not monitor.done:          # budget = total_steps environment steps
    ... act, env.step(a), learn ...
    monitor.step(agent)          # counts the step; every eval_every steps runs evaluate()
    monitor.episode(ret)         # at each episode end, logs the training return
```

So the budget is always counted in **environment steps**, whatever the method does with
them (one gradient step per env step for DDPG/TD3/SAC/DQN, one update per rollout for
PPO/A2C, one per batch of episodes for REINFORCE, one per generation for CEM). That is the
fair axis for a comparison: interaction with the world is what is expensive in RL.

## The three loop shapes

| shape | methods | where |
|---|---|---|
| off-policy: act, store in replay buffer, one gradient step on a random batch | `ddpg`, `td3`, `sac`, `dqn`, `double-dqn`, `dueling-dqn`, `discretized-dqn` | `algorithms/base.py: OffPolicyAgent.learn` |
| on-policy: collect a rollout with the current policy, compute advantages, update, throw the data away | `a2c`, `ppo`, `reinforce` | `algorithms/policy_gradient.py` |
| episodic search / tabular-style TD | `cem` (whole episodes per candidate), `sarsa` (update every step, traces) | `algorithms/cem.py`, `algorithms/sarsa.py` |

The off-policy methods start with `warmup_steps = 1000` uniform random actions: the
buffer needs varied data before the first gradient step, otherwise the critic fits the
handful of transitions of a single untrained policy.

## Defaults, and why

| setting | value | why |
|---|---|---|
| network width | 128 x 2 (off-policy), 64 x 2 (on-policy), 16 (CEM) | Pendulum's state is 3 numbers; bigger nets only slow a CPU down. CEM searches the weights directly, so it needs very few of them |
| `gamma` | 0.99, but 0.95 for PPO / A2C / REINFORCE on Pendulum | Pendulum's reward is dense and the useful horizon is short; a smaller gamma makes returns less noisy, which on-policy methods need most |
| `reward_scale` | 0.1 for PPO / A2C on Pendulum | rewards go down to -16 per step; scaled values keep the critic's targets around 1-10 and its loss from dominating the shared optimizer |
| `tau` | 0.005 | target nets move 0.5% per step: slow targets stop the critic from chasing its own tail |
| `exploration_noise` | 0.1 x max torque | DDPG/TD3 actors are deterministic; without noise they never try anything new |
| SAC `init_alpha` | 0.2, tuned automatically to entropy -1 | no noise schedule to tune: the temperature finds itself |
| DQN epsilon | 1.0 -> 0.05 over the first 20% (Pendulum), 1.0 -> 0.01 (Acrobot) | explore broadly first, then exploit; on Acrobot a small final epsilon keeps the training data close to the greedy policy that is evaluated |
| DQN target copy | every 1000 updates (Pendulum), 250 with lr 5e-4 (Acrobot) | a fixed target for a while = a supervised regression problem between copies |
| `anneal_lr` | on for PPO / A2C (Pendulum), REINFORCE (Acrobot) | linear decay to 0 stops late collapses of an already good policy |
| REINFORCE (Acrobot) | lr 1e-2, 4 episodes per batch | a uniform policy rarely reaches the goal; many large updates are needed to escape -500 |
| CEM (Pendulum) | population 50, 3 episodes per candidate, 1.5M steps | several start states per candidate stop the elites from overfitting one start |
| PPO | rollout 2048 (Pendulum) / 1024 (Acrobot), 10 epochs, minibatch 64, clip 0.2, GAE 0.95 | the values from the PPO paper's MuJoCo setting |
| A2C | rollout 32 (Pendulum) / 16 (Acrobot) steps, one step per rollout | A2C's point: small, frequent, on-policy updates |
| SARSA | 8 tilings x 6 tiles per dim, alpha 0.5 / 8, lambda 0.9, epsilon 0 | Sutton & Barto's Acrobot setting, scaled down |

## Budgets of the comparison

| | Pendulum | Acrobot |
|---|---|---|
| DDPG / TD3 / SAC | 15 000 steps (75 episodes) | n/a |
| discretized DQN | 30 000 | n/a |
| DQN / double / dueling | n/a | 60 000 |
| PPO | 100 000 | 150 000 |
| A2C | 150 000 | 150 000 |
| REINFORCE | 600 000 | 100 000 |
| CEM | 1 500 000 (about 2 min: no gradients) | 150 000 |
| SARSA(lambda) | n/a | 50 000 |

The off-policy methods take one gradient step per environment step, so they get the
smallest budgets; the on-policy methods throw data away after each update and get 7-20
times more. `scripts/run_all.sh` runs them all (one process at a time).

## CPU threads

`--threads 1` (the default) calls `torch.set_num_threads(1)`. For networks of this size
one thread is the fastest: the matrices are tiny and multi-thread synchronisation costs
more than it saves. Set `OMP_NUM_THREADS=1` in the shell too when several jobs share a
machine.

## Reproducibility

`--seed` seeds Python, NumPy, PyTorch, the environment and its action space. Runs are
repeatable on the same machine; across machines floating point differences make RL runs
diverge after a while, which is normal. For a real conclusion, run several seeds
(`pendulum-compare --train --seed 1`, `--runs-dir runs_seed1`).
