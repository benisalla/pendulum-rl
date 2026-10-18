# Tests

```bash
pytest -q            # the fast suite: CPU, one thread
pytest -q -m slow    # opt-in: real SAC swing-up check (a few minutes on a loaded CPU)
```

The suite sets `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` in `tests/conftest.py`.

| file | what it checks | why it matters |
|---|---|---|
| `test_envs_buffers.py` | env shapes, the discretization wrapper's torques, the replay buffer ring and sampling, GAE with lambda = 1 (equals Monte Carlo returns), lambda = 0 (equals one-step TD errors), truncation bootstrap, rollout cut with `last_value` | GAE and terminated/truncated handling are the two places where RL code is most often silently wrong |
| `test_networks.py` | actor bounds, critic shapes, twin critics differ, the squashed-Gaussian log-prob equals the change-of-variables formula, the SAC sample is reparameterised (gradients reach the actor), dueling head's mean advantage is zero, Polyak update formula | each network's one invariant |
| `test_algorithms.py` | every (env, algo) pair (16 of them) trains for 400 steps, saves, reloads to the same actions, and evaluates; config validation; TD3 target min; SAC's temperature goes down when the policy is more random than the target entropy | the whole pipeline for every method |
| `test_learning.py` | DQN overfits a fixed batch (TD loss of that batch near 0); tile-coded SARSA(lambda) beats the random policy on Acrobot within 15 000 steps; (slow) SAC reaches better than -400 on Pendulum in 5 000 steps | the methods actually learn, not only run |
| `test_cli_viz.py` | `pendulum-train`, `pendulum-eval`, `pendulum-play --gif` and `pendulum-compare` end to end in a temp dir; heatmap grid | the commands in the README work |

Why so few "learning" tests: a real learning check costs minutes of CPU per method. The
per-method learning evidence is the full comparison in [evaluation.md](evaluation.md);
the tests keep to what is fast and deterministic.
