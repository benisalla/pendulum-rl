# Examples

Short scripts that use the `pendulum_rl` package from Python, in the order one would use
it. All run on one CPU thread in minutes; run them from the repository root after
`pip install -e ".[dev]"`. Everything they write goes to `examples/output/` (gitignored).

| script | what it shows | needs |
|---|---|---|
| [01_train_sac_pendulum.py](01_train_sac_pendulum.py) | `default_config` + `train`: SAC swings the pendulum up in 6 000 steps; the learned entropy temperature | nothing |
| [02_evaluate_heatmap_gif.py](02_evaluate_heatmap_gif.py) | `load_agent`, `evaluate`, value / torque heatmaps over (theta, theta_dot), a headless GIF | 01 |
| [03_ppo_clipping_and_gae.py](03_ppo_clipping_and_gae.py) | PPO piece by piece: one rollout, GAE advantages, how far the ratio moves and how often it is clipped; then 40k steps of PPO | nothing |
| [04_acrobot_sarsa_vs_cem.py](04_acrobot_sarsa_vs_cem.py) | Acrobot with no neural network: the tile coder's active features, SARSA(lambda) and the cross-entropy method | nothing |
| [05_train_any_method.py](05_train_any_method.py) | one command per method (`--algo td3`, `--env acrobot --algo dueling-dqn`, `--list`) with small budgets | nothing |
| [06_compare_methods.py](06_compare_methods.py) | the comparison table, curves and heatmaps of every run made by 01-05 | any of 01-05 |

```bash
python examples/01_train_sac_pendulum.py
python examples/02_evaluate_heatmap_gif.py
python examples/03_ppo_clipping_and_gae.py
python examples/04_acrobot_sarsa_vs_cem.py
python examples/05_train_any_method.py --algo td3
python examples/06_compare_methods.py
```

The same steps as CLI commands are in the main README (`pendulum-train`, `pendulum-eval`,
`pendulum-play`, `pendulum-compare`); the Kaggle/Colab notebook runs the full comparison.
