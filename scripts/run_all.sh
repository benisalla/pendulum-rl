#!/usr/bin/env bash
# Every real run behind the README tables, one training process at a time.
# Usage: bash scripts/run_all.sh  [LOCK=/path/to/lockfile to queue behind other jobs]
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { if [ -n "${LOCK:-}" ]; then flock "$LOCK" "$@"; else "$@"; fi; }
PY=${PY:-.venv/bin/python}
run $PY -m pendulum_rl.compare --env pendulum --train --algos ddpg td3
run $PY -m pendulum_rl.compare --env pendulum --train --algos sac discretized-dqn
run $PY -m pendulum_rl.compare --env pendulum --train --algos ppo a2c reinforce cem    # ~10 min
run $PY -m pendulum_rl.compare --env acrobot --train --algos dqn double-dqn
run $PY -m pendulum_rl.compare --env acrobot --train --algos dueling-dqn sarsa cem
run $PY -m pendulum_rl.compare --env acrobot --train --algos ppo a2c reinforce
# final figures and tables from all runs (+ GIFs), then the README's before/after GIFs
$PY -m pendulum_rl.compare --env pendulum --gif
$PY -m pendulum_rl.compare --env acrobot --gif
$PY scripts/make_gifs.py
