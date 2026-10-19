"""One configuration dataclass for every algorithm, plus the defaults that make each one work.

Every field is also a ``pendulum-train --flag`` (underscores become dashes). Fields that an
algorithm does not use are simply ignored by it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

# algorithm -> environments it applies to
ALGOS: dict[str, tuple[str, ...]] = {
    # continuous control (Pendulum)
    "ddpg": ("pendulum",),
    "td3": ("pendulum",),
    "sac": ("pendulum",),
    "discretized-dqn": ("pendulum",),
    # both
    "ppo": ("pendulum", "acrobot"),
    "a2c": ("pendulum", "acrobot"),
    "reinforce": ("pendulum", "acrobot"),
    "cem": ("pendulum", "acrobot"),
    # discrete actions (Acrobot)
    "dqn": ("acrobot",),
    "double-dqn": ("acrobot",),
    "dueling-dqn": ("acrobot",),
    "sarsa": ("acrobot",),
}


@dataclass
class TrainConfig:
    env: str = "pendulum"
    algo: str = "sac"
    total_steps: int = 20_000         # environment steps (the budget every method is compared on)
    seed: int = 0
    threads: int = 1
    eval_every: int = 2_000           # evaluate the greedy / deterministic policy every N steps
    eval_episodes: int = 5
    final_eval_episodes: int = 20
    out_dir: str = "runs"             # checkpoints and metrics go to out_dir/env/algo/

    # shared
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    reward_scale: float = 1.0         # multiply rewards before learning (curves stay unscaled)

    # off-policy: DDPG, TD3, SAC, DQN family
    batch_size: int = 128
    buffer_size: int = 100_000
    warmup_steps: int = 1_000         # uniform random actions before learning starts
    tau: float = 0.005                # Polyak factor for target networks
    critic_lr: float = 1e-3
    exploration_noise: float = 0.1    # DDPG / TD3 Gaussian action noise (fraction of max torque)
    policy_delay: int = 2             # TD3: actor + targets updated every 2 critic updates
    target_noise: float = 0.2         # TD3 target policy smoothing
    noise_clip: float = 0.5
    init_alpha: float = 0.2           # SAC entropy temperature, tuned automatically
    auto_alpha: bool = True
    eps_start: float = 1.0            # DQN epsilon-greedy schedule
    eps_end: float = 0.05
    eps_fraction: float = 0.3         # fraction of total_steps over which epsilon decays
    target_update: int = 500          # DQN hard target copy every N steps
    n_bins: int = 5                   # discretized-dqn: torque bins

    # on-policy: REINFORCE, A2C, PPO
    n_steps: int = 2048               # rollout length per update (A2C / PPO)
    gae_lambda: float = 0.95
    clip: float = 0.2                 # PPO ratio clip
    epochs: int = 10                  # PPO passes over each rollout
    minibatch: int = 64
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    init_log_std: float = 0.0
    episodes_per_batch: int = 10      # REINFORCE episodes per gradient step
    anneal_lr: bool = False           # policy gradients: decay the learning rate linearly to 0 over the budget

    # cross-entropy method
    cem_population: int = 50
    cem_elite_frac: float = 0.2
    cem_episodes: int = 1             # episodes (start states) per candidate; the score is their mean
    cem_init_std: float = 1.0
    cem_extra_std: float = 0.1        # extra noise, decayed, keeps the search from collapsing

    # SARSA(lambda) with tile coding
    n_tilings: int = 8
    n_tiles: int = 6                  # tiles per dimension per tiling
    sarsa_alpha: float = 0.5          # divided by n_tilings internally
    sarsa_lambda: float = 0.9
    sarsa_eps: float = 0.0            # optimistic zero init already explores (rewards are -1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Acrobot DQN: with lr 1e-3, a target copy every 500 steps and epsilon ending at 0.05, the
# training episodes improved but the *greedy* policy kept falling back to -500 (it would hold
# the arm still). A smaller lr, more frequent target copies and epsilon -> 0.01 (so training
# visits the states the greedy policy visits) make the greedy policy stable.
ACROBOT_DQN = dict(lr=5e-4, target_update=250, eps_end=0.01, eps_fraction=0.2)

# Settings that make each (env, algo) pair learn within a CPU-minutes budget.
DEFAULTS: dict[tuple[str, str], dict] = {
    ("pendulum", "ddpg"): dict(total_steps=15_000, eval_every=1_000),
    ("pendulum", "td3"): dict(total_steps=15_000, eval_every=1_000),
    ("pendulum", "sac"): dict(total_steps=15_000, eval_every=1_000),
    ("pendulum", "discretized-dqn"): dict(total_steps=30_000, n_bins=5, lr=5e-4, eps_fraction=0.2, target_update=1_000),
    # Pendulum's reward is dense and the useful horizon short: gamma 0.9-0.95 helps the
    # on-policy methods, whose advantage estimates are otherwise very noisy.
    ("pendulum", "ppo"): dict(total_steps=100_000, eval_every=5_000, n_steps=2048, gamma=0.95, lr=1e-3, hidden=64,
                              reward_scale=0.1, epochs=10, minibatch=64, gae_lambda=0.95, anneal_lr=True),
    ("pendulum", "a2c"): dict(total_steps=150_000, eval_every=5_000, n_steps=32, gamma=0.95, lr=7e-4, hidden=64,
                              reward_scale=0.1, gae_lambda=0.95, max_grad_norm=0.5, anneal_lr=True),
    ("pendulum", "reinforce"): dict(total_steps=600_000, eval_every=20_000, gamma=0.95, lr=3e-3, hidden=64,
                                    episodes_per_batch=10),
    # CEM: one generation = population x episodes, so it needs many env steps, but they are
    # cheap (NumPy policy, no gradient): 1.5M steps take about 2 minutes.
    ("pendulum", "cem"): dict(total_steps=1_500_000, eval_every=50_000, hidden=16, cem_population=50, cem_episodes=3),
    ("acrobot", "dqn"): dict(total_steps=60_000, eval_every=2_500, **ACROBOT_DQN),
    ("acrobot", "double-dqn"): dict(total_steps=60_000, eval_every=2_500, **ACROBOT_DQN),
    ("acrobot", "dueling-dqn"): dict(total_steps=60_000, eval_every=2_500, **ACROBOT_DQN),
    ("acrobot", "ppo"): dict(total_steps=150_000, eval_every=5_000, n_steps=1024, hidden=64, lr=3e-4, ent_coef=0.0),
    ("acrobot", "a2c"): dict(total_steps=150_000, eval_every=5_000, n_steps=16, hidden=64, lr=7e-4, gae_lambda=1.0,
                             ent_coef=0.01),
    # REINFORCE on Acrobot: a uniform policy almost never reaches the goal, so it needs many
    # small batches (many updates) and a large step to escape -500, then annealing to stay there.
    ("acrobot", "reinforce"): dict(total_steps=100_000, eval_every=5_000, hidden=64, lr=1e-2, episodes_per_batch=4,
                                   anneal_lr=True),
    ("acrobot", "cem"): dict(total_steps=150_000, eval_every=5_000, hidden=16, cem_population=30),
    ("acrobot", "sarsa"): dict(total_steps=50_000, eval_every=2_500),
}


def default_config(env: str = "pendulum", algo: str = "sac", **overrides) -> TrainConfig:
    """The repo's tuned defaults for ``(env, algo)``, then your overrides on top."""
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}, choose from {list(ALGOS)}")
    if env not in ALGOS[algo]:
        raise ValueError(f"{algo} does not apply to {env} (it supports {ALGOS[algo]})")
    cfg = TrainConfig(env=env, algo=algo)
    return replace(cfg, **{**DEFAULTS.get((env, algo), {}), **overrides})
