# The two environments

Both come from `gymnasium[classic-control]` and are created by `pendulum_rl.envs.make_env`.
They are small, fast (about 40 microseconds per step) and need no download.

## Pendulum-v1: continuous control

A rod hangs from a motor. The agent applies a torque and must swing it up and keep it
upright. The motor is too weak to lift the rod directly, so a good policy first rocks it
back and forth to gain energy, then brakes as it arrives at the top.

| | |
|---|---|
| observation | `(cos theta, sin theta, theta_dot)`, theta = 0 is upright, theta_dot in [-8, 8] |
| action | one number, the torque, in [-2, 2] (a `Box`, so **continuous**) |
| reward | `-(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)` every step, between about -16.3 and 0 |
| episode | always 200 steps (truncated, never terminated); random start angle and velocity |
| scores | random policy about -1200; good swing-up about -150; we call it **solved at -200** |

**Why cos/sin and not theta?** theta jumps from pi to -pi at the bottom; a network fed the
angle directly would see a discontinuity where nothing physical happens. `(cos, sin)` is
continuous all the way round.

**Why truncation matters.** Episodes end at 200 steps only because of a clock. The state
after step 200 is not worse than any other state, so value targets must still bootstrap
from it. The replay buffer stores `terminated` (a real end) separately from `truncated`
(the clock), and every target uses `(1 - terminated)` only; GAE does the same with an
extra V of the final observation. Mixing them up is a classic bug that makes Pendulum
critics wrong for the last steps of every episode.

### Discretized Pendulum

`make_env("pendulum", n_bins=5)` wraps the env with `DiscretizedAction`: the action space
becomes `Discrete(5)` and action k means torque `linspace(-2, 2, 5)[k]` = -2, -1, 0, 1, 2.
That is what `discretized-dqn` sees. Five torques are enough for a swing-up; more bins give
finer control but more outputs to learn.

## Acrobot-v1: discrete actions, sparse-ish reward

Two links hang from a fixed point; only the elbow has a motor. The goal is to swing the
tip above a line one link-length above the shoulder.

| | |
|---|---|
| observation | 6 numbers: `cos/sin` of both joint angles, both angular velocities |
| action | `Discrete(3)`: elbow torque -1, 0 or +1 |
| reward | -1 every step until the goal (then the episode **terminates**) |
| episode | at most 500 steps (truncated) |
| scores | random policy about -500 (never reaches the goal); good policies -60 to -100; **solved at -100** |

The reward carries no hint of progress: every step costs -1 until the very end. A method
first has to reach the goal at least once by exploration before it can learn anything,
which is why some methods sit at -500 for a while and then jump.

`SarsaLambda` reconstructs the 4 underlying state variables `(theta1, theta2, theta1_dot,
theta2_dot)` with `atan2` before tile coding them (`sarsa.acrobot_features`).

## Rendering and GIFs

`make_env(name, render=True)` uses `render_mode="rgb_array"` (pygame draws into an
off-screen surface). On a machine without a display set `SDL_VIDEODRIVER=dummy`;
`viz.save_gif` and `pendulum-play --gif` do it for you.
