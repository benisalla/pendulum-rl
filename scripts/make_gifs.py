"""Regenerate the README's before/after GIFs and the per-method GIF grids.

    python scripts/make_gifs.py                 # needs the runs of scripts/run_all.sh in runs/

* assets/before_after_pendulum.gif: untrained SAC | trained SAC, same start state
* assets/before_after_acrobot.gif:  untrained <best Acrobot method> | trained, same start state
* assets/grid_pendulum.gif, assets/grid_acrobot.gif: one panel per method, same start state

Each panel has a caption with the method and the episode return. "Untrained" is the same
network, freshly initialised (what the method starts from), acting deterministically.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pendulum_rl.config import ALGOS, TrainConfig  # noqa: E402
from pendulum_rl.envs import make_env  # noqa: E402
from pendulum_rl.registry import env_bins, load_agent, make_agent  # noqa: E402


def rollout_frames(agent, seed: int, size: int) -> tuple[list[Image.Image], float]:
    env = make_env(agent.cfg.env, render=True, n_bins=env_bins(agent.cfg))
    s, _ = env.reset(seed=seed)
    frames, total, done = [Image.fromarray(env.render()).resize((size, size))], 0.0, False
    while not done:
        s, r, term, trunc, _ = env.step(agent.act(s))
        total, done = total + r, term or trunc
        frames.append(Image.fromarray(env.render()).resize((size, size)))
    env.close()
    return frames, total


def font(px: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(panels: list[tuple[str, list[Image.Image]]], cols: int, size: int, every: int, cap_px: int) -> list[Image.Image]:
    """Tile the panels (caption above each); finished episodes hold their last frame."""
    rows = (len(panels) + cols - 1) // cols
    cap_h = cap_px + 8
    n = max(len(f) for _, f in panels)
    f = font(cap_px)
    out = []
    for t in range(0, n, every):
        canvas = Image.new("RGB", (cols * size, rows * (size + cap_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (caption, frames) in enumerate(panels):
            x, y = (i % cols) * size, (i // cols) * (size + cap_h)
            canvas.paste(frames[min(t, len(frames) - 1)], (x, y + cap_h))
            w = draw.textlength(caption, font=f)
            draw.text((x + (size - w) / 2, y + 4), caption, fill="black", font=f)
        out.append(canvas.quantize(colors=64, dither=Image.Dither.NONE))
    return out


def save(frames: list[Image.Image], path: Path, ms: int) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"{path}  {len(frames)} frames  {path.stat().st_size / 1e6:.2f} MB")


def hanging_seed(start: int = 10_000) -> int:
    """First seed >= start whose Pendulum start state hangs near the bottom, nearly still:
    the hardest start, the one that shows the swing-up."""
    env = make_env("pendulum")
    for seed in range(start, start + 1000):
        c, s, d = env.reset(seed=seed)[0]
        if c < -0.95 and abs(d) < 0.5:
            return seed
    return start


def best_method(runs: Path, env: str) -> str:
    finals = {p.parent.name: json.loads(p.read_text())["final"]["mean"] for p in (runs / env).glob("*/metrics.json")}
    return max(finals, key=finals.get)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--assets", default="assets")
    p.add_argument("--pendulum-algo", default="sac")
    p.add_argument("--acrobot-algo", default=None, help="default: best final return in runs/acrobot")
    p.add_argument("--seed", type=int, default=None, help="default: a hanging start for Pendulum, 10000 for Acrobot")
    args = p.parse_args()
    runs, assets = Path(args.runs_dir), Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)

    for env, algo, every in (("pendulum", args.pendulum_algo, 2),
                             ("acrobot", args.acrobot_algo or best_method(runs, "acrobot"), 4)):
        trained = load_agent(runs / env / algo / "model.pt")
        untrained = make_agent(TrainConfig.from_dict({**trained.cfg.to_dict(), "seed": 123}))
        seed = args.seed if args.seed is not None else (hanging_seed() if env == "pendulum" else 10_000)
        f0, r0 = rollout_frames(untrained, seed, 260)
        f1, r1 = rollout_frames(trained, seed, 260)
        frames = compose([(f"untrained {algo}: return {round(r0)}", f0), (f"trained {algo}: return {round(r1)}", f1)],
                         cols=2, size=260, every=every, cap_px=15)
        save(frames, assets / f"before_after_{env}.gif", 40 * every)
        print(f"  {env}: start seed {seed}, untrained {r0:.1f}, trained {algo} {r1:.1f}")

        panels = []
        for a in [a for a, envs in ALGOS.items() if env in envs]:
            path = runs / env / a / "model.pt"
            if path.exists():
                fr, ret = rollout_frames(load_agent(path), seed, 150)
                panels.append((f"{a}: {round(ret)}", fr))
        grid_every = 2 if env == "acrobot" else 4
        save(compose(panels, cols=4, size=150, every=grid_every, cap_px=12), assets / f"grid_{env}.gif", 40 * grid_every)


if __name__ == "__main__":
    main()
