import importlib
import json

from pendulum_rl import compare, play

# the package re-exports the train() / evaluate() functions, so import the modules explicitly
train_mod = importlib.import_module("pendulum_rl.train")
evaluate_mod = importlib.import_module("pendulum_rl.evaluate")
from pendulum_rl.viz import pendulum_grid


def test_cli_train_eval_play_compare(tmp_path, capsys):
    out = str(tmp_path / "runs")
    for algo in ("ddpg", "ppo"):
        train_mod.main(["--env", "pendulum", "--algo", algo, "--total-steps", "400", "--eval-every", "400",
                        "--eval-episodes", "1", "--final-eval-episodes", "1", "--warmup-steps", "100",
                        "--n-steps", "200", "--out-dir", out, "--quiet"])
    ckpt = str(tmp_path / "runs" / "pendulum" / "ddpg" / "model.pt")
    capsys.readouterr()
    evaluate_mod.main([ckpt, "--episodes", "1"])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert res["algo"] == "ddpg" and res["episodes"] == 1
    gif = tmp_path / "ep.gif"
    play.main([ckpt, "--episodes", "1", "--gif", str(gif)])
    assert gif.exists() and gif.stat().st_size > 1000
    assets = tmp_path / "assets"
    compare.main(["--env", "pendulum", "--algos", "ddpg", "ppo", "--runs-dir", out, "--assets", str(assets)])
    assert (assets / "pendulum_curves.png").exists() and (assets / "pendulum_heatmaps.png").exists()
    table = (assets / "pendulum_results.md").read_text()
    assert "`ddpg`" in table and "`ppo`" in table


def test_pendulum_grid_covers_upright_and_hanging():
    theta, theta_dot, obs = pendulum_grid(11)
    assert obs.shape == (121, 3)
    assert theta.min() == -theta.max() and 0.0 in theta and theta_dot.max() == 8
