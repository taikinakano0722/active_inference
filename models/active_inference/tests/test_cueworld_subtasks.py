"""cueworld subtasks アダプタの確認。環境の規則そのものは subtasks 側のテストが見る。"""

from __future__ import annotations

import numpy as np

from active_inference.envs.cueworld_subtasks import (
    make_communication_env,
    make_sanity_env,
)
from active_inference.experiments.cueworld_subtasks import main as run_experiment


def _one_hot(idx: int, n: int) -> np.ndarray:
    a = np.zeros(n, dtype=np.float32)
    a[idx] = 1.0
    return a


def test_spaces_and_contract():
    env = make_communication_env(max_steps=5, seed=0)
    assert env.obs_space["obs"].shape == (4,)
    assert env.act_space["action"].discrete and env.act_space["action"].n == 4
    obs, info = env.reset()
    assert set(obs) == {"obs"} and obs["obs"].shape == (4,)
    assert info["is_first"] is True
    # 短期・n_meaningful=1: 行動 0 の次は親 2。one-hot でも int でも受ける。
    obs, reward, done, info = env.step({"action": _one_hot(0, 4)})
    assert info["parent_symbol"] == 2 and info["action"] == 0
    assert reward == 0.0 and done is False
    obs, _, _, info = env.step(0)
    assert info["parent_symbol"] == 2
    for _ in range(3):
        _, _, done, info = env.step(env.sample_action())
    assert done is True and info["truncated"] is True and info["is_last"] is True


def test_sanity_deterministic_action():
    env = make_sanity_env(max_steps=10, seed=3)
    _, info = env.reset()
    a = info["deterministic_action"]
    for _ in range(5):
        obs, _, _, info = env.step(_one_hot(a, env.n_actions))
        assert int(np.argmax(obs["obs"])) == info["deterministic_symbol"]


def test_smoke_train_and_eval(tmp_path=None):
    """ごく短い学習と評価が通り、指標が出ること。"""
    out = "/tmp/_test_cueworld_subtasks.json"
    for extra in (
        ["--env", "communication", "--dependency", "short"],
        ["--env", "communication", "--dependency", "long", "--condition", "yoke", "--n-tapes", "3"],
        ["--env", "sanity", "--action-mode", "candidate"],
    ):
        res = run_experiment(
            extra
            + [
                "--max-steps", "12", "--seq-len", "8", "--rounds", "2",
                "--random-episodes", "4", "--collect-episodes", "2",
                "--train-ratio", "8", "--batch-size", "4",
                "--eval-episodes", "2", "--horizon", "2", "--out", out,
            ]
        )
        for k in ("agent", "random"):
            assert np.isclose(sum(res[k]["action_hist"]), 1.0, atol=1e-3)
            assert 0.0 <= res[k]["frac_determined"] <= 1.0


if __name__ == "__main__":
    test_spaces_and_contract()
    test_sanity_deterministic_action()
    test_smoke_train_and_eval()
    print("test_cueworld_subtasks: ok")
