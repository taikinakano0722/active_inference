"""Point1DEnv の動作確認。"""

from __future__ import annotations

import numpy as np

from active_inference.envs.point1d import Point1DEnv


def test_discrete_reset_step():
    env = Point1DEnv(discrete=True, max_speed=0.1, max_steps=20, seed=0)
    obs, info = env.reset()
    assert set(obs.keys()) == {"position", "goal"}
    assert obs["position"].shape == (1,)
    assert abs(obs["position"][0] - obs["goal"][0]) >= 0.25
    assert info["is_first"] is True

    # 正方向へ何歩か進める。
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    start = float(obs["position"][0])
    for _ in range(3):
        obs, reward, done, info = env.step(action)
        assert 0.0 <= reward <= 1.0
        assert not done
    assert float(obs["position"][0]) > start


def test_continuous_clip_and_one_step_not_enough():
    env = Point1DEnv(discrete=False, max_speed=0.1, max_steps=50, seed=1)
    obs, _ = env.reset()
    gap = abs(float(obs["position"][0]) - float(obs["goal"][0]))
    assert gap > env.max_speed

    # 上限を超える速度はクリップされる。
    obs2, _, _, info = env.step(np.array([10.0], dtype=np.float32))
    assert abs(info["speed"]) <= env.max_speed + 1e-6
    moved = abs(float(obs2["position"][0]) - float(obs["position"][0]))
    assert moved <= env.max_speed + 1e-6


def test_episode_ends():
    env = Point1DEnv(discrete=True, max_steps=5, seed=2)
    env.reset()
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(env.sample_action())
    assert done is True


if __name__ == "__main__":
    test_discrete_reset_step()
    test_continuous_clip_and_one_step_not_enough()
    test_episode_ends()
    print("test_point1d: ok")
